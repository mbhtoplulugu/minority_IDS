# data_preprocessing.py
# UNSW-NB15 iin tek noktadan, cache'li ve model-agnostik n-ileme
# - XGB davrann bozmaz (ordinal encode + minmax ile [0,1] aral)
# - MLP/Siamese ile ayn Xt_* cache'ini paylar
# - Tekrarlanabilir split + hafif bellek kullanm (float32)

import os, glob, joblib, warnings, re
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import check_random_state

warnings.filterwarnings("ignore")

# ---------- helpers ----------
def _stage(msg: str): print(f"[Stage] {msg}")
def _info(msg: str):  print(f"[Info]  {msg}")

def _as_f32(x: np.ndarray) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32, order="C")
def to_float32(X):
    return _as_f32(X)
def _print_dist(tag: str, y: np.ndarray):
    vc = pd.Series(y).value_counts()
    ratios = (vc / vc.sum()).sort_values(ascending=False)
    s = ", ".join([f"{k}:{p*100:.2f}%" for k,p in ratios.items()])
    print(f"[Dist] {tag}: {s}")

# ---------- config ----------
@dataclass
class DPConfig:
    files_glob: str ="C:/Users/mbhto/source/repos/UNSW-NB15/UNSWNB15_[0-5].csv"
    features_csv: str ="C:/Users/mbhto/source/repos/UNSW-NB15/NUSW-NB15_features.csv"
    target: str = "attack_cat"
    cache_dir: str = "."
    # cache dosya adlar (pipeline ile uyumlu)
    xy_cache: str = "cache_xy.pkl"
    raw_df_cache: str = "raw_df.joblib"
    # split
    valid_size: float = 0.15
    test_size: float  = 0.15
    random_state: int = 42
    # kategorik lekleme davran
    scale_categoricals: bool = True
    use_minmax_for_cat: bool = True
    # saysal stunlar iin scaler
    scale_numeric: bool = True
    # tr/kolon ad hatalarna kar gvenli seim
    strict_feature_list: bool = False
    # float32 cast
    cast_float32: bool = True
    group_key: str|None = None

_CANON = {
    'normal':'normal','generic':'generic','exploits':'exploits','dos':'dos',
    'reconnaissance':'reconnaissance','analysis':'analysis','backdoor':'backdoor',
    'fuzzers':'fuzzers','shellcode':'shellcode','worms':'worms'
}
_ALIASES = {
    ' fuz zers':'fuzzers','fuzzer':'fuzzers','fuzzer ':'fuzzers',
    'backdoors':'backdoor','do s':'dos','dos ':'dos','DoS':'dos',
    'recon':'reconnaissance',' reconnaissance ':'reconnaissance',
    ' shellcode ':'shellcode','shell code':'shellcode',
    'exploit':'exploits','exploit s':'exploits',
    'nan':'normal','none':'normal','':'normal'
}

def _canon_label(s: str) -> str:
    s = (s or 'normal').strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = s.replace('_',' ').replace('-',' ')
    s = _ALIASES.get(s, s)
    if s in ('fuzzer','worm','exploit'): s += 's'
    return _CANON.get(s, s)

def coerce_to_int(df:pd.DataFrame, cols:List[str])->pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def fix_binary_columns(df:pd.DataFrame, cols:List[str])->pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            df[c] = df[c].apply(lambda v: 1 if pd.notna(v) and v>1 else v).fillna(0).astype('Int64')
    return df

def normalize_attack_cat(df:pd.DataFrame, col:str='attack_cat')->pd.DataFrame:
    if col in df.columns:
        s = df[col].astype(str).map(_canon_label)
        s = s.replace({pd.NA:'normal', None:'normal'}).fillna('normal')
        df[col] = s
    return df

def apply_domain_cleanups(df:pd.DataFrame, verbose:bool=True)->pd.DataFrame:
    if verbose: _stage("Domain cleanups")
    df = coerce_to_int(df, ['dsport','sport','ct_ftp_cmd'])
    df = fix_binary_columns(df, ['is_sm_ips_ports','is_ftp_login'])
    df = normalize_attack_cat(df, 'attack_cat')
    return df

# ---------- IO & feature list ----------
def _normalize_colnames(names:List[str])->List[str]:
    return [str(x).strip().replace(' ','').lower() for x in names]
def _load_raw_csvs(files_glob: str,features_csv:str) -> pd.DataFrame:
    _stage("CSV birletirme")
    fns = sorted(glob.glob(files_glob))
    assert len(fns)>0, f"CSV bulunamad: {files_glob}"
    dfs = [pd.read_csv(fp, header=None, low_memory=False) for fp in fns]
    df = pd.concat(dfs, ignore_index=True)
    cols = pd.read_csv(features_csv, encoding='ISO-8859-1')['Name'].tolist()
    cols = _normalize_colnames(cols)
    # Kolon says kontrol
    if df.shape[1] != len(cols):
        _info(f"Kolon uyumsuzluu data={df.shape[1]} vs features={len(cols)}")
        if df.shape[1] > len(cols):
            df = df.iloc[:, :len(cols)]
        else:
            for _ in range(len(cols) - df.shape[1]):
                df[df.shape[1]] = np.nan
    df.columns = cols
    df = apply_domain_cleanups(df, verbose=True)
    return df

# ---------- Tail feature integration ----------
def save_all_features(df: pd.DataFrame, cache_dir: str):
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    all_cols = list(df.drop(columns=['attack_cat','label'], errors='ignore').columns)
    joblib.dump(all_cols, cdir/"all_features.joblib")
    _info(f"Tm kolonlar kaydedildi: {len(all_cols)} kolon")
def _read_feature_list(features_csv: str) -> List[str]:
    """
    zellik isimlerini features.csv'den okur.
    - CSV'de 'Name' veya ilk stun zellik adlarn iersin.
    - Hedef kolon (attack_cat) listede olsa bile training'de karlr.
    """
    fdf = pd.read_csv(features_csv, encoding='cp1252')  # Windows encoding
    if "Name" in fdf.columns:
        cols = fdf["Name"].astype(str).tolist()
    else:
        cols = fdf.iloc[:, 0].astype(str).tolist()
    cols = [c for c in cols if isinstance(c, str) and len(c)]
    return cols

# ---------- core preprocess ----------
def _build_preprocess(df: pd.DataFrame, feature_cols: List[str], cfg: DPConfig):
    """
    ColumnTransformer + Pipeline kur.
    - Kategorik: SimpleImputer(most_frequent)  OrdinalEncoder(unknown=-1)  (MinMaxScaler veya StandardScaler)
    - Saysal   : SimpleImputer(median)  (StandardScaler)
    """
    use_cols = [c for c in feature_cols if c in df.columns and c != cfg.target]
    if cfg.strict_feature_list:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature listesinde olup df'de olmayan kolonlar: {missing}")

    X = df[use_cols].copy()

    # tip kefi
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in use_cols if c not in cat_cols]

    # kategorik pipeline
    cat_pipe_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]
    if cfg.scale_categoricals:
        if cfg.use_minmax_for_cat:
            cat_pipe_steps.append(("scale", MinMaxScaler()))   # XGB ile gvenli
        else:
            # with_mean=False dense iin sorun karmaz; sabit lek
            cat_pipe_steps.append(("scale", StandardScaler(with_mean=True)))
    cat_pipe = SkPipeline(cat_pipe_steps) if cat_cols else "drop"

    # saysal pipeline
    num_pipe_steps = [("impute", SimpleImputer(strategy="median"))]
    if cfg.scale_numeric:
        num_pipe_steps.append(("scale", StandardScaler()))
    num_pipe = SkPipeline(num_pipe_steps) if num_cols else "drop"

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,   # zorla dense
    )

    # k float32'e eviren hafif bir adm ekleyelim
    to_f32 = FunctionTransformer(to_float32, accept_sparse=False)

    pipeline = SkPipeline([
        ("pre", pre),
        ("to_f32", to_f32),
    ])

    meta = {
        "use_cols": use_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }
    return pipeline, meta
def _fit_transform_cache(df: pd.DataFrame, cfg: DPConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    df  split(stratify)  fit(preprocess on train)  transform(train/valid/test)  joblib cache
    """
    rng = check_random_state(cfg.random_state)
    target = cfg.target
    assert target in df.columns, f"Target '{target}' dataframe'de yok"

    # hedef al
    y_all = df[target].astype(str).values
    # zellik listesi
    feature_cols = _read_feature_list(cfg.features_csv)
    from sklearn.model_selection import StratifiedShuffleSplit

    if cfg.group_key and cfg.group_key in df.columns:
        # group_key varsa yine alnr ama stratified blme yaplr
        g = df[cfg.group_key]
        gss1 = StratifiedShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
        idx_trv, idx_te = next(gss1.split(df, y_all))
        df_trv, df_te = df.iloc[idx_trv], df.iloc[idx_te]

        gss2 = StratifiedShuffleSplit(n_splits=1, test_size=cfg.valid_size/(1-cfg.test_size),
                                      random_state=cfg.random_state)
        y_trv = y_all[idx_trv]
        idx_tr, idx_v = next(gss2.split(df_trv, y_trv))
        df_tr, df_v = df_trv.iloc[idx_tr], df_trv.iloc[idx_v]
    else:
        # nce test'i ayr (stratify)
        df_trv, df_te = train_test_split(
            df, test_size=cfg.test_size, stratify=df[target],
            random_state=cfg.random_state, shuffle=True
        )
        # sonra valid'i ayr
        valid_frac_of_trv = cfg.valid_size / (1.0 - cfg.test_size)
        df_tr, df_v = train_test_split(
            df_trv, test_size=valid_frac_of_trv, stratify=df_trv[target],
            random_state=cfg.random_state, shuffle=True
        )

    _print_dist("train", df_tr[target].values)
    _print_dist("valid", df_v[target].values)
    _print_dist("test",  df_te[target].values)

    # preprocess pipeline (yalnz train zerinde fit)
    pp, meta = _build_preprocess(df_tr, feature_cols, cfg)

    X_tr = pp.fit_transform(df_tr)
    X_v  = pp.transform(df_v)
    X_te = pp.transform(df_te)

    if cfg.cast_float32:
        X_tr, X_v, X_te = _as_f32(X_tr), _as_f32(X_v), _as_f32(X_te)

    y_tr = df_tr[target].astype(str).values
    y_v  = df_v[target].astype(str).values
    y_te = df_te[target].astype(str).values

    # LabelEncoder: snf sras pipeline boyunca sabit kalsn
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(df[target].astype(str).values)  # tm veri zerinde fit: snf evreni sabitlenir
    y_tr_enc = le.transform(y_tr)  # (not used outside; ama sanity iin)
    _ = y_tr_enc  # lint sessiz

    # cache yaz
    cdir = Path(cfg.cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    joblib.dump(X_tr, cdir / "Xt_tr.joblib", compress=3)
    joblib.dump(X_v,  cdir / "Xt_v.joblib",  compress=3)
    joblib.dump(X_te, cdir / "Xt_te.joblib", compress=3)
    joblib.dump(y_tr, cdir / "y_tr.joblib",  compress=3)
    joblib.dump(y_v,  cdir / "y_v.joblib",   compress=3)
    joblib.dump(y_te, cdir / "y_te.joblib",  compress=3)

    joblib.dump(le,   cdir / "label_encoder.joblib", compress=3)
    joblib.dump(pp,   cdir / "preprocess.joblib",    compress=3)
    joblib.dump(meta, cdir / "preprocess_meta.joblib", compress=3)

    # xy_cache (opsiyonel, mevcut pipeline ile uyumlu)
    joblib.dump(
        {
            "target": cfg.target,
            "feature_cols": feature_cols,
            "use_cols": meta["use_cols"],
        },
        cdir / cfg.xy_cache,
        compress=3
    )

    return X_tr, y_tr, X_v, y_v, X_te, y_te
'''def _fit_transform_cache(df: pd.DataFrame, cfg: DPConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    df  split(stratify)  fit(preprocess on train)  transform(train/valid/test)  joblib cache
    """
    rng = check_random_state(cfg.random_state)
    target = cfg.target
    assert target in df.columns, f"Target '{target}' dataframe'de yok"

    # hedef al
    y_all = df[target].astype(str).values
    # zellik listesi
    feature_cols = _read_feature_list(cfg.features_csv)
    from sklearn.model_selection import GroupShuffleSplit
    if cfg.group_key and cfg.group_key in df.columns:
        g = df[cfg.group_key]
        gss1 = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
        idx_trv, idx_te = next(gss1.split(df, groups=g))
        df_trv, df_te = df.iloc[idx_trv], df.iloc[idx_te]

        gss2 = GroupShuffleSplit(n_splits=1, test_size=cfg.valid_size/(1-cfg.test_size),
                                 random_state=cfg.random_state)
        g_tr = g.iloc[idx_trv]
        idx_tr, idx_v = next(gss2.split(df_trv, groups=g_tr))
        df_tr, df_v = df_trv.iloc[idx_tr], df_trv.iloc[idx_v]
    else:
        # nce test'i ayr (stratify)
        df_trv, df_te = train_test_split(
            df, test_size=cfg.test_size, stratify=df[target],
            random_state=cfg.random_state, shuffle=True
        )
        # sonra valid'i ayr
        valid_frac_of_trv = cfg.valid_size / (1.0 - cfg.test_size)
        df_tr, df_v = train_test_split(
            df_trv, test_size=valid_frac_of_trv, stratify=df_trv[target],
            random_state=cfg.random_state, shuffle=True
        )

    _print_dist("train", df_tr[target].values)
    _print_dist("valid", df_v[target].values)
    _print_dist("test",  df_te[target].values)

    # preprocess pipeline (yalnz train zerinde fit)
    pp, meta = _build_preprocess(df_tr, feature_cols, cfg)

    X_tr = pp.fit_transform(df_tr)
    X_v  = pp.transform(df_v)
    X_te = pp.transform(df_te)

    if cfg.cast_float32:
        X_tr, X_v, X_te = _as_f32(X_tr), _as_f32(X_v), _as_f32(X_te)

    y_tr = df_tr[target].astype(str).values
    y_v  = df_v[target].astype(str).values
    y_te = df_te[target].astype(str).values

    # LabelEncoder: snf sras pipeline boyunca sabit kalsn
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(df[target].astype(str).values)  # tm veri zerinde fit: snf evreni sabitlenir
    y_tr_enc = le.transform(y_tr)  # (not used outside; ama sanity iin)
    _ = y_tr_enc  # lint sessiz

    # cache yaz
    cdir = Path(cfg.cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    joblib.dump(X_tr, cdir / "Xt_tr.joblib", compress=3)
    joblib.dump(X_v,  cdir / "Xt_v.joblib",  compress=3)
    joblib.dump(X_te, cdir / "Xt_te.joblib", compress=3)
    joblib.dump(y_tr, cdir / "y_tr.joblib",  compress=3)
    joblib.dump(y_v,  cdir / "y_v.joblib",   compress=3)
    joblib.dump(y_te, cdir / "y_te.joblib",  compress=3)

    joblib.dump(le,   cdir / "label_encoder.joblib", compress=3)
    joblib.dump(pp,   cdir / "preprocess.joblib",    compress=3)
    joblib.dump(meta, cdir / "preprocess_meta.joblib", compress=3)

    # xy_cache (opsiyonel, mevcut pipeline ile uyumlu)
    joblib.dump(
        {
            "target": cfg.target,
            "feature_cols": feature_cols,
            "use_cols": meta["use_cols"],
        },
        cdir / cfg.xy_cache,
        compress=3
    )

    return X_tr, y_tr, X_v, y_v, X_te, y_te'''

# ---------- public API ----------
# ensure_preprocessed iinde arlacak
def ensure_preprocessed(cfg: DPConfig):
    cdir = Path(cfg.cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    raw_df_cache = cdir / cfg.raw_df_cache
    if raw_df_cache.exists():
        df = joblib.load(raw_df_cache)
        _info(f"Raw DF cache yklendi: {df.shape}")
    else:
        df = _load_raw_csvs(cfg.files_glob,cfg.features_csv)
        joblib.dump(df, raw_df_cache)
        _info(f"Raw DF cache kaydedildi: {df.shape}")

    # --- Tm kolonlar kaydet (drop_cols dmeden) ---
    save_all_features(df, cfg.cache_dir)
    print("Columns:", df.columns.tolist())

    X = df.drop(columns=[cfg.target,'label'], errors='ignore').copy()
    y = df[cfg.target].copy()

    need_build = False
    for name in ["Xt_tr.joblib", "Xt_v.joblib", "Xt_te.joblib",
                 "y_tr.joblib", "y_v.joblib", "y_te.joblib",
                 "label_encoder.joblib", "preprocess.joblib"]:
        if not (cdir / name).exists():
            need_build = True
            break

    if not need_build:
        _info("Preprocess cache bulundu -> Xt_* yukleniyor")
        Xt_tr = joblib.load(cdir / "Xt_tr.joblib")
        Xt_v  = joblib.load(cdir / "Xt_v.joblib")
        Xt_te = joblib.load(cdir / "Xt_te.joblib")
        y_tr  = joblib.load(cdir / "y_tr.joblib")
        y_v   = joblib.load(cdir / "y_v.joblib")
        y_te  = joblib.load(cdir / "y_te.joblib")
        le    = joblib.load(cdir / "label_encoder.joblib")

        # gvenlik: tip/biim
        Xt_tr, Xt_v, Xt_te = _as_f32(Xt_tr), _as_f32(Xt_v), _as_f32(Xt_te)
        y_tr = np.asarray(y_tr); y_v = np.asarray(y_v); y_te = np.asarray(y_te)
        return Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te, le

    # yoksa hamdan ret
    _stage("Ham veri ykleniyor")
    if (cdir / cfg.raw_df_cache).exists():
        df_all = joblib.load(cdir / cfg.raw_df_cache)
        _info(f"raw_df cache yklendi: shape={df_all.shape}")
    else:
        df_all = _load_raw_csvs(cfg.files_glob)
        joblib.dump(df_all, cdir / cfg.raw_df_cache, compress=3)

    _stage("n-ileme + split + cache")
    Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te = _fit_transform_cache(df_all, cfg)
    le = joblib.load(cdir / "label_encoder.joblib")
    return Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te, le
# ---------- CLI (istee bal) ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="UNSW-NB15 preprocess & cache")
    ap.add_argument("--files-glob", type=str, default="C:/Users/mbhto/source/repos/UNSW-NB15/UNSWNB15_[0-5].csv")
    ap.add_argument("--features-csv", type=str, default="C:/Users/mbhto/source/repos/UNSW-NB15/NUSW-NB15_features.csv")
    ap.add_argument("--cache-dir", type=str, default=".")
    ap.add_argument("--valid-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--no-scale-cat", action="store_true")
    ap.add_argument("--no-minmax-cat", action="store_true")
    ap.add_argument("--no-scale-num", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    cfg = DPConfig(
        files_glob=args.files_glob,
        features_csv=args.features_csv,
        cache_dir=args.cache_dir,
        valid_size=args.valid_size,
        test_size=args.test_size,
        scale_categoricals=(not args.no_scale_cat),
        use_minmax_for_cat=(not args.no_minmax_cat),
        scale_numeric=(not args.no_scale_num),
        strict_feature_list=args.strict,
    )
    ensure_preprocessed(cfg)
