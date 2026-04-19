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

# ---------- Evrensel Kolon Esleme (Dataset-Agnostic) ----------
# CICIDS17 kolonlarini ic standart isimlerimize donusturur.
# UNSW-NB15 zaten bu isimleri kullandigi icin rename islemi hicbir sey degistirmez.
#
# ONEMLI: Sadece feature_engineering_fixed.py'da KULLANILAN kolonlar eslestirilir.
# Semantik olarak yanlis veya duplike eslemeler kaldirilmistir.
#
# feature_engineering_fixed.py'daki kullanim:
#   dur   -> packet_rate, byte_rate, throughput, activity_intensity, tcp_setup_inefficiency
#   spkts -> packet_rate, packet_asymmetry, avg_packet_size, small_packet_ratio, communication_efficiency
#   dpkts -> packet_asymmetry, communication_efficiency, d_loss_ratio
#   sbytes-> byte_rate, payload_asymmetry, avg_packet_size, communication_efficiency, payload_density
#   dbytes-> payload_asymmetry, communication_efficiency
#   dsport-> non_standard_port, scans_common_port, uses_high_port, port_diversity_ratio
#   sttl  -> ttl_asymmetry
#   dttl  -> ttl_asymmetry
#   sjit  -> jitter_asymmetry
#   djit  -> jitter_asymmetry
#   sloss -> s_loss_ratio, loss_asymmetry
#   dloss -> d_loss_ratio, loss_asymmetry
#   tcprtt-> tcp_setup_inefficiency
#   synack-> tcp_ack_syn_ratio
#   ackdat-> tcp_ack_syn_ratio
#   swin/dwin -> (henuz kullanilmiyor ama potansiyel)
#   smean/dmean -> (henuz kullanilmiyor ama semantik dogru)
GLOBAL_COL_MAP = {
    # === Temel Akis Ozellikleri (Feature Engineering icin KRITIK) ===
    'Flow Duration': 'dur',
    'Total Fwd Packets': 'spkts',
    'Total Backward Packets': 'dpkts',
    'Total Length of Fwd Packets': 'sbytes',
    'Total Length of Bwd Packets': 'dbytes',
    'Destination Port': 'dsport',
    # === Paket Boyut Istatistikleri ===
    'Fwd Packet Length Mean': 'smean',
    'Bwd Packet Length Mean': 'dmean',
    # === Hiz / Oran ===
    'Flow Bytes/s': 'rate',
    'Flow Packets/s': 'srate',
    # === Inter-Arrival Time (Jitter kaynaklari) ===
    'Flow IAT Mean': 'sinpkt',
    'Flow IAT Std': 'dinpkt',
    'Fwd IAT Total': 'sjit',
    'Bwd IAT Total': 'djit',
    # === Pencere Boyutlari ===
    'Init_Win_bytes_forward': 'swin',
    'Init_Win_bytes_backward': 'dwin',
    # === Header / TTL (CICIDS'de tam TTL yok, Header Length en yakin proxy) ===
    'Fwd Header Length': 'sttl',
    'Bwd Header Length': 'dttl',
    # === Paket Kayip Proxy (CICIDS'de sloss/dloss yok, min/max packet en yakin proxy) ===
    'Min Packet Length': 'sloss',
    'Max Packet Length': 'dloss',
    # === TCP Zamanlama Proxyleri (CICIDS'de tcprtt/synack/ackdat yok, Active * en yakin proxy) ===
    'Active Mean': 'tcprtt',
    'Active Std': 'synack',
    'Active Max': 'ackdat',
    # === Hedef Degisken ===
    'Label': 'attack_cat',
}

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
    # ===== DUAL-MODE DATASET SECIMI =====
    dataset_mode: str = 'unsw'    # 'unsw' veya 'cicids'
    train_csv: str|None = None    # CICIDS modu icin: cicids_train.csv yolu
    test_csv: str|None = None     # CICIDS modu icin: cicids_test.csv yolu
    exclude_weak: bool = False    # Zayif siniflari veri setinden tamamen cikarir

_CANON = {
    'normal':'normal','generic':'generic','exploits':'exploits','dos':'dos',
    'reconnaissance':'reconnaissance', 'analysis':'analysis', 'backdoor':'backdoor',
    'fuzzers':'fuzzers', 'shellcode':'shellcode', 'worms':'worms',
    'bot':'bot', 'heartbleed':'heartbleed', 'infiltration':'infiltration',
    'portscan':'portscan', 'bruteforce':'bruteforce', 'patator':'bruteforce'
}
_ALIASES = {
    ' fuz zers':'fuzzers','fuzzer':'fuzzers','fuzzer ':'fuzzers',
    'backdoors':'backdoor','do s':'dos','dos ':'dos','DoS':'dos',
    'recon':'reconnaissance',' reconnaissance ':'reconnaissance',
    ' shellcode ':'shellcode','shell code':'shellcode',
    'exploit':'exploits','exploit s':'exploits',
    'sql injection':'exploits','web attack':'exploits',
    'nan':'normal','none':'normal','':'normal'
}

def _canon_label(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    # 1. Lowercase ve bastaki/sondaki bosluklari sil
    s = s.strip().lower()
    # 2. Turkce/Ozel karakter temizligi (sadece a-z, 0-9 ve bosluk kalsin)
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    # 3. Fazla bosluklari teke indir
    s = re.sub(r'\s+', ' ', s).strip()
    # 4. Alt-tire ve tireleri bosluga cevir (zaten 2. adimda siliniyor ama netlik icin)
    s = s.replace('_',' ').replace('-',' ')
    # 5. Alias ve canon mapping
    s = _ALIASES.get(s, s)
    if s in ('fuzzer','worm','exploit'): s += 's'
    # Bilinen sinifsa kanonik adini don, bilinmiyorsa temizlenmis halini birak
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
        _info(f"Normalizing labels in '{col}'...")
        # Adim adim temizle
        df[col] = df[col].astype(str).map(_canon_label)
        df[col] = df[col].replace({pd.NA:'normal', 'nan':'normal', None:'normal'}).fillna('normal')
        
        unique_classes = df[col].unique()
        _info(f"Unique classes found ({len(unique_classes)}): {list(unique_classes)}")
    return df

def apply_domain_cleanups(df:pd.DataFrame, verbose:bool=True)->pd.DataFrame:
    if verbose: _stage("Domain cleanups")
    # 1. Kolon isimlerindeki bosluk ve ozel karakterleri temizle (CICIDS17 icin kritik)
    df.columns = df.columns.str.strip()
    # 2. Evrensel kolon esleme uygula (CICIDS17 -> standart, UNSW icin no-op)
    df.rename(columns=GLOBAL_COL_MAP, inplace=True)
    # 3. Infinity degerlerini NaN yap (CICIDS17 Flow Bytes/s icin)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 4. Sayisal kolon duzeltmeleri (sadece varsa)
    coerce_cols = [c for c in ['dsport','sport','ct_ftp_cmd'] if c in df.columns]
    df = coerce_to_int(df, coerce_cols)
    binary_cols = [c for c in ['is_sm_ips_ports','is_ftp_login'] if c in df.columns]
    df = fix_binary_columns(df, binary_cols)
    # 5. Hedef degisken normalizasyonu
    if 'attack_cat' in df.columns:
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
    # zellik listesi - CICIDS modunda features_csv olmayabilir, CSV kolonlarindan cikar
    if cfg.dataset_mode == 'cicids' or not cfg.features_csv or not Path(cfg.features_csv).exists():
        feature_cols = [c for c in df.columns if c not in (target, 'label', 'Label')]
        _info(f"Feature list auto-detected from DataFrame: {len(feature_cols)} features")
    else:
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

# ---------- CICIDS CSV loader ----------
def _load_cicids_csvs(train_csv: str, test_csv: str) -> pd.DataFrame:
    """CICIDS17 icin hazir temizlenmis CSV'leri yukleyip birlestiren loader.
    cicids_preprocessor.py tarafindan uretilmis cicids_train.csv ve cicids_test.csv
    dosyalarini okur.
    """
    _stage("CICIDS17 CSV yukleniyor (hazir temizlenmis)")
    assert Path(train_csv).exists(), f"CICIDS train CSV bulunamadi: {train_csv}"
    assert Path(test_csv).exists(), f"CICIDS test CSV bulunamadi: {test_csv}"
    
    df_tr = pd.read_csv(train_csv, low_memory=False)
    df_te = pd.read_csv(test_csv, low_memory=False)
    _info(f"CICIDS train: {df_tr.shape}, test: {df_te.shape}")
    
    # Birlestir (split islemi _fit_transform_cache icinde yapilacak)
    df = pd.concat([df_tr, df_te], ignore_index=True)
    
    # Domain temizlikleri uygula (kolon strip, inf->nan, label normalization)
    df = apply_domain_cleanups(df, verbose=True)
    
    _info(f"CICIDS combined: {df.shape}")
    return df

# ---------- public API ----------
# ensure_preprocessed iinde arlacak
def ensure_preprocessed(cfg: DPConfig):
    cdir = Path(cfg.cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    # --- Cache kontrolu ---
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

    # --- Hamdan uretim ---
    if cfg.dataset_mode == 'cicids':
        # ===== CICIDS MODU =====
        _stage("CICIDS17 modu: hazir CSV'lerden yukleniyor")
        train_csv = cfg.train_csv or (str(Path('cicids_train.csv')) if Path('cicids_train.csv').exists() else str(cdir / 'cicids_train.csv'))
        test_csv = cfg.test_csv or (str(Path('cicids_test.csv')) if Path('cicids_test.csv').exists() else str(cdir / 'cicids_test.csv'))
        
        raw_df_cache = cdir / 'raw_df_cicids.joblib'
        if raw_df_cache.exists():
            df_all = joblib.load(raw_df_cache)
            _info(f"CICIDS raw_df cache yuklendi: {df_all.shape}")
        else:
            df_all = _load_cicids_csvs(train_csv, test_csv)
            joblib.dump(df_all, raw_df_cache, compress=3)
            _info(f"CICIDS raw_df cache kaydedildi: {df_all.shape}")
    else:
        # ===== UNSW MODU (mevcut davranis) =====
        _stage("UNSW-NB15 modu: ham CSV'lerden yukleniyor")
        raw_df_cache = cdir / cfg.raw_df_cache
        if raw_df_cache.exists():
            df_all = joblib.load(raw_df_cache)
            _info(f"Raw DF cache yuklendi: {df_all.shape}")
        else:
            df_all = _load_raw_csvs(cfg.files_glob, cfg.features_csv)
            joblib.dump(df_all, raw_df_cache)
            _info(f"Raw DF cache kaydedildi: {df_all.shape}")

    if cfg.exclude_weak:
        exclude_list = ['xss', 'infiltration'] if cfg.dataset_mode == 'cicids' else ['analysis', 'backdoor']
        _stage(f"YAPTIGINIZ SECIM UYGULANIYOR: {exclude_list} siniflari datadan tamamen SILINIYOR!")
        df_len_before = len(df_all)
        df_all = df_all[~df_all[cfg.target].isin(exclude_list)].copy()
        _info(f"Veri boyutu degisti: {df_len_before} -> {len(df_all)}")

    # --- Tum kolonlar kaydet ---
    save_all_features(df_all, cfg.cache_dir)
    _info(f"Columns: {df_all.columns.tolist()}")

    _stage("On-isleme + split + cache")
    Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te = _fit_transform_cache(df_all, cfg)
    le = joblib.load(cdir / "label_encoder.joblib")
    return Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te, le
# ---------- CLI (istee bal) ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Dataset-Agnostic IDS Preprocess & Cache")
    # Dataset mode
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument("--unsw", action="store_true", default=True, help="UNSW-NB15 modu (default)")
    mode_group.add_argument("--cicids", action="store_true", help="CICIDS17 modu")
    # UNSW arguments
    ap.add_argument("--files-glob", type=str, default="C:/Users/mbhto/source/repos/UNSW-NB15/UNSWNB15_[0-5].csv")
    ap.add_argument("--features-csv", type=str, default="C:/Users/mbhto/source/repos/UNSW-NB15/NUSW-NB15_features.csv")
    # CICIDS arguments
    ap.add_argument("--train-csv", type=str, default=None, help="CICIDS train CSV yolu")
    ap.add_argument("--test-csv", type=str, default=None, help="CICIDS test CSV yolu")
    # Common arguments
    ap.add_argument("--cache-dir", type=str, default=".")
    ap.add_argument("--valid-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--no-scale-cat", action="store_true")
    ap.add_argument("--no-minmax-cat", action="store_true")
    ap.add_argument("--no-scale-num", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--exclude-weak-classes", action="store_true")
    args = ap.parse_args()

    dataset_mode = 'cicids' if args.cicids else 'unsw'
    cache_mode_name = dataset_mode + "_excluded" if args.exclude_weak_classes else dataset_mode

    cfg = DPConfig(
        files_glob=args.files_glob,
        features_csv=args.features_csv,
        cache_dir=str(Path(args.cache_dir) / cache_mode_name),
        valid_size=args.valid_size,
        test_size=args.test_size,
        scale_categoricals=(not args.no_scale_cat),
        use_minmax_for_cat=(not args.no_minmax_cat),
        scale_numeric=(not args.no_scale_num),
        strict_feature_list=args.strict,
        dataset_mode=dataset_mode,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        exclude_weak=args.exclude_weak_classes,
    )
    ensure_preprocessed(cfg)
