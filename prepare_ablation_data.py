"""
prepare_ablation_data.py
========================
UNSW-NB15 ve CICIDS14 veri setlerini ham CSV'den okuyarak
5 ablasyon aşaması için diske kaydeder.

Aşamalar:
  stage_A_normalized   — RobustScaler normalize edilmiş ham özellikler
  stage_B_feature_eng  — A + mühendislik özellikleri (row istatistikleri + çarpım)
  stage_C_rus          — B train seti üzerinde RandomUnderSampler
  stage_D_tomek        — C train seti üzerinde TomekLinks
  stage_E_smote        — D train seti üzerinde SMOTE

Kolon uyumsuzluğu (column mismatch) önleme stratejisi:
  - Tüm veri setleri için ColumnTransformer pipeline'ı train'de fit edilir.
  - Test seti aynı pipeline ile transform edilir.
  - Çıktı daima numpy float32 array'i — pandas kolon isimlerinden bağımsız.
  - Feature isimleri meta.json içinde saklanır (opsiyonel referans).
  - C/D/E aşamalarında test seti her zaman B aşamasından alınır.

Kullanım:
  python prepare_ablation_data.py --dataset all --output-dir ablation_data
  python prepare_ablation_data.py --dataset cicids14
  python prepare_ablation_data.py --dataset unsw --unsw-glob "C:/data/UNSWNB15_*.csv"
"""

import re
import glob
import json
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent

# ─── Sabitler ────────────────────────────────────────────────────────────────

UNSW_FILES_GLOB    = "C:/Users/mbhto/source/repos/UNSW-NB15/UNSWNB15_[0-5].csv"
UNSW_FEATURES_CSV  = "C:/Users/mbhto/source/repos/UNSW-NB15/NUSW-NB15_features.csv"

CICIDS14_TRAIN_CSV = BASE / "cicids14_train.csv"
CICIDS14_TEST_CSV  = BASE / "cicids14_test.csv"

# UNSW'de attack_cat → kanonik isim eşlemesi (data_preprocessing.py ile birebir)
_UNSW_CANON = {
    'normal':'normal','generic':'generic','exploits':'exploits','dos':'dos',
    'reconnaissance':'reconnaissance', 'analysis':'analysis', 'backdoor':'backdoor',
    'fuzzers':'fuzzers', 'shellcode':'shellcode', 'worms':'worms',
}
_UNSW_ALIASES = {
    'fuzzer':'fuzzers','fuzzer ':'fuzzers',' fuz zers':'fuzzers',
    'backdoors':'backdoor','do s':'dos','dos ':'dos',
    'recon':'reconnaissance',' reconnaissance ':'reconnaissance',
    ' shellcode ':'shellcode','shell code':'shellcode',
    'exploit':'exploits','exploit s':'exploits',
    'nan':'normal','none':'normal','':'normal',
}

# CICIDS17 → 14 sınıflı kanonik isim eşlemesi (cicids_preprocessor_14.py ile birebir)
CICIDS14_LABEL_MAP = {
    'BENIGN': 'normal',
    'Bot': 'bot',
    'DDoS': 'ddos',
    'DoS Hulk': 'dos_hulk',
    'DoS GoldenEye': 'dos_goldeneye',
    'DoS slowloris': 'dos_slowloris',
    'DoS Slowhttptest': 'dos_slowhttptest',
    'Heartbleed': 'heartbleed',
    'Infiltration': 'infiltration',
    'PortScan': 'portscan',
    'FTP-Patator': 'ftp_patator',
    'SSH-Patator': 'ssh_patator',
}

# ─── Yardımcılar ──────────────────────────────────────────────────────────────

def _info(m):  print(f"[INFO]  {m}")
def _stage(m): print(f"\n{'='*60}\n[STAGE] {m}\n{'='*60}")


def _canon_unsw_label(s) -> str:
    if not isinstance(s, str): s = str(s)
    s = s.strip().lower()
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace('_', ' ').replace('-', ' ')
    s = _UNSW_ALIASES.get(s, s)
    if s in ('fuzzer', 'worm', 'exploit'): s += 's'
    return _UNSW_CANON.get(s, s)


def _canon_cicids14_label(s) -> str:
    if not isinstance(s, str): s = str(s)
    s = s.strip()
    if s in CICIDS14_LABEL_MAP:
        return CICIDS14_LABEL_MAP[s]
    sl = s.lower()
    if 'web attack' in sl and 'brute' in sl:  return 'web_attack_brute'
    if 'web attack' in sl and 'xss'   in sl:  return 'web_attack_xss'
    if 'web attack' in sl and 'sql'   in sl:  return 'web_attack_sql'
    cleaned = re.sub(r'[^a-zA-Z0-9 ]', ' ', s).strip().lower()
    cleaned = re.sub(r'\s+', '_', cleaned)
    if cleaned in ('benign', 'normal', ''): return 'normal'
    return cleaned


def _safe_feat_names(names):
    """LGBM/XGBoost JSON için güvenli özellik isimleri üret."""
    result = []
    for f in names:
        f = str(f).replace(' ', '_').replace(':', '_')
        f = f.replace('[', '_').replace(']', '_')
        f = f.replace('{', '_').replace('}', '_')
        f = f.replace('"', '_').replace("'", '_')
        result.append(f)
    return result


# ─── ColumnTransformer Pipeline ───────────────────────────────────────────────

def build_ct_pipeline(df_train: pd.DataFrame, feature_cols: list):
    """
    Train setinde fit'lenen ColumnTransformer döndürür.
    Kategorik → OrdinalEncoder + MinMaxScaler
    Sayısal   → SimpleImputer(median) + RobustScaler
    Çıktı daima yoğun float32 numpy array.
    """
    X = df_train[feature_cols]
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols  = [c for c in feature_cols if c not in cat_cols]

    num_pipe = SkPipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  RobustScaler()),
    ])
    cat_pipe = SkPipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ord",    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ("scale",  MinMaxScaler()),
    ])

    transformers = []
    if num_cols: transformers.append(("num", num_pipe, num_cols))
    if cat_cols: transformers.append(("cat", cat_pipe, cat_cols))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )
    ct.fit(df_train[feature_cols])

    # Sıralı özellik isimleri (num önce, sonra cat — CT'nin çıktısıyla birebir)
    ordered_names = num_cols + cat_cols
    return ct, ordered_names


def ct_transform(ct, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Verilen DataFrame'i mevcut CT ile transform edip float32 array döndürür."""
    X = df[feature_cols].copy()
    out = ct.transform(X)
    if hasattr(out, "toarray"):
        out = out.toarray()
    return np.asarray(out, dtype=np.float32)


# ─── Mühendislik Özellikleri ──────────────────────────────────────────────────

def add_engineered_features(X_norm: np.ndarray, feat_names: list,
                             X_train_ref: np.ndarray = None) -> tuple:
    """
    feature_engineering_fixed.py ile birebir aynı özellikleri üretir.
    Girdi: normalize edilmiş numpy array + orijinal özellik isimleri listesi.
    Çıktı: (X_enhanced: np.ndarray[float32], new_feat_names: list[str])

    Üretilen ek özellikler (feature_engineering_fixed.py ile uyumlu):
      Grup A — İstatistiksel anomali göstergeleri:
        zscore_count, extreme_count
      Grup A' — Moment istatistikleri:
        row_mean, row_std, row_skew, row_kurt
      Grup B — Anomali skoru:
        isolation_forest_score   (sadece train ref verilirse)
      Grup C — Alan bilgisi özellikleri (koşullu, ~21 özellik):
        communication_efficiency, activity_intensity, is_long_duration,
        uses_non_standard_port, packet_rate, byte_rate, payload_asymmetry,
        packet_asymmetry, avg_packet_size, is_very_short_duration,
        src_port_entropy, scans_common_port, uses_small_packets,
        payload_density, shellcode_size_range, uses_high_port,
        moderate_throughput, s_loss_ratio, d_loss_ratio, loss_asymmetry,
        ttl_asymmetry, tcp_setup_inefficiency, tcp_ack_syn_ratio,
        jitter_asymmetry

    X_train_ref: IsolationForest'i fit etmek için kullanılan referans veri.
        None ise X_norm üzerinde fit edilir.
    """
    from scipy import stats as _stats
    from sklearn.ensemble import IsolationForest

    extra = []
    extra_names = []

    # ── Grup A: İstatistiksel Anomali Göstergeleri ────────────────────────────
    # zscore_count: her satırda |z| > 3 olan sütun sayısı
    zscores = np.abs(_stats.zscore(X_norm, axis=0))
    zscore_count = (zscores > 3).sum(axis=1).astype(np.float32)
    extra.append(zscore_count.reshape(-1, 1))
    extra_names.append("zscore_count")

    # extreme_count: %1 / %99 dışında kalan sütun sayısı
    p1  = np.percentile(X_norm, 1,  axis=0)
    p99 = np.percentile(X_norm, 99, axis=0)
    extreme_count = ((X_norm <= p1) | (X_norm >= p99)).sum(axis=1).astype(np.float32)
    extra.append(extreme_count.reshape(-1, 1))
    extra_names.append("extreme_count")

    # ── Grup A': Moment İstatistikleri ───────────────────────────────────────
    extra.append(X_norm.mean(axis=1, keepdims=True).astype(np.float32))
    extra_names.append("row_mean")
    extra.append(X_norm.std(axis=1, keepdims=True).astype(np.float32))
    extra_names.append("row_std")
    extra.append(_stats.skew(X_norm, axis=1).reshape(-1, 1).astype(np.float32))
    extra_names.append("row_skew")
    extra.append(_stats.kurtosis(X_norm, axis=1).reshape(-1, 1).astype(np.float32))
    extra_names.append("row_kurt")

    # ── Grup B: Isolation Forest Anomali Skoru ────────────────────────────────
    try:
        ref = X_train_ref if X_train_ref is not None else X_norm
        # Büyük veri setlerinde subsample ile fit et (hız için)
        n_ref = min(len(ref), 50_000)
        rng = np.random.default_rng(42)
        idx_ref = rng.choice(len(ref), n_ref, replace=False)
        iso = IsolationForest(random_state=42, n_jobs=-1, contamination=0.1)
        iso.fit(ref[idx_ref])
        iso_scores = iso.decision_function(X_norm).reshape(-1, 1).astype(np.float32)
        extra.append(iso_scores)
        extra_names.append("isolation_forest_score")
    except Exception as e:
        _info(f"  IsolationForest atlandı: {e}")

    # ── Grup C: Alan Bilgisi Özellikleri ─────────────────────────────────────
    # feature_engineering_fixed.create_attack_specific_features ile birebir aynı
    df = pd.DataFrame(X_norm, columns=feat_names)

    def _add(arr, name):
        v = np.asarray(arr, dtype=np.float32).flatten()
        extra.append(v.reshape(-1, 1))
        extra_names.append(name)

    EPS = 1e-10

    # 1. Communication efficiency
    if all(c in df.columns for c in ['sbytes', 'dbytes', 'spkts', 'dpkts']):
        _add((df['sbytes'] + df['dbytes']) / (df['spkts'] + df['dpkts'] + EPS),
             'communication_efficiency')

    # 2. Activity intensity
    if all(c in df.columns for c in ['spkts', 'dpkts', 'dur']):
        _add((df['spkts'] + df['dpkts']) / (df['dur'] + EPS), 'activity_intensity')

    # 3. Is long duration
    if 'dur' in df.columns:
        thr = df['dur'].quantile(0.9)
        _add((df['dur'] > thr).astype(float), 'is_long_duration')

    # 4. Non-standard port
    if 'dsport' in df.columns:
        std_ports = [80, 443, 22, 21, 25, 53, 110, 143, 993, 995]
        _add((~df['dsport'].isin(std_ports)).astype(float), 'uses_non_standard_port')

    # 5. Packet rate
    if all(c in df.columns for c in ['spkts', 'dur']):
        _add(df['spkts'] / (df['dur'] + EPS), 'packet_rate')

    # 6. Byte rate
    if all(c in df.columns for c in ['sbytes', 'dur']):
        _add(df['sbytes'] / (df['dur'] + EPS), 'byte_rate')

    # 7. Payload asymmetry
    if all(c in df.columns for c in ['sbytes', 'dbytes']):
        _add(np.abs(df['sbytes'] - df['dbytes']) / (df['sbytes'] + df['dbytes'] + EPS),
             'payload_asymmetry')

    # 8. Packet asymmetry
    if all(c in df.columns for c in ['spkts', 'dpkts']):
        _add(np.abs(df['spkts'] - df['dpkts']) / (df['spkts'] + df['dpkts'] + EPS),
             'packet_asymmetry')

    # 9. Avg packet size
    if all(c in df.columns for c in ['sbytes', 'spkts']):
        _add(df['sbytes'] / (df['spkts'] + EPS), 'avg_packet_size')

    # 10. Very short duration
    if 'dur' in df.columns:
        thr_short = df['dur'].quantile(0.1)
        _add((df['dur'] <= thr_short).astype(float), 'is_very_short_duration')

    # 11. Src port entropy (srcip gerektirir — yoksa atla)
    if 'sport' in df.columns and 'srcip' in df.columns:
        ent = (df.groupby('srcip')['sport'].transform('nunique') /
               (df.groupby('srcip')['sport'].transform('count') + EPS))
        _add(ent, 'src_port_entropy')

    # 12. Scans common port
    if 'dsport' in df.columns:
        scan_ports = [21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 993, 995, 1433, 3389]
        _add(df['dsport'].isin(scan_ports).astype(float), 'scans_common_port')

    # 13. Small packet ratio
    if all(c in df.columns for c in ['sbytes', 'spkts']):
        _add((df['sbytes'] / (df['spkts'] + EPS) < 100).astype(float), 'uses_small_packets')

    # 14. Payload density
    if all(c in df.columns for c in ['sbytes', 'spkts']):
        _add(df['sbytes'] / (df['spkts'] + EPS), 'payload_density')

    # 15. Shellcode size range
    if 'sbytes' in df.columns:
        _add(((df['sbytes'] >= 100) & (df['sbytes'] <= 2000)).astype(float), 'shellcode_size_range')

    # 16. High port usage
    if 'dsport' in df.columns:
        _add((df['dsport'] > 1024).astype(float), 'uses_high_port')

    # 17. Moderate throughput
    if all(c in df.columns for c in ['dur', 'sbytes']):
        thr_val = df['sbytes'] / (df['dur'] + EPS)
        p25, p75 = np.percentile(thr_val, 25), np.percentile(thr_val, 75)
        _add(((thr_val > p25) & (thr_val < p75)).astype(float), 'moderate_throughput')

    # 18. Loss ratios & asymmetry
    if all(c in df.columns for c in ['sloss', 'spkts', 'dloss', 'dpkts']):
        s_loss = df['sloss'] / (df['spkts'] + EPS)
        d_loss = df['dloss'] / (df['dpkts'] + EPS)
        _add(s_loss, 's_loss_ratio')
        _add(d_loss, 'd_loss_ratio')
        _add(np.abs(s_loss - d_loss), 'loss_asymmetry')

    # 19. TTL asymmetry
    if all(c in df.columns for c in ['sttl', 'dttl']):
        _add(np.abs(df['sttl'] - df['dttl']) / (df['sttl'] + df['dttl'] + EPS),
             'ttl_asymmetry')

    # 20. TCP setup inefficiency
    if all(c in df.columns for c in ['tcprtt', 'dur']):
        _add(df['tcprtt'] / (df['dur'] + EPS), 'tcp_setup_inefficiency')

    # 21. TCP ack-syn ratio
    if all(c in df.columns for c in ['ackdat', 'synack']):
        _add(df['ackdat'] / (df['synack'] + EPS), 'tcp_ack_syn_ratio')

    # 22. Jitter asymmetry
    if all(c in df.columns for c in ['sjit', 'djit']):
        _add(np.abs(df['sjit'] - df['djit']) / (df['sjit'] + df['djit'] + EPS),
             'jitter_asymmetry')

    # ── Birleştir ────────────────────────────────────────────────────────────
    X_enh = np.hstack([X_norm.astype(np.float32)] + extra)
    all_names = feat_names + extra_names

    added = len(extra_names)
    _info(f"  Feature Engineering: {X_norm.shape[1]} → {X_enh.shape[1]} özellik "
          f"(+{added} eklendi: stat={6 + int('isolation_forest_score' in extra_names)}, "
          f"alan={added - 6 - int('isolation_forest_score' in extra_names)})")

    return X_enh.astype(np.float32), all_names


# ─── Stage Kaydetme ──────────────────────────────────────────────────────────

def save_stage(stage_dir: Path, ds_name: str, stage_name: str, desc: str,
               X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray,  y_test:  np.ndarray,
               le: LabelEncoder, feat_names: list):
    """
    Bir ablasyon aşamasının verilerini diske yazar.
    Tüm X'ler numpy float32 array — kolon uyumsuzluğu imkânsız.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Boyut eşitliği garantisi
    assert X_train.shape[1] == X_test.shape[1], (
        f"[{stage_name}] Kolon uyumsuzluğu: train={X_train.shape[1]}, test={X_test.shape[1]}"
    )

    safe_names = _safe_feat_names(feat_names)

    joblib.dump(X_train.astype(np.float32),  stage_dir / "X_train.joblib",       compress=3)
    joblib.dump(X_test.astype(np.float32),   stage_dir / "X_test.joblib",        compress=3)
    joblib.dump(y_train,                     stage_dir / "y_train.joblib",       compress=3)
    joblib.dump(y_test,                      stage_dir / "y_test.joblib",        compress=3)
    joblib.dump(le,                          stage_dir / "label_encoder.joblib")

    utr, ctr = np.unique(y_train, return_counts=True)
    ute, cte = np.unique(y_test,  return_counts=True)

    meta = {
        "dataset":    ds_name,
        "stage":      stage_name,
        "description": desc,
        "n_train":    int(len(X_train)),
        "n_test":     int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "n_classes":  int(len(le.classes_)),
        "classes":    list(le.classes_.astype(str)),
        "feature_names": safe_names,
        "class_dist_train": {str(le.classes_[k]): int(v) for k, v in zip(utr, ctr)},
        "class_dist_test":  {str(le.classes_[k]): int(v) for k, v in zip(ute, cte)},
    }
    with open(stage_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    _info(f"  ✓ {stage_name}: train={X_train.shape}, test={X_test.shape} → {stage_dir}")


# ─── 5 Aşama Üretici ─────────────────────────────────────────────────────────

def run_stages(X_tr_norm: np.ndarray, y_tr: np.ndarray,
               X_te_norm: np.ndarray, y_te: np.ndarray,
               le: LabelEncoder, feat_names: list,
               ds_name: str, out_path: Path,
               n_select: int = 75,
               strategy: str = "full"):
    """
    Normalize edilmiş train/test array'lerinden 5 ablasyon aşaması üretir.

    n_select : Stage B SelectKBest k değeri.
    strategy : Resampling stratejisi.
        "full"      — RUS → Tomek(not_minority) → SMOTE  [mevcut davranış]
        "rus_only"  — RUS, Tomek yok, SMOTE yok
                      Stage D = Stage C kopyası, Stage E = Stage C kopyası
        "rus_smote" — RUS → Tomek(majority_only, min_cls>=50) → SMOTE
                      Tomek sadece majority sınıfını temizler,
                      <50 örnekli sınıf varsa Tomek tamamen atlanır
    """
    _info(f"[{ds_name}] Strateji: {strategy}")
    n_orig = X_tr_norm.shape[1]

    # ── Aşama A: Normalized ──────────────────────────────────────────────────
    _info(f"[{ds_name}] Aşama A: Normalized")
    save_stage(out_path / "stage_A_normalized",
               ds_name, "stage_A_normalized",
               "RobustScaler ile normalize ham özellikler",
               X_tr_norm, y_tr, X_te_norm, y_te, le, feat_names)

    # ── Aşama B: Feature Engineering + SelectKBest ───────────────────────────
    _info(f"[{ds_name}] Aşama B: Feature Engineering")
    X_tr_enh, enh_names = add_engineered_features(X_tr_norm, feat_names, X_train_ref=X_tr_norm)
    X_te_enh, _         = add_engineered_features(X_te_norm, feat_names, X_train_ref=X_tr_norm)
    assert X_tr_enh.shape[1] == X_te_enh.shape[1], "B: Kolon uyumsuzluğu!"

    pool_size = X_tr_enh.shape[1]
    k = min(n_select, pool_size)
    if k < pool_size:
        from sklearn.feature_selection import SelectKBest, f_classif
        _info(f"  SelectKBest: {pool_size} → {k} özellik (f_classif, train fit)")
        selector = SelectKBest(f_classif, k=k)
        X_tr_enh = selector.fit_transform(X_tr_enh, y_tr).astype(np.float32)
        X_te_enh = selector.transform(X_te_enh).astype(np.float32)
        mask = selector.get_support()
        enh_names = [n for n, m in zip(enh_names, mask) if m]
    else:
        _info(f"  SelectKBest atlandı: havuz ({pool_size}) ≤ hedef ({k}), tümü korunuyor")

    save_stage(out_path / "stage_B_feature_eng",
               ds_name, "stage_B_feature_eng",
               f"Normalize + Mühendislik + SelectKBest(k={k})",
               X_tr_enh, y_tr, X_te_enh, y_te, le, enh_names)

    X_te_final = X_te_enh.copy()

    # ── Aşama C: RUS (tüm stratejilerde aynı) ────────────────────────────────
    _info(f"[{ds_name}] Aşama C: Random UnderSampler")
    uniq, cnts = np.unique(y_tr, return_counts=True)
    strat_rus = {c: min(int(cnt), 200_000) for c, cnt in zip(uniq, cnts)}
    if all(cnt <= 200_000 for cnt in cnts):
        max_cnt = int(max(cnts))
        cap = max(max_cnt // 3, int(min(cnts)))
        strat_rus = {c: min(int(cnt), cap) for c, cnt in zip(uniq, cnts)}
        _info(f"  RUS: dinamik cap={cap}")
    rus = RandomUnderSampler(sampling_strategy=strat_rus, random_state=42)
    X_tr_rus, y_tr_rus = rus.fit_resample(X_tr_enh, y_tr)

    save_stage(out_path / "stage_C_rus",
               ds_name, "stage_C_rus", "B + RandomUnderSampler",
               X_tr_rus, y_tr_rus, X_te_final, y_te, le, enh_names)

    # ── Aşama D: Tomek (stratejiye göre) ─────────────────────────────────────
    _info(f"[{ds_name}] Aşama D: Tomek ({strategy})")
    vc_d = pd.Series(y_tr_rus).value_counts()
    min_cls_d = int(vc_d.min())

    if strategy == "rus_only":
        # Tomek yok — C'yi kopyala
        _info(f"  Tomek atlandı (strateji=rus_only), C kopyalanıyor")
        X_tr_tomek, y_tr_tomek = X_tr_rus.copy(), y_tr_rus.copy()
        tomek_desc = "C kopyası (rus_only stratejisi)"

    elif strategy == "rus_smote":
        # Tomek: sadece majority sınıfı temizle, <50 örnekli sınıf varsa atla
        TOMEK_MIN_CLS = 50
        if min_cls_d < TOMEK_MIN_CLS:
            _info(f"  Tomek atlandı (min sınıf={min_cls_d} < {TOMEK_MIN_CLS}), C kopyalanıyor")
            X_tr_tomek, y_tr_tomek = X_tr_rus.copy(), y_tr_rus.copy()
            tomek_desc = f"C kopyası (min_cls={min_cls_d} < {TOMEK_MIN_CLS})"
        else:
            tl = TomekLinks(sampling_strategy="majority", n_jobs=-1)
            X_tr_tomek, y_tr_tomek = tl.fit_resample(X_tr_rus, y_tr_rus)
            tomek_desc = "C + Tomek(majority only)"
            _info(f"  Tomek(majority): {len(y_tr_rus):,} → {len(y_tr_tomek):,}")

    else:  # "full"
        if min_cls_d < 20:
            _info(f"  TomekLinks atlandı (min sınıf={min_cls_d} < 20), C kopyalanıyor")
            X_tr_tomek, y_tr_tomek = X_tr_rus.copy(), y_tr_rus.copy()
            tomek_desc = "C kopyası (min_cls<20)"
        else:
            tl = TomekLinks(sampling_strategy="not minority", n_jobs=-1)
            X_tr_tomek, y_tr_tomek = tl.fit_resample(X_tr_rus, y_tr_rus)
            tomek_desc = "C + Tomek(not_minority)"

    save_stage(out_path / "stage_D_tomek",
               ds_name, "stage_D_tomek", tomek_desc,
               X_tr_tomek, y_tr_tomek, X_te_final, y_te, le, enh_names)

    # ── Aşama E: SMOTE (stratejiye göre) ─────────────────────────────────────
    _info(f"[{ds_name}] Aşama E: SMOTE ({strategy})")
    vc_e = pd.Series(y_tr_tomek).value_counts()
    median_cnt = int(vc_e.median())
    min_cnt_e  = int(vc_e.min())

    if strategy == "rus_only":
        # SMOTE yok — C'yi kopyala
        _info(f"  SMOTE atlandı (strateji=rus_only), C kopyalanıyor")
        X_tr_smote, y_tr_smote = X_tr_rus.copy(), y_tr_rus.copy()
        smote_desc = "C kopyası (rus_only stratejisi)"

    else:  # "full" veya "rus_smote"
        over_dict = {}
        for c, cnt in vc_e.items():
            if cnt < median_cnt:
                over_dict[c] = min(int(cnt) * 2, median_cnt)

        if not over_dict:
            _info("  SMOTE atlandı (tüm sınıflar dengeli), D kopyalanıyor")
            X_tr_smote, y_tr_smote = X_tr_tomek.copy(), y_tr_tomek.copy()
            smote_desc = "D kopyası (dengeli)"
        elif min_cnt_e < 2:
            _info(f"  SMOTE atlandı (min sınıf={min_cnt_e} < 2), D kopyalanıyor")
            X_tr_smote, y_tr_smote = X_tr_tomek.copy(), y_tr_tomek.copy()
            smote_desc = "D kopyası (min_cls<2)"
        else:
            k_sm = max(1, min(5, min_cnt_e - 1))
            smote = SMOTE(sampling_strategy=over_dict, k_neighbors=k_sm, random_state=42)
            X_tr_smote, y_tr_smote = smote.fit_resample(X_tr_tomek, y_tr_tomek)
            smote_desc = f"D + SMOTE(k={k_sm})"
            _info(f"  SMOTE: {len(y_tr_tomek):,} → {len(y_tr_smote):,}")

    save_stage(out_path / "stage_E_smote",
               ds_name, "stage_E_smote", smote_desc,
               X_tr_smote, y_tr_smote, X_te_final, y_te, le, enh_names)

    _info(f"[{ds_name}] Tüm aşamalar tamamlandı ({strategy}) → {out_path}")


# ─── UNSW-NB15 Hazırlayıcı ───────────────────────────────────────────────────

def prepare_unsw(out_dir: Path, files_glob: str = UNSW_FILES_GLOB,
                 features_csv: str = UNSW_FEATURES_CSV,
                 random_state: int = 42, test_size: float = 0.20,
                 n_select: int = 75, strategy: str = "full"):
    _stage("UNSW-NB15 HAZIRLANIYOR")
    folder_name = "unsw" if strategy == "full" else f"unsw_{strategy}"
    out_path = out_dir / folder_name

    # ── 1. CSV'leri bul ve birleştir ─────────────────────────────────────────
    csv_files = sorted(glob.glob(files_glob))
    if not csv_files:
        _info(f"HATA: '{files_glob}' ile eşleşen dosya bulunamadı. --unsw-glob parametresini kontrol et.")
        return

    _info(f"{len(csv_files)} dosya yükleniyor: {[Path(f).name for f in csv_files]}")

    # Kolon isimlerini features.csv'den oku
    feat_csv_path = Path(features_csv)
    if feat_csv_path.exists():
        feat_df = pd.read_csv(feat_csv_path, encoding="latin1")
        # features.csv'deki kolon isim sütunu: 'Name' veya ilk sütun
        name_col = "Name" if "Name" in feat_df.columns else feat_df.columns[0]
        col_names = feat_df[name_col].str.strip().tolist()
        _info(f"Features.csv'den {len(col_names)} kolon ismi okundu")
    else:
        col_names = None
        _info("Features.csv bulunamadı — CSV başlıkları kullanılacak")

    frames = []
    for f in csv_files:
        try:
            if col_names:
                df_chunk = pd.read_csv(f, header=0, names=col_names[:49], low_memory=False)
            else:
                df_chunk = pd.read_csv(f, low_memory=False)
            frames.append(df_chunk)
        except Exception as e:
            _info(f"  {f} yüklenirken hata: {e}")

    df = pd.concat(frames, ignore_index=True)
    _info(f"Toplam {len(df)} satır yüklendi, {df.shape[1]} sütun")

    # ── 2. Temizlik ───────────────────────────────────────────────────────────
    df.columns = df.columns.str.strip().str.lower()

    # Hedef sütun
    target_col = "attack_cat"
    if target_col not in df.columns:
        # Olası alternatif isimler
        for alt in ["label", "class", "category"]:
            if alt in df.columns:
                df.rename(columns={alt: target_col}, inplace=True)
                break

    if target_col not in df.columns:
        _info(f"HATA: '{target_col}' sütunu bulunamadı. Mevcut: {list(df.columns)[:10]}")
        return

    # Gereksiz sütunları at
    for drop_col in ["id", "label"]:
        if drop_col in df.columns:
            df.drop(columns=[drop_col], inplace=True)

    # Infinity / NaN temizliği
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Sayısal dönüşüm gerektiren kategorik sütunlar
    for c in ["dsport", "sport", "ct_ftp_cmd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in ["is_sm_ips_ports", "is_ftp_login"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            df[c] = df[c].apply(lambda v: 1 if pd.notna(v) and v > 1 else v).fillna(0)

    # Etiket kanonizasyonu
    df[target_col] = df[target_col].astype(str).apply(_canon_unsw_label)
    df = df[df[target_col].notna() & (df[target_col] != "")]
    _info(f"UNSW sınıflar: {sorted(df[target_col].unique())}")

    # ── 3. Feature / Label ayır ───────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c != target_col]
    X_df   = df[feature_cols].copy()
    y_raw  = df[target_col].copy()

    # ── 4. Label encode ───────────────────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw.astype(str))
    _info(f"UNSW: {len(le.classes_)} sınıf — {list(le.classes_)}")

    # ── 5. Stratified split ───────────────────────────────────────────────────
    X_tr_df, X_te_df, y_tr, y_te = train_test_split(
        X_df, y_enc, test_size=test_size,
        stratify=y_enc, random_state=random_state
    )
    _info(f"Split: train={len(X_tr_df)}, test={len(X_te_df)}")

    # ── 6. ColumnTransformer pipeline (train'de fit, test'e transform) ────────
    ct, ordered_names = build_ct_pipeline(X_tr_df, feature_cols)
    X_tr_norm = ct_transform(ct, X_tr_df, feature_cols)
    X_te_norm = ct_transform(ct, X_te_df, feature_cols)
    _info(f"Normalize boyut: {X_tr_norm.shape}")

    # ── 7. 5 Aşama üret ──────────────────────────────────────────────────────
    run_stages(X_tr_norm, y_tr, X_te_norm, y_te, le, ordered_names, "unsw", out_path,
               n_select=n_select, strategy=strategy)


# ─── CICIDS14 Hazırlayıcı ─────────────────────────────────────────────────────

def prepare_cicids14(out_dir: Path,
                     train_csv: Path = CICIDS14_TRAIN_CSV,
                     test_csv: Path  = CICIDS14_TEST_CSV,
                     n_select: int = 75, strategy: str = "full"):
    _stage("CICIDS17 (14 Sınıf) HAZIRLANIYOR")
    folder_name = "cicids14" if strategy == "full" else f"cicids14_{strategy}"
    out_path = out_dir / folder_name

    # Dosya yolu çözümle (birden fazla olası konum)
    def _find(default: Path, name: str) -> Path:
        candidates = [
            default,
            BASE / name,
            BASE / "cicids14" / name,
            BASE / "CICIDS17" / name,
        ]
        for p in candidates:
            if Path(p).exists():
                return Path(p)
        return default  # bulunamazsa hata verecek

    tr_path = _find(train_csv, "cicids14_train.csv")
    te_path = _find(test_csv,  "cicids14_test.csv")

    if not tr_path.exists():
        _info(f"HATA: cicids14_train.csv bulunamadı (denenen: {tr_path})")
        _info("  --cicids-train / --cicids-test parametreleriyle yolu belirt.")
        return

    _info(f"Train: {tr_path}")
    _info(f"Test : {te_path}")

    # ── 1. Yükle ─────────────────────────────────────────────────────────────
    df_tr = pd.read_csv(tr_path, low_memory=False)
    df_te = pd.read_csv(te_path, low_memory=False)

    # ── 2. Temizlik + Rename + Proxy türetme ─────────────────────────────────
    # GLOBAL_COL_MAP: CICIDS17 kolon adlarını UNSW standardına çevirir.
    # Hem boşluklu (orijinal) hem alt çizgili (preprocessed) formları kapsar.
    GLOBAL_COL_MAP = {
        # ── Boşluklu orijinal isimleri ──
        'Destination Port':              'dsport',
        'Flow Duration':                 'dur',
        'Total Fwd Packets':             'spkts',
        'Total Backward Packets':        'dpkts',
        'Total Length of Fwd Packets':   'sbytes',
        'Total Length of Bwd Packets':   'dbytes',
        'Fwd Packet Length Mean':        'smean',
        'Bwd Packet Length Mean':        'dmean',
        'Flow Bytes/s':                  'rate',
        'Flow Packets/s':                'srate',
        'Bwd Packets/s':                 'bwd_pkt_rate',
        'Flow IAT Mean':                 'sinpkt',
        'Flow IAT Std':                  'dinpkt',
        'Fwd IAT Std':                   'sjit',
        'Bwd IAT Total':                 'djit',
        'Init_Win_bytes_forward':        'swin',
        'Init_Win_bytes_backward':       'dwin',
        'Fwd Header Length':             'sttl',
        'Bwd Header Length':             'dttl',
        'Min Packet Length':             'sloss',
        'Max Packet Length':             'dloss',
        'Active Mean':                   'tcprtt',
        'Active Std':                    'synack',
        'Active Max':                    'ackdat',
        'Label':                         'attack_cat',
        # ── Alt çizgili preprocessed isimleri ──
        'Destination_Port':              'dsport',
        'Flow_Duration':                 'dur',
        'Total_Fwd_Packets':             'spkts',
        'Total_Backward_Packets':        'dpkts',
        'Total_Length_of_Fwd_Packets':   'sbytes',
        'Total_Length_of_Bwd_Packets':   'dbytes',
        'Fwd_Packet_Length_Mean':        'smean',
        'Bwd_Packet_Length_Mean':        'dmean',
        'Flow_Bytes/s':                  'rate',
        'Flow_Packets/s':                'srate',
        'Bwd_Packets/s':                 'bwd_pkt_rate',
        'Flow_IAT_Mean':                 'sinpkt',
        'Flow_IAT_Std':                  'dinpkt',
        'Fwd_IAT_Std':                   'sjit',
        'Bwd_IAT_Total':                 'djit',
        'Fwd_Header_Length':             'sttl',
        'Bwd_Header_Length':             'dttl',
        'Min_Packet_Length':             'sloss',
        'Max_Packet_Length':             'dloss',
        'Active_Mean':                   'tcprtt',
        'Active_Std':                    'synack',
        'Active_Max':                    'ackdat',
    }

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip()
        # Hedef sütun yeniden adlandır
        for alt in ["Label", "label", "class"]:
            if alt in df.columns and "attack_cat" not in df.columns:
                df.rename(columns={alt: "attack_cat"}, inplace=True)
                break
        # CICIDS17 → UNSW standart isim eşlemesi
        df.rename(columns=GLOBAL_COL_MAP, inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Sayısal dönüşüm
        for c in df.columns:
            if c == "attack_cat":
                continue
            if df[c].dtype == object:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _derive_proxies(df: pd.DataFrame) -> pd.DataFrame:
        """
        CICIDS14'te doğrudan bulunmayan UNSW kolonlarını proxy ile türetir.
        Ham (normalize edilmemiş) DataFrame üzerinde çalışır.

        dpkts  : Total_Backward_Packets yoksa → spkts * Down/Up_Ratio
                 veya bwd_pkt_rate * dur
        dbytes : Total_Length_of_Bwd_Packets yoksa → dmean * dpkts
                 veya sbytes * Down/Up_Ratio (daha kaba)

        sttl/dttl, sport/srcip → CICIDS'de hiç kaydedilmiyor, proxy yok.
        """
        EPS = 1e-10
        df = df.copy()

        # ── dpkts proxy ──────────────────────────────────────────────────────
        if 'dpkts' not in df.columns:
            if 'spkts' in df.columns and 'Down/Up_Ratio' in df.columns:
                df['dpkts'] = (df['spkts'] * df['Down/Up_Ratio']).fillna(0)
                _info("  Proxy: dpkts = spkts × Down/Up_Ratio")
            elif 'bwd_pkt_rate' in df.columns and 'dur' in df.columns:
                df['dpkts'] = (df['bwd_pkt_rate'] * df['dur']).fillna(0)
                _info("  Proxy: dpkts = bwd_pkt_rate × dur")

        # ── dbytes proxy ─────────────────────────────────────────────────────
        if 'dbytes' not in df.columns:
            if 'dmean' in df.columns and 'dpkts' in df.columns:
                df['dbytes'] = (df['dmean'] * df['dpkts']).fillna(0)
                _info("  Proxy: dbytes = dmean × dpkts")
            elif 'sbytes' in df.columns and 'Down/Up_Ratio' in df.columns:
                df['dbytes'] = (df['sbytes'] * df['Down/Up_Ratio']).fillna(0)
                _info("  Proxy: dbytes = sbytes × Down/Up_Ratio")

        return df

    df_tr = _derive_proxies(_clean(df_tr))
    df_te = _derive_proxies(_clean(df_te))

    if "attack_cat" not in df_tr.columns:
        _info(f"HATA: 'attack_cat' sütunu bulunamadı. Mevcut: {list(df_tr.columns)[:10]}")
        return

    # Etiket kanonizasyonu
    df_tr["attack_cat"] = df_tr["attack_cat"].astype(str).apply(_canon_cicids14_label)
    df_te["attack_cat"] = df_te["attack_cat"].astype(str).apply(_canon_cicids14_label)
    _info(f"Train sınıflar: {sorted(df_tr['attack_cat'].unique())}")
    _info(f"Test  sınıflar: {sorted(df_te['attack_cat'].unique())}")

    # ── 3. Feature / Label ayır ───────────────────────────────────────────────
    feature_cols = [c for c in df_tr.columns if c != "attack_cat"]
    # Test setinde olmayan kolonları çıkar, sadece ortak kolonları kullan
    common_cols = [c for c in feature_cols if c in df_te.columns]
    if len(common_cols) < len(feature_cols):
        dropped = set(feature_cols) - set(common_cols)
        _info(f"  Test'te olmayan {len(dropped)} sütun kaldırıldı: {list(dropped)[:5]}")
    feature_cols = common_cols

    X_tr_df = df_tr[feature_cols].fillna(0)
    X_te_df = df_te[feature_cols].fillna(0)
    y_tr_raw = df_tr["attack_cat"].copy()
    y_te_raw = df_te["attack_cat"].copy()

    # ── 4. Label encode (train sınıfları baz alınır) ─────────────────────────
    le = LabelEncoder()
    le.fit(y_tr_raw.astype(str))
    # Test'te train'de olmayan sınıflar varsa 'unknown' olarak işaretle
    known = set(le.classes_)
    y_te_raw_mapped = y_te_raw.apply(lambda x: x if x in known else le.classes_[0])
    y_tr = le.transform(y_tr_raw.astype(str))
    y_te = le.transform(y_te_raw_mapped.astype(str))
    _info(f"CICIDS14: {len(le.classes_)} sınıf — {list(le.classes_)}")

    # ── 5. ColumnTransformer pipeline ─────────────────────────────────────────
    ct, ordered_names = build_ct_pipeline(X_tr_df, feature_cols)
    X_tr_norm = ct_transform(ct, X_tr_df, feature_cols)
    X_te_norm = ct_transform(ct, X_te_df, feature_cols)
    _info(f"Normalize boyut: {X_tr_norm.shape}")

    # ── 6. 5 Aşama üret ───────────────────────────────────────────────────────
    run_stages(X_tr_norm, y_tr, X_te_norm, y_te, le, ordered_names, "cicids14", out_path,
               n_select=n_select, strategy=strategy)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Ablasyon verisi hazırlayıcı — UNSW-NB15 ve CICIDS14"
    )
    ap.add_argument("--dataset",      choices=["unsw", "cicids14", "all"], default="all",
                    help="Hangi veri seti hazırlansın (varsayılan: all)")
    ap.add_argument("--output-dir",   type=str, default="ablation_data",
                    help="Çıktı klasörü (varsayılan: ablation_data)")
    ap.add_argument("--unsw-glob",    type=str, default=UNSW_FILES_GLOB,
                    help="UNSW-NB15 CSV glob deseni")
    ap.add_argument("--unsw-features-csv", type=str, default=UNSW_FEATURES_CSV,
                    help="UNSW features.csv yolu")
    ap.add_argument("--cicids-train", type=str, default=str(CICIDS14_TRAIN_CSV),
                    help="CICIDS14 train CSV yolu")
    ap.add_argument("--cicids-test",  type=str, default=str(CICIDS14_TEST_CSV),
                    help="CICIDS14 test CSV yolu")
    ap.add_argument("--test-size",    type=float, default=0.20,
                    help="UNSW için test split oranı (varsayılan: 0.20)")
    ap.add_argument("--n-select",     type=int,   default=75,
                    help="Stage B SelectKBest k değeri (varsayılan: 75)")
    ap.add_argument("--strategy",     type=str,   default="full",
                    choices=["full", "rus_only", "rus_smote"],
                    help="Resampling stratejisi: full | rus_only | rus_smote (varsayılan: full)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("unsw", "all"):
        prepare_unsw(
            out_dir,
            files_glob=args.unsw_glob,
            features_csv=args.unsw_features_csv,
            test_size=args.test_size,
            n_select=args.n_select,
            strategy=args.strategy,
        )

    if args.dataset in ("cicids14", "all"):
        prepare_cicids14(
            out_dir,
            train_csv=Path(args.cicids_train),
            test_csv=Path(args.cicids_test),
            n_select=args.n_select,
            strategy=args.strategy,
        )

    print(f"\n{'='*60}")
    print(f"  Hazırlık tamamlandı. Çıktılar: {out_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
