# cicids_preprocessor.py
# CICIDS2017 HAM CSV -> cicids_train.csv + cicids_test.csv
# Best-practice cleaning pipeline:
#   1. Column strip
#   2. Duplicate removal
#   3. Metadata/leakage column removal
#   4. Infinity -> NaN -> median impute
#   5. Constant feature removal
#   6. Correlation filter (|r| > 0.99)
#   7. Binary flag forcing (> 0 -> 1)
#   8. Label mapping -> canon names
#   9. Stratified 70/30 train/test split
#  10. NaN cleanup + float32 downcasting

import os, glob, argparse, re, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def _info(m): print(f"[Info]  {m}")
def _stage(m): print(f"[Stage] {m}")

# ======================== LABEL MAPPING ========================
# CICIDS17 label -> kanonik IDS sinif isimleri
CICIDS_LABEL_MAP = {
    'BENIGN': 'normal',
    'Bot': 'bot',
    'DDoS': 'dos',
    'DoS Hulk': 'dos',
    'DoS GoldenEye': 'dos',
    'DoS slowloris': 'dos',
    'DoS Slowhttptest': 'dos',
    'Heartbleed': 'heartbleed',
    'Infiltration': 'infiltration',
    'PortScan': 'portscan',
    'FTP-Patator': 'bruteforce',
    'SSH-Patator': 'bruteforce',
}

def _canon_cicids_label(s: str) -> str:
    """CICIDS17 etiketlerini kanonik isimlere donusturur."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    # Exact match dene
    if s in CICIDS_LABEL_MAP:
        return CICIDS_LABEL_MAP[s]
    # Web Attack varyantlari (bozuk karakterli)
    s_lower = s.lower()
    if 'web attack' in s_lower and 'brute' in s_lower:
        return 'bruteforce'
    if 'web attack' in s_lower and 'xss' in s_lower:
        return 'xss'
    if 'web attack' in s_lower and 'sql' in s_lower:
        return 'sql_injection'
    # Genel regex temizlik
    cleaned = re.sub(r'[^a-zA-Z0-9 ]', ' ', s).strip().lower()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    if cleaned in ('benign', 'normal', ''): return 'normal'
    return cleaned

# ======================== METADATA COLUMNS ========================
# Bu kolonlar model icin veri sizdirma (leakage) kaynagidir
METADATA_COLS = [
    'Flow ID', 'Source IP', 'Src IP', 'Destination IP', 'Dst IP',
    'Timestamp', 'Source Port',  # Source Port bazı çalışmalarda silinir
]

# Binary flag kolonlari (> 0 -> 1 zorlanacak)
BINARY_FLAG_COLS = [
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'CWE Flag Count', 'ECE Flag Count',
    'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags',
]

# ======================== MAIN PIPELINE ========================

def load_and_merge_csvs(input_dir: str) -> pd.DataFrame:
    """8 CSV dosyasini chunk-by-chunk okur ve birlestirir."""
    _stage("Loading and merging CICIDS17 CSVs...")
    csv_files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    
    _info(f"Found {len(csv_files)} CSV files")
    
    dfs = []
    for i, f in enumerate(csv_files):
        fn = os.path.basename(f)
        _info(f"  [{i+1}/{len(csv_files)}] Reading {fn}...")
        try:
            df_chunk = pd.read_csv(f, encoding='latin1', low_memory=False)
            _info(f"    -> {len(df_chunk)} rows, {len(df_chunk.columns)} cols")
            dfs.append(df_chunk)
        except Exception as e:
            _info(f"    ERROR reading {fn}: {e}")
            continue
    
    df = pd.concat(dfs, ignore_index=True)
    _info(f"Total merged: {len(df)} rows, {len(df.columns)} cols")
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Kolon isimlerindeki bosluk ve duplike kolon sorunlarini giderir."""
    _stage("Cleaning column names...")
    df.columns = df.columns.str.strip()
    
    # Duplike kolon isimleri ('Fwd Header Length' ve 'Fwd Header Length.1')
    # .1 olanini kaldir
    dup_cols = [c for c in df.columns if c.endswith('.1')]
    if dup_cols:
        _info(f"  Dropping duplicate columns: {dup_cols}")
        df = df.drop(columns=dup_cols, errors='ignore')
    
    return df


def remove_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Metadata/leakage kolonlarini siler."""
    _stage("Removing metadata/leakage columns...")
    to_drop = [c for c in METADATA_COLS if c in df.columns]
    if to_drop:
        _info(f"  Dropping: {to_drop}")
        df = df.drop(columns=to_drop)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Tam duplike satirlari siler."""
    _stage("Removing duplicate rows...")
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    _info(f"  {n_before} -> {n_after} ({n_before - n_after} duplicates removed)")
    return df.reset_index(drop=True)


def fix_numeric_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Infinity -> NaN, sayisal olmayan degerleri NaN yapar, NaN imputation."""
    _stage("Fixing numeric issues (Inf, NaN, type coercion)...")
    
    label_col = None
    if 'Label' in df.columns:
        label_col = df['Label'].copy()
        df = df.drop(columns=['Label'])
    elif 'attack_cat' in df.columns:
        label_col = df['attack_cat'].copy()
        df = df.drop(columns=['attack_cat'])
    
    # Tum kolonlari numeric'e zorla (non-numeric -> NaN)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Infinity -> NaN
    n_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    df = df.replace([np.inf, -np.inf], np.nan)
    _info(f"  Replaced {n_inf} infinity values with NaN")
    
    # NaN -> median imputation (kolon bazli)
    n_nan = df.isna().sum().sum()
    if n_nan > 0:
        _info(f"  Imputing {n_nan} NaN values with column medians...")
        for c in df.columns:
            if df[c].isna().any():
                median_val = df[c].median()
                if pd.isna(median_val):
                    median_val = 0
                df[c] = df[c].fillna(median_val)
    
    # Negatif deger temizligi (bazi flow feature'lar negatif olamaz)
    # Sadece duration/length/count kolonlarinda
    neg_cols = [c for c in df.columns if any(k in c.lower() for k in 
                ['length', 'packets', 'bytes', 'duration', 'size', 'count'])]
    for c in neg_cols:
        neg_count = (df[c] < 0).sum()
        if neg_count > 0:
            df[c] = df[c].clip(lower=0)
            _info(f"  Clipped {neg_count} negative values in '{c}'")
    
    # Label kolonunu geri ekle
    if label_col is not None:
        df['Label'] = label_col.values
    
    return df


def fix_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag kolonlarini 0/1'e zorlar."""
    _stage("Fixing binary flag columns (> 0 -> 1)...")
    for c in BINARY_FLAG_COLS:
        if c in df.columns:
            non_binary = ((df[c] != 0) & (df[c] != 1)).sum()
            if non_binary > 0:
                _info(f"  {c}: {non_binary} non-binary values -> forcing to 0/1")
            df[c] = (df[c] > 0).astype(int)
    return df


def remove_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tek unique degeri olan feature'lari siler."""
    _stage("Removing constant (single-value) features...")
    label_col = 'Label' if 'Label' in df.columns else 'attack_cat' if 'attack_cat' in df.columns else None
    
    to_drop = []
    for c in df.columns:
        if c == label_col:
            continue
        if df[c].nunique() <= 1:
            to_drop.append(c)
    
    if to_drop:
        _info(f"  Dropping {len(to_drop)} constant features: {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        _info("  No constant features found")
    
    return df


def correlation_filter(df: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    """Pearson |r| > threshold olan feature ciftlerinden birini siler.
    
    Neden: CICIDS17'de bircok feature neredeyse ayni bilgiyi tasiyor
    (orn. Total Fwd Packets ≈ Subflow Fwd Packets). Bu fazlalik:
    - Tree-based modellerde feature importance dagitarak onemli feature'larin 
      etkisini zayiflatir
    - MLP gibi modellerde multicollinearity ile gradyan karasizligina yol acar
    - Bellegi gereksiz mesgul eder
    
    ANCAK: threshold=0.99 sadece NEREDEYSE AYNI olan feature'lari siler.
    0.95 veya daha dusuk deger kullanmiyoruz cunku farkli modeller farkli 
    kombinasyonlardan faydalanabilir.
    """
    _stage(f"Correlation filtering (|r| > {threshold})...")
    
    label_col = 'Label' if 'Label' in df.columns else 'attack_cat' if 'attack_cat' in df.columns else None
    
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != label_col]
    
    if len(numeric_cols) < 2:
        _info("  Not enough numeric columns for correlation filter")
        return df
    
    _info(f"  Computing correlation matrix for {len(numeric_cols)} features...")
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Ust ucgen matrisi al (diagonal + alt ucgen haric)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # |r| > threshold olan kolonlari bul
    to_drop = set()
    pairs_found = []
    for col in upper.columns:
        high_corr = upper.index[upper[col] > threshold].tolist()
        if high_corr:
            for hc in high_corr:
                if hc not in to_drop:
                    pairs_found.append((col, hc, corr_matrix.loc[col, hc]))
                    to_drop.add(col)
                    break
    
    if to_drop:
        _info(f"  Found {len(pairs_found)} highly correlated pairs:")
        for c1, c2, r in pairs_found[:10]:  # ilk 10'unu goster
            _info(f"    {c1} <-> {c2}: r={r:.4f}")
        if len(pairs_found) > 10:
            _info(f"    ... and {len(pairs_found) - 10} more")
        
        _info(f"  Dropping {len(to_drop)} features: {sorted(to_drop)}")
        df = df.drop(columns=list(to_drop))
    else:
        _info("  No highly correlated features found")
    
    return df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """CICIDS17 labellerini kanonik IDS sinif isimlerine donusturur."""
    _stage("Mapping labels to canonical names...")
    
    label_col = 'Label' if 'Label' in df.columns else None
    if label_col is None:
        _info("  No 'Label' column found, skipping")
        return df
    
    df[label_col] = df[label_col].apply(_canon_cicids_label)
    
    # 'Label' -> 'attack_cat' (standart isim)
    df = df.rename(columns={'Label': 'attack_cat'})
    
    vc = df['attack_cat'].value_counts()
    _info("  Label distribution:")
    for k, v in vc.items():
        pct = v / len(df) * 100
        _info(f"    {k}: {v} ({pct:.2f}%)")
    
    return df


def downcast_types(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32, int64 -> int32 (bellek optimizasyonu)."""
    _stage("Downcasting data types for memory optimization...")
    
    mem_before = df.memory_usage(deep=True).sum() / 1024**2
    
    for c in df.select_dtypes(include=['float64']).columns:
        df[c] = df[c].astype(np.float32)
    
    for c in df.select_dtypes(include=['int64']).columns:
        col_min, col_max = df[c].min(), df[c].max()
        if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            df[c] = df[c].astype(np.int32)
    
    mem_after = df.memory_usage(deep=True).sum() / 1024**2
    _info(f"  Memory: {mem_before:.1f} MB -> {mem_after:.1f} MB ({(1-mem_after/mem_before)*100:.1f}% reduction)")
    
    return df


def split_and_save(df: pd.DataFrame, output_dir: str, test_ratio: float = 0.30,
                   random_state: int = 42):
    """Stratified train/test split yapar ve CSV olarak kaydeder."""
    _stage(f"Stratified train/test split (test={test_ratio:.0%})...")
    
    target_col = 'attack_cat'
    
    # Cok kucuk siniflar icin stratify sorun cikarabilir, 
    # minimum 2 ornek olmali her sinifta
    vc = df[target_col].value_counts()
    small_classes = vc[vc < 2].index.tolist()
    if small_classes:
        _info(f"  WARNING: Classes with < 2 samples will be handled: {small_classes}")
        # Bu cok kucuk siniflari en yakin buyuk sinifa merge et
        # (genellikle tek basina model egitilemez)
    
    df_train, df_test = train_test_split(
        df, test_size=test_ratio, 
        stratify=df[target_col], 
        random_state=random_state, 
        shuffle=True
    )
    
    # Dağılımları göster
    _info(f"  Train: {len(df_train)} rows")
    for k, v in df_train[target_col].value_counts().items():
        _info(f"    {k}: {v}")
    
    _info(f"  Test: {len(df_test)} rows")
    for k, v in df_test[target_col].value_counts().items():
        _info(f"    {k}: {v}")
    
    # Kaydet
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = out_dir / 'cicids_train.csv'
    test_path = out_dir / 'cicids_test.csv'
    
    _info(f"  Saving {train_path}...")
    df_train.to_csv(train_path, index=False)
    
    _info(f"  Saving {test_path}...")
    df_test.to_csv(test_path, index=False)
    
    _info(f"  Final feature count: {len(df.columns) - 1} (excluding target)")
    _info(f"  Final train columns: {list(df_train.columns)}")
    
    return df_train, df_test


def run_cicids_preprocessing(input_dir: str, output_dir: str, 
                              corr_threshold: float = 0.99,
                              test_ratio: float = 0.30,
                              random_state: int = 42,
                              skip_corr: bool = False):
    """Tam CICIDS17 on-isleme pipeline'i."""
    print("=" * 70)
    print("  CICIDS2017 PREPROCESSING PIPELINE")
    print("=" * 70)
    t0 = time.time()
    
    # 1. Yukle ve birlestir
    df = load_and_merge_csvs(input_dir)
    
    # 2. Kolon temizligi
    df = clean_columns(df)
    
    # 3. Metadata kolonlarini sil
    df = remove_metadata(df)
    
    # 4. Duplike satirlari sil
    df = remove_duplicates(df)
    
    # 5. Numeric sorunlar (inf/nan/negatif)
    df = fix_numeric_issues(df)
    
    # 6. Binary flag zorlama
    df = fix_binary_flags(df)
    
    # 7. Constant feature silme
    df = remove_constant_features(df)
    
    # 8. Korelasyon filtresi
    if not skip_corr:
        df = correlation_filter(df, threshold=corr_threshold)
    else:
        _info("Correlation filter SKIPPED (--skip-corr)")
    
    # 9. Label mapping
    df = map_labels(df)
    
    # 10. Tip downcasting
    df = downcast_types(df)
    
    # 11. Son NaN kontrolu
    remaining_nan = df.isna().sum().sum()
    if remaining_nan > 0:
        _info(f"WARNING: {remaining_nan} NaN values remaining! Filling with 0...")
        df = df.fillna(0)
    
    # 12. Split ve kaydet
    df_train, df_test = split_and_save(df, output_dir, test_ratio, random_state)
    
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  CICIDS17 PREPROCESSING COMPLETE ({elapsed:.1f}s)")
    print(f"  Output: {output_dir}/cicids_train.csv, cicids_test.csv")
    print(f"  Features: {len(df.columns) - 1} | Classes: {df['attack_cat'].nunique()}")
    print("=" * 70)
    
    return df_train, df_test


# ======================== CLI ========================
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="CICIDS2017 Dataset Preprocessing Pipeline")
    ap.add_argument('--input-dir', type=str, default='CICIDS17',
                    help='CICIDS17 CSV dosyalarinin bulundugu dizin')
    ap.add_argument('--output-dir', type=str, default='.',
                    help='Cikti CSV dosyalarinin kaydedilecegi dizin')
    ap.add_argument('--corr-threshold', type=float, default=0.99,
                    help='Korelasyon filtresi esigi (default: 0.99)')
    ap.add_argument('--test-ratio', type=float, default=0.30,
                    help='Test seti orani (default: 0.30)')
    ap.add_argument('--skip-corr', action='store_true',
                    help='Korelasyon filtresini atla')
    ap.add_argument('--random-state', type=int, default=42,
                    help='Random seed (default: 42)')
    
    args = ap.parse_args()
    
    run_cicids_preprocessing(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        corr_threshold=args.corr_threshold,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        skip_corr=args.skip_corr
    )
