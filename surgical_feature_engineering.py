"""
Feature Engineering 2.0 - Surgical Expert Feature Augmentation
Only used by binary surgical experts (backdoor, analysis, dos).
Does NOT touch the base models or their 36-feature pipeline.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

def create_surgical_features(raw_df: pd.DataFrame) -> np.ndarray:
    """
    Create discriminative features from raw data columns that are NOT in the
    base 36-feature set. These are designed specifically to separate:
    - Backdoor from Exploits (port patterns, TTL anomalies)
    - Analysis from DoS (traffic symmetry, packet rates)
    - DoS from Normal (flow intensity, byte density)
    
    Returns numpy array of shape (n_samples, n_new_features)
    """
    df = raw_df.copy()
    
    # --- 1. Port-Based Features (Backdoor vs Exploits separation) ---
    sport = df['sport'].fillna(0).astype(float)
    dsport = df['dsport'].fillna(0).astype(float)
    
    # High ephemeral port usage (backdoors often use high ports)
    feat_src_port_high = (sport > 49152).astype(np.float32)
    feat_dst_port_high = (dsport > 49152).astype(np.float32)
    
    # Well-known port flags
    feat_dst_http = ((dsport == 80) | (dsport == 443) | (dsport == 8080)).astype(np.float32)
    feat_dst_ssh = (dsport == 22).astype(np.float32)
    feat_dst_dns = (dsport == 53).astype(np.float32)
    feat_dst_ftp = ((dsport == 20) | (dsport == 21)).astype(np.float32)
    
    # Port asymmetry (src vs dst port ratio)
    feat_port_ratio = np.where(dsport > 0, sport / (dsport + 1e-6), 0).astype(np.float32)
    feat_port_ratio = np.clip(feat_port_ratio, 0, 1000)
    
    # --- 2. Traffic Intensity Features (DoS vs Normal) ---
    sbytes = df['sbytes'].fillna(0).astype(float)
    dbytes = df['dbytes'].fillna(0).astype(float)
    spkts = df.get('spkts', pd.Series(np.zeros(len(df)))).fillna(0).astype(float)
    dpkts = df.get('dpkts', pd.Series(np.zeros(len(df)))).fillna(0).astype(float)
    dur = df['dur'].fillna(0).astype(float)
    sload = df.get('sload', pd.Series(np.zeros(len(df)))).fillna(0).astype(float)
    dload = df.get('dload', pd.Series(np.zeros(len(df)))).fillna(0).astype(float)
    
    # Bytes per packet (DoS sends tiny packets in bulk)
    feat_sbytes_per_pkt = np.where(spkts > 0, sbytes / spkts, 0).astype(np.float32)
    feat_dbytes_per_pkt = np.where(dpkts > 0, dbytes / dpkts, 0).astype(np.float32)
    
    # Packet rate (packets per second - DoS has extreme rates)
    feat_pkt_rate = np.where(dur > 0, (spkts + dpkts) / (dur + 1e-6), 0).astype(np.float32)
    feat_pkt_rate = np.clip(feat_pkt_rate, 0, 1e8)
    
    # Byte rate
    feat_byte_rate = np.where(dur > 0, (sbytes + dbytes) / (dur + 1e-6), 0).astype(np.float32)
    feat_byte_rate = np.clip(feat_byte_rate, 0, 1e12)
    
    # --- 3. Traffic Symmetry Features (Analysis vs Exploits) ---
    total_bytes = sbytes + dbytes + 1e-6
    feat_byte_asymmetry = ((sbytes - dbytes) / total_bytes).astype(np.float32)  # -1 to 1
    
    total_pkts = spkts + dpkts + 1e-6
    feat_pkt_asymmetry = ((spkts - dpkts) / total_pkts).astype(np.float32)
    
    # Load imbalance
    total_load = sload + dload + 1e-6
    feat_load_asymmetry = ((sload - dload) / total_load).astype(np.float32)
    
    # --- 4. Cross-Features (Interaction terms for separation) ---
    feat_sbytes_x_dpkts = (sbytes * dpkts).astype(np.float32)
    feat_sbytes_x_dpkts = np.log1p(feat_sbytes_x_dpkts)  # log scale
    
    # TTL anomaly feature (backdoors often have unusual TTL values)
    sttl = df['sttl'].fillna(0).astype(float)
    dttl = df['dttl'].fillna(0).astype(float)
    feat_ttl_diff = (np.abs(sttl - dttl)).astype(np.float32)
    feat_ttl_sum = (sttl + dttl).astype(np.float32)
    
    # --- 5. Jitter and Timing Features ---
    sjit = df.get('sjit', pd.Series(np.zeros(len(df)))).fillna(0).astype(float)
    djit = df.get('djit', pd.Series(np.zeros(len(df)))).fillna(0).astype(float)
    sintpkt = df.get('sintpkt', pd.Series(np.zeros(len(df)))).fillna(0).astype(float)
    dintpkt = df.get('dintpkt', pd.Series(np.zeros(len(df)))).fillna(0).astype(float)
    
    feat_jit_ratio = np.where(djit > 0, sjit / (djit + 1e-6), 0).astype(np.float32)
    feat_jit_ratio = np.clip(feat_jit_ratio, 0, 1000)
    feat_intpkt_ratio = np.where(dintpkt > 0, sintpkt / (dintpkt + 1e-6), 0).astype(np.float32)
    feat_intpkt_ratio = np.clip(feat_intpkt_ratio, 0, 1000)
    
    # Stack all features
    features = np.column_stack([
        feat_src_port_high,     # 0
        feat_dst_port_high,     # 1
        feat_dst_http,          # 2
        feat_dst_ssh,           # 3
        feat_dst_dns,           # 4
        feat_dst_ftp,           # 5
        feat_port_ratio,        # 6
        feat_sbytes_per_pkt,    # 7
        feat_dbytes_per_pkt,    # 8
        feat_pkt_rate,          # 9
        feat_byte_rate,         # 10
        feat_byte_asymmetry,    # 11
        feat_pkt_asymmetry,     # 12
        feat_load_asymmetry,    # 13
        feat_sbytes_x_dpkts,    # 14
        feat_ttl_diff,          # 15
        feat_ttl_sum,           # 16
        feat_jit_ratio,         # 17
        feat_intpkt_ratio,      # 18
    ])
    
    return features.astype(np.float32)


FEATURE_NAMES = [
    'src_port_high', 'dst_port_high', 'dst_http', 'dst_ssh', 'dst_dns', 'dst_ftp',
    'port_ratio', 'sbytes_per_pkt', 'dbytes_per_pkt', 'pkt_rate', 'byte_rate',
    'byte_asymmetry', 'pkt_asymmetry', 'load_asymmetry', 'sbytes_x_dpkts',
    'ttl_diff', 'ttl_sum', 'jit_ratio', 'intpkt_ratio'
]


if __name__ == '__main__':
    print("Loading raw data...")
    raw_df = joblib.load('raw_df.joblib')
    
    print("Creating surgical features...")
    feats = create_surgical_features(raw_df)
    print(f"New features shape: {feats.shape}")
    print(f"Feature names: {FEATURE_NAMES}")
    print(f"NaN count: {np.isnan(feats).sum()}")
    print(f"Inf count: {np.isinf(feats).sum()}")
    
    # Save split features
    n_tr = len(joblib.load('y_tr.joblib'))  # 1778032
    n_v = len(joblib.load('y_v.joblib'))    # 381007
    
    feats_tr = feats[:n_tr]
    feats_v = feats[n_tr:n_tr+n_v]
    feats_te = feats[n_tr+n_v:]
    
    print(f"Train: {feats_tr.shape}, Val: {feats_v.shape}, Test: {feats_te.shape}")
    
    # Combine with original features
    Xt_tr = joblib.load('Xt_tr.joblib')
    Xt_v = joblib.load('Xt_v.joblib')
    Xt_te = joblib.load('Xt_te.joblib')
    
    Xt_tr_aug = np.hstack([Xt_tr, feats_tr])
    Xt_v_aug = np.hstack([Xt_v, feats_v])
    Xt_te_aug = np.hstack([Xt_te, feats_te])
    
    print(f"Augmented Train: {Xt_tr_aug.shape} (36 orig + {len(FEATURE_NAMES)} new = {Xt_tr_aug.shape[1]})")
    
    joblib.dump(Xt_tr_aug, 'Xt_tr_surgical.joblib', compress=3)
    joblib.dump(Xt_v_aug, 'Xt_v_surgical.joblib', compress=3)
    joblib.dump(Xt_te_aug, 'Xt_te_surgical.joblib', compress=3)
    
    print("Saved Xt_tr_surgical.joblib, Xt_v_surgical.joblib, Xt_te_surgical.joblib")
