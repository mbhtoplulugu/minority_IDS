import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings("ignore")

def _info(m): print(f"[Info]  {m}")
def _stage(m): print(f"[Stage] {m}")

def generate_dataset_agnostic_features(cfg, n_features=65):
    """
    Dataset-agnostic feature engineering.
    Detects available columns from preprocess_meta and generates derived features.
    If enhanced cache exists for the current dataset, it loads it instead of re-calculating.
    """
    cache_dir = Path(getattr(cfg, 'cache_dir', '.'))
    
    # 1. Smart Caching: Check if already exists
    # We use a pattern to check for any existing enhanced features in this cache_dir
    existing = list(cache_dir.glob("Xt_tr_enhanced*.joblib"))
    if existing:
        # Sort by stem length or name to get the "best" one if multiple exist, 
        # but usually we just want the latest/exact one.
        latest = sorted(existing)[-1]
        import re
        match = re.search(r'enhanced(\d+)', latest.stem)
        actual_n = int(match.group(1)) if match else n_features
        _info(f"Enhanced features found in {cache_dir}: {latest.name}. Skipping generation.")
        return actual_n

    _stage(f"Generating Dynamic Dataset-Agnostic Features in {cache_dir}")

    # 2. Load Base Data
    try:
        Xt_tr = joblib.load(cache_dir / "Xt_tr.joblib")
        Xt_v  = joblib.load(cache_dir / "Xt_v.joblib")
        Xt_te = joblib.load(cache_dir / "Xt_te.joblib")
        y_tr  = joblib.load(cache_dir / "y_tr.joblib")
        meta  = joblib.load(cache_dir / "preprocess_meta.joblib")
    except Exception as e:
        _info(f"Error loading base data from {cache_dir}: {e}")
        return None

    # Column Mapping
    num_cols = meta.get("num_cols", [])
    cat_cols = meta.get("cat_cols", [])
    all_cols_in_matrix = num_cols + cat_cols
    col_map = {name: i for i, name in enumerate(all_cols_in_matrix)}

    _info(f"Targeting {len(all_cols_in_matrix)} base features for augmentation.")

    def get_derived_features(X_arr):
        new_feats = []
        
        # Helper to get indices safely
        def get_idx(name):
            return col_map.get(name, -1)

        # A. Packet & Byte Rates (DoS / Probe Detection)
        idx_dur = get_idx('dur')
        idx_spkts = get_idx('spkts')
        idx_dpkts = get_idx('dpkts')
        idx_sbytes = get_idx('sbytes')
        idx_dbytes = get_idx('dbytes')

        if idx_dur >= 0:
            dur_eps = X_arr[:, idx_dur] + 1e-9
            if idx_spkts >= 0:
                new_feats.append(X_arr[:, idx_spkts] / dur_eps) # spkts_rate
            if idx_dpkts >= 0:
                new_feats.append(X_arr[:, idx_dpkts] / dur_eps) # dpkts_rate
            if idx_sbytes >= 0:
                new_feats.append(X_arr[:, idx_sbytes] / dur_eps) # sbyte_rate
            if idx_dbytes >= 0:
                new_feats.append(X_arr[:, idx_dbytes] / dur_eps) # dbyte_rate

        # B. Asymmetry (Analysis / Exploit Detection)
        if idx_sbytes >= 0 and idx_dbytes >= 0:
            sum_bytes = X_arr[:, idx_sbytes] + X_arr[:, idx_dbytes] + 1e-9
            new_feats.append((X_arr[:, idx_sbytes] - X_arr[:, idx_dbytes]) / sum_bytes) # byte_asym
        
        if idx_spkts >= 0 and idx_dpkts >= 0:
            sum_pkts = X_arr[:, idx_spkts] + X_arr[:, idx_dpkts] + 1e-9
            new_feats.append((X_arr[:, idx_spkts] - X_arr[:, idx_dpkts]) / sum_pkts) # pkt_asym

        # C. Communication Efficiency (Payload Density - Botnet/C2 Detection)
        if idx_sbytes >= 0 and idx_spkts >= 0:
            new_feats.append(X_arr[:, idx_sbytes] / (X_arr[:, idx_spkts] + 1e-9)) # s_payload_density
        if idx_dbytes >= 0 and idx_dpkts >= 0:
            new_feats.append(X_arr[:, idx_dbytes] / (X_arr[:, idx_dpkts] + 1e-9)) # d_payload_density

        # D. Port Specifics (Recon / Backdoor)
        idx_dsport = get_idx('dsport')
        if idx_dsport >= 0:
            # We assume dsport is normalized or original. Let's create dummy-like flags for standard targets
            # Since data is scaled, we can't do exact matching easily if Scaler was applied.
            # But the 'dynamic' nature implies we should have done this BEFORE scaling if possible.
            # However, preprocess.joblib handles scaling. 
            pass

        # E. TCP Performance (Fuzzers / Slowloris)
        idx_tcprtt = get_idx('tcprtt')
        idx_synack = get_idx('synack')
        idx_ackdat = get_idx('ackdat')
        if idx_tcprtt >= 0 and idx_dur >= 0:
            new_feats.append(X_arr[:, idx_tcprtt] / (X_arr[:, idx_dur] + 1e-9))
        if idx_synack >= 0 and idx_ackdat >= 0:
            new_feats.append(X_arr[:, idx_synack] / (X_arr[:, idx_ackdat] + 1e-9))

        if not new_feats:
            return None
        
        res = np.column_stack(new_feats).astype(np.float32)
        # Clean NaNs and Infs
        return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)

    # Calculate for all splits
    feats_tr = get_derived_features(Xt_tr)
    feats_v  = get_derived_features(Xt_v)
    feats_te = get_derived_features(Xt_te)

    if feats_tr is None:
        _info("No derived features generated (columns not found). Creating fallback enhanced set.")
        Xt_tr_enh, Xt_v_enh, Xt_te_enh = Xt_tr, Xt_v, Xt_te
    else:
        # Scale new features (Crucial for MLP/AE)
        scaler = StandardScaler()
        feats_tr_scaled = scaler.fit_transform(feats_tr)
        feats_v_scaled  = scaler.transform(feats_v)
        feats_te_scaled = scaler.transform(feats_te)

        Xt_tr_enh = np.hstack([Xt_tr, feats_tr_scaled])
        Xt_v_enh  = np.hstack([Xt_v, feats_v_scaled])
        Xt_te_enh = np.hstack([Xt_te, feats_te_scaled])
        _info(f"Augmented base features with {feats_tr.shape[1]} derived metrics.")

    # 3. Dimensionality Management (SelectKBest)
    actual_dim = Xt_tr_enh.shape[1]
    if actual_dim > n_features:
        _info(f"Pruning features from {actual_dim} down to {n_features} via SelectKBest...")
        # Since y_tr might be categorical strings, encode for selection
        from sklearn.preprocessing import LabelEncoder
        le_temp = LabelEncoder()
        y_tr_enc = le_temp.fit_transform(y_tr)
        
        selector = SelectKBest(f_classif, k=n_features)
        Xt_tr_final = selector.fit_transform(Xt_tr_enh, y_tr_enc)
        Xt_v_final  = selector.transform(Xt_v_enh)
        Xt_te_final = selector.transform(Xt_te_enh)
    else:
        Xt_tr_final, Xt_v_final, Xt_te_final = Xt_tr_enh, Xt_v_enh, Xt_te_enh

    # 4. Save Cache
    final_n = Xt_tr_final.shape[1]
    joblib.dump(Xt_tr_final, cache_dir / f"Xt_tr_enhanced{final_n}.joblib", compress=3)
    joblib.dump(Xt_v_final,  cache_dir / f"Xt_v_enhanced{final_n}.joblib",  compress=3)
    joblib.dump(Xt_te_final, cache_dir / f"Xt_te_enhanced{final_n}.joblib", compress=3)
    _info(f"Enhanced features saved: {cache_dir / f'Xt_tr_enhanced{final_n}.joblib'}")

    return final_n

if __name__ == '__main__':
    # Can be run standalone if cache exists
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', type=str, required=True)
    ap.add_argument('--n-features', type=int, default=65)
    args = ap.parse_args()
    
    class FakeCfg:
        def __init__(self, cdir): self.cache_dir = cdir
    
    generate_dataset_agnostic_features(FakeCfg(args.cache_dir), n_features=args.n_features)
