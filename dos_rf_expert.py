import os, joblib, pickle, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

def _info(m): print(f"[Info]  {m}")
def _stage(m): print(f"[Stage] {m}")

def train_dos_expert(cache_dir, n_features=65):
    """
    DOS sinifi icin ozel Binary RF Expert.
    Enhanced featurelar ile egitilir.
    """
    _stage(f"Training DOS RF Expert (n_features={n_features})")
    
    cdir = Path(cache_dir)
    
    # 1. Veri Yukleme (Enhanced)
    # Glob-based arama
    enhanced_found = False
    candidates = sorted(cdir.glob('Xt_tr_enhanced*.joblib'))
    for tr_f in candidates:
        suffix = tr_f.stem.replace('Xt_tr_', '')
        v_f  = cdir / f'Xt_v_{suffix}.joblib'
        te_f = cdir / f'Xt_te_{suffix}.joblib'
        if v_f.exists() and te_f.exists():
            Xt_tr = joblib.load(tr_f)
            Xt_v  = joblib.load(v_f)
            Xt_te = joblib.load(te_f)
            _info(f"Loaded enhanced features ({suffix}): {Xt_tr.shape}")
            enhanced_found = True
            break
            
    if not enhanced_found:
        _info("Enhanced features not found, using original features.")
        Xt_tr = joblib.load(cdir / 'Xt_tr.joblib')
        Xt_v  = joblib.load(cdir / 'Xt_v.joblib')
        Xt_te = joblib.load(cdir / 'Xt_te.joblib')

    y_tr = joblib.load(cdir / 'y_tr.joblib')
    y_v  = joblib.load(cdir / 'y_v.joblib')
    y_te = joblib.load(cdir / 'y_te.joblib')
    le   = joblib.load(cdir / 'label_encoder.joblib')
    
    # DOS binary label olustur
    y_tr_bin = (y_tr == 'dos').astype(int)
    y_v_bin  = (y_v  == 'dos').astype(int)
    y_te_bin = (y_te == 'dos').astype(int)
    
    # === OPTIMIZATION: Scaling check ===
    # Features are already scaled (StandardScaler), but RF can handle them.
    # However, log1p was destroying negative values. We remove it.
    _info("Using features as-is (already standardized). Removed destructive log1p.")
    
    _info(f"DOS samples in train: {y_tr_bin.sum()} / {len(y_tr_bin)}")

    # 2. Model: BalancedBagging with RandomForest
    # sampling_strategy='not minority' majority snfn undersample eder.
    model = BalancedBaggingClassifier(
        estimator=RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42),
        sampling_strategy='not minority', 
        replacement=False,
        random_state=42
    )
    
    _info("Fitting Optimized BalancedBagging RandomForest model...")
    model.fit(Xt_tr, y_tr_bin)

    
    # 3. Degerlendirme
    y_pred_v = model.predict(Xt_v)
    f1_v = f1_score(y_v_bin, y_pred_v)
    _info(f"Validation DOS Binary F1: {f1_v:.4f}")
    
    y_pred_te = model.predict(Xt_te)
    f1_te = f1_score(y_te_bin, y_pred_te)
    _info(f"Test DOS Binary F1: {f1_te:.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(y_te_bin, y_pred_te))

    # 4. Kaydet
    model_path = cdir / 'dos_rf_expert.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    _info(f"Model saved to {model_path}")
    
    # OOF-like olasiliklari kaydet (ensemble icin)
    # Valid
    out_v = np.zeros((len(y_v), len(le.classes_)), dtype=np.float32)
    c_idx = int(np.where(le.classes_ == 'dos')[0][0])
    out_v[:, c_idx] = model.predict_proba(Xt_v)[:, 1]
    np.savez(cdir / "proba_fast_expert_dos_oof_valid.npz", proba=out_v, classes=le.classes_.astype(str))
    
    # Test
    out_te = np.zeros((len(y_te), len(le.classes_)), dtype=np.float32)
    out_te[:, c_idx] = model.predict_proba(Xt_te)[:, 1]
    np.savez(cdir / "proba_fast_expert_dos_oof_test.npz", proba=out_te, classes=le.classes_.astype(str))
    
    _info("OOF probability files updated for ensemble.")
    
    return f1_te

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=str, default='.')
    parser.add_argument('--n-features', type=int, default=65)
    args = parser.parse_args()
    train_dos_expert(args.cache_dir, args.n_features)
