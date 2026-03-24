import argparse, joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

def _info(m): print(f"[Info]  {m}")
def _stage(m): print(f"[Stage] {m}")

def run_hist_gb_oof(cfg, n_features=50, n_splits=5):
    """
    Subagent Research: Low-RAM CPU Model for UNSW-NB15
    - Model: HistGradientBoostingClassifier
    - Discretizes features into integers (max_bins=63) cutting RAM usage by 75% compared to RF.
    - Uses limited depth/leaves (max_leaf_nodes=31) to prevent explosions.
    """
    cdir = Path(cfg.cache_dir)
    _stage(f"HistGradientBoostingClassifier OOF Generation (n_features={n_features}, folds={n_splits})")
    
    # Yukleme - Enhanced Features (glob-based arama)
    enhanced_found = False
    
    # 1) Exact match dene
    exact_files = [
        cdir / f'Xt_tr_enhanced{n_features}.joblib',
        cdir / f'Xt_v_enhanced{n_features}.joblib',
        cdir / f'Xt_te_enhanced{n_features}.joblib'
    ]
    if all(f.exists() for f in exact_files):
        Xt_tr = joblib.load(exact_files[0])
        Xt_v  = joblib.load(exact_files[1])
        Xt_te = joblib.load(exact_files[2])
        _info(f"[HistGB] Enhanced features found (exact {n_features}): {Xt_tr.shape}")
        enhanced_found = True
    
    # 2) Glob ile en iyi enhanced dosyay bul
    if not enhanced_found:
        candidates = sorted(cdir.glob('Xt_tr_enhanced*.joblib'))
        for tr_f in candidates:
            suffix = tr_f.stem.replace('Xt_tr_', '')  # e.g. 'enhanced58'
            v_f  = cdir / f'Xt_v_{suffix}.joblib'
            te_f = cdir / f'Xt_te_{suffix}.joblib'
            if v_f.exists() and te_f.exists():
                Xt_tr = joblib.load(tr_f)
                Xt_v  = joblib.load(v_f)
                Xt_te = joblib.load(te_f)
                _info(f"[HistGB] Enhanced features found ({suffix}): {Xt_tr.shape}")
                enhanced_found = True
                break
    
    # 3) Fallback: original
    if not enhanced_found:
        _info("[HistGB] Enhanced features not found, reverting to original features.")
        Xt_tr = joblib.load(cdir / 'Xt_tr.joblib')
        Xt_v  = joblib.load(cdir / 'Xt_v.joblib')
        Xt_te = joblib.load(cdir / 'Xt_te.joblib')

    y_tr = joblib.load(cdir / 'y_tr.joblib')
    y_v  = joblib.load(cdir / 'y_v.joblib')
    y_te = joblib.load(cdir / 'y_te.joblib')
    
    le = joblib.load(cdir / 'label_encoder.joblib')
    classes = le.classes_.astype(str)
    
    ytr_enc = le.transform(np.asarray(y_tr))
    yv_enc  = le.transform(np.asarray(y_v))
    yte_enc = le.transform(np.asarray(y_te))
    
    # Prepare OOF Arrays
    oof_train = np.zeros((len(ytr_enc), len(classes)), dtype=np.float32)
    valid_preds = np.zeros((len(yv_enc), len(classes)), dtype=np.float32)
    test_preds  = np.zeros((len(yte_enc), len(classes)), dtype=np.float32)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(Xt_tr, ytr_enc)):
        _info(f"HistGB Fold {fold+1}/{n_splits} starting...")
        X_fold_tr, y_fold_tr = Xt_tr[trn_idx], ytr_enc[trn_idx]
        X_fold_val, y_fold_val = Xt_tr[val_idx], ytr_enc[val_idx]
        
        # HistGB Parameters specified by the Research Agent for 16GB
        clf = HistGradientBoostingClassifier(
            max_iter=200,             
            max_leaf_nodes=31,        
            max_bins=63,              # CRITICAL: Memory saver
            class_weight='balanced',  
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42 + fold
        )
        
        clf.fit(X_fold_tr, y_fold_tr)
        
        oof_train[val_idx] = clf.predict_proba(X_fold_val)
        valid_preds += clf.predict_proba(Xt_v) / n_splits
        test_preds  += clf.predict_proba(Xt_te) / n_splits
        
    _stage("HistGradientBoosting Validation Set Evaluation")
    yh_v = valid_preds.argmax(axis=1)
    rep_v = classification_report(yv_enc, yh_v, target_names=classes)
    print("VALIDATION REPORT:")
    print(rep_v)

    _stage("HistGradientBoosting Test Set Evaluation")
    yh_t = test_preds.argmax(axis=1)
    rep_t = classification_report(yte_enc, yh_t, target_names=classes)
    print("TEST REPORT:")
    print(rep_t)
    
    # Calculate Macro F1 explicitly
    _, _, fts, _ = precision_recall_fscore_support(yte_enc, yh_t, average='macro', zero_division=0)
    _info(f"HistGB Test Macro F1: {fts:.4f}")

    # Kaydet
    np.savez(cdir / "proba_histgb_oof_train.npz", proba=oof_train, classes=classes)
    np.savez(cdir / "proba_histgb_oof_valid.npz", proba=valid_preds, classes=classes)
    np.savez(cdir / "proba_histgb_oof_test.npz",  proba=test_preds, classes=classes)
    _info("Saved HistGB OOF probabilities.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', type=str, default='.')
    ap.add_argument('--n-features', type=int, default=65)
    args = ap.parse_args()
    
    from collections import namedtuple
    Config = namedtuple('Config', ['cache_dir'])
    cfg = Config(cache_dir=args.cache_dir)
    run_hist_gb_oof(cfg, n_features=args.n_features)
