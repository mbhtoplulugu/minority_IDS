"""
stack_experts.py - Level-2 Meta-Model (OOF Stacking)
----------------------------------------------------
- Loads OOF (Out-Of-Fold) probabilities from Layer-1 models:
  - XGBoost (proba_xgb_oof_*.npz)
  - Experts (proba_expert_{cls}_oof_*.npz)
- Concatenates them into a new feature matrix.
- Trains a Level-2 Logistic Regression Meta-Classifier on the train set.
- Predicts on VALID and TEST, applying F1-optimized thresholds.
"""
import argparse, joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

def _info(m): print(f"[Info]  {m}")
def _stage(m): print(f"[Stage] {m}")

def _pick_first_exist(cdir, patterns):
    for pat in patterns:
        p = cdir/pat
        if p.exists():
            return p
    return None

def _load_aligned(path, classes):
    data = np.load(path, allow_pickle=True)
    P = data['proba']
    src = data['classes'].astype(str)
    out = np.zeros((P.shape[0], len(classes)), dtype=np.float32)
    idx = {c:i for i,c in enumerate(src)}
    for j,c in enumerate(classes):
        if c in idx and idx[c] < P.shape[1]:
            out[:,j] = P[:, idx[c]]
    return out

def stack_with_experts(cache_dir: str, focus=None):
    cdir = Path(cache_dir)
    le = joblib.load(cdir/'label_encoder.joblib')
    classes = le.classes_.astype(str)

    if focus is not None:
        focus = [str(c) for c in focus if str(c) in classes]
        if focus:
            _info(f"Using requested expert classes: {focus}")
        else:
            _info("Requested expert classes did not match known classes; stacking base models only.")
    else:
        # Auto-detect expert classes from available OOF files — dataset agnostic.
        # Scans for proba_expert_<cls>_oof_train.npz and extracts <cls> from the name.
        focus = sorted({
            p.stem
            .replace('proba_expert_', '')
            .replace('_oof_train', '')
            for p in cdir.glob('proba_expert_*_oof_train.npz')
        } & set(classes))  # only keep classes the label encoder knows

        if focus:
            _info(f"Auto-detected expert classes: {focus}")
        else:
            _info("No expert OOF files found — stacking base models only.")

    y_tr = joblib.load(cdir/'y_tr.joblib')
    y_v  = joblib.load(cdir/'y_v.joblib')
    y_te = joblib.load(cdir/'y_te.joblib')
    ytr_enc = le.transform(np.asarray(y_tr))
    yv_enc  = le.transform(np.asarray(y_v))
    yt_enc  = le.transform(np.asarray(y_te))

    _stage("Assembling Level-1 OOF Features")
    
    # Feature block lists
    feat_train, feat_valid, feat_test = [], [], []

    # 1. Base XGBoost (OOF)
    xgb_train = _pick_first_exist(cdir, ['proba_xgb_oof_train.npz'])
    xgb_valid = _pick_first_exist(cdir, ['proba_xgb_oof_valid.npz'])
    xgb_test  = _pick_first_exist(cdir, ['proba_xgb_oof_test.npz'])
    
    if xgb_train and xgb_valid and xgb_test:
        _info("Loaded base XGBoost OOF probabilities")
        feat_train.append(_load_aligned(xgb_train, classes))
        feat_valid.append(_load_aligned(xgb_valid, classes))
        feat_test.append(_load_aligned(xgb_test, classes))
    else:
        raise RuntimeError("Missing XGBoost OOF predictions. Please run XGBoost training with OOF first.")

    # 1b. Base LightGBM (OOF)
    lgbm_train = _pick_first_exist(cdir, ['proba_lgbm_oof_train.npz'])
    lgbm_valid = _pick_first_exist(cdir, ['proba_lgbm_oof_valid.npz'])
    lgbm_test  = _pick_first_exist(cdir, ['proba_lgbm_oof_test.npz'])
    
    if lgbm_train and lgbm_valid and lgbm_test:
        _info("Loaded base LightGBM OOF probabilities")
        feat_train.append(_load_aligned(lgbm_train, classes))
        feat_valid.append(_load_aligned(lgbm_valid, classes))
        feat_test.append(_load_aligned(lgbm_test, classes))
    else:
        _info("LightGBM OOF not found, skipping feature.")

    # 1c. Base LightGBM V2 (OOF)
    lgbmV2_train = _pick_first_exist(cdir, ['proba_lgbmV2_oof_train.npz'])
    lgbmV2_valid = _pick_first_exist(cdir, ['proba_lgbmV2_oof_valid.npz'])
    lgbmV2_test  = _pick_first_exist(cdir, ['proba_lgbmV2_oof_test.npz'])
    
    if lgbmV2_train and lgbmV2_valid and lgbmV2_test:
        _info("Loaded base LightGBM V2 OOF probabilities")
        feat_train.append(_load_aligned(lgbmV2_train, classes))
        feat_valid.append(_load_aligned(lgbmV2_valid, classes))
        feat_test.append(_load_aligned(lgbmV2_test, classes))
    else:
        _info("LightGBM V2 OOF not found, skipping feature.")

    # 1d. Base RF (OOF)
    rf_train = _pick_first_exist(cdir, ['proba_rf_oof_train.npz'])
    rf_valid = _pick_first_exist(cdir, ['proba_rf_oof_valid.npz'])
    rf_test  = _pick_first_exist(cdir, ['proba_rf_oof_test.npz'])
    
    if rf_train and rf_valid and rf_test:
        _info("Loaded base Random Forest OOF probabilities")
        feat_train.append(_load_aligned(rf_train, classes))
        feat_valid.append(_load_aligned(rf_valid, classes))
        feat_test.append(_load_aligned(rf_test, classes))
    else:
        _info("Random Forest OOF not found, skipping feature.")

    # 1c. Base MLP (OOF)
    mlp_train = _pick_first_exist(cdir, ['proba_mlp_oof_train.npz'])
    mlp_valid = _pick_first_exist(cdir, ['proba_mlp_oof_valid.npz'])
    mlp_test  = _pick_first_exist(cdir, ['proba_mlp_oof_test.npz'])
    if mlp_train and mlp_valid and mlp_test:
        _info("Loaded base MLP OOF probabilities")
        feat_train.append(_load_aligned(mlp_train, classes))
        feat_valid.append(_load_aligned(mlp_valid, classes))
        feat_test.append(_load_aligned(mlp_test, classes))

    # 2. Experts (OOF)
    for cls in focus:
        if cls not in classes: continue
        t_p = cdir/f"proba_expert_{cls}_oof_train.npz"
        v_p = cdir/f"proba_expert_{cls}_oof_valid.npz"
        te_p = cdir/f"proba_expert_{cls}_oof_test.npz"
        if t_p.exists() and v_p.exists() and te_p.exists():
            _info(f"Loaded {cls} Expert OOF probabilities")
            feat_train.append(_load_aligned(t_p, classes))
            feat_valid.append(_load_aligned(v_p, classes))
            feat_test.append(_load_aligned(te_p, classes))
        else:
            _info(f"Expert OOF for {cls} not found, skipping feature.")

    X_meta_tr = np.hstack(feat_train)
    X_meta_v  = np.hstack(feat_valid)
    X_meta_te = np.hstack(feat_test)

    _info(f"Loaded Level-1 Models and Experts. Meta-Train Shape: {X_meta_tr.shape}")

    # Diagnostic: show per-class sample counts so starvation is visible
    _stage("Meta-Train Class Distribution")
    unique_cls, counts = np.unique(ytr_enc, return_counts=True)
    for ci, cnt in zip(unique_cls, counts):
        flag = "  <<< STARVED" if cnt < 10 else ""
        _info(f"  {classes[ci]:<30} {cnt:>6} samples{flag}")

    _stage("Training Meta-Model (Logistic Regression + Balanced Weights)")
    # class_weight='balanced' is critical: without it, rare classes (heartbleed,
    # infiltration, web attack sql) are overwhelmed by majority classes and the
    # LR never picks them as argmax → F1=0.
    # C=4.0 relaxes regularisation slightly so minority-class weights can be learned.
    meta_model = LogisticRegression(
        max_iter=5000,
        multi_class='multinomial',
        solver='lbfgs',
        C=4.0,
        class_weight='balanced',
    )
    meta_model.fit(X_meta_tr, ytr_enc)

    Pv_ensemble = meta_model.predict_proba(X_meta_v)
    Pt_ensemble = meta_model.predict_proba(X_meta_te)

    yhat_v = np.argmax(Pv_ensemble, axis=1)
    yhat_t = np.argmax(Pt_ensemble, axis=1)

    # --- COMPARISON TABLE LOGIC ---
    _stage("Detailed Class-wise Model Comparison (Test Set)")
    def get_f1_per_class(P, use_argmax=False):
        yh = P.argmax(axis=1)
        _, _, f1_arr, _ = precision_recall_fscore_support(yt_enc, yh, labels=np.arange(len(classes)), zero_division=0)
        return f1_arr

    # Extract all probabilities again for metrics printing
    comp_metrics = {}
    acc_metrics = {}

    if xgb_test:
        P_xgb = _load_aligned(xgb_test, classes)
        comp_metrics['XGBoost'] = get_f1_per_class(P_xgb, use_argmax=True)
        acc_metrics['XGBoost'] = accuracy_score(yt_enc, P_xgb.argmax(axis=1))
        
    if lgbm_test:
        P_lgbm = _load_aligned(lgbm_test, classes)
        comp_metrics['LightGBM'] = get_f1_per_class(P_lgbm, use_argmax=True)
        acc_metrics['LightGBM'] = accuracy_score(yt_enc, P_lgbm.argmax(axis=1))
        
    if lgbmV2_test:
        P_lgbmV2 = _load_aligned(lgbmV2_test, classes)
        comp_metrics['LightGBM_V2'] = get_f1_per_class(P_lgbmV2, use_argmax=True)
        acc_metrics['LightGBM_V2'] = accuracy_score(yt_enc, P_lgbmV2.argmax(axis=1))
        
    if rf_test:
        P_rf = _load_aligned(rf_test, classes)
        comp_metrics['RandomForest'] = get_f1_per_class(P_rf, use_argmax=True)
        acc_metrics['RandomForest'] = accuracy_score(yt_enc, P_rf.argmax(axis=1))
        
    if mlp_test:
        P_mlp = _load_aligned(mlp_test, classes)
        comp_metrics['MLP'] = get_f1_per_class(P_mlp, use_argmax=True)
        acc_metrics['MLP'] = accuracy_score(yt_enc, P_mlp.argmax(axis=1))
        
    for cls in focus:  # focus is already auto-detected and validated against classes
        te_p = cdir/f"proba_expert_{cls}_oof_test.npz"
        if te_p.exists():
            P_exp = _load_aligned(te_p, classes)
            comp_metrics[f'Expert ({cls})'] = get_f1_per_class(P_exp, use_argmax=True)
            acc_metrics[f'Expert ({cls})'] = accuracy_score(yt_enc, P_exp.argmax(axis=1))

    comp_metrics['Meta-Model (Ensemble)'] = get_f1_per_class(Pt_ensemble)
    acc_metrics['Meta-Model (Ensemble)'] = accuracy_score(yt_enc, Pt_ensemble.argmax(axis=1))

    print("\n" + "="*80)
    print(f"{'Class':<15} |", end="")
    model_names = list(comp_metrics.keys())
    for m in model_names:
        print(f" {m[:14]:<14} |", end="")
    print("\n" + "-"*80)
    
    for i, cls in enumerate(classes):
        print(f"{cls:<15} |", end="")
        for m in model_names:
            val = comp_metrics[m][i]
            print(f" {val:.4f}{' '*8} |", end="")
        print()
    
    print("-" * 80)
    print(f"{'Macro F1':<15} |", end="")
    for m in model_names:
        mac = np.mean(comp_metrics[m])
        print(f" {mac:.4f}{' '*8} |", end="")
    print("\n" + "-"*80)
    
    print(f"{'Accuracy':<15} |", end="")
    for m in model_names:
        acc = acc_metrics[m]
        print(f" {acc:.4f}{' '*8} |", end="")
    print("\n" + "="*80 + "\n")
    
    # Save Meta Model Predictions
    
    # Save Meta Model Predictions
    np.savez(cdir/"proba_stack_meta_valid.npz", proba=Pv_ensemble, classes=classes)
    np.savez(cdir/"proba_stack_meta_test.npz", proba=Pt_ensemble, classes=classes)

def main():
    ap = argparse.ArgumentParser(description='Level-2 OOF Stacking Meta-Classifier')
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument('--unsw',     action='store_true', default=True, help='UNSW-NB15 modu (default)')
    mode_group.add_argument('--cicids',   action='store_true', help='CICIDS17 modu')
    mode_group.add_argument('--cicids14', action='store_true', help='CICIDS17 14 sinifli modu')

    ap.add_argument('--cache-dir', type=str, default='.')
    ap.add_argument('--exclude-weak-classes', action='store_true')
    ap.add_argument('--focus', nargs='*', default=None, help='Optional expert classes to include in stacking')
    args = ap.parse_args()

    dataset_mode    = 'cicids14' if args.cicids14 else ('cicids' if args.cicids else 'unsw')
    cache_mode_name = dataset_mode + "_excluded" if args.exclude_weak_classes else dataset_mode
    target_cache_dir = Path(args.cache_dir) / cache_mode_name

    stack_with_experts(cache_dir=str(target_cache_dir), focus=args.focus)

if __name__ == '__main__':
    main()
