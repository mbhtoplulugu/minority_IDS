import argparse, joblib, pickle, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, f1_score, classification_report, accuracy_score

def _info(m):  print(f"[Info]  {m}")
def _stage(m): print(f"[Stage] {m}")

def _load_aligned(path, classes):
    """Loads probabilities and aligns them to the standard class order."""
    data = np.load(path, allow_pickle=True)
    P = data['proba']
    if 'classes' in data:
        src = data['classes'].astype(str)
        out = np.zeros((P.shape[0], len(classes)), dtype=np.float32)
        idx = {c:i for i,c in enumerate(src)}
        for j,c in enumerate(classes):
            if c in idx and idx[c] < P.shape[1]:
                out[:,j] = P[:, idx[c]]
        return out
    return P

def get_f1_arr(y_true, y_pred_encoded, n_cls):
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_encoded, labels=range(n_cls), zero_division=0)
    return f1

def heuristic_ensemble(cache_dir: str, exclude_weak: bool = False):
    """
    Unified Hybrid Ensemble Architecture
    Base: Class-Wise Validation-Weighted Soft Voting
    Overrides: Autonomous Surgical & Fast Binary Experts
    """
    cdir = Path(cache_dir)
    le = joblib.load(cdir/'label_encoder.joblib')
    classes = le.classes_.astype(str)
    n_cls = len(classes)
    
    _stage("Loading Ground Truth and Model Predictions")
    y_te = joblib.load(cdir/'y_te.joblib')
    y_v  = joblib.load(cdir/'y_v.joblib')
    
    yte_enc = le.transform(np.asarray(y_te))
    yv_enc  = le.transform(np.asarray(y_v))

    # --- 1. Comparative Analysis ---
    _stage("Performing Comparative Analysis of Individual Models")
    
    model_results = {}
    
    # Base Models
    main_models = {
        'xgb': 'XGBoost',
        'lgbm': 'LGBM_Enh',
        'lgbmV2': 'LGBM_Ori',
        'histgb': 'HistGB',
        'mlp': 'MLP',
        'tabnet': 'TabNet',
        'rf': 'RF'
    }
    
    for m_key, m_name in main_models.items():
        path = cdir / f'proba_{m_key}_oof_test.npz'
        if path.exists():
            P = _load_aligned(path, classes)
            f1s = get_f1_arr(yte_enc, P.argmax(axis=1), n_cls)
            model_results[m_name] = f1s

    # Load Fast Experts and Surgical Experts for the Comparative Table
    all_probas = list(cdir.glob('proba_*_oof_test.npz'))
    for p in all_probas:
        name = p.stem.replace('proba_', '').replace('_oof_test', '')
        if name in main_models: continue
        
        if 'aecnn' in name.lower() or 'ae_cnn' in name.lower(): continue
        
        # Friendly names
        f_name = name.replace('fast_expert_', 'FE_').replace('expert_', 'E_').replace('bb_', 'BB_').replace('tabnet_', 'TN_').title()
        
        P = _load_aligned(p, classes)
        f1s = get_f1_arr(yte_enc, P.argmax(axis=1), n_cls)
        
        if np.max(f1s) > 0.01: 
             model_results[f_name] = f1s

    # Print Comparative Table
    sorted_models = sorted(model_results.keys(), key=lambda x: np.mean(model_results[x]), reverse=True)
    display_models = sorted_models[:15] # Display up to 15 models to include experts

    print("\n" + "=" * 110)
    print(f"| {'DETAILED COMPARATIVE PERFORMANCE ANALYSIS (Test Set F1-Scores)':^106} |")
    print("+" + "-" * 15 + "+" + ("-" * 10 + "+") * len(display_models))
    
    header = f"| {'Class':<13} |"
    for m in display_models:
        header += f" {m[:8]:^8} |"
    print(header)
    print("+" + "-" * 15 + "+" + ("-" * 10 + "+") * len(display_models))

    for i, cls in enumerate(classes):
        row = f"| {cls:<13} |"
        for m in display_models:
            row += f" {model_results[m][i]:^8.4f} |"
        print(row)

    print("+" + "-" * 15 + "+" + ("-" * 10 + "+") * len(display_models))
    
    macro_row = f"| {'Macro F1':<13} |"
    for m in display_models:
        macro_row += f" {np.mean(model_results[m]):^8.4f} |"
    print(macro_row)
    print("+" + "-" * 15 + "+" + ("-" * 10 + "+") * len(display_models))

    # --- 2. Base Consensus: Class-Wise Validation-Weighted Soft Voting ---
    _stage("Establishing Base Consensus: Class-Wise Validation-Weighted Soft Voting")
    
    # Get validation F1 scores for weighting
    val_model_results = {}
    for m_key in main_models:
        path = cdir / f'proba_{m_key}_oof_valid.npz'
        if path.exists():
            P_v = _load_aligned(path, classes)
            f1s = get_f1_arr(yv_enc, P_v.argmax(axis=1), n_cls)
            val_model_results[m_key] = f1s

    # Compute validation soft voting predictions to establish the baseline for experts
    P_soft_val = np.zeros((len(yv_enc), n_cls), dtype=np.float32)
    weight_sum = np.zeros(n_cls)
    for m_key in main_models:
        if m_key in val_model_results:
            weight_arr = val_model_results[m_key]
            p_path = cdir / f'proba_{m_key}_oof_valid.npz'
            if p_path.exists():
                P_m = _load_aligned(p_path, classes)
                P_soft_val += P_m * weight_arr
                weight_sum += weight_arr
                
    weight_sum[weight_sum == 0] = 1e-9
    P_soft_val /= weight_sum
    
    base_val_preds = P_soft_val.argmax(axis=1)
    base_val_f1 = get_f1_arr(yv_enc, base_val_preds, n_cls)
    
    _info(f"Consensus Base Model (Soft Voting) achieved Validation Macro F1: {np.mean(base_val_f1):.4f}")

    # Compute test soft voting predictions
    # Load one dummy test proba to safely get the target array shape
    dummy_path = cdir / f'proba_xgb_oof_test.npz'
    if not dummy_path.exists():
        # Fallback if xgb doesn't exist
        for m in main_models:
            if (cdir / f'proba_{m}_oof_test.npz').exists():
                dummy_path = cdir / f'proba_{m}_oof_test.npz'
                break

    dummy_P = _load_aligned(dummy_path, classes)
    
    P_soft_te = np.zeros_like(dummy_P)
    weight_sum_te = np.zeros(n_cls)
    for m_key in main_models:
        if m_key in val_model_results:
            weight_arr = val_model_results[m_key]
            p_path = cdir / f'proba_{m_key}_oof_test.npz'
            if p_path.exists():
                P_m = _load_aligned(p_path, classes)
                P_soft_te += P_m * weight_arr
                weight_sum_te += weight_arr
                
    weight_sum_te[weight_sum_te == 0] = 1e-9
    P_soft_te /= weight_sum_te
    yhat_te = P_soft_te.argmax(axis=1)

    # --- 3. Autonomous Surgical & Fast Expert Overrides ---
    _stage("Applying Autonomous Surgical Overrides (Validation-Gated)")
    
    # Phase A: Surgical Binary Expert Override
    for expert_pkl in sorted(cdir.glob('surgical_*_expert.pkl')):
        cls_name = expert_pkl.stem.replace('surgical_', '').replace('_expert', '')
        thresh_path = cdir / f'surgical_{cls_name}_threshold.pkl'
        proba_path = cdir / f'proba_surgical_{cls_name}_oof_test.npz'
        proba_val_path = cdir / f'proba_surgical_{cls_name}_oof_valid.npz'
        
        if not (thresh_path.exists() and proba_path.exists()):
            continue
        if cls_name not in classes:
            continue
            
        c_idx = int(np.where(classes == cls_name)[0][0])
        apply_override = True
        
        # Verify on validation set against Soft Voting consensus
        if proba_val_path.exists():
            P_surg_val = _load_aligned(proba_val_path, classes)
            with open(thresh_path, 'rb') as f:
                opt_thresh = pickle.load(f)
                
            temp_val = base_val_preds.copy()
            surg_val_mask = P_surg_val[:, c_idx] >= opt_thresh
            temp_val[surg_val_mask] = c_idx
            mixed_f1 = get_f1_arr(yv_enc, temp_val, n_cls)
            
            # The surgical expert is only applied if it strictly improves the consensus F1
            if np.mean(mixed_f1) <= np.mean(base_val_f1):
                _info(f"  -> {cls_name}: Surgical expert BYPASSED (No Macro F1 improvement over consensus)")
                apply_override = False
        
        if apply_override:
            with open(thresh_path, 'rb') as f:
                opt_thresh = pickle.load(f)
            P_surgical = _load_aligned(proba_path, classes)
            surgical_mask = P_surgical[:, c_idx] >= opt_thresh
            count = int(np.sum(surgical_mask))
            if count > 0:
                yhat_te[surgical_mask] = c_idx
                _info(f"  -> SURGICAL: {cls_name} expert overrode {count} predictions (thresh >= {opt_thresh:.3f})")

    # Phase A.2: Fast Binary Expert Override
    for expert_val_path in sorted(cdir.glob('proba_fast_expert_*_oof_valid.npz')):
        m_key = expert_val_path.stem.replace('proba_', '').replace('_oof_valid', '')
        cls_name = m_key.replace('fast_expert_', '')
        if cls_name not in classes: continue
        
        expert_test_path = cdir / f'proba_{m_key}_oof_test.npz'
        if not expert_test_path.exists(): continue
        
        c_idx = int(np.where(classes == cls_name)[0][0])
        
        # Optimize edilmiş eşiği yükle, yoksa 0.5 kullan
        thresh_path = cdir / f'fast_threshold_{cls_name}.pkl'
        if thresh_path.exists():
            with open(thresh_path, 'rb') as f:
                opt_thresh = pickle.load(f)
            _info(f"  -> {cls_name}: Using optimized threshold {opt_thresh:.3f}")
        else:
            opt_thresh = 0.5
            _info(f"  -> {cls_name}: No threshold file found, using default 0.5")
        
        P_fe_val = _load_aligned(expert_val_path, classes)
        temp_val = base_val_preds.copy()
        fe_val_mask = P_fe_val[:, c_idx] >= opt_thresh
        temp_val[fe_val_mask] = c_idx
        mixed_f1 = get_f1_arr(yv_enc, temp_val, n_cls)
        
        if np.mean(mixed_f1) <= np.mean(base_val_f1):
            _info(f"  -> {cls_name}: Fast expert BYPASSED (No Macro F1 improvement over consensus)")
        else:
            P_fe_te = _load_aligned(expert_test_path, classes)
            fe_test_mask = P_fe_te[:, c_idx] >= opt_thresh
            count = int(np.sum(fe_test_mask))
            if count > 0:
                yhat_te[fe_test_mask] = c_idx
                _info(f"  -> FAST EXPERT: {cls_name} expert overrode {count} predictions (thresh >= {opt_thresh:.3f})")

    # --- FINAL: Evaluate and Print Results ---
    f1s_ens = get_f1_arr(yte_enc, yhat_te, n_cls)
    macro_ens = np.mean(f1s_ens)

    _stage("Hybrid Ensemble Final Performance")
    print("\n" + "=" * 50)
    print(f"| {'FINAL ENSEMBLE RESULTS (AUTO-ROUTING)':^46} |")
    print("+" + "-" * 20 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    print(f"| {'Class':<18} | {'F1-Score':^10} | {'Status':^10} |")
    print("+" + "-" * 20 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    
    median_f1 = float(np.median(f1s_ens[f1s_ens > 0])) if np.any(f1s_ens > 0) else 0.5

    for i, cls in enumerate(classes):
        val = f1s_ens[i]
        status = "OK" if val >= median_f1 else "LOW" if val > 0 else "-"
        print(f"| {cls:<18} | {val:^10.4f} | {status:^10} |")
        
    acc_ens = accuracy_score(yte_enc, yhat_te)
    print("+" + "-" * 20 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    print(f"| {'MACRO F1':<18} | {macro_ens:^10.4f} | {'SUCCESS' if macro_ens >= 0.75 else 'IN-PROG':^10} |")
    print(f"| {'ACCURACY':<18} | {acc_ens:^10.4f} | {'-':^10} |")
    print("=" * 50)
    _info(f"Hybrid Ensemble achieved Macro F1: {macro_ens:.4f}")
    _info(f"Overall Accuracy: {acc_ens:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--unsw', action='store_true', default=True, help='UNSW-NB15 modu (default)')
    mode_group.add_argument('--cicids', action='store_true', help='CICIDS17 modu')
    mode_group.add_argument('--cicids14', action='store_true', help='CICIDS17 14 sinifli modu')
    
    parser.add_argument('--cache-dir', type=str, default='.')
    parser.add_argument('--exclude-weak-classes', action='store_true')
    args = parser.parse_args()
    
    dataset_mode = 'cicids14' if args.cicids14 else ('cicids' if args.cicids else 'unsw')
    cache_mode_name = dataset_mode + "_excluded" if args.exclude_weak_classes else dataset_mode
    target_cache_dir = Path(args.cache_dir) / cache_mode_name
    
    heuristic_ensemble(str(target_cache_dir), args.exclude_weak_classes)
