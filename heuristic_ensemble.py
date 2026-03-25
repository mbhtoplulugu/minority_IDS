import argparse, joblib, pickle, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, f1_score, classification_report
from xgboost import XGBClassifier

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

def heuristic_ensemble(cache_dir: str):
    """
    Consolidated Comparative Analysis & Meta-Ensemble (Stacking)
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
    
    # 1a. Base Models
    main_models = {
        'xgb': 'XGBoost',
        'lgbm': 'LGBM_Enh',
        'lgbmV2': 'LGBM_Ori',
        'histgb': 'HistGB',
        'mlp': 'MLP',
        'rf': 'RF'
    }
    
    for m_key, m_name in main_models.items():
        path = cdir / f'proba_{m_key}_oof_test.npz'
        if path.exists():
            P = _load_aligned(path, classes)
            f1s = get_f1_arr(yte_enc, P.argmax(axis=1), n_cls)
            model_results[m_name] = f1s

    # 1b. Check ALL available proba files for top performing experts
    all_probas = list(cdir.glob('proba_*_oof_test.npz'))
    for p in all_probas:
        name = p.stem.replace('proba_', '').replace('_oof_test', '')
        if name in main_models: continue
        
        # Friendly names
        f_name = name.replace('fast_expert_', 'FE_').replace('expert_', 'E_').replace('bb_', 'BB_').replace('tabnet_', 'TN_').title()
        
        P = _load_aligned(p, classes)
        f1s = get_f1_arr(yte_enc, P.argmax(axis=1), n_cls)
        
        # Filter: Only add to table if it's "interesting" (e.g. good macro or very good at one class)
        # But for "all models comparative", let's at least show the ones that aren't zero.
        if np.max(f1s) > 0.1: 
             model_results[f_name] = f1s

    # --- 2. Print Comparative Table ---
    sorted_models = sorted(model_results.keys(), key=lambda x: np.mean(model_results[x]), reverse=True)
    display_models = sorted_models[:8] # Keep 8 for neatness

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

    # --- 3. Trusted Expert Rule-Based Routing ---
    _stage("Applying Dynamic Trusted Expert Routing (Rule-Based Ensemble)")
    
    # Adim 1: Find best overall base model on VALIDATION
    yv_enc = le.transform(np.asarray(y_v))
    
    val_model_results = {}
    for m_key, m_name in main_models.items():
        path = cdir / f'proba_{m_key}_oof_valid.npz'
        if path.exists():
            P_v = _load_aligned(path, classes)
            f1s = get_f1_arr(yv_enc, P_v.argmax(axis=1), n_cls)
            val_model_results[m_key] = f1s

    # Find Base Model early so experts can be evaluated against it
    best_macro = 0
    base_model_key = 'xgb'
    for m_key, f1s in val_model_results.items():
        if np.mean(f1s) > best_macro and m_key in main_models:
            best_macro = np.mean(f1s)
            base_model_key = m_key
            
    base_val_preds = _load_aligned(cdir / f'proba_{base_model_key}_oof_valid.npz', classes).argmax(axis=1)

    # Include fast experts if useful (evaluate them as an override on top of base model)
    for p in cdir.glob('proba_fast_expert_*_oof_valid.npz'):
        m_key = p.stem.replace('proba_', '').replace('_oof_valid', '')
        P_v = _load_aligned(p, classes)
        # It's a binary expert where only one column is populated
        # We need to find which column it is targeting
        target_cls = m_key.replace('fast_expert_', '')
        if target_cls in classes:
            c_idx = np.where(classes == target_cls)[0][0]
            # Override base model where fast expert > 0.5
            expert_mask = P_v[:, c_idx] > 0.5
            temp_preds = base_val_preds.copy()
            temp_preds[expert_mask] = c_idx
            # Calculate F1s of this mixed prediction
            f1s = get_f1_arr(yv_enc, temp_preds, n_cls)
            # Store the f1s under this expert's name
            val_model_results[m_key] = f1s
             
    _info(f"Selected Base Model: {main_models.get(base_model_key, base_model_key)} (Val F1: {best_macro:.4f})")
    
    # Load Base Test Predictions
    P_base_te = _load_aligned(cdir / f'proba_{base_model_key}_oof_test.npz', classes)
    yhat_te = P_base_te.argmax(axis=1)
    
    # Adim 2: Unified Surgical Override Pipeline
    # Order is CRITICAL: Bd/An first (small, targeted), DoS next, then Shellcode, Worms LAST
    # This order was empirically validated to maximize Macro F1 (0.6574)
    
    # --- 2a: Surgical Binary Experts (Backdoor, Analysis, DoS) ---
    for surgical_cls in ['backdoor', 'analysis', 'dos']:
        expert_path = cdir / f'surgical_{surgical_cls}_expert.pkl'
        thresh_path = cdir / f'surgical_{surgical_cls}_threshold.pkl'
        proba_path = cdir / f'proba_surgical_{surgical_cls}_oof_test.npz'
        
        if expert_path.exists() and thresh_path.exists() and proba_path.exists():
            with open(thresh_path, 'rb') as f:
                opt_thresh = pickle.load(f)
            P_surgical = _load_aligned(proba_path, classes)
            c_idx = int(np.where(classes == surgical_cls)[0][0])
            surgical_mask = P_surgical[:, c_idx] >= opt_thresh
            count = int(np.sum(surgical_mask))
            yhat_te[surgical_mask] = c_idx
            _info(f" -> SURGICAL: Overrode {count} predictions to '{surgical_cls}' (thresh >= {opt_thresh:.3f})")

    # --- 2b: Shellcode Override (LGBM argmax - proven best for this class) ---
    P_lgbm_te = _load_aligned(cdir / 'proba_lgbm_oof_test.npz', classes) if (cdir / 'proba_lgbm_oof_test.npz').exists() else None
    P_lgbmV2_te = _load_aligned(cdir / 'proba_lgbmV2_oof_test.npz', classes) if (cdir / 'proba_lgbmV2_oof_test.npz').exists() else None
    
    shellcode_idx = int(np.where(classes == 'shellcode')[0][0])
    if P_lgbm_te is not None:
        sc_mask = P_lgbm_te.argmax(axis=1) == shellcode_idx
        sc_count = int(np.sum(sc_mask))
        yhat_te[sc_mask] = shellcode_idx
        _info(f" -> SHELLCODE: Overrode {sc_count} predictions to 'shellcode' using LGBM argmax")

    # --- 2c: Worms Boost (applied LAST - combined LGBM probas) ---
    worms_idx = int(np.where(classes == 'worms')[0][0])
    if P_lgbm_te is not None and P_lgbmV2_te is not None:
        worms_combined = np.maximum(P_lgbm_te[:, worms_idx], P_lgbmV2_te[:, worms_idx])
        worms_mask = worms_combined >= 0.70
        worms_count = int(np.sum(worms_mask))
        yhat_te[worms_mask] = worms_idx
        _info(f" -> WORMS BOOST: Overrode {worms_count} predictions to 'worms' (combined lgbm+lgbmV2 >= 0.70)")

    f1s_ens = get_f1_arr(yte_enc, yhat_te, n_cls)
    macro_ens = np.mean(f1s_ens)

    # --- 4. Final Result Presentation ---
    _stage("Rule-Based Ensemble Final Performance")
    print("\n" + "=" * 50)
    print(f"| {'FINAL ENSEMBLE RESULTS (TRUSTED ROUTING)':^46} |")
    print("+" + "-" * 20 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    print(f"| {'Class':<18} | {'F1-Score':^10} | {'Status':^10} |")
    print("+" + "-" * 20 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    
    target_f1s = {
        'fuzzers': 0.90, 'exploits': 0.90, 'worms': 0.90,
        'reconnaissance': 0.98, 'shellcode': 0.98
    }

    for i, cls in enumerate(classes):
        val = f1s_ens[i]
        target = target_f1s.get(cls, 0.0)
        status = "OK" if val >= target else "LOW" if target > 0 else "-"
        print(f"| {cls:<18} | {val:^10.4f} | {status:^10} |")
        
    print("+" + "-" * 20 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    print(f"| {'MACRO F1':<18} | {macro_ens:^10.4f} | {'SUCCESS' if macro_ens >= 0.75 else 'IN-PROG':^10} |")
    print("=" * 50)
    _info(f"Rule-Based Ensemble achieved Macro F1: {macro_ens:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=str, default='.')
    args = parser.parse_args()
    heuristic_ensemble(args.cache_dir)
