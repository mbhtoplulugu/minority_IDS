import joblib, pickle, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

def _load_aligned(path, classes):
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

def run_ensemble_test(cache_dir: str, config: dict):
    cdir = Path(cache_dir)
    le = joblib.load(cdir/'label_encoder.joblib')
    classes = le.classes_.astype(str)
    n_cls = len(classes)
    
    y_te = joblib.load(cdir/'y_te.joblib')
    y_v  = joblib.load(cdir/'y_v.joblib')
    yte_enc = le.transform(np.asarray(y_te))
    yv_enc  = le.transform(np.asarray(y_v))
    
    main_models = {
        'xgb': 'XGBoost',
        'lgbm': 'LGBM_Enh',
        'lgbmV2': 'LGBM_Ori',
        'histgb': 'HistGB',
        'mlp': 'MLP',
        'tabnet': 'TabNet',
        'rf': 'RF'
    }
    
    # Load validation results
    val_model_results = {}
    for m_key, m_name in main_models.items():
        path = cdir / f'proba_{m_key}_oof_valid.npz'
        if path.exists():
            P_v = _load_aligned(path, classes)
            f1s = get_f1_arr(yv_enc, P_v.argmax(axis=1), n_cls)
            val_model_results[m_key] = f1s
            
    best_macro = 0
    base_model_key = 'xgb'
    for m_key, f1s in val_model_results.items():
        if np.mean(f1s) > best_macro and m_key in main_models:
            best_macro = np.mean(f1s)
            base_model_key = m_key
            
    P_base_te = _load_aligned(cdir / f'proba_{base_model_key}_oof_test.npz', classes)
    base_val_preds_xgb = _load_aligned(cdir / f'proba_{base_model_key}_oof_valid.npz', classes).argmax(axis=1)
    
    # 1. Base Predictions
    if config.get('use_soft_voting_base', False):
        # Compute Weighted Soft Voting on Validation first to get base_val_f1
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
        yhat_val = P_soft_val.argmax(axis=1)
        base_val_f1 = get_f1_arr(yv_enc, yhat_val, n_cls)
        base_val_preds = yhat_val.copy()
        
        # Test Set base predictions
        P_soft_te = np.zeros_like(P_base_te)
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
    else:
        yhat_te = P_base_te.argmax(axis=1)
        base_val_f1 = val_model_results.get(base_model_key, np.zeros(n_cls))
        base_val_preds = base_val_preds_xgb.copy()
        
    # Phase A Override
    if config.get('run_phase_a', False):
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
            
            if proba_val_path.exists():
                P_surg_val = _load_aligned(proba_val_path, classes)
                with open(thresh_path, 'rb') as f:
                    opt_thresh = pickle.load(f)
                temp_val = base_val_preds.copy()
                surg_val_mask = P_surg_val[:, c_idx] >= opt_thresh
                temp_val[surg_val_mask] = c_idx
                mixed_f1 = get_f1_arr(yv_enc, temp_val, n_cls)
                
                if np.mean(mixed_f1) <= np.mean(base_val_f1):
                    apply_override = False
            
            if apply_override:
                with open(thresh_path, 'rb') as f:
                    opt_thresh = pickle.load(f)
                P_surgical = _load_aligned(proba_path, classes)
                surgical_mask = P_surgical[:, c_idx] >= opt_thresh
                yhat_te[surgical_mask] = c_idx

    # Phase A.2 Override
    if config.get('run_phase_a2', False):
        for expert_val_path in sorted(cdir.glob('proba_fast_expert_*_oof_valid.npz')):
            m_key = expert_val_path.stem.replace('proba_', '').replace('_oof_valid', '')
            cls_name = m_key.replace('fast_expert_', '')
            if cls_name not in classes: continue
            
            expert_test_path = cdir / f'proba_{m_key}_oof_test.npz'
            if not expert_test_path.exists(): continue
            
            c_idx = int(np.where(classes == cls_name)[0][0])
            P_fe_val = _load_aligned(expert_val_path, classes)
            temp_val = base_val_preds.copy()
            fe_val_mask = P_fe_val[:, c_idx] > 0.5
            temp_val[fe_val_mask] = c_idx
            mixed_f1 = get_f1_arr(yv_enc, temp_val, n_cls)
            
            if np.mean(mixed_f1) > np.mean(base_val_f1):
                P_fe_te = _load_aligned(expert_test_path, classes)
                fe_test_mask = P_fe_te[:, c_idx] > 0.5
                yhat_te[fe_test_mask] = c_idx

    # Phase B Override
    if config.get('run_phase_b', False):
        for m_key in list(val_model_results.keys()):
            if m_key == base_model_key or m_key not in main_models:
                continue
            test_path = cdir / f'proba_{m_key}_oof_test.npz'
            if not test_path.exists():
                continue
            alt_f1 = val_model_results[m_key]
            P_alt_te = _load_aligned(test_path, classes)
            
            for c_idx in range(n_cls):
                cls_name = classes[c_idx]
                if cls_name in ('normal', 'generic'):
                    continue
                improvement = alt_f1[c_idx] - base_val_f1[c_idx]
                if improvement > 0.01:
                    alt_mask = P_alt_te.argmax(axis=1) == c_idx
                    yhat_te[alt_mask] = c_idx

    # Phase C Override
    if config.get('run_phase_c', False):
        boost_type = config.get('phase_c_type', 'max') # 'max', 'agreement', 'sum'
        for c_idx in range(n_cls):
            cls_name = classes[c_idx]
            if cls_name in ('normal', 'generic'):
                continue
            if base_val_f1[c_idx] >= 0.50:
                continue
            
            probas_list = []
            for m_key in main_models:
                p_path = cdir / f'proba_{m_key}_oof_test.npz'
                if p_path.exists():
                    P_m = _load_aligned(p_path, classes)
                    probas_list.append(P_m[:, c_idx])
            
            if len(probas_list) >= 2:
                probas_arr = np.array(probas_list)
                
                if boost_type == 'max':
                    combined_proba = np.max(probas_arr, axis=0)
                    boost_mask = combined_proba >= 0.85
                elif boost_type == 'agreement':
                    agreement_count = np.sum(probas_arr >= 0.40, axis=0)
                    max_proba = np.max(probas_arr, axis=0)
                    boost_mask = (agreement_count >= 2) & (max_proba >= 0.85)
                elif boost_type == 'sum':
                    sorted_probas = np.sort(probas_arr, axis=0)
                    combined_proba = sorted_probas[-1] + sorted_probas[-2]
                    boost_mask = combined_proba >= 1.30
                else:
                    boost_mask = np.zeros(len(yhat_te), dtype=bool)
                
                yhat_te[boost_mask] = c_idx

    f1s = get_f1_arr(yte_enc, yhat_te, n_cls)
    return classes, f1s, np.mean(f1s)

if __name__ == '__main__':
    datasets = {
        'UNSW-NB15': r'c:\Users\mbhto\source\repos\minority_IDS',
        'CICIDS-17': r'c:\Users\mbhto\source\repos\minority_IDS\cicids'
    }
    
    configs = {
        '1. Soft Voting Base Only': {
            'use_soft_voting_base': True,
            'run_phase_a': False,
            'run_phase_a2': False,
            'run_phase_b': False,
            'run_phase_c': False,
        },
        '2. Strict Heuristic (Original)': {
            'use_soft_voting_base': False,
            'run_phase_a': True,
            'run_phase_a2': True,
            'run_phase_b': True,
            'run_phase_c': True,
            'phase_c_type': 'max'
        },
        '3. Soft Voting + Phase A/A.2 Overrides Only': {
            'use_soft_voting_base': True,
            'run_phase_a': True,
            'run_phase_a2': True,
            'run_phase_b': False,
            'run_phase_c': False,
        },
        '4. Unified Heuristic (Soft Voting Base + Safe Phase C Agreement)': {
            'use_soft_voting_base': True,
            'run_phase_a': True,
            'run_phase_a2': True,
            'run_phase_b': True,
            'run_phase_c': True,
            'phase_c_type': 'agreement'
        },
        '5. Unified Heuristic (Soft Voting Base + Safe Phase C Sum)': {
            'use_soft_voting_base': True,
            'run_phase_a': True,
            'run_phase_a2': True,
            'run_phase_b': True,
            'run_phase_c': True,
            'phase_c_type': 'sum'
        },
        '6. Original Strict + Safe Phase C Agreement (XGBoost Base)': {
            'use_soft_voting_base': False,
            'run_phase_a': True,
            'run_phase_a2': True,
            'run_phase_b': True,
            'run_phase_c': True,
            'phase_c_type': 'agreement'
        },
        '7. Original Strict + Safe Phase C Sum (XGBoost Base)': {
            'use_soft_voting_base': False,
            'run_phase_a': True,
            'run_phase_a2': True,
            'run_phase_b': True,
            'run_phase_c': True,
            'phase_c_type': 'sum'
        }
    }
    
    for d_name, d_path in datasets.items():
        if not os.path.exists(d_path):
            print(f"Directory {d_path} not found. Skipping {d_name}.")
            continue
        print(f"\n=================== DATASET: {d_name} ===================")
        for c_name, cfg in configs.items():
            classes, f1s, macro = run_ensemble_test(d_path, cfg)
            print(f"\n--- Strategy: {c_name} ---")
            print(f"Macro F1: {macro:.4f}")
            for c, f in zip(classes, f1s):
                print(f"  {c:<15}: {f:.4f}")