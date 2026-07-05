import joblib, pickle
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

def evaluate_expert_contributions(cache_dir='.'):
    cdir = Path(cache_dir)
    le = joblib.load(cdir/'label_encoder.joblib')
    classes = le.classes_.astype(str)
    n_cls = len(classes)
    
    y_te = joblib.load(cdir/'y_te.joblib')
    yte_enc = le.transform(np.asarray(y_te))

    # --- 1. Base Consensus (Soft Voting) on TEST ---
    main_models = ['xgb', 'lgbm', 'lgbmV2', 'histgb', 'mlp', 'tabnet', 'rf']
    
    # Val weights for soft voting
    y_v = joblib.load(cdir/'y_v.joblib')
    yv_enc = le.transform(np.asarray(y_v))
    
    val_model_results = {}
    for m_key in main_models:
        path = cdir / f'proba_{m_key}_oof_valid.npz'
        if path.exists():
            P_v = _load_aligned(path, classes)
            f1s = get_f1_arr(yv_enc, P_v.argmax(axis=1), n_cls)
            val_model_results[m_key] = f1s

    dummy_path = cdir / f'proba_xgb_oof_test.npz'
    if not dummy_path.exists():
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
    base_te_preds = P_soft_te.argmax(axis=1)
    base_te_f1 = get_f1_arr(yte_enc, base_te_preds, n_cls)
    
    from sklearn.metrics import accuracy_score
    base_te_acc = accuracy_score(yte_enc, base_te_preds)
    
    print("\n" + "="*110)
    print(f"| {'IKILI UZMANLARIN (BINARY EXPERTS) SINIF BAZLI KATKI ANALIZI':^106} |")
    print("="*110)
    print(f"| {'Sınıf Adı':<15} | {'Uzman Türü':<15} | {'Base F1':<8} | {'Yeni F1':<8} | {'F1 Fark':<8} | {'Base Acc':<8} | {'Yeni Acc':<8} | {'Acc Fark':<8} |")
    print("-" * 110)

    # --- 2. FAST EXPERTS EVALUATION ON TEST ---
    yhat_te = base_te_preds.copy()
    
    for expert_test_path in sorted(cdir.glob('proba_fast_expert_*_oof_test.npz')):
        m_key = expert_test_path.stem.replace('proba_', '').replace('_oof_test', '')
        cls_name = m_key.replace('fast_expert_', '')
        if cls_name not in classes: continue
        
        c_idx = int(np.where(classes == cls_name)[0][0])
        P_fe_te = _load_aligned(expert_test_path, classes)
        
        fe_test_mask = P_fe_te[:, c_idx] > 0.5
        temp_te = base_te_preds.copy()
        temp_te[fe_test_mask] = c_idx
        
        mixed_te_f1 = get_f1_arr(yte_enc, temp_te, n_cls)
        new_te_acc = accuracy_score(yte_enc, temp_te)
        
        base_f1 = base_te_f1[c_idx]
        new_f1 = mixed_te_f1[c_idx]
        diff = new_f1 - base_f1
        acc_diff = new_te_acc - base_te_acc
        
        print(f"| {cls_name:<15} | {'Fast Expert':<15} | {base_f1:<8.4f} | {new_f1:<8.4f} | {diff:>+8.4f} | {base_te_acc:<8.4f} | {new_te_acc:<8.4f} | {acc_diff:>+8.4f} |")

    # --- 3. SURGICAL EXPERTS EVALUATION ON TEST ---
    for expert_test_path in sorted(cdir.glob('proba_surgical_*_oof_test.npz')):
        m_key = expert_test_path.stem.replace('proba_', '').replace('_oof_test', '')
        cls_name = m_key.replace('surgical_', '').replace('_expert', '')
        if cls_name not in classes: continue
        
        thresh_path = cdir / f'surgical_{cls_name}_threshold.pkl'
        if not thresh_path.exists(): continue
            
        c_idx = int(np.where(classes == cls_name)[0][0])
        P_surg_te = _load_aligned(expert_test_path, classes)
        
        with open(thresh_path, 'rb') as f:
            opt_thresh = pickle.load(f)
            
        surg_test_mask = P_surg_te[:, c_idx] >= opt_thresh
        temp_te = base_te_preds.copy()
        temp_te[surg_test_mask] = c_idx
        
        mixed_te_f1 = get_f1_arr(yte_enc, temp_te, n_cls)
        new_te_acc = accuracy_score(yte_enc, temp_te)
        
        base_f1 = base_te_f1[c_idx]
        new_f1 = mixed_te_f1[c_idx]
        diff = new_f1 - base_f1
        acc_diff = new_te_acc - base_te_acc
        
        print(f"| {cls_name:<15} | {'Surgical Expert':<15} | {base_f1:<8.4f} | {new_f1:<8.4f} | {diff:>+8.4f} | {base_te_acc:<8.4f} | {new_te_acc:<8.4f} | {acc_diff:>+8.4f} |")

    print("=" * 110)
    print("Not: 'Base F1/Acc', tüm temel modellerin soft-voting ile elde ettiği sonuçtur.")
    print("     'Yeni F1/Acc', temel model kararının üzerine uzmanın tahminleri eklendiğinde sistemin ulaştığı sonuçtur.")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="İkili Uzman Katkı Analizi Aracı")
    
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument('--unsw', action='store_true', default=True, help='UNSW-NB15 modu (default)')
    mode_group.add_argument('--cicids', action='store_true', help='CICIDS17 modu')
    mode_group.add_argument('--cicids14', action='store_true', help='CICIDS17 14 sinifli modu')
    
    ap.add_argument('--cache-dir', type=str, default='.')
    args = ap.parse_args()
    
    dataset_mode = 'cicids14' if args.cicids14 else ('cicids' if args.cicids else 'unsw')
    target_cache_dir = Path(args.cache_dir) / dataset_mode
    
    print(f"Mod: {dataset_mode.upper()} - Dizin: {target_cache_dir} üzerinden analiz ediliyor...")
    evaluate_expert_contributions(str(target_cache_dir))
