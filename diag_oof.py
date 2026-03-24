import joblib, numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, classification_report

def check_oof(cache_dir='.'):
    cdir = Path(cache_dir)
    y_te = joblib.load(cdir/'y_te.joblib')
    le = joblib.load(cdir/'label_encoder.joblib')
    y_true = le.transform(y_te)
    classes = le.classes_
    
    files = {
        'XGB_Orig': 'proba_xgb_oof_test.npz',
        'LGBM_V2_Orig': 'proba_lgbmV2_oof_test.npz',
        'LGBM_Enh': 'proba_lgbm_oof_test.npz',
        'MLP_Enh': 'proba_mlp_oof_test.npz',
        'HistGB_Enh': 'proba_histgb_oof_test.npz',
        'DoS_Expert': 'proba_fast_expert_dos_oof_test.npz'
    }
    
    print(f"{'Model':<15} | {'Macro F1':<10} | {'DoS F1':<10} | {'Exploits F1':<11}")
    print("-" * 55)
    
    for name, fname in files.items():
        p = cdir / fname
        if not p.exists():
            print(f"{name:<15} | Not Found")
            continue
            
        data = np.load(p, allow_pickle=True)
        P = data['proba']
        src_classes = data['classes']
        
        # Align
        aligned_P = np.zeros((P.shape[0], len(classes)))
        idx_map = {c:i for i,c in enumerate(src_classes)}
        for j, c in enumerate(classes):
            if c in idx_map:
                aligned_P[:, j] = P[:, idx_map[c]]
        
        y_pred = aligned_P.argmax(axis=1)
        macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        dos_idx = np.where(classes == 'dos')[0][0]
        exp_idx = np.where(classes == 'exploits')[0][0]
        
        dos_f1 = f1_score(y_true == dos_idx, y_pred == dos_idx, zero_division=0)
        exp_f1 = f1_score(y_true == exp_idx, y_pred == exp_idx, zero_division=0)
        
        print(f"{name:<15} | {macro:.4f}     | {dos_f1:.4f}     | {exp_f1:.4f}")

if __name__ == '__main__':
    check_oof()
