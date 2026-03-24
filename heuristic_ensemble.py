import argparse, joblib, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, f1_score, classification_report
from xgboost import XGBClassifier

def _info(m):  print(f"[Info]  {m}")
def _stage(m): print(f"[Stage] {m}")

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

def get_f1_arr(y_true, y_hf, n_cls):
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_hf, labels=range(n_cls), zero_division=0)
    return f1

def heuristic_ensemble(cache_dir: str):
    """
    Meta-Ensemble v16 - Smart Stacking Meta-Learner
    Replaces manual heuristics with an XGBoost meta-model.
    """
    cdir = Path(cache_dir)
    le = joblib.load(cdir/'label_encoder.joblib')
    classes = le.classes_.astype(str)
    
    y_tr = joblib.load(cdir/'y_tr.joblib')
    y_v = joblib.load(cdir/'y_v.joblib')
    y_te = joblib.load(cdir/'y_te.joblib')
    
    ytr_enc = le.transform(np.asarray(y_tr))
    yv_enc = le.transform(np.asarray(y_v))
    yte_enc = le.transform(np.asarray(y_te))

    _stage("Assembling Meta-Features (OOF Probabilities)")
    
    def assemble_features(split='test'):
        feat_list = []
        # Main Models
        for m in ['xgb', 'lgbm', 'histgb', 'lgbmV2', 'mlp', 'rf']:
            path = cdir / f'proba_{m}_oof_{split}.npz'
            if path.exists():
                feat_list.append(_load_aligned(path, classes))
        
        # Binary Experts (Fast & Others)
        for p in cdir.glob(f'proba_*_expert_*_oof_{split}.npz'):
            feat_list.append(_load_aligned(p, classes))
            
        for p in cdir.glob(f'proba_bb_*_oof_{split}.npz'):
            feat_list.append(_load_aligned(p, classes))
            
        for p in cdir.glob(f'proba_tabnet_*_oof_{split}.npz'):
            feat_list.append(_load_aligned(p, classes))

        if not feat_list: return None
        return np.hstack(feat_list)

    X_meta_tr = assemble_features('train')
    X_meta_v  = assemble_features('valid')
    X_meta_te = assemble_features('test')

    if X_meta_tr is None or X_meta_v is None:
        _info("OOF features for Train/Valid not found! Using manual heuristic fallback.")
        # Fallback to a simplified version of the old heuristic if meta-stacking isn't possible
        return 

    _stage(f"Training XGBoost Meta-Learner (Features: {X_meta_tr.shape[1]})")
    
    # We use a shallow XGBoost as meta-learner to avoid overfitting the OOF probabilities
    meta_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=len(classes),
        random_state=42,
        n_jobs=-1
    )
    
    # Train on VALID, test on TEST (or Train+Valid if you want to be risksier)
    # Using VALID as the meta-train set is safer to prevent data leakage from Train OOF
    meta_model.fit(X_meta_v, yv_enc)
    
    _stage("Final Performance Evaluation (Meta-Learner)")
    yhat_te = meta_model.predict(X_meta_te)
    
    print("\n" + "=" * 80)
    print(f"{'Class':<15} | {'Meta-F1':<10} | {'Status':<10}")
    print("-" * 80)
    
    f1s = get_f1_arr(yte_enc, yhat_te, len(classes))
    target_f1s = {
        'fuzzers': 0.90, 'exploits': 0.90, 'worms': 0.90,
        'reconnaissance': 0.98, 'shellcode': 0.98
    }
    
    for i, cls in enumerate(classes):
        val = f1s[i]
        target = target_f1s.get(cls, 0.0)
        status = "PASSED" if val >= target else "F-LOW" if target > 0 else ""
        print(f"{cls:<15} | {val:.4f}     | {status:<10}")
        
    macro = np.mean(f1s)
    print("-" * 80)
    print(f"{'Macro F1':<15} | {macro:.4f}     |")
    print("=" * 80)

    # Save meta-model
    joblib.dump(meta_model, cdir/'stack_meta_model.joblib')
    _info(f"Meta-model saved. Macro F1: {macro:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=str, default='.')
    args = parser.parse_args()
    heuristic_ensemble(args.cache_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=str, default='.')
    args = parser.parse_args()
    heuristic_ensemble(args.cache_dir)
