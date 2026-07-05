import numpy as np
import pandas as pd
import joblib
import pickle
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

def _info(m): print(f"[Info]  {m}")

def autonomous_surgical_experts(cache_dir, target_classes, n_features=65):
    """
    Otonom Cerrahi Uzman Eğitimi
    Sınıfların neyle karıştığını XGBoost validation proba'sından (confusion matrix)
    kendi kendine bulur ve hard-negative mining yapar.
    """
    print("\n" + "="*60)
    print("AUTONOMOUS SURGICAL EXPERTS SYSTEM")
    print("="*60)
    
    cache_path = Path(cache_dir)
    
    # 1. Veri Yükleme (Öncelikli olarak enhanced)
    enhanced_found = False
    candidates = sorted(cache_path.glob('Xt_tr_enhanced*.joblib'))
    if candidates:
        tr_f = candidates[-1]  # En son / en çok feature'ı olanı al
        suffix = tr_f.stem.replace('Xt_tr_', '')
        v_f = cache_path / f'Xt_v_{suffix}.joblib'
        te_f = cache_path / f'Xt_te_{suffix}.joblib'
        if v_f.exists() and te_f.exists():
            Xt_tr = joblib.load(tr_f)
            Xt_v  = joblib.load(v_f)
            Xt_te = joblib.load(te_f)
            enhanced_found = True
            print(f"Loaded enhanced data ({suffix}): {Xt_tr.shape[1]} features")
    
    if not enhanced_found:
        print("Enhanced features not found, loading original features...")
        try:
            Xt_tr = joblib.load(cache_path / 'Xt_tr.joblib')
            Xt_v  = joblib.load(cache_path / 'Xt_v.joblib')
            Xt_te = joblib.load(cache_path / 'Xt_te.joblib')
        except Exception as e:
            print(f"ERROR: Cannot load data. {e}")
            return None
            
    try:
        y_tr = joblib.load(cache_path / 'y_tr.joblib')
        y_v  = joblib.load(cache_path / 'y_v.joblib')
        y_te = joblib.load(cache_path / 'y_te.joblib')
        le   = joblib.load(cache_path / 'label_encoder.joblib')
    except Exception as e:
        print(f"ERROR: Cannot load labels. {e}")
        return None
        
    classes = le.classes_.astype(str)
    
    # XGBoost Valid probas for hard-negative detection
    xgb_val_path = cache_path / 'proba_xgb_oof_valid.npz'
    if not xgb_val_path.exists():
        print(f"ERROR: {xgb_val_path} not found. Cannot perform autonomous hard-negative mining.")
        return None
        
    xgb_data = np.load(xgb_val_path, allow_pickle=True)
    P_xgb_v = xgb_data['proba']
    y_pred_v = np.argmax(P_xgb_v, axis=1)
    y_v_enc = le.transform(y_v)
    
    cm = confusion_matrix(y_v_enc, y_pred_v)
    
    results = {}
    
    for cls_name in target_classes:
        if cls_name not in classes:
            print(f"Skipping {cls_name}, not in classes.")
            continue
            
        c_idx = int(np.where(classes == cls_name)[0][0])
        print(f"\n{'='*50}")
        print(f"[{cls_name.upper()}] Autonomous Surgical Expert")
        print(f"{'='*50}")
        
        # --- AUTONOMOUS HARD-NEGATIVE MINING ---
        confusing_scores = []
        for i in range(len(classes)):
            if i == c_idx:
                continue
            fn_count = cm[c_idx, i]
            fp_count = cm[i, c_idx]
            total_confusion = fn_count + fp_count
            if total_confusion > 0:
                confusing_scores.append((classes[i], total_confusion))
                
        confusing_scores.sort(key=lambda x: x[1], reverse=True)
        top_confusing = [c for c, score in confusing_scores[:4]]
        
        print(f"  Autonomous Hard-Negative Detection: Top confusing classes -> {top_confusing}")
        
        y_bin_tr = (le.transform(y_tr) == c_idx).astype(int)
        y_bin_v  = (y_v_enc == c_idx).astype(int)
        y_bin_te = (le.transform(y_te) == c_idx).astype(int)
        
        pos_idx = np.where(y_bin_tr == 1)[0]
        
        if len(pos_idx) < 5:
            print(f"  Skipping {cls_name}: Not enough positive samples ({len(pos_idx)})")
            continue
            
        conf_idxs = [int(np.where(classes == c)[0][0]) for c in top_confusing]
        conf_neg_idx = np.where(np.isin(le.transform(y_tr), conf_idxs) & (y_bin_tr == 0))[0]
        other_neg_idx = np.where((~np.isin(le.transform(y_tr), conf_idxs)) & (y_bin_tr == 0))[0]
        
        n_conf = min(50000, len(conf_neg_idx))
        n_other = min(15000, len(other_neg_idx))
        
        np.random.seed(42)
        sel_conf = np.random.choice(conf_neg_idx, n_conf, replace=False) if len(conf_neg_idx) > 0 else []
        sel_other = np.random.choice(other_neg_idx, n_other, replace=False) if len(other_neg_idx) > 0 else []
        
        sel = np.concatenate([pos_idx, sel_conf, sel_other]).astype(int)
        np.random.shuffle(sel)
        
        X_sub = Xt_tr[sel]
        y_sub = y_bin_tr[sel]
        
        print(f"  Training shape: pos={len(pos_idx)}, conf_neg={len(sel_conf)}, other_neg={len(sel_other)}")
        
        if len(pos_idx) < 10000:
            smote_ratio = min(0.4, (len(pos_idx)*3) / max(1, len(sel_conf) + len(sel_other)))
            if smote_ratio > 0.05:
                try:
                    k = min(5, len(pos_idx) - 1)
                    smote = SMOTE(sampling_strategy=smote_ratio, k_neighbors=k, random_state=42)
                    X_sub, y_sub = smote.fit_resample(X_sub, y_sub)
                    print(f"  Autonomous SMOTE applied. New pos={np.sum(y_sub==1)}")
                except Exception as e:
                    print(f"  SMOTE failed: {e}")
                    
        if not HAS_LGBM:
            print("  LGBM not installed, skipping.")
            continue
            
        model = LGBMClassifier(
            n_estimators=800, max_depth=8, num_leaves=63,
            learning_rate=0.02, subsample=0.7, colsample_bytree=0.7,
            min_child_samples=5, scale_pos_weight=2.0,
            objective='binary', random_state=42, n_jobs=-1, verbose=-1
        )
        print("  Training LGBM Surgical Model...")
        model.fit(X_sub, y_sub)
        
        proba_v = model.predict_proba(Xt_v)[:, 1]
        
        best_f1 = 0; best_th = 0.5
        for th in np.arange(0.1, 0.95, 0.01):
            pred = (proba_v >= th).astype(int)
            if pred.sum() == 0: continue
            f1 = f1_score(y_bin_v, pred)
            if f1 > best_f1: 
                best_f1 = f1
                best_th = th
                
        proba_te = model.predict_proba(Xt_te)[:, 1]
        pred_te = (proba_te >= best_th).astype(int)
        f1_te = f1_score(y_bin_te, pred_te)
        
        print(f"  Optimized Threshold: {best_th:.3f}")
        print(f"  Validation F1: {best_f1:.4f} | Test F1: {f1_te:.4f}")
        
        model_path = cache_path / f'surgical_{cls_name}_expert.pkl'
        th_path = cache_path / f'surgical_{cls_name}_threshold.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(th_path, 'wb') as f:
            pickle.dump(best_th, f)
            
        out_v = np.zeros((len(Xt_v), len(classes)), dtype=np.float32)
        out_v[:, c_idx] = proba_v
        np.savez(cache_path / f'proba_surgical_{cls_name}_oof_valid.npz', proba=out_v, classes=classes)
        
        out_te = np.zeros((len(Xt_te), len(classes)), dtype=np.float32)
        out_te[:, c_idx] = proba_te
        np.savez(cache_path / f'proba_surgical_{cls_name}_oof_test.npz', proba=out_te, classes=classes)
        
        results[cls_name] = {'f1': f1_te, 'threshold': best_th}
        
    print("\n" + "="*60)
    print("AUTONOMOUS SURGICAL EXPERTS COMPLETED")
    print("="*60)
    return results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', type=str, required=True)
    ap.add_argument('--target', nargs='+', required=True)
    args = ap.parse_args()
    autonomous_surgical_experts(args.cache_dir, args.target)
