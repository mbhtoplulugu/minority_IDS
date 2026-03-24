import numpy as np
import pickle
from pathlib import Path
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def _info(m): print(f"[Info]  {m}")

def optimize_thresholds_fast(cache_dir, minority_classes, n_features=50):
    """Fast binary experts iin threshold optimizasyonu"""
    print("\n" + "="*60)
    print("FAST THRESHOLD OPTIMIZATION")
    print("="*60)
    
    cache_path = Path(cache_dir)
    
    # Enhanced features ykleme (glob-based)
    enhanced_found = False
    exact_v = cache_path / f'Xt_v_enhanced{n_features}.joblib'
    exact_te = cache_path / f'Xt_te_enhanced{n_features}.joblib'
    
    if exact_v.exists() and exact_te.exists():
        Xt_v = joblib.load(exact_v)
        Xt_te = joblib.load(exact_te)
        _info(f"Loaded enhanced features (exact {n_features}).")
        enhanced_found = True
    
    if not enhanced_found:
        candidates = sorted(cache_path.glob('Xt_v_enhanced*.joblib'))
        for v_f in candidates:
            suffix = v_f.stem.replace('Xt_v_', '')
            te_f = cache_path / f'Xt_te_{suffix}.joblib'
            if te_f.exists():
                Xt_v = joblib.load(v_f)
                Xt_te = joblib.load(te_f)
                _info(f"Loaded enhanced features ({suffix}).")
                enhanced_found = True
                break
    
    if not enhanced_found:
        Xt_v = joblib.load(cache_path / 'Xt_v.joblib')
        Xt_te = joblib.load(cache_path / 'Xt_te.joblib')
        _info("Enhanced not found, using original features.")
        
    try:
        y_v  = joblib.load(cache_path / 'y_v.joblib')
        y_te = joblib.load(cache_path / 'y_te.joblib')
        le   = joblib.load(cache_path / 'label_encoder.joblib')
        classes = le.classes_.astype(str)
    except Exception as e:
        print(f"ERROR: Cannot load data. {e}")
        return
    
    optimized_results = {}
    
    for minority_class in minority_classes:
        model_path = cache_path / f'fast_binary_expert_{minority_class}.pkl'
        
        if not model_path.exists():
            print(f"[{minority_class.upper()}] Model not found, skipping...")
            continue
            
        print(f"[{minority_class.upper()}] Optimizing threshold:")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        try:
            minority_encoded = le.transform([minority_class])[0]
            c_id = int(np.where(classes==minority_class)[0][0])
        except ValueError:
            continue
            
        y_val_encoded = le.transform(y_v)
        y_binary_val = (y_val_encoded == minority_encoded).astype(int)
        
        if hasattr(model, 'predict_proba'):
            y_proba_v = model.predict_proba(Xt_v)[:, 1]
            y_proba_te = model.predict_proba(Xt_te)[:, 1]
        else:
            y_proba_v = model.decision_function(Xt_v)
            y_proba_v = (y_proba_v - y_proba_v.min()) / (y_proba_v.max() - y_proba_v.min())
            y_proba_te = model.decision_function(Xt_te)
            y_proba_te = (y_proba_te - y_proba_te.min()) / (y_proba_te.max() - y_proba_te.min())
            
        
        best_f1 = 0
        best_threshold = 0.5
        best_precision = 0
        best_recall = 0
        
        if minority_class == 'worms':
            thresholds = np.arange(0.01, 0.8, 0.01)
        else:
            thresholds = np.arange(0.05, 0.95, 0.02)
        
        for threshold in thresholds:
            y_pred = (y_proba_v >= threshold).astype(int)
            if len(np.unique(y_pred)) < 2:  
                continue
            f1 = f1_score(y_binary_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision_score(y_binary_val, y_pred, zero_division=0)
                best_recall = recall_score(y_binary_val, y_pred, zero_division=0)
        
        print(f"  Optimal threshold: {best_threshold:.3f}")
        print(f"  Optimized F1: {best_f1:.4f} P: {best_precision:.4f} R: {best_recall:.4f}")
        
        threshold_path = cache_path / f'fast_threshold_{minority_class}.pkl'
        with open(threshold_path, 'wb') as f:
            pickle.dump(best_threshold, f)
            
        # We need to save the test and valid probabilities with the proper naming for the pipeline Stack
        out_v = np.zeros((len(y_v), len(classes)), dtype=np.float32)
        out_v[:, c_id] = y_proba_v
        np.savez(cache_path / f"proba_fast_expert_{minority_class}_oof_valid.npz", proba=out_v, classes=classes)
        
        out_te = np.zeros((len(y_te), len(classes)), dtype=np.float32)
        out_te[:, c_id] = y_proba_te
        np.savez(cache_path / f"proba_fast_expert_{minority_class}_oof_test.npz", proba=out_te, classes=classes)
        
        optimized_results[minority_class] = { 'threshold': best_threshold, 'f1_score': best_f1 }
    
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION COMPLETED")
    for class_name, result in optimized_results.items():
        print(f"{class_name.upper():<12} Threshold: {result['threshold']:.3f} F1: {result['f1_score']:.4f}")
    
    return optimized_results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', type=str, default='.')
    ap.add_argument('--focus', nargs='*', default=['analysis', 'backdoor', 'dos', 'worms'])
    ap.add_argument('--n-features', type=int, default=65)
    args = ap.parse_args()
    optimize_thresholds_fast(args.cache_dir, args.focus, args.n_features)
