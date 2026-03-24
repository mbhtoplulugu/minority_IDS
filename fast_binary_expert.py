import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import pickle
from pathlib import Path
import joblib

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

def _info(m): print(f"[Info]  {m}")

def fast_binary_experts(cache_dir, minority_classes, n_features=50):
    """Gelitirilmi binary expert eitimi - LightGBM/XGBoost + SMOTE destei"""
    print("\n" + "="*60)
    print("FAST BINARY EXPERTS SYSTEM (Enhanced v2)")
    print("="*60)
    
    cache_path = Path(cache_dir)
    
    # 1. Ykleme - enhanced features ncelikli (glob-based)
    enhanced_found = False
    
    # Exact match dene
    exact_files = [
        cache_path / f'Xt_tr_enhanced{n_features}.joblib',
        cache_path / f'Xt_v_enhanced{n_features}.joblib'
    ]
    if all(f.exists() for f in exact_files):
        Xt_tr = joblib.load(exact_files[0])
        Xt_v  = joblib.load(exact_files[1])
        enhanced_found = True
        print(f"Loaded enhanced data (exact {n_features}): {Xt_tr.shape[1]} features")
    
    # Glob ile bul
    if not enhanced_found:
        candidates = sorted(cache_path.glob('Xt_tr_enhanced*.joblib'))
        for tr_f in candidates:
            suffix = tr_f.stem.replace('Xt_tr_', '')
            v_f = cache_path / f'Xt_v_{suffix}.joblib'
            if v_f.exists():
                Xt_tr = joblib.load(tr_f)
                Xt_v  = joblib.load(v_f)
                enhanced_found = True
                print(f"Loaded enhanced data ({suffix}): {Xt_tr.shape[1]} features")
                break
    
    # Fallback: original
    if not enhanced_found:
        print(f"Enhanced features not found, falling back to standard...")
        try:
            Xt_tr = joblib.load(cache_path / 'Xt_tr.joblib')
            Xt_v  = joblib.load(cache_path / 'Xt_v.joblib')
        except Exception as e2:
            print(f"ERROR: Cannot load data. {e2}")
            return None
    
    try:
        y_tr  = joblib.load(cache_path / 'y_tr.joblib')
        y_v   = joblib.load(cache_path / 'y_v.joblib')
        le    = joblib.load(cache_path / 'label_encoder.joblib')
    except Exception as e:
        print(f"ERROR: Cannot load labels. {e}")
        return None
    
    classes = le.classes_.astype(str)
    results = {}
    
    for minority_class in minority_classes:
        print(f"\n{'='*50}")
        print(f"[{minority_class.upper()}] Fast Binary Expert (Enhanced):")
        print(f"{'='*50}")
        
        try:
            minority_encoded = le.transform([minority_class])[0]
        except ValueError:
            print(f"  Skipping {minority_class}: Class not found")
            results[minority_class] = {'f1_score': 0.0, 'method': 'NotFound', 'model': None}
            continue
            
        y_train_encoded = le.transform(y_tr)
        y_val_encoded = le.transform(y_v)
        
        y_binary_train = (y_train_encoded == minority_encoded).astype(int)
        y_binary_val = (y_val_encoded == minority_encoded).astype(int)
        
        unique_classes = np.unique(y_binary_train)
        if len(unique_classes) < 2:
            print(f"  Skipping {minority_class}: Only {len(unique_classes)} class found")
            results[minority_class] = {'f1_score': 0.0, 'method': 'Skipped', 'model': None}
            continue
        
        original_ratio = Counter(y_binary_train)
        n_pos_orig = original_ratio[1]
        n_neg_orig = original_ratio[0]
        print(f"  Original ratio: 1:{n_neg_orig/max(n_pos_orig, 1):.1f} (pos={n_pos_orig}, neg={n_neg_orig})")
        
        # Snfa gre adaptif rnekleme stratejisi
        # Backdoor ve Fuzzers iin daha fazla veri al
        if minority_class in ['backdoor', 'fuzzers']:
            max_samples = 150000  # Daha byk sample
            max_imbalance = 8     # Daha dengeli
        elif minority_class == 'worms':
            max_samples = 100000
            max_imbalance = 5
        else:
            max_samples = 80000
            max_imbalance = 10
        
        minority_indices = np.where(y_binary_train == 1)[0]
        majority_indices = np.where(y_binary_train == 0)[0]
        n_minority = len(minority_indices)
        
        # Dengeleme: majority'yi snrla
        n_majority_target = min(len(majority_indices), max(max_samples - n_minority, n_minority * max_imbalance))
        
        np.random.seed(42)
        selected_majority = np.random.choice(majority_indices, n_majority_target, replace=False)
        selected_indices = np.concatenate([minority_indices, selected_majority])
        np.random.shuffle(selected_indices)
        
        X_train_sub = Xt_tr[selected_indices]
        y_binary_train_sub = y_binary_train[selected_indices]
        print(f"  Subsampled: {len(Xt_tr)} -> {len(X_train_sub)} (pos={n_minority}, neg={n_majority_target})")
        
        # SMOTE ile minority snf oalt (zellikle backdoor iin kritik)
        if HAS_SMOTE and n_minority >= 6:  # SMOTE en az k_neighbors+1 sample ister
            try:
                smote_ratio = min(0.3, n_minority / n_majority_target * 3)  # 3x oalt ama max %30'a
                if minority_class in ['backdoor', 'worms']:
                    smote_ratio = min(0.5, n_minority / n_majority_target * 5)  # Daha agresif
                
                k_neighbors = min(5, n_minority - 1)
                smote = SMOTE(sampling_strategy=smote_ratio, k_neighbors=k_neighbors, random_state=42)
                X_train_sub, y_binary_train_sub = smote.fit_resample(X_train_sub, y_binary_train_sub)
                new_counts = Counter(y_binary_train_sub)
                print(f"  SMOTE applied: pos={new_counts[1]}, neg={new_counts[0]} (ratio=1:{new_counts[0]/max(new_counts[1],1):.1f})")
            except Exception as e:
                _info(f"SMOTE failed ({e}), continuing without oversampling")
        
        # Class weights hesaplama
        class_weights = compute_class_weight('balanced', 
                                           classes=np.unique(y_binary_train_sub), 
                                           y=y_binary_train_sub)
        weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        imbalance_ratio = Counter(y_binary_train_sub)[0] / max(1, Counter(y_binary_train_sub)[1])
        if imbalance_ratio > 50:
            weight_dict[1] = min(weight_dict[1] * 3, 50.0)
        elif imbalance_ratio > 20:
            weight_dict[1] *= 2
        
        print(f"  Class weights: {{0: {weight_dict[0]:.3f}, 1: {weight_dict[1]:.3f}}}")
        
        # Her snf iin en iyi modeli bulmak zere geni model havuzu
        models = {}
        
        # 1. LightGBM (en gl binary classifier)
        if HAS_LGBM:
            scale_pos = weight_dict[1] / max(weight_dict[0], 1e-6)
            models['LightGBM'] = LGBMClassifier(
                n_estimators=300, max_depth=8, num_leaves=31,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=min(scale_pos, 50.0),
                objective='binary', random_state=42, n_jobs=-1, verbose=-1
            )
            # Daha agresif LightGBM varyant
            models['LightGBM_Aggressive'] = LGBMClassifier(
                n_estimators=500, max_depth=10, num_leaves=63,
                learning_rate=0.03, subsample=0.7, colsample_bytree=0.7,
                scale_pos_weight=min(scale_pos * 1.5, 80.0),
                min_child_samples=5,
                objective='binary', random_state=42, n_jobs=-1, verbose=-1
            )
        
        # 2. XGBoost
        if HAS_XGB:
            scale_pos_xgb = weight_dict[1] / max(weight_dict[0], 1e-6)
            models['XGBoost'] = XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=min(scale_pos_xgb, 50.0),
                objective='binary:logistic', random_state=42, 
                n_jobs=-1, verbosity=0, tree_method='hist'
            )
        
        # 3. Random Forest
        if minority_class in ['backdoor', 'worms']:
            models['RandomForest'] = RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_split=3, min_samples_leaf=1,
                class_weight=weight_dict, random_state=42, n_jobs=-1
            )
        else:
            models['RandomForest'] = RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_split=10, min_samples_leaf=5,
                class_weight=weight_dict, random_state=42, n_jobs=-1
            )
        
        # 4. Logistic Regression
        models['LogisticRegression'] = LogisticRegression(
            class_weight=weight_dict, C=10.0, max_iter=2000, 
            solver='liblinear', random_state=42
        )
        
        best_f1 = 0
        best_model = None
        best_model_name = None
        all_model_results = []
        
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            try:
                model.fit(X_train_sub, y_binary_train_sub)
                y_pred_val = model.predict(Xt_v)
                
                f1 = f1_score(y_binary_val, y_pred_val)
                precision = precision_score(y_binary_val, y_pred_val, zero_division=0)
                recall = recall_score(y_binary_val, y_pred_val, zero_division=0)
                
                print(f"    F1: {f1:.4f} P: {precision:.4f} R: {recall:.4f}")
                all_model_results.append((model_name, f1, precision, recall))
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
                    best_model_name = model_name
                    
            except Exception as e:
                print(f"    {model_name} failed: {e}")
                continue
        
        print(f"\n  >>> Best: {best_model_name} (F1: {best_f1:.4f})")
        
        # Tm model sonular karlatrma tablosu
        if all_model_results:
            print(f"  {'Model':<25} {'F1':>8} {'Prec':>8} {'Recall':>8}")
            print(f"  {'-'*51}")
            for mn, mf, mp, mr in sorted(all_model_results, key=lambda x: -x[1]):
                marker = " <<<" if mn == best_model_name else ""
                print(f"  {mn:<25} {mf:>8.4f} {mp:>8.4f} {mr:>8.4f}{marker}")
        
        # Modeli kaydet
        if best_model is not None:
            model_path = cache_path / f'fast_binary_expert_{minority_class}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
        
        results[minority_class] = { 'f1_score': best_f1, 'method': best_model_name }
    
    # Son genel zet
    print("\n" + "="*60)
    print("FAST BINARY EXPERTS COMPLETED (Enhanced v2)")
    print(f"{'Class':<15} {'F1':>8} {'Method':<25}")
    print("-" * 50)
    for class_name, result in results.items():
        print(f"{class_name.upper():<15} {result['f1_score']:>8.4f} {result['method']:<25}")
    print("=" * 60)
    return results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', type=str, default='.')
    ap.add_argument('--focus', nargs='*', default=['analysis', 'backdoor', 'dos', 'fuzzers', 'worms'])
    ap.add_argument('--n-features', type=int, default=65)
    args = ap.parse_args()
    fast_binary_experts(args.cache_dir, args.focus, args.n_features)
