import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import argparse
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.over_sampling import ADASYN
from sklearn.metrics import f1_score, precision_score, recall_score

def _info(m): print(f"[Info]  {m}")
def _stage(m): print(f"[Stage] {m}")

def train_balanced_bagging_expert(cache_dir, target_classes=['analysis', 'backdoor', 'fuzzers'], n_features=51):
    """
    Trains specialized Balanced Bagging classifiers for extremely sparse classes.
    Uses the Augmented dataset (n_features+1) which includes the AE Anomaly Score.
    
    Architecture:
    1. Severe Undersampling of Majority (to make ADASYN manageable)
    2. ADASYN (Adaptive Synthetic Sampling) to focus on hard-to-learn minority examples
    3. BalancedBaggingClassifier wrapped around HistGradientBoosting for fast, memory-efficient tree ensembles
    """
    _stage("Balanced Bagging + ADASYN Expert Training")
    cdir = Path(cache_dir)
    
    # 1. Veri Yukleme (Augmented)
    try:
        Xt_tr = joblib.load(cdir / f'Xt_tr_augmented{n_features}.joblib')
        Xt_v  = joblib.load(cdir / f'Xt_v_augmented{n_features}.joblib')
        _info(f"Augmented zellikleri yuklendi: {n_features} kolon")
    except Exception as e:
        _info(f"Augmented veri bulunamadi ({e}). Orijinal zelliklere donuluyor.")
        Xt_tr = joblib.load(cdir / 'Xt_tr.joblib')
        Xt_v  = joblib.load(cdir / 'Xt_v.joblib')
        
    y_tr = joblib.load(cdir / 'y_tr.joblib')
    y_v = joblib.load(cdir / 'y_v.joblib')
    le = joblib.load(cdir / 'label_encoder.joblib')
    classes = le.classes_.astype(str)
    
    y_tr_enc = le.transform(np.asarray(y_tr))
    y_v_enc = le.transform(np.asarray(y_v))
    
    for target in target_classes:
        if target not in classes:
            _info(f"Hedef {target} etiketlerde yok. Atlaniyor.")
            continue
            
        _stage(f"Training Balanced Bagging for [{target.upper()}]")
        c_idx = le.transform([target])[0]
        
        y_train_bin = (y_tr_enc == c_idx).astype(int)
        y_val_bin = (y_v_enc == c_idx).astype(int)
        
        counts = np.bincount(y_train_bin)
        minority_count = counts[1]
        
        # 2. ADASYN oncesi Asiri Undersampling
        # Eger milyarlarca satir olursa ADASYN kilitlenir. 1:10 oraninda kesiyoruz.
        ratio = 10 
        n_maj = min(counts[0], minority_count * ratio)
        n_maj = max(50000, n_maj) 
        
        minority_idx = np.where(y_train_bin == 1)[0]
        majority_idx = np.where(y_train_bin == 0)[0]
        
        np.random.seed(42)
        chosen_majority = np.random.choice(majority_idx, n_maj, replace=False)
        train_indices = np.concatenate([minority_idx, chosen_majority])
        np.random.shuffle(train_indices)
        
        X_train_sub = Xt_tr[train_indices]
        y_train_sub = y_train_bin[train_indices]
        
        _info(f"Alt-kumme Egitim Verisi (Oncesi): {X_train_sub.shape[0]} (Minority: {minority_count})")
        
        # 3. ADASYN ile Minoriteyi cogalt (Sadece egitim setinde)
        adasyn = ADASYN(sampling_strategy='minority', n_neighbors=5, random_state=42)
        try:
            X_resampled, y_resampled = adasyn.fit_resample(X_train_sub, y_train_sub)
            _info(f"ADASYN Sonrasi Egitim Verisi: {X_resampled.shape[0]}")
        except Exception as e:
            _info(f"ADASYN Hatasi ({e}). ADASYN atlanip Base kullaniliyor.")
            X_resampled, y_resampled = X_train_sub, y_train_sub
            
        # 4. Balanced Bagging + HistGBM
        # HistGBM hizli agaclar kurar. BalancedBagging ise her agaca resample edilmis esit paketler sunar.
        base_estimator = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, random_state=42)
        bagging_clf = BalancedBaggingClassifier(
            estimator=base_estimator,
            n_estimators=10, # 10 farkli base agac 
            sampling_strategy='auto', # Her bag kendi icinde dengelenecek
            random_state=42,
            n_jobs=-1
        )
        
        _info("Model egitiliyor (Balanced Bagging wraps HistGBM)...")
        bagging_clf.fit(X_resampled, y_resampled)
        
        # Eval
        preds_val = bagging_clf.predict(Xt_v)
        f1 = f1_score(y_val_bin, preds_val)
        p = precision_score(y_val_bin, preds_val, zero_division=0)
        r = recall_score(y_val_bin, preds_val, zero_division=0)
        
        _info(f"[{target.upper()}] Validation -> F1: {f1:.4f}  Prec: {p:.4f}  Rec: {r:.4f}")
        
        # Predict on Test/Validation to save OOF probabilities
        try:
            Xt_te = joblib.load(cdir / f'Xt_te_augmented{n_features}.joblib')
        except:
            Xt_te = joblib.load(cdir / 'Xt_te.joblib')
            
        proba_val = bagging_clf.predict_proba(Xt_v)
        proba_test = bagging_clf.predict_proba(Xt_te)
        
        P_val_full = np.zeros((Xt_v.shape[0], len(classes)))
        P_val_full[:, c_idx] = proba_val[:, 1]
        
        P_test_full = np.zeros((Xt_te.shape[0], len(classes)))
        P_test_full[:, c_idx] = proba_test[:, 1]
        
        joblib.dump(bagging_clf, cdir / f"balanced_bagging_{target}.joblib")
        np.savez_compressed(cdir / f'proba_bb_{target}_oof_valid.npz', proba=P_val_full)
        np.savez_compressed(cdir / f'proba_bb_{target}_oof_test.npz', proba=P_test_full)
        _info(f"[{target.upper()}] OOF sonuclari kaydedildi.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', type=str, default='.')
    ap.add_argument('--n-features', type=int, default=51)
    args = ap.parse_args()
    
    train_balanced_bagging_expert(args.cache_dir, n_features=args.n_features)
