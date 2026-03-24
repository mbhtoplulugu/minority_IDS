import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import argparse
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import torch

def _info(m): print(f"[Info]  {m}")
def _stage(m): print(f"[Stage] {m}")

def train_tabnet_expert(cache_dir, target_classes=['dos', 'exploits'], n_features=51):
    """
    Trains a specialized TabNet architecture for highly overlapping classes (DoS/Exploits).
    We use the augmented dataset (n_features+1) which includes the AE Anomaly Score.
    TabNet uses sequential attention to select specific features per sample, ignoring noise.
    """
    _stage("TabNet Expert Training (DoS & Exploits)")
    cdir = Path(cache_dir)
    
    # Yukle Augmented Features (51 features, 50 original + 1 AE score)
    try:
        Xt_tr = joblib.load(cdir / f'Xt_tr_augmented{n_features}.joblib')
        Xt_v  = joblib.load(cdir / f'Xt_v_augmented{n_features}.joblib')
        _info(f"Augmented zellikleri yuklendi: {n_features} kolon")
    except Exception as e:
        _info(f"Augmented veri bulunamadi ({e}). Sadece orijinal zellikleri kullaniyoruz.")
        Xt_tr = joblib.load(cdir / 'Xt_tr.joblib')
        Xt_v  = joblib.load(cdir / 'Xt_v.joblib')
        
    y_tr = joblib.load(cdir / 'y_tr.joblib')
    y_v = joblib.load(cdir / 'y_v.joblib')
    le = joblib.load(cdir / 'label_encoder.joblib')
    classes = le.classes_.astype(str)
    
    y_tr_enc = le.transform(np.asarray(y_tr))
    y_v_enc = le.transform(np.asarray(y_v))
    
    # TabNet hyperparameters for highly imbalanced sparse data
    # n_d, n_a: Dimensions of the network
    # n_steps: Number of architecture steps (attention routing)
    # gamma: Attention coefficient
    tabnet_params = dict(
        n_d=64, n_a=64, n_steps=5, 
        gamma=1.5, n_independent=2, n_shared=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":50, "gamma":0.90},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax', # helps with very noisy datasets
        verbose=1
    )
    
    for target in target_classes:
        if target not in classes:
            _info(f"Hedef {target} etiketlerde yok. Atlaniyor.")
            continue
            
        _stage(f"Training TabNet Expert for [{target.upper()}]")
        c_idx = le.transform([target])[0]
        
        # Binary target olustur
        y_train_bin = (y_tr_enc == c_idx).astype(int)
        y_val_bin = (y_v_enc == c_idx).astype(int)
        
        # Heavy Undersampling for Base Class (Normal, vb.) 
        # Cok genis veriler TabNet'i asiri yavaslatir, sadece zorlu kisimlari alalim
        minority_idx = np.where(y_train_bin == 1)[0]
        majority_idx = np.where(y_train_bin == 0)[0]
        
        ratio = 5 # 1:5 oraninda tutacagiz
        n_maj = min(len(majority_idx), len(minority_idx) * ratio)
        n_maj = max(50000, n_maj) # En az 50k majority kalsin otekileri ogrenmesi icin
        
        np.random.seed(42)
        chosen_majority = np.random.choice(majority_idx, n_maj, replace=False)
        train_indices = np.concatenate([minority_idx, chosen_majority])
        np.random.shuffle(train_indices)
        
        X_train_sub = Xt_tr[train_indices]
        y_train_sub = y_train_bin[train_indices]
        
        # Class weights
        counts = np.bincount(y_train_sub)
        weight_0 = 1.0
        weight_1 = counts[0] / max(1, counts[1])
        _info(f"Class Weights -> 0: {weight_0:.2f}, 1: {weight_1:.2f}")
        
        clf = TabNetClassifier(**tabnet_params)
        
        _info(f"Egitim Verisi: {X_train_sub.shape[0]} ornek.")
        clf.fit(
            X_train=X_train_sub, y_train=y_train_sub,
            eval_set=[(Xt_v, y_val_bin)],
            eval_name=['val'],
            eval_metric=['auc'],
            max_epochs=30,
            patience=5,
            batch_size=2048,
            virtual_batch_size=256,
            weights=1, # pytorch tabnet can balanced weights intrinsically if set to 1
            drop_last=False
        )
        
        # Eval
        preds_val = clf.predict(Xt_v)
        f1 = f1_score(y_val_bin, preds_val)
        p = precision_score(y_val_bin, preds_val, zero_division=0)
        r = recall_score(y_val_bin, preds_val, zero_division=0)
        
        _info(f"[{target.upper()}] TabNet Validation -> F1: {f1:.4f}  Prec: {p:.4f}  Rec: {r:.4f}")
        
        # Predict on Test/Validation to save OOF probabilities for Heuristic Ensemble
        try:
            Xt_te = joblib.load(cdir / f'Xt_te_augmented{n_features}.joblib')
        except:
            Xt_te = joblib.load(cdir / 'Xt_te.joblib')
            
        proba_val = clf.predict_proba(Xt_v)
        proba_test = clf.predict_proba(Xt_te)
        
        # Bu formati genis matrise cevir (OOF Stacking uyumu icin)
        P_val_full = np.zeros((Xt_v.shape[0], len(classes)))
        P_val_full[:, c_idx] = proba_val[:, 1]
        
        P_test_full = np.zeros((Xt_te.shape[0], len(classes)))
        P_test_full[:, c_idx] = proba_test[:, 1]
        
        joblib.dump(clf, cdir / f"tabnet_expert_{target}.joblib")
        np.savez_compressed(cdir / f'proba_tabnet_{target}_oof_valid.npz', proba=P_val_full)
        np.savez_compressed(cdir / f'proba_tabnet_{target}_oof_test.npz', proba=P_test_full)
        _info(f"[{target.upper()}] OOF sonuclari ve model kaydedildi.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', type=str, default='.')
    ap.add_argument('--n-features', type=int, default=51) # 50 orijinal + 1 AE Anomaly Skoru
    args = ap.parse_args()
    
    train_tabnet_expert(args.cache_dir, n_features=args.n_features)
