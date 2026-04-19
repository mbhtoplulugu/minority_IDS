import os, joblib, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

def _info(m): print(f"[Info]  {m}")
def _stage(m): print(f"[Stage] {m}")

def run_tabnet_oof(cfg, n_features=75, cached_data=None):
    """
    Runs TabNet Multi-class OOF training and prediction.
    """
    _stage(f"TabNet Multi-class OOF Training (n_features={n_features})")
    cdir = Path(cfg.cache_dir)
    
    if cached_data is not None:
        Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te = cached_data
    else:
        # Try to load enhanced features first
        try:
            Xt_tr = joblib.load(cdir / f"Xt_tr_enhanced{n_features}.joblib")
            Xt_v  = joblib.load(cdir / f"Xt_v_enhanced{n_features}.joblib")
            Xt_te = joblib.load(cdir / f"Xt_te_enhanced{n_features}.joblib")
            _info(f"Loaded enhanced features: {n_features}")
        except:
            Xt_tr = joblib.load(cdir / "Xt_tr.joblib")
            Xt_v  = joblib.load(cdir / "Xt_v.joblib")
            Xt_te = joblib.load(cdir / "Xt_te.joblib")
            _info("Loaded original features (enhanced not found).")
            
        y_tr = joblib.load(cdir / "y_tr.joblib")
        y_v = joblib.load(cdir / "y_v.joblib")
        y_te = joblib.load(cdir / "y_te.joblib")

    le = joblib.load(cdir / "label_encoder.joblib")
    y_tr_enc = le.transform(np.asarray(y_tr))
    y_v_enc = le.transform(np.asarray(y_v))
    y_te_enc = le.transform(np.asarray(y_te))
    
    n_classes = len(le.classes_)
    device = "cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu"
    _info(f"Using device: {device}")

    # TabNet Parameters
    tabnet_params = dict(
        n_d=32, n_a=32, n_steps=3,
        gamma=1.3, n_independent=2, n_shared=2,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        device_name=device,
        verbose=1
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
    
    oof_train = np.zeros((len(y_tr_enc), n_classes), dtype=np.float32)
    valid_preds = np.zeros((len(y_v_enc), n_classes), dtype=np.float32)
    test_preds = np.zeros((len(y_te_enc), n_classes), dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(Xt_tr, y_tr_enc)):
        _info(f"Fold {fold+1}/5 starting...")
        X_fold_tr, y_fold_tr = Xt_tr[trn_idx], y_tr_enc[trn_idx]
        X_fold_val, y_fold_val = Xt_tr[val_idx], y_tr_enc[val_idx]
        
        # We can use a subset of training data if it's too slow, 
        # but let's try with a larger batch size first.
        clf = TabNetClassifier(**tabnet_params)
        clf.fit(
            X_train=X_fold_tr, y_train=y_fold_tr,
            eval_set=[(X_fold_val, y_fold_val)],
            eval_name=['val'],
            eval_metric=['accuracy'],
            max_epochs=20,
            patience=5,
            batch_size=4096,
            virtual_batch_size=256,
            weights=1, # Balanced weights
            drop_last=False
        )
        
        oof_train[val_idx] = clf.predict_proba(X_fold_val)
        valid_preds += clf.predict_proba(Xt_v) / skf.n_splits
        test_preds += clf.predict_proba(Xt_te) / skf.n_splits

    # Save OOF
    from unsw_nb15_pipeline import save_proba_npz
    save_proba_npz(str(cdir / "proba_tabnet_oof_train.npz"), oof_train, le=le)
    save_proba_npz(str(cdir / "proba_tabnet_oof_valid.npz"), valid_preds, le=le)
    save_proba_npz(str(cdir / "proba_tabnet_oof_test.npz"), test_preds, le=le)

    y_pred_te = test_preds.argmax(1)
    acc = accuracy_score(y_te_enc, y_pred_te)
    _info(f"TabNet OOF Test Accuracy: {acc:.4f}")
    print(classification_report(y_te_enc, y_pred_te, target_names=le.classes_, digits=4))

    # Final model on all train data
    _stage("Training Final TabNet Model on all data")
    final_clf = TabNetClassifier(**tabnet_params)
    final_clf.fit(
        X_train=Xt_tr, y_train=y_tr_enc,
        eval_set=[(Xt_v, y_v_enc)],
        eval_name=['valid'],
        eval_metric=['accuracy'],
        max_epochs=25,
        patience=5,
        batch_size=4096,
        virtual_batch_size=256,
        weights=1
    )
    joblib.dump(final_clf, cdir / "model_tabnet_final.joblib")
    _info(f"Final TabNet model saved to {cdir / 'model_tabnet_final.joblib'}")

if __name__ == '__main__':
    # Add dummy/test execution if needed
    pass
