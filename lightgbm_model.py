from pathlib import Path
import joblib
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from lightgbm import LGBMClassifier
from scipy.special import softmax

try:
    from unsw_nb15_pipeline import preprocess_and_cache, apply_smart_resampling, save_proba_npz, _stage, _info
except ImportError:
    pass

warnings.filterwarnings("ignore")

def _predict_proba_lgbm(model, X):
    """
    When LightGBM uses a custom objective, predict_proba() returns raw
    log-likelihood scores (not normalized). Apply softmax to fix this.
    """
    raw = model.predict_proba(X)
    if raw.min() < 0 or raw.max() > 1 or not np.allclose(raw.sum(axis=1), 1.0, atol=0.01):
        # Raw scores detected, apply row-wise softmax
        raw = softmax(raw, axis=1)
    return raw.astype(np.float32)

def focal_loss_multiclass_lgbm(y_true, y_pred):
    """Custom Multiclass Focal Loss for LightGBM"""
    gamma = 2.0
    alpha = np.array([5.0, 6.0, 1.5, 1.1, 1.8, 0.8, 0.7, 1.1, 1.5, 15.0]) # Same weights as before
    
    y = y_true if not hasattr(y_true, 'get_label') else y_true.get_label()
    y = y.astype(int)
    n_classes = 10
    
    is_1d = (y_pred.ndim == 1)
    if is_1d:
        preds = y_pred.reshape(n_classes, -1).T
    else:
        preds = y_pred
    
    p = softmax(preds, axis=1)
    y_true_oh = np.eye(n_classes)[y]
    
    grad = (p - y_true_oh)
    hess = p * (1.0 - p) 
    
    for i in range(n_classes):
        alpha_factor = alpha[i] if i < len(alpha) else 1.0
        weight_factor = alpha_factor * np.power(1.0 - p[:, i], gamma)
        grad[:, i] *= weight_factor
        hess[:, i] *= weight_factor
        
    if is_1d:
        return grad.T.flatten(), hess.T.flatten()
    return grad, hess

def run_lgbm_oof(cfg, n_features=50, cached_data=None):
    _stage("LightGBM 5-Fold OOF Training")
    
    cache_dir = Path(getattr(cfg, 'cache_dir', '.'))
    from sklearn.preprocessing import LabelEncoder
    
    # Optimize: Cache'den ykle, preprocess tekrarlanmasn
    if cached_data is not None:
        _info("[LGBM] Using pre-loaded cached data")
        Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te = cached_data
    else:
        from experts import _load_cache
        Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te, _ = _load_cache(cache_dir, n_features)
    
    le = joblib.load(cache_dir / 'label_encoder.joblib')
    
    y_tr_arr = np.asarray(y_tr)
    yv_enc = le.transform(np.asarray(y_v))
    yte_enc = le.transform(np.asarray(y_te))
    
    # Enhanced features ncelii (cached_data kullanlmadysa _load_cache zaten hallediyor,
    # ama cached_data geldi ise enhanced'a gei yapmamz gerekir)
    if cached_data is not None:
        enhanced_found = False
        # 1) Exact match
        exact_files = [
            cache_dir / f"Xt_tr_enhanced{n_features}.joblib",
            cache_dir / f"Xt_v_enhanced{n_features}.joblib", 
            cache_dir / f"Xt_te_enhanced{n_features}.joblib"
        ]
        if all(f.exists() for f in exact_files):
            Xt_tr = joblib.load(exact_files[0])
            Xt_v = joblib.load(exact_files[1])
            Xt_te = joblib.load(exact_files[2])
            _info(f"[LGBM] Using enhanced features (exact {n_features}): {Xt_tr.shape}")
            enhanced_found = True
        
        # 2) Glob fallback
        if not enhanced_found:
            candidates = sorted(cache_dir.glob('Xt_tr_enhanced*.joblib'))
            for tr_f in candidates:
                suffix = tr_f.stem.replace('Xt_tr_', '')
                v_f  = cache_dir / f'Xt_v_{suffix}.joblib'
                te_f = cache_dir / f'Xt_te_{suffix}.joblib'
                if v_f.exists() and te_f.exists():
                    Xt_tr = joblib.load(tr_f)
                    Xt_v  = joblib.load(v_f)
                    Xt_te = joblib.load(te_f)
                    _info(f"[LGBM] Using enhanced features ({suffix}): {Xt_tr.shape}")
                    enhanced_found = True
                    break
        
        if not enhanced_found:
            _info(f"[LGBM] No enhanced features found, using original features.")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state + 10) # Different seed for diversity
    
    oof_train = np.zeros((len(y_tr), len(le.classes_)), dtype=np.float32)
    test_preds = np.zeros((len(y_te), len(le.classes_)), dtype=np.float32)
    valid_preds = np.zeros((len(y_v), len(le.classes_)), dtype=np.float32)

    device_type = 'gpu' if getattr(cfg, 'use_gpu', False) else 'cpu'

    for fold, (trn_idx, val_idx) in enumerate(skf.split(Xt_tr, y_tr_arr)):
        _info(f"LGBM Fold {fold+1}/5 starting...")
        X_fold_tr, y_fold_tr = Xt_tr[trn_idx], y_tr_arr[trn_idx]
        X_fold_val = Xt_tr[val_idx]
        
        X_res, y_res = apply_smart_resampling(X_fold_tr, y_fold_tr, cfg)
        yres_enc = le.transform(np.asarray(y_res))
        
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.08,
            num_leaves=127,
            max_depth=9,
            min_data_in_leaf=15,
            random_state=cfg.random_state + fold + 10,
            n_jobs=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            objective=focal_loss_multiclass_lgbm
        )
        
        # Fit needs to handle custom objective
        model.fit(X_res, yres_enc)
        
        oof_train[val_idx] = _predict_proba_lgbm(model, X_fold_val)
        valid_preds += _predict_proba_lgbm(model, Xt_v) / skf.n_splits
        test_preds += _predict_proba_lgbm(model, Xt_te) / skf.n_splits
        
    save_proba_npz(str(cache_dir/"proba_lgbm_oof_train.npz"), oof_train, le=le)
    save_proba_npz(str(cache_dir/"proba_lgbm_oof_valid.npz"), valid_preds, le=le)
    save_proba_npz(str(cache_dir/"proba_lgbm_oof_test.npz"), test_preds, le=le)
    
    _stage("Training Final LGBM Model on all train data")
    X_res_all, y_res_all = apply_smart_resampling(Xt_tr, y_tr_arr, cfg)
    yres_all_enc = le.transform(np.asarray(y_res_all))
    
    final_model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.08,
        num_leaves=127,
        max_depth=9,
        min_data_in_leaf=15,
        random_state=cfg.random_state + 10,
        n_jobs=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
        objective=focal_loss_multiclass_lgbm
    )
    final_model.fit(X_res_all, yres_all_enc)
    
    lgbm_model_path = cache_dir / "model_lgbm_final.joblib"
    joblib.dump(final_model, lgbm_model_path)
    _info(f"Final LGBM modeli kaydedildi: {lgbm_model_path}")
    
    y_pred_te = test_preds.argmax(1)
    acc = accuracy_score(yte_enc, y_pred_te)
    _,_,f,_ = precision_recall_fscore_support(yte_enc, y_pred_te, average='macro', zero_division=0)
    _info(f"OOF LGBM TEST -> Acc:{acc:.4f} MacroF1:{f:.4f}")
    print(classification_report(yte_enc, y_pred_te, target_names=le.classes_, digits=3))
