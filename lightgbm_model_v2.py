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

def run_lgbm_oof_v2(cfg, n_features=50, cached_data=None):
    _stage("LightGBM V2 (Balanced Weights) 5-Fold OOF Training")
    
    cache_dir = Path(getattr(cfg, 'cache_dir', '.'))
    from sklearn.preprocessing import LabelEncoder
    
    # Optimize: Cache'den ykle, preprocess tekrarlanmasn
    if cached_data is not None:
        _info(f"[LGBM V2] Using pre-loaded cached data with {cached_data[0].shape[1]} features")
        Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te = cached_data
    else:
        from experts import _load_cache
        Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te, _ = _load_cache(cache_dir, n_features)
    
    le = joblib.load(cache_dir / 'label_encoder.joblib')
    
    y_tr_arr = np.asarray(y_tr)
    yv_enc = le.transform(np.asarray(y_v))
    yte_enc = le.transform(np.asarray(y_te))
    
    # Enhanced features ncelii (cached_data kullanldysa enhanced'a gei yap)
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
            _info(f"[LGBM V2] Using enhanced features (exact {n_features}): {Xt_tr.shape}")
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
                    _info(f"[LGBM V2] Using enhanced features ({suffix}): {Xt_tr.shape}")
                    enhanced_found = True
                    break
        
        if not enhanced_found:
            _info(f"[LGBM V2] Enhanced features not found, using original features.")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state + 20)
    
    oof_train = np.zeros((len(y_tr), len(le.classes_)), dtype=np.float32)
    test_preds = np.zeros((len(y_te), len(le.classes_)), dtype=np.float32)
    valid_preds = np.zeros((len(y_v), len(le.classes_)), dtype=np.float32)

    # Calculate dynamic class weights based on current classes
    unique_y = np.unique(y_tr_arr)
    # We want to boost minority classes and penalize 'normal'
    # 'normal' usually maps to index 5 in CICIDS or some other in UNSW
    # Let's find index for 'normal' and 'generic'
    normal_idx = -1
    generic_idx = -1
    for i, cls_name in enumerate(le.classes_):
        if str(cls_name).lower() == 'normal': normal_idx = i
        if str(cls_name).lower() == 'generic': generic_idx = i
    
    # Create dynamic weights: Default 1.0, Normal 0.1, others based on scarcity
    # But for LGBM, 'balanced' often works well too. Let's use a hybrid approach.
    dynamic_weights = {}
    for i in range(len(le.classes_)):
        if i == normal_idx: dynamic_weights[i] = 0.2
        elif i == generic_idx: dynamic_weights[i] = 1.0
        else: dynamic_weights[i] = 2.0 # Attack classes

    for fold, (trn_idx, val_idx) in enumerate(skf.split(Xt_tr, y_tr_arr)):
        _info(f"LGBM V2 Fold {fold+1}/5 starting...")
        X_fold_tr, y_fold_tr = Xt_tr[trn_idx], y_tr_arr[trn_idx]
        X_fold_val = Xt_tr[val_idx]
        
        # We apply smart resampling which caps the maximum classes and floors the minimum
        X_res, y_res = apply_smart_resampling(X_fold_tr, y_fold_tr, cfg)
        yres_enc = le.transform(np.asarray(y_res))
        
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.08,
            num_leaves=127,
            max_depth=9,
            min_data_in_leaf=15,
            random_state=cfg.random_state + fold + 20,
            n_jobs=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            objective='multiclass',
            class_weight=dynamic_weights
        )
        
        # Standard fit without custom objective
        model.fit(X_res, yres_enc)
        
        oof_train[val_idx] = model.predict_proba(X_fold_val)
        valid_preds += model.predict_proba(Xt_v) / skf.n_splits
        test_preds += model.predict_proba(Xt_te) / skf.n_splits
        
    save_proba_npz(str(cache_dir/"proba_lgbmV2_oof_train.npz"), oof_train, le=le)
    save_proba_npz(str(cache_dir/"proba_lgbmV2_oof_valid.npz"), valid_preds, le=le)
    save_proba_npz(str(cache_dir/"proba_lgbmV2_oof_test.npz"), test_preds, le=le)
    
    _stage("Training Final LGBM V2 Model on all train data")
    X_res_all, y_res_all = apply_smart_resampling(Xt_tr, y_tr_arr, cfg)
    yres_all_enc = le.transform(np.asarray(y_res_all))
    
    final_model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.08,
        num_leaves=127,
        max_depth=9,
        min_data_in_leaf=15,
        random_state=cfg.random_state + 20,
        n_jobs=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
        objective='multiclass',
        class_weight=dynamic_weights
    )
    final_model.fit(X_res_all, yres_all_enc)
    
    lgbm_model_path = cache_dir / "model_lgbmV2_final.joblib"
    joblib.dump(final_model, lgbm_model_path)
    _info(f"Final LGBM V2 modeli kaydedildi: {lgbm_model_path}")
    
    y_pred_te = test_preds.argmax(1)
    acc = accuracy_score(yte_enc, y_pred_te)
    _,_,f,_ = precision_recall_fscore_support(yte_enc, y_pred_te, average='macro', zero_division=0)
    _info(f"OOF LGBM V2 TEST -> Acc:{acc:.4f} MacroF1:{f:.4f}")
    print(classification_report(yte_enc, y_pred_te, target_names=le.classes_, digits=3))
