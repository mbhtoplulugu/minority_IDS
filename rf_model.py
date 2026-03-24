from pathlib import Path
import joblib
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.ensemble import BalancedBaggingClassifier

try:
    from unsw_nb15_pipeline import preprocess_and_cache, apply_smart_resampling, save_proba_npz, _stage, _info
except ImportError:
    pass

warnings.filterwarnings("ignore")

def run_rf_oof(cfg, n_features=50):
    _stage("Balanced Bagging + Extra Trees 5-Fold OOF Training")
    
    # Base prep and cache
    Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te = preprocess_and_cache(cfg)
    
    cache_dir = Path(getattr(cfg, 'cache_dir', '.'))
    from sklearn.preprocessing import LabelEncoder
    le = joblib.load(cache_dir / 'label_encoder.joblib')
    
    y_tr_arr = np.asarray(y_tr)
    yv_enc = le.transform(np.asarray(y_v))
    yte_enc = le.transform(np.asarray(y_te))
    
    # Enhanced features ncelii
    enhanced_files = [
        cache_dir / f"Xt_tr_enhanced{n_features}.joblib",
        cache_dir / f"Xt_v_enhanced{n_features}.joblib", 
        cache_dir / f"Xt_te_enhanced{n_features}.joblib"
    ]
    
    if all(f.exists() for f in enhanced_files):
        Xt_tr = joblib.load(enhanced_files[0])
        Xt_v = joblib.load(enhanced_files[1])
        Xt_te = joblib.load(enhanced_files[2])
        _info(f"[RF] Using enhanced features: {Xt_tr.shape}")
    else:
        _info(f"[RF] Enhanced features not found, reverting to original features.")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state + 30)
    
    oof_train = np.zeros((len(y_tr), len(le.classes_)), dtype=np.float32)
    test_preds = np.zeros((len(y_te), len(le.classes_)), dtype=np.float32)
    valid_preds = np.zeros((len(y_v), len(le.classes_)), dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(Xt_tr, y_tr_arr)):
        _info(f"RF (BB-ET) Fold {fold+1}/5 starting...")
        X_fold_tr, y_fold_tr = Xt_tr[trn_idx], y_tr_arr[trn_idx]
        X_fold_val = Xt_tr[val_idx]
        
        # apply_smart_resampling helps cap the majority classes
        X_res, y_res = apply_smart_resampling(X_fold_tr, y_fold_tr, cfg)
        yres_enc = le.transform(np.asarray(y_res))
        
        # Base CPU Model
        base_model = ExtraTreesClassifier(
            n_estimators=50,      # Kept relatively low to save memory in bagging
            max_depth=25,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            max_features='sqrt',
            n_jobs=-1,
            random_state=cfg.random_state + fold + 30
        )
        
        # Meta Bagging Estimator (Imbalanced-Learn)
        model = BalancedBaggingClassifier(
            estimator=base_model,
            sampling_strategy='all', # dynamically balance all classes in each bootstrap
            replacement=False,       # sampling without replacement is gentler on RAM
            n_estimators=10,         # 10 bags of 50 trees = 500 trees total
            random_state=cfg.random_state + fold + 30,
            n_jobs=-1
        )
        
        model.fit(X_res, yres_enc)
        
        oof_train[val_idx] = model.predict_proba(X_fold_val)
        valid_preds += model.predict_proba(Xt_v) / skf.n_splits
        test_preds += model.predict_proba(Xt_te) / skf.n_splits
        
    save_proba_npz(str(cache_dir/"proba_rf_oof_train.npz"), oof_train, le=le)
    save_proba_npz(str(cache_dir/"proba_rf_oof_valid.npz"), valid_preds, le=le)
    save_proba_npz(str(cache_dir/"proba_rf_oof_test.npz"), test_preds, le=le)
    
    _stage("Training Final RF (BB-ET) Model on all train data")
    X_res_all, y_res_all = apply_smart_resampling(Xt_tr, y_tr_arr, cfg)
    yres_all_enc = le.transform(np.asarray(y_res_all))
    
    final_base_model = ExtraTreesClassifier(
        n_estimators=50,
        max_depth=25,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        max_features='sqrt',
        n_jobs=-1,
        random_state=cfg.random_state + 30
    )
    final_model = BalancedBaggingClassifier(
        estimator=final_base_model,
        sampling_strategy='all',
        replacement=False,
        n_estimators=10,
        random_state=cfg.random_state + 30,
        n_jobs=-1
    )
    
    final_model.fit(X_res_all, yres_all_enc)
    
    rf_model_path = cache_dir / "model_rf_final.joblib"
    joblib.dump(final_model, rf_model_path)
    _info(f"Final RF modeli kaydedildi: {rf_model_path}")
    
    y_pred_te = test_preds.argmax(1)
    acc = accuracy_score(yte_enc, y_pred_te)
    _,_,f,_ = precision_recall_fscore_support(yte_enc, y_pred_te, average='macro', zero_division=0)
    _info(f"OOF RF TEST -> Acc:{acc:.4f} MacroF1:{f:.4f}")
    print(classification_report(yte_enc, y_pred_te, target_names=le.classes_, digits=3))
