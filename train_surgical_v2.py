"""
Surgical Expert v2 - Training with Augmented Features (55 features)
Trains binary experts for backdoor, analysis, dos using:
- Original 36 features + 19 new engineered features
- Hard-negative mining + optimized thresholds
"""
import numpy as np, joblib, pickle
from sklearn.metrics import f1_score, precision_score, recall_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

le = joblib.load('label_encoder.joblib')
classes = le.classes_.astype(str)
ytr = le.transform(joblib.load('y_tr.joblib'))
yv = le.transform(joblib.load('y_v.joblib'))
yte = le.transform(joblib.load('y_te.joblib'))

# Load augmented features (36 orig + 19 new = 55)
Xt_tr = joblib.load('Xt_tr_surgical.joblib')
Xt_v = joblib.load('Xt_v_surgical.joblib')
Xt_te = joblib.load('Xt_te_surgical.joblib')

print(f"Feature dimensions: {Xt_tr.shape[1]} (36 orig + 19 engineered)")

# Also load original features for comparison
Xt_tr_orig = joblib.load('Xt_tr.joblib')
Xt_v_orig = joblib.load('Xt_v.joblib')
Xt_te_orig = joblib.load('Xt_te.joblib')

np.random.seed(42)

targets = {
    'backdoor': {
        'confusing': ['exploits', 'dos', 'fuzzers', 'analysis'],
        'n_conf': 15000, 'n_other': 5000, 'smote_ratio': 0.3,
    },
    'analysis': {
        'confusing': ['exploits', 'dos', 'fuzzers', 'backdoor'],
        'n_conf': 15000, 'n_other': 5000, 'smote_ratio': 0.3,
    },
    'dos': {
        'confusing': ['exploits', 'backdoor', 'analysis', 'fuzzers', 'generic', 'reconnaissance'],
        'n_conf': 50000, 'n_other': 10000, 'smote_ratio': None,  # DoS has enough samples
    },
}

results = {}

for cls_name, cfg in targets.items():
    c_idx = int(np.where(classes == cls_name)[0][0])
    y_bin_tr = (ytr == c_idx).astype(int)
    y_bin_v = (yv == c_idx).astype(int)
    y_bin_te = (yte == c_idx).astype(int)
    
    pos_idx = np.where(y_bin_tr == 1)[0]
    conf_idxs = [int(np.where(classes == c)[0][0]) for c in cfg['confusing']]
    conf_neg_idx = np.where(np.isin(ytr, conf_idxs) & (y_bin_tr == 0))[0]
    other_neg_idx = np.where((~np.isin(ytr, conf_idxs)) & (y_bin_tr == 0))[0]
    
    n_conf = min(cfg['n_conf'], len(conf_neg_idx))
    n_other = min(cfg['n_other'], len(other_neg_idx))
    sel_conf = np.random.choice(conf_neg_idx, n_conf, replace=False)
    sel_other = np.random.choice(other_neg_idx, n_other, replace=False)
    sel = np.concatenate([pos_idx, sel_conf, sel_other])
    np.random.shuffle(sel)
    
    print(f"\n{'='*60}")
    print(f"Training {cls_name} expert (pos={len(pos_idx)}, conf_neg={n_conf}, other_neg={n_other})")
    
    # Train with BOTH original and augmented features and compare
    for feat_name, X_tr_feat, X_v_feat, X_te_feat in [
        ('36-orig', Xt_tr_orig, Xt_v_orig, Xt_te_orig),
        ('55-aug', Xt_tr, Xt_v, Xt_te),
    ]:
        X_sub = X_tr_feat[sel]
        y_sub = y_bin_tr[sel]
        
        # Apply SMOTE if configured
        if cfg['smote_ratio'] is not None:
            k = min(5, len(pos_idx) - 1)
            smote = SMOTE(sampling_strategy=cfg['smote_ratio'], k_neighbors=k, random_state=42)
            X_sub, y_sub = smote.fit_resample(X_sub, y_sub)
        
        model = LGBMClassifier(
            n_estimators=800, max_depth=8, num_leaves=63,
            learning_rate=0.02, subsample=0.7, colsample_bytree=0.7,
            min_child_samples=5, scale_pos_weight=1,
            objective='binary', random_state=42, n_jobs=-1, verbose=-1
        )
        model.fit(X_sub, y_sub)
        
        proba_v = model.predict_proba(X_v_feat)[:, 1]
        best_f1 = 0; best_th = 0.5
        for th in np.arange(0.05, 0.95, 0.005):
            pred = (proba_v >= th).astype(int)
            if pred.sum() == 0: continue
            f1 = f1_score(y_bin_v, pred)
            if f1 > best_f1: best_f1 = f1; best_th = th
        
        proba_te = model.predict_proba(X_te_feat)[:, 1]
        pred_te = (proba_te >= best_th).astype(int)
        f1_te = f1_score(y_bin_te, pred_te)
        prec = precision_score(y_bin_te, pred_te, zero_division=0)
        rec = recall_score(y_bin_te, pred_te)
        
        print(f"  {feat_name}: val_F1={best_f1:.4f}  test_F1={f1_te:.4f}  P={prec:.4f}  R={rec:.4f}  th={best_th:.3f}  pred={pred_te.sum()}")
        
        # Save if augmented and better
        if feat_name == '55-aug':
            results[cls_name] = {
                'model': model, 'threshold': best_th, 'f1_te': f1_te,
                'proba_v': proba_v, 'proba_te': proba_te
            }

# Save all augmented experts
print(f"\n{'='*60}")
print("Saving augmented surgical experts...")
for cls_name, res in results.items():
    c_idx = int(np.where(classes == cls_name)[0][0])
    
    with open(f'surgical_{cls_name}_expert.pkl', 'wb') as f:
        pickle.dump(res['model'], f)
    with open(f'surgical_{cls_name}_threshold.pkl', 'wb') as f:
        pickle.dump(res['threshold'], f)
    
    # Save proba arrays
    out_v = np.zeros((len(Xt_v), len(classes)), dtype=np.float32)
    out_v[:, c_idx] = res['proba_v']
    np.savez(f'proba_surgical_{cls_name}_oof_valid.npz', proba=out_v, classes=classes)
    
    out_te = np.zeros((len(Xt_te), len(classes)), dtype=np.float32)
    out_te[:, c_idx] = res['proba_te']
    np.savez(f'proba_surgical_{cls_name}_oof_test.npz', proba=out_te, classes=classes)
    
    print(f"  {cls_name}: F1={res['f1_te']:.4f} th={res['threshold']:.3f}")

# Final projection with all augmented experts
print(f"\n{'='*60}")
print("FINAL ENSEMBLE PROJECTION:")
P_xgb = np.load('proba_xgb_oof_test.npz', allow_pickle=True)['proba']
P_lgbm = np.load('proba_lgbm_oof_test.npz', allow_pickle=True)['proba']
P_lgbmV2 = np.load('proba_lgbmV2_oof_test.npz', allow_pickle=True)['proba']

base = P_xgb.argmax(axis=1).copy()

# Apply surgical overrides in optimal order: Bd -> An -> DoS -> Shellcode -> Worms
for cls_name in ['backdoor', 'analysis', 'dos']:
    c_idx = int(np.where(classes == cls_name)[0][0])
    base[results[cls_name]['proba_te'] >= results[cls_name]['threshold']] = c_idx

s_idx = int(np.where(classes == 'shellcode')[0][0])
base[P_lgbm.argmax(axis=1) == s_idx] = s_idx

w_idx = int(np.where(classes == 'worms')[0][0])
worms_combined = np.maximum(P_lgbm[:, w_idx], P_lgbmV2[:, w_idx])
base[worms_combined >= 0.70] = w_idx

f1s = f1_score(yte, base, average=None)
for i, c in enumerate(classes):
    print(f"  {c:18s} {f1s[i]:.4f}")
print(f"  {'MACRO F1':18s} {np.mean(f1s):.4f}")
