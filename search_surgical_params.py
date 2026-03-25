"""Systematic hyper-parameter search for Backdoor & Analysis surgical experts."""
import numpy as np, joblib, pickle
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

le = joblib.load('label_encoder.joblib')
classes = le.classes_.astype(str)
ytr = le.transform(joblib.load('y_tr.joblib'))
yv = le.transform(joblib.load('y_v.joblib'))
yte = le.transform(joblib.load('y_te.joblib'))
Xt_tr = joblib.load('Xt_tr.joblib')
Xt_v = joblib.load('Xt_v.joblib')
Xt_te = joblib.load('Xt_te.joblib')

np.random.seed(42)
bd_idx = int(np.where(classes=='backdoor')[0][0])
a_idx = int(np.where(classes=='analysis')[0][0])

best_results = {}

for target_name, target_idx in [('backdoor', bd_idx), ('analysis', a_idx)]:
    y_bin_tr = (ytr == target_idx).astype(int)
    y_bin_v = (yv == target_idx).astype(int)
    y_bin_te = (yte == target_idx).astype(int)
    pos_idx = np.where(y_bin_tr == 1)[0]
    
    confusing = ['exploits', 'dos', 'fuzzers', 'analysis'] if target_name == 'backdoor' else ['exploits', 'dos', 'fuzzers', 'backdoor']
    conf_idxs = [int(np.where(classes == c)[0][0]) for c in confusing]
    conf_neg_idx = np.where(np.isin(ytr, conf_idxs) & (y_bin_tr == 0))[0]
    other_neg_idx = np.where((~np.isin(ytr, conf_idxs)) & (y_bin_tr == 0))[0]
    
    results = []
    for n_conf, n_other in [(10000, 2000), (15000, 5000), (20000, 5000), (30000, 5000)]:
        for smote_r in [0.2, 0.3, 0.5, 0.7, 1.0]:
            for depth in [6, 8, 10]:
                for n_est in [500, 800, 1200]:
                    for spw in [1, 2, 3]:
                        sel_conf = np.random.choice(conf_neg_idx, min(n_conf, len(conf_neg_idx)), replace=False)
                        sel_other = np.random.choice(other_neg_idx, min(n_other, len(other_neg_idx)), replace=False)
                        sel = np.concatenate([pos_idx, sel_conf, sel_other])
                        np.random.shuffle(sel)
                        X_sub = Xt_tr[sel]; y_sub = y_bin_tr[sel]
                        
                        k = min(5, len(pos_idx) - 1)
                        smote = SMOTE(sampling_strategy=smote_r, k_neighbors=k, random_state=42)
                        X_sub, y_sub = smote.fit_resample(X_sub, y_sub)
                        
                        model = LGBMClassifier(
                            n_estimators=n_est, max_depth=depth, num_leaves=2**depth-1,
                            learning_rate=0.02, subsample=0.7, colsample_bytree=0.7,
                            min_child_samples=3, scale_pos_weight=spw,
                            objective='binary', random_state=42, n_jobs=-1, verbose=-1
                        )
                        model.fit(X_sub, y_sub)
                        
                        proba_v = model.predict_proba(Xt_v)[:, 1]
                        best_f1 = 0; best_th = 0.5
                        for th in np.arange(0.10, 0.95, 0.01):
                            pred = (proba_v >= th).astype(int)
                            if pred.sum() == 0: continue
                            f1 = f1_score(y_bin_v, pred)
                            if f1 > best_f1: best_f1 = f1; best_th = th
                        
                        results.append((best_f1, n_conf, n_other, smote_r, depth, n_est, spw, best_th, model))
    
    results.sort(key=lambda x: -x[0])
    print(f'\n=== {target_name.upper()} TOP-10 ===')
    for r in results[:10]:
        model_r = r[8]
        proba_te = model_r.predict_proba(Xt_te)[:, 1]
        f1_te = f1_score(y_bin_te, (proba_te >= r[7]).astype(int))
        print(f'  val_F1={r[0]:.4f} test_F1={f1_te:.4f} conf={r[1]} other={r[2]} smote={r[3]} d={r[4]} n={r[5]} spw={r[6]} th={r[7]:.2f}')
    
    # Save the best
    best = results[0]
    best_model = best[8]
    best_th = best[7]
    proba_v_best = best_model.predict_proba(Xt_v)[:, 1]
    proba_te_best = best_model.predict_proba(Xt_te)[:, 1]
    f1_te_best = f1_score(y_bin_te, (proba_te_best >= best_th).astype(int))
    
    best_results[target_name] = {
        'model': best_model, 'threshold': best_th,
        'proba_v': proba_v_best, 'proba_te': proba_te_best,
        'f1_te': f1_te_best
    }
    print(f'  >>> BEST {target_name}: val={best[0]:.4f} test={f1_te_best:.4f} th={best_th:.2f}')

# Save best experts
for cls_name, res in best_results.items():
    c_idx = int(np.where(classes == cls_name)[0][0])
    with open(f'surgical_{cls_name}_expert.pkl', 'wb') as f:
        pickle.dump(res['model'], f)
    with open(f'surgical_{cls_name}_threshold.pkl', 'wb') as f:
        pickle.dump(res['threshold'], f)
    out_v = np.zeros((len(Xt_v), len(classes)), dtype=np.float32)
    out_v[:, c_idx] = res['proba_v']
    np.savez(f'proba_surgical_{cls_name}_oof_valid.npz', proba=out_v, classes=classes)
    out_te = np.zeros((len(Xt_te), len(classes)), dtype=np.float32)
    out_te[:, c_idx] = res['proba_te']
    np.savez(f'proba_surgical_{cls_name}_oof_test.npz', proba=out_te, classes=classes)
    print(f'Saved {cls_name}: F1={res["f1_te"]:.4f} th={res["threshold"]:.2f}')

# Also restore the DoS expert with original 36 features (it was better)
print("\nRetraining DoS expert with original features...")
dos_idx = int(np.where(classes=='dos')[0][0])
y_bin_tr = (ytr == dos_idx).astype(int)
y_bin_v = (yv == dos_idx).astype(int)
y_bin_te = (yte == dos_idx).astype(int)
pos_idx = np.where(y_bin_tr == 1)[0]
confusing = ['exploits', 'backdoor', 'analysis', 'fuzzers', 'generic', 'reconnaissance']
conf_idxs = [int(np.where(classes == c)[0][0]) for c in confusing]
conf_neg_idx = np.where(np.isin(ytr, conf_idxs) & (y_bin_tr == 0))[0]
other_neg_idx = np.where((~np.isin(ytr, conf_idxs)) & (y_bin_tr == 0))[0]
sel_conf = np.random.choice(conf_neg_idx, 50000, replace=False)
sel_other = np.random.choice(other_neg_idx, 10000, replace=False)
sel = np.concatenate([pos_idx, sel_conf, sel_other])
np.random.shuffle(sel)

model = LGBMClassifier(
    n_estimators=800, max_depth=8, num_leaves=63,
    learning_rate=0.02, subsample=0.7, colsample_bytree=0.7,
    min_child_samples=5, scale_pos_weight=1,
    objective='binary', random_state=42, n_jobs=-1, verbose=-1
)
model.fit(Xt_tr[sel], y_bin_tr[sel])

proba_v = model.predict_proba(Xt_v)[:, 1]
best_f1 = 0; best_th = 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (proba_v >= th).astype(int)
    if pred.sum() == 0: continue
    f1 = f1_score(y_bin_v, pred)
    if f1 > best_f1: best_f1 = f1; best_th = th

proba_te = model.predict_proba(Xt_te)[:, 1]
f1_te = f1_score(y_bin_te, (proba_te >= best_th).astype(int))
print(f'DoS expert: val_F1={best_f1:.4f} test_F1={f1_te:.4f} th={best_th:.3f}')

with open('surgical_dos_expert.pkl', 'wb') as f: pickle.dump(model, f)
with open('surgical_dos_threshold.pkl', 'wb') as f: pickle.dump(best_th, f)
out_v = np.zeros((len(Xt_v), len(classes)), dtype=np.float32)
out_v[:, dos_idx] = proba_v
np.savez('proba_surgical_dos_oof_valid.npz', proba=out_v, classes=classes)
out_te = np.zeros((len(Xt_te), len(classes)), dtype=np.float32)
out_te[:, dos_idx] = proba_te
np.savez('proba_surgical_dos_oof_test.npz', proba=out_te, classes=classes)
print('Saved DoS expert.')
