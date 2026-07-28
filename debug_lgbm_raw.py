"""
LGBM RAW verisi üzerinde düşük performansı debug etmek için.
"""
import joblib, numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

for ds in ['unsw', 'cicids']:
    p = Path(ds)
    if not p.exists(): continue

    print(f"\n{'='*55}\n  {ds.upper()}\n{'='*55}")

    Xt_tr = joblib.load(p / 'Xt_tr.joblib')
    Xt_te = joblib.load(p / 'Xt_te.joblib')
    y_tr  = joblib.load(p / 'y_tr.joblib')
    y_te  = joblib.load(p / 'y_te.joblib')
    le    = joblib.load(p / 'label_encoder.joblib')

    y_tr_enc = le.transform(np.asarray(y_tr))
    y_te_enc = le.transform(np.asarray(y_te))

    print(f"  Train shape: {Xt_tr.shape}, dtype={Xt_tr.dtype}")
    print(f"  NaN: {np.isnan(Xt_tr).sum()}, Inf: {np.isinf(Xt_tr).sum()}")
    print(f"  Classes: {le.classes_}")

    # Test 1: class_weight=balanced
    try:
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=150, max_depth=6, num_leaves=31,
                           learning_rate=0.1, n_jobs=-1, random_state=42,
                           verbose=-1, class_weight="balanced")
        m.fit(Xt_tr, y_tr_enc)
        yhat = m.predict_proba(Xt_te).argmax(axis=1)
        acc = accuracy_score(y_te_enc, yhat)
        _, _, mf1, _ = precision_recall_fscore_support(y_te_enc, yhat, average='macro', zero_division=0)
        print(f"  LGBM balanced     → Acc={acc:.4f}, MacroF1={mf1:.4f}")
    except Exception as e:
        print(f"  LGBM balanced ERROR: {e}")

    # Test 2: class_weight=None
    try:
        m2 = LGBMClassifier(n_estimators=150, max_depth=6, num_leaves=31,
                            learning_rate=0.1, n_jobs=-1, random_state=42,
                            verbose=-1, class_weight=None)
        m2.fit(Xt_tr, y_tr_enc)
        yhat2 = m2.predict_proba(Xt_te).argmax(axis=1)
        acc2 = accuracy_score(y_te_enc, yhat2)
        _, _, mf12, _ = precision_recall_fscore_support(y_te_enc, yhat2, average='macro', zero_division=0)
        print(f"  LGBM no_weight    → Acc={acc2:.4f}, MacroF1={mf12:.4f}")
    except Exception as e:
        print(f"  LGBM no_weight ERROR: {e}")

    # Test 3: predict sonucu kontrol (hepsini bir sinifa mi atiyor?)
    try:
        unique_preds, counts = np.unique(yhat, return_counts=True)
        print(f"  Tahmin dagilimi (balanced):")
        for u, c in zip(le.classes_[unique_preds], counts):
            print(f"    {u:20s}: {c}")
    except: pass

print("\nDone.")
