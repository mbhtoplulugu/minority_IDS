import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import precision_recall_fscore_support

# Kok dizindeki proba vs unsw/ dizinindeki proba karsilastir
for ds_dir, le_dir in [('.', '.'), ('unsw', 'unsw')]:
    p_proba = Path(ds_dir) / 'proba_lgbm_oof_test.npz'
    p_le    = Path(le_dir) / 'label_encoder.joblib'
    p_yte   = Path(le_dir) / 'y_te.joblib'
    if not all(f.exists() for f in [p_proba, p_le, p_yte]):
        print(f'{ds_dir}: dosyalar eksik')
        continue
    le = joblib.load(p_le)
    classes = le.classes_.astype(str)
    y_te = joblib.load(p_yte)
    yte_enc = le.transform(np.asarray(y_te))
    data = np.load(p_proba, allow_pickle=True)
    P = data['proba']; src = data['classes'].astype(str)
    out = np.zeros((P.shape[0], len(classes)), dtype=np.float32)
    idx = {c: i for i, c in enumerate(src)}
    for j, c in enumerate(classes): 
        if c in idx: out[:, j] = P[:, idx[c]]
    yhat = out.argmax(axis=1)
    _, _, mf1, _ = precision_recall_fscore_support(yte_enc, yhat, average='macro', zero_division=0)
    print(f'dir={ds_dir:6s} | proba_shape={P.shape} | n_classes={len(classes)} | MacroF1={mf1:.4f}')
    print(f'  proba classes : {list(src)}')
    print(f'  le    classes : {list(classes)}')
    print(f'  y_te  len     : {len(y_te)}')
    print(f'  proba rows    : {P.shape[0]}')
