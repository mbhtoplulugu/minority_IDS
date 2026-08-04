"""
mcnemar_test.py
===============
Topluluk stratejisi bileşenleri arasında McNemar istatistiksel anlamlılık testi.

Karşılaştırılan çift kombinasyonlar:
  A vs B:  Baseline (eşit ağırlık) vs +ClassWeight
  A vs C:  Baseline vs +Boost
  A vs D:  Baseline vs +Full (tam sistem)
  B vs D:  +ClassWeight vs +Full

Her karşılaştırma için:
  - Genel (tüm örnekler)
  - Sınıf bazlı (her sınıf ayrı)

McNemar istatistiği (süreklilik düzeltmeli):
  χ² = (|n01 - n10| - 1)² / (n01 + n10)
  df = 1, H0: iki sistem arasında fark yoktur

Kullanım:
  python mcnemar_test.py --cache-dir . --dataset unsw
  python mcnemar_test.py --cache-dir . --dataset cicids14
  python mcnemar_test.py --cache-dir . --dataset all
"""
import argparse, pickle, warnings
import numpy as np
from pathlib import Path
import joblib
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore")

MAIN_MODELS = {
    'xgb':    'XGBoost',
    'lgbm':   'LGBM',
    'lgbmV2': 'LGBM_V2',
    'histgb': 'HistGB',
    'mlp':    'MLP',
    'tabnet': 'TabNet',
}

def _info(m):  print(f"[Info]  {m}")
def _stage(m): print(f"\n{'='*65}\n[Stage] {m}\n{'='*65}")


def _load_aligned(path, classes):
    data = np.load(path, allow_pickle=True)
    P = data['proba']
    if 'classes' in data:
        src = data['classes'].astype(str)
        out = np.zeros((P.shape[0], len(classes)), dtype=np.float32)
        idx = {c: i for i, c in enumerate(src)}
        for j, c in enumerate(classes):
            if c in idx and idx[c] < P.shape[1]:
                out[:, j] = P[:, idx[c]]
        return out
    return P


def _apply_boost(P_base, P_exp, c_idx, threshold, alpha):
    if P_exp is None:
        return P_base
    conf = P_exp[:, c_idx]
    mask = (P_base.argmax(axis=1) != c_idx) & (conf >= threshold)
    if not mask.any():
        return P_base
    boosted = P_base.copy().astype(np.float32)
    boosted[mask, c_idx] = np.minimum(
        0.999,
        (1 - alpha) * boosted[mask, c_idx] + alpha * conf[mask])
    rs = boosted[mask].sum(axis=1, keepdims=True)
    rs[rs <= 0] = 1.0
    boosted[mask] /= rs
    return boosted


def _mcnemar(n01, n10):
    """McNemar χ² (süreklilik düzeltmeli) + p değeri döndürür."""
    denom = n01 + n10
    if denom == 0:
        return float('nan'), float('nan')
    chi2  = (abs(n01 - n10) - 1) ** 2 / denom
    p_val = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p_val


def _sig_label(p):
    if np.isnan(p):   return '   '
    if p < 0.001:     return '***'
    if p < 0.01:      return '** '
    if p < 0.05:      return '*  '
    return '   '


def _build_predictions(cdir, classes, n_cls, yte_enc, yv_enc):
    """
    4 ablasyon konfigürasyonu için tahmin dizileri üretir:
      A — baseline     : eşit ağırlıklı soft voting
      B — +classweight : sınıf-özel validation-weighted voting
      C — +boost       : B + uncertainty boost (fast + surgical)
      D — +full        : C + override
    """
    P_te_list  = []
    P_val_list = []
    mk_list    = []
    for m_key in MAIN_MODELS:
        pt = cdir / f'proba_{m_key}_oof_test.npz'
        pv = cdir / f'proba_{m_key}_oof_valid.npz'
        if pt.exists() and pv.exists():
            P_te_list.append(_load_aligned(pt, classes))
            P_val_list.append(_load_aligned(pv, classes))
            mk_list.append(m_key)

    if not P_te_list:
        return None

    # A: Baseline
    P_A = np.mean(P_te_list, axis=0)

    # B: +ClassWeight
    val_f1 = {}
    for i, m_key in enumerate(mk_list):
        yhat_v = P_val_list[i].argmax(axis=1)
        _, _, f1, _ = precision_recall_fscore_support(
            yv_enc, yhat_v, labels=range(n_cls), zero_division=0)
        val_f1[m_key] = f1

    P_B_te  = np.zeros((len(yte_enc), n_cls), dtype=np.float32)
    P_B_val = np.zeros((len(yv_enc),  n_cls), dtype=np.float32)
    w_sum   = np.zeros(n_cls)
    for i, m_key in enumerate(mk_list):
        if m_key in val_f1:
            w = val_f1[m_key]
            P_B_te  += P_te_list[i]  * w
            P_B_val += P_val_list[i] * w
            w_sum   += w
    w_sum[w_sum == 0] = 1e-9
    P_B_te  /= w_sum
    P_B_val /= w_sum

    # C: +Boost
    P_C_te  = P_B_te.copy()
    P_C_val = P_B_val.copy()

    for ep in sorted(cdir.glob('proba_fast_expert_*_oof_test.npz')):
        cls_name = ep.stem.replace('proba_fast_expert_', '').replace('_oof_test', '')
        if cls_name not in classes:
            continue
        vp  = cdir / f'proba_fast_expert_{cls_name}_oof_valid.npz'
        thr = cdir / f'fast_threshold_{cls_name}.pkl'
        if not vp.exists():
            continue
        c_idx = int(np.where(classes == cls_name)[0][0])
        opt   = float(pickle.load(open(thr, 'rb'))) if thr.exists() else 0.5
        P_C_te  = _apply_boost(P_C_te,  _load_aligned(ep, classes), c_idx, opt, 0.20)
        P_C_val = _apply_boost(P_C_val, _load_aligned(vp, classes), c_idx, opt, 0.20)

    for ep in sorted(cdir.glob('proba_surgical_*_oof_test.npz')):
        cls_name = (ep.stem.replace('proba_surgical_', '')
                           .replace('_oof_test', '')
                           .replace('_expert', ''))
        if cls_name not in classes:
            continue
        vp  = cdir / f'proba_surgical_{cls_name}_oof_valid.npz'
        thr = cdir / f'surgical_{cls_name}_threshold.pkl'
        if not vp.exists():
            continue
        c_idx = int(np.where(classes == cls_name)[0][0])
        opt   = float(pickle.load(open(thr, 'rb'))) if thr.exists() else 0.5
        P_C_te  = _apply_boost(P_C_te,  _load_aligned(ep, classes), c_idx, opt, 0.35)
        P_C_val = _apply_boost(P_C_val, _load_aligned(vp, classes), c_idx, opt, 0.35)

    # D: +Full (boost + override)
    yhat_D = P_C_te.argmax(axis=1).copy()
    base_v  = P_C_val.argmax(axis=1).copy()

    for ep in sorted(cdir.glob('proba_fast_expert_*_oof_test.npz')):
        cls_name = ep.stem.replace('proba_fast_expert_', '').replace('_oof_test', '')
        if cls_name not in classes:
            continue
        vp  = cdir / f'proba_fast_expert_{cls_name}_oof_valid.npz'
        thr = cdir / f'fast_threshold_{cls_name}.pkl'
        if not vp.exists():
            continue
        c_idx = int(np.where(classes == cls_name)[0][0])
        opt   = float(pickle.load(open(thr, 'rb'))) if thr.exists() else 0.5
        Pv    = _load_aligned(vp, classes)
        tmp   = base_v.copy()
        tmp[Pv[:, c_idx] >= opt] = c_idx
        _, _, f1b, _ = precision_recall_fscore_support(
            yv_enc, base_v, labels=range(n_cls), zero_division=0)
        _, _, f1n, _ = precision_recall_fscore_support(
            yv_enc, tmp,    labels=range(n_cls), zero_division=0)
        if f1n[c_idx] > f1b[c_idx] and np.mean(f1n) >= np.mean(f1b) - 0.02:
            Pte = _load_aligned(ep, classes)
            mask = Pte[:, c_idx] >= opt
            if mask.sum() > 0:
                yhat_D[mask] = c_idx
            base_v = tmp

    return {
        'A_baseline':    P_A.argmax(axis=1),
        'B_classweight': P_B_te.argmax(axis=1),
        'C_boost':       P_C_te.argmax(axis=1),
        'D_full':        yhat_D,
    }


def run_mcnemar_analysis(cdir, classes, n_cls, yte_enc, yv_enc, ds_name):
    _stage(f"McNEMAR TESTİ — {ds_name.upper()}")

    preds = _build_predictions(cdir, classes, n_cls, yte_enc, yv_enc)
    if preds is None:
        _info("Model dosyaları bulunamadı.")
        return

    # Karşılaştırılacak çiftler
    pairs = [
        ('A_baseline',    'B_classweight', 'Baseline  vs +ClassWeight'),
        ('A_baseline',    'C_boost',       'Baseline  vs +Boost      '),
        ('A_baseline',    'D_full',        'Baseline  vs +Full       '),
        ('B_classweight', 'D_full',        '+ClassWgt vs +Full       '),
        ('C_boost',       'D_full',        '+Boost    vs +Full       '),
    ]

    # ── Genel McNemar tablosu ─────────────────────────────────────────────────
    W = 10
    sep = ('+' + '-'*32 + '+' + '-'*(W+2) + '+' + '-'*(W+2) + '+' +
           '-'*(W+2) + '+' + '-'*16 + '+')

    print('\n  GENEL McNEMAR SONUÇLARI (Tüm Örnekler)')
    print(sep)
    print('| {:<30} | {:^{w}} | {:^{w}} | {:^{w}} | {:14} |'.format(
        'Karşılaştırma', 'n01', 'n10', 'χ²', 'p değeri', w=W))
    print(sep)

    for key_a, key_b, label in pairs:
        y_a = preds[key_a]
        y_b = preds[key_b]
        cor_a = (y_a == yte_enc)
        cor_b = (y_b == yte_enc)
        n01 = int((~cor_a &  cor_b).sum())   # sadece B doğru
        n10 = int(( cor_a & ~cor_b).sum())   # sadece A doğru
        chi2, p = _mcnemar(n01, n10)
        sig = _sig_label(p)
        p_str = f'{p:.4f} {sig}' if not np.isnan(p) else 'N/A'
        chi2_str = f'{chi2:.3f}' if not np.isnan(chi2) else 'N/A'
        print('| {:<30} | {:^{w},} | {:^{w},} | {:^{w}} | {:14} |'.format(
            label, n01, n10, chi2_str, p_str, w=W))

    print(sep)
    print('  * p<0.05  ** p<0.01  *** p<0.001\n')

    # ── Sınıf bazlı McNemar (Baseline vs Full) ────────────────────────────────
    print('  SINIF BAZLI McNEMAR — Baseline vs +Full (En önemli karşılaştırma)')
    y_a = preds['A_baseline']
    y_b = preds['D_full']
    cor_a = (y_a == yte_enc)
    cor_b = (y_b == yte_enc)

    W2 = 7
    sep2 = ('+' + '-'*24 + '+' + '-'*(W2+2) + '+' + '-'*(W2+2) + '+' +
            '-'*(W2+2) + '+' + '-'*(W2+2) + '+' + '-'*16 + '+')
    print(sep2)
    print('| {:<22} | {:^{w}} | {:^{w}} | {:^{w}} | {:^{w}} | {:14} |'.format(
        'Sınıf', 'n01', 'n10', 'n01+n10', 'χ²', 'p değeri', w=W2))
    print(sep2)

    for i, cls_name in enumerate(classes):
        mask = (yte_enc == i)
        if mask.sum() == 0:
            continue
        cb = cor_a[mask]
        cf = cor_b[mask]
        n01 = int((~cb &  cf).sum())
        n10 = int(( cb & ~cf).sum())
        chi2, p = _mcnemar(n01, n10)
        sig    = _sig_label(p)
        p_str  = f'{p:.4f} {sig}' if not np.isnan(p) else 'N/A       '
        chi2_s = f'{chi2:.3f}' if not np.isnan(chi2) else 'N/A'
        print('| {:<22} | {:^{w},} | {:^{w},} | {:^{w},} | {:^{w}} | {:14} |'.format(
            cls_name[:22], n01, n10, n01+n10, chi2_s, p_str, w=W2))

    print(sep2)
    print('  n01: sadece +Full doğru  |  n10: sadece Baseline doğru')
    print('  * p<0.05  ** p<0.01  *** p<0.001\n')


def main():
    ap = argparse.ArgumentParser(description='McNemar İstatistiksel Anlamlılık Testi')
    ap.add_argument('--cache-dir', type=str, default='.')
    ap.add_argument('--dataset', type=str, default='all',
                    choices=['unsw', 'cicids14', 'all'])
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    datasets  = ['unsw', 'cicids14'] if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        cdir = cache_dir / ds
        if not cdir.exists():
            _info(f"{cdir} bulunamadı, atlandı.")
            continue
        le      = joblib.load(cdir / 'label_encoder.joblib')
        classes = le.classes_.astype(str)
        n_cls   = len(classes)
        y_te    = joblib.load(cdir / 'y_te.joblib')
        y_v     = joblib.load(cdir / 'y_v.joblib')
        yte_enc = le.transform(np.asarray(y_te))
        yv_enc  = le.transform(np.asarray(y_v))
        run_mcnemar_analysis(cdir, classes, n_cls, yte_enc, yv_enc, ds)


if __name__ == '__main__':
    main()
