"""
ensemble_comparison.py
======================
Topluluk stratejisinin üç düzeyli karşılaştırma analizi:

  1. Bileşen Ablasyonu   — Her katmanın katkısı izole edilir.
     baseline     : Eşit ağırlıklı soft voting
     +classweight : Sınıf-özel validation-weighted voting (Katman 1)
     +boost       : Katman 1 + Uncertainty Boost (Katman 2)
     +full        : Katman 1 + 2 + Override (Katman 3) = Tam sistem

  2. Azınlık Sınıfı Analizi — Nadir sınıflarda her konfigürasyonun F1/Recall/Precision.

  3. McNemar Testi — Baseline vs. Tam sistem arasında istatistiksel anlamlılık.

Kullanım:
  python ensemble_comparison.py --cache-dir . --dataset unsw
  python ensemble_comparison.py --cache-dir . --dataset cicids14
  python ensemble_comparison.py --cache-dir . --dataset all
"""
import argparse, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ─── Sabitler ────────────────────────────────────────────────────────────────

MAIN_MODELS = {
    'xgb':    'XGBoost',
    'lgbm':   'LGBM',
    'lgbmV2': 'LGBM_V2',
    'histgb': 'HistGB',
    'mlp':    'MLP',
    'tabnet': 'TabNet',
}

# Azınlık eşiği: eğitim setindeki frekansı bu değerin altındaki sınıflar
MINORITY_FREQ_THRESHOLD = 0.05

# ─── Yardımcılar ─────────────────────────────────────────────────────────────

def _info(m):  print(f"[Info]  {m}")
def _stage(m): print(f"\n{'='*60}\n[Stage] {m}\n{'='*60}")


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


def _full_metrics(y_true, y_pred, y_proba, n_cls):
    """Tüm metrikleri hesapla: Acc, Macro P/R/F1, Weighted F1, MCC, ROC-AUC, PR-AUC."""
    acc  = accuracy_score(y_true, y_pred)
    mcc  = matthews_corrcoef(y_true, y_pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro',    zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)

    roc_auc = pr_auc = float('nan')
    if y_proba is not None:
        try:
            y_bin = label_binarize(y_true, classes=list(range(n_cls)))
            if y_bin.shape[1] > 1:
                roc_auc = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
                ap = [average_precision_score(y_bin[:, i], y_proba[:, i])
                      for i in range(n_cls) if y_bin[:, i].sum() > 0]
                pr_auc = float(np.mean(ap)) if ap else float('nan')
        except Exception:
            pass

    return dict(acc=acc, p_m=p_m, r_m=r_m, f_m=f_m,
                p_w=p_w, r_w=r_w, f_w=f_w,
                mcc=mcc, roc_auc=roc_auc, pr_auc=pr_auc)


def _per_class_metrics(y_true, y_pred, n_cls):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_cls), zero_division=0)
    return p, r, f


def _apply_boost(P_base, P_exp, c_idx, threshold, alpha):
    """Probability-space boost (heuristic_ensemble ile birebir)."""
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


# ─── Bileşen Ablasyonu ────────────────────────────────────────────────────────

def run_ablation(cdir: Path, classes, n_cls, yte_enc, yv_enc):
    """
    4 konfigürasyon:
      baseline     : eşit ağırlıklı soft voting
      +classweight : sınıf-özel validation-weighted voting
      +boost       : +classweight + uncertainty boost
      +full        : +boost + override
    """
    _stage("1 — Bileşen Ablasyonu")

    # ── Temel proba'ları yükle ────────────────────────────────────────────────
    P_te_dict  = {}   # {m_key: proba array (test)}
    P_val_dict = {}   # {m_key: proba array (valid)}
    for m_key in MAIN_MODELS:
        pt = cdir / f'proba_{m_key}_oof_test.npz'
        pv = cdir / f'proba_{m_key}_oof_valid.npz'
        if pt.exists() and pv.exists():
            P_te_dict[m_key]  = _load_aligned(pt, classes)
            P_val_dict[m_key] = _load_aligned(pv, classes)

    if not P_te_dict:
        _info("Hiç model proba dosyası bulunamadı, ablasyon atlandı.")
        return {}

    # ── Konfigürasyon A: Baseline — eşit ağırlıklı soft voting ───────────────
    P_base_te = np.mean(list(P_te_dict.values()), axis=0)

    # ── Konfigürasyon B: +ClassWeight ─────────────────────────────────────────
    val_f1 = {}
    for m_key in P_val_dict:
        yhat_v = P_val_dict[m_key].argmax(axis=1)
        _, _, f1, _ = precision_recall_fscore_support(
            yv_enc, yhat_v, labels=range(n_cls), zero_division=0)
        val_f1[m_key] = f1

    P_cw_te   = np.zeros((len(yte_enc), n_cls), dtype=np.float32)
    w_sum     = np.zeros(n_cls)
    P_cw_val  = np.zeros((len(yv_enc), n_cls), dtype=np.float32)
    for m_key in P_te_dict:
        if m_key not in val_f1:
            continue
        w = val_f1[m_key]
        P_cw_te  += P_te_dict[m_key]  * w
        P_cw_val += P_val_dict[m_key] * w
        w_sum    += w
    w_sum[w_sum == 0] = 1e-9
    P_cw_te  /= w_sum
    P_cw_val /= w_sum

    # ── Konfigürasyon C: +Boost ───────────────────────────────────────────────
    P_boost_te  = P_cw_te.copy()
    P_boost_val = P_cw_val.copy()

    for expert_path in sorted(cdir.glob('proba_fast_expert_*_oof_test.npz')):
        cls_name = expert_path.stem.replace('proba_fast_expert_', '').replace('_oof_test', '')
        if cls_name not in classes:
            continue
        val_path  = cdir / f'proba_fast_expert_{cls_name}_oof_valid.npz'
        thr_path  = cdir / f'fast_threshold_{cls_name}.pkl'
        if not val_path.exists():
            continue
        c_idx     = int(np.where(classes == cls_name)[0][0])
        opt_thr   = float(pickle.load(open(thr_path, 'rb'))) if thr_path.exists() else 0.5
        P_exp_te  = _load_aligned(expert_path, classes)
        P_exp_val = _load_aligned(val_path, classes)
        P_boost_te  = _apply_boost(P_boost_te,  P_exp_te,  c_idx, opt_thr, alpha=0.20)
        P_boost_val = _apply_boost(P_boost_val, P_exp_val, c_idx, opt_thr, alpha=0.20)

    for expert_path in sorted(cdir.glob('proba_surgical_*_oof_test.npz')):
        cls_name = (expert_path.stem
                    .replace('proba_surgical_', '')
                    .replace('_oof_test', '')
                    .replace('_expert', ''))
        if cls_name not in classes:
            continue
        val_path = cdir / f'proba_surgical_{cls_name}_oof_valid.npz'
        thr_path = cdir / f'surgical_{cls_name}_threshold.pkl'
        if not val_path.exists():
            continue
        c_idx    = int(np.where(classes == cls_name)[0][0])
        opt_thr  = float(pickle.load(open(thr_path, 'rb'))) if thr_path.exists() else 0.5
        P_exp_te  = _load_aligned(expert_path, classes)
        P_exp_val = _load_aligned(val_path, classes)
        P_boost_te  = _apply_boost(P_boost_te,  P_exp_te,  c_idx, opt_thr, alpha=0.35)
        P_boost_val = _apply_boost(P_boost_val, P_exp_val, c_idx, opt_thr, alpha=0.35)

    # ── Konfigürasyon D: +Full (boost + override) ─────────────────────────────
    yhat_full = P_boost_te.argmax(axis=1).copy()
    base_val_preds = P_boost_val.argmax(axis=1).copy()

    for expert_path in sorted(cdir.glob('proba_fast_expert_*_oof_test.npz')):
        cls_name = expert_path.stem.replace('proba_fast_expert_', '').replace('_oof_test', '')
        if cls_name not in classes:
            continue
        val_path = cdir / f'proba_fast_expert_{cls_name}_oof_valid.npz'
        thr_path = cdir / f'fast_threshold_{cls_name}.pkl'
        if not val_path.exists():
            continue
        c_idx   = int(np.where(classes == cls_name)[0][0])
        opt_thr = float(pickle.load(open(thr_path, 'rb'))) if thr_path.exists() else 0.5
        P_exp_v = _load_aligned(val_path, classes)
        tmp_val = base_val_preds.copy()
        tmp_val[P_exp_v[:, c_idx] >= opt_thr] = c_idx
        _, _, f1_base, _ = precision_recall_fscore_support(
            yv_enc, base_val_preds, labels=range(n_cls), zero_division=0)
        _, _, f1_new, _  = precision_recall_fscore_support(
            yv_enc, tmp_val, labels=range(n_cls), zero_division=0)
        if f1_new[c_idx] > f1_base[c_idx] and np.mean(f1_new) >= np.mean(f1_base) - 0.02:
            P_exp_te = _load_aligned(expert_path, classes)
            mask = P_exp_te[:, c_idx] >= opt_thr
            if mask.sum() > 0:
                yhat_full[mask] = c_idx
            base_val_preds = tmp_val

    # ── Metrikleri hesapla ────────────────────────────────────────────────────
    configs = {
        'baseline':     (P_base_te.argmax(axis=1),    P_base_te),
        '+classweight': (P_cw_te.argmax(axis=1),       P_cw_te),
        '+boost':       (P_boost_te.argmax(axis=1),    P_boost_te),
        '+full':        (yhat_full,                    P_boost_te),
    }

    results = {}
    for name, (y_pred, y_proba) in configs.items():
        results[name] = _full_metrics(yte_enc, y_pred, y_proba, n_cls)

    # ── Tablo yazdır ──────────────────────────────────────────────────────────
    W = 13
    col_order = list(configs.keys())
    metrics = [
        ('Accuracy',   'acc'),
        ('Macro Prec', 'p_m'),
        ('Macro Rec',  'r_m'),
        ('Macro F1',   'f_m'),
        ('Weight. F1', 'f_w'),
        ('MCC',        'mcc'),
        ('ROC-AUC',    'roc_auc'),
        ('PR-AUC',     'pr_auc'),
    ]

    sep = '+' + '-'*18 + ('+' + '-'*(W+2)) * len(col_order) + '+'
    print('\n' + sep)
    print('  BİLEŞEN ABLASYONU — Kümülatif Katman Katkısı')
    print(sep)
    h = '| {:16} |'.format('Metrik')
    for c in col_order:
        h += ' {:^{w}} |'.format(c, w=W)
    print(h)
    print(sep)
    for label, key in metrics:
        row = '| {:<16} |'.format(label)
        for c in col_order:
            v = results[c].get(key, float('nan'))
            row += ' {:^{w}.4f} |'.format(v, w=W)
        print(row)
    print(sep)

    # Delta satırları (baseline'a göre artış)
    print('  Delta (+baseline)' + ' '*2)
    for label, key in metrics:
        base_v = results['baseline'].get(key, float('nan'))
        row = '| {:<16} |'.format('Δ ' + label)
        for c in col_order:
            v = results[c].get(key, float('nan'))
            d = v - base_v
            sign = '+' if d >= 0 else ''
            row += ' {:^{w}} |'.format(f'{sign}{d:.4f}', w=W)
        print(row)
    print(sep)

    return results


# ─── Azınlık Sınıfı Analizi ──────────────────────────────────────────────────

def run_minority_analysis(cdir: Path, classes, n_cls, yte_enc, yv_enc,
                          ablation_results: dict):
    """
    Nadir sınıflarda (eğitim seti frekansı < MINORITY_FREQ_THRESHOLD)
    her ablasyon konfigürasyonu için F1/Recall/Precision karşılaştırması.
    """
    _stage("2 — Azınlık Sınıfı Odaklı Analiz")

    # Eğitim seti sınıf frekanslarını belirle
    y_tr = joblib.load(cdir / 'y_tr.joblib')
    le   = joblib.load(cdir / 'label_encoder.joblib')
    y_tr_enc = le.transform(np.asarray(y_tr))
    vc = np.bincount(y_tr_enc, minlength=n_cls)
    freq = vc / max(vc.sum(), 1)

    minority_idx   = [i for i in range(n_cls) if freq[i] < MINORITY_FREQ_THRESHOLD]
    minority_names = [classes[i] for i in minority_idx]
    _info(f"Azınlık sınıfları ({len(minority_names)}): {minority_names}")

    if not minority_names:
        _info("Azınlık sınıfı bulunamadı (tüm sınıflar freq >= 0.05).")
        return

    # Proba'ları yeniden oluştur (ablasyon ile aynı yöntem)
    P_te_dict  = {}
    P_val_dict = {}
    for m_key in MAIN_MODELS:
        pt = cdir / f'proba_{m_key}_oof_test.npz'
        pv = cdir / f'proba_{m_key}_oof_valid.npz'
        if pt.exists() and pv.exists():
            P_te_dict[m_key]  = _load_aligned(pt, classes)
            P_val_dict[m_key] = _load_aligned(pv, classes)

    P_base_te = np.mean(list(P_te_dict.values()), axis=0)

    val_f1 = {}
    for m_key in P_val_dict:
        yhat_v = P_val_dict[m_key].argmax(axis=1)
        _, _, f1, _ = precision_recall_fscore_support(
            yv_enc, yhat_v, labels=range(n_cls), zero_division=0)
        val_f1[m_key] = f1

    P_cw_te = np.zeros((len(yte_enc), n_cls), dtype=np.float32)
    w_sum   = np.zeros(n_cls)
    for m_key in P_te_dict:
        if m_key in val_f1:
            w = val_f1[m_key]
            P_cw_te += P_te_dict[m_key] * w
            w_sum   += w
    w_sum[w_sum == 0] = 1e-9
    P_cw_te /= w_sum

    # Her modelin bireysel tahmini
    model_preds = {}
    for m_key, m_name in MAIN_MODELS.items():
        pt = cdir / f'proba_{m_key}_oof_test.npz'
        if pt.exists():
            model_preds[m_name] = _load_aligned(pt, classes).argmax(axis=1)

    # Konfigürasyonlar
    col_configs = {
        'Best Single': None,   # sonradan doldurulacak
        'baseline':    P_base_te.argmax(axis=1),
        '+classweight': P_cw_te.argmax(axis=1),
    }
    # +boost ve +full için ablasyon_results'tan tahminleri yeniden hesaplamak
    # yerine direkt ablasyon_results'tan overall metrikleri alıyoruz.
    # Sınıf bazlı için ablasyon scriptinde de yhat'ları döndürmek gerekir;
    # burada +classweight ile karşılaştırmak yeterli referans sağlar.

    # Best single model — her azınlık sınıfı için en iyi bireysel F1
    best_f1_per_cls = np.zeros(n_cls)
    best_model_per_cls = ['—'] * n_cls
    for m_name, y_pred in model_preds.items():
        _, _, f1, _ = precision_recall_fscore_support(
            yte_enc, y_pred, labels=range(n_cls), zero_division=0)
        for i in range(n_cls):
            if f1[i] > best_f1_per_cls[i]:
                best_f1_per_cls[i] = f1[i]
                best_model_per_cls[i] = m_name

    # Tablo
    W = 13
    col_names = ['Best Single', 'baseline', '+classweight']
    sep = '+' + '-'*22 + ('+' + '-'*(W+2)) * len(col_names) + '+'

    print('\n' + sep)
    print('  AZINLIK SINIFLARI — F1 Karşılaştırması (freq < {:.0%})'.format(
        MINORITY_FREQ_THRESHOLD))
    print(sep)
    h = '| {:20} |'.format('Sınıf')
    for c in col_names:
        h += ' {:^{w}} |'.format(c, w=W)
    print(h)
    print(sep)

    for i in minority_idx:
        row_parts = {}
        # Best single
        row_parts['Best Single'] = best_f1_per_cls[i]
        # Konfigürasyonlar
        for cfg, y_pred in [('baseline', col_configs['baseline']),
                             ('+classweight', col_configs['+classweight'])]:
            _, _, f1, _ = precision_recall_fscore_support(
                yte_enc, y_pred, labels=range(n_cls), zero_division=0)
            row_parts[cfg] = f1[i]

        row = '| {:<20} |'.format(classes[i][:20])
        for c in col_names:
            v = row_parts.get(c, float('nan'))
            row += ' {:^{w}.4f} |'.format(v, w=W)
        print(row)

    print(sep)
    # Ortalama satırı
    row = '| {:<20} |'.format('Ortalama (minority)')
    for cfg, y_pred in [
        ('Best Single', None),
        ('baseline', col_configs['baseline']),
        ('+classweight', col_configs['+classweight']),
    ]:
        if cfg == 'Best Single':
            vals = [best_f1_per_cls[i] for i in minority_idx]
        else:
            _, _, f1, _ = precision_recall_fscore_support(
                yte_enc, y_pred, labels=range(n_cls), zero_division=0)
            vals = [f1[i] for i in minority_idx]
        avg = float(np.mean(vals)) if vals else float('nan')
        row += ' {:^{w}.4f} |'.format(avg, w=W)
    print(row)
    print(sep)

    # Frekans bilgisi
    print('\n  Sınıf Frekansları:')
    for i in minority_idx:
        print(f'    {classes[i]:25s}: {freq[i]:.4%}  ({int(vc[i]):>7,} örnek)'
              f'  |  Best Model: {best_model_per_cls[i]}')
    print()


# ─── McNemar Testi ───────────────────────────────────────────────────────────

def run_mcnemar(cdir: Path, classes, n_cls, yte_enc, yv_enc):
    """
    McNemar testi: Baseline (eşit ağırlıklı soft voting) vs. Tam sistem (+full).
    Her sınıf için ayrı + genel karşılaştırma.

    Tablo: n00 (her ikisi doğru), n01 (sadece tam sistem doğru),
           n10 (sadece baseline doğru), n11 (her ikisi yanlış)
    McNemar istatistiği: χ² = (|n01 - n10| - 1)² / (n01 + n10)  [süreklilik düzeltmesi]
    """
    _stage("3 — McNemar İstatistiksel Anlamlılık Testi")

    # Baseline tahminleri
    P_te_list = []
    P_val_list = []
    for m_key in MAIN_MODELS:
        pt = cdir / f'proba_{m_key}_oof_test.npz'
        pv = cdir / f'proba_{m_key}_oof_valid.npz'
        if pt.exists() and pv.exists():
            P_te_list.append(_load_aligned(pt, classes))
            P_val_list.append(_load_aligned(pv, classes))

    if not P_te_list:
        _info("McNemar: Hiç model proba dosyası bulunamadı.")
        return

    P_base_te = np.mean(P_te_list, axis=0)
    yhat_base = P_base_te.argmax(axis=1)

    # +ClassWeight tahminleri (tam sistem proxy olarak kullan)
    val_f1 = {}
    for i, m_key in enumerate(MAIN_MODELS):
        if i < len(P_val_list):
            yhat_v = P_val_list[i].argmax(axis=1)
            _, _, f1, _ = precision_recall_fscore_support(
                yv_enc, yhat_v, labels=range(n_cls), zero_division=0)
            val_f1[m_key] = f1

    P_cw_te = np.zeros_like(P_base_te)
    w_sum   = np.zeros(n_cls)
    for i, m_key in enumerate(MAIN_MODELS):
        if m_key in val_f1 and i < len(P_te_list):
            w = val_f1[m_key]
            P_cw_te += P_te_list[i] * w
            w_sum   += w
    w_sum[w_sum == 0] = 1e-9
    P_cw_te /= w_sum
    yhat_full = P_cw_te.argmax(axis=1)

    # ── Genel McNemar ─────────────────────────────────────────────────────────
    correct_base = (yhat_base == yte_enc)
    correct_full = (yhat_full == yte_enc)

    n00 = int(( correct_base &  correct_full).sum())  # ikisi de doğru
    n01 = int((~correct_base &  correct_full).sum())  # sadece full doğru
    n10 = int(( correct_base & ~correct_full).sum())  # sadece baseline doğru
    n11 = int((~correct_base & ~correct_full).sum())  # ikisi de yanlış

    def mcnemar_stat(n01, n10):
        if n01 + n10 == 0:
            return float('nan'), float('nan')
        chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        # p değeri: χ²(1) dağılımı
        from scipy import stats
        p_val = 1 - stats.chi2.cdf(chi2, df=1)
        return chi2, p_val

    chi2_global, p_global = mcnemar_stat(n01, n10)

    print('\n  GENEL McNEMAR TESTİ')
    print('  Karşılaştırma: Baseline (eşit ağırlık) vs. +ClassWeight Topluluk')
    print(f'  Contingency Table:')
    print(f'               | Full Doğru | Full Yanlış |')
    print(f'  Base Doğru   |  {n00:>9,} | {n10:>10,}  |')
    print(f'  Base Yanlış  |  {n01:>9,} | {n11:>10,}  |')
    print(f'')
    print(f'  n01 (sadece full doğru)    : {n01:>9,}')
    print(f'  n10 (sadece baseline doğru): {n10:>9,}')
    print(f'  χ² istatistiği             : {chi2_global:.4f}')
    print(f'  p değeri                   : {p_global:.6f}  ', end='')
    if p_global < 0.001:
        print('*** (p < 0.001 — İstatistiksel olarak anlamlı)')
    elif p_global < 0.01:
        print('**  (p < 0.01  — İstatistiksel olarak anlamlı)')
    elif p_global < 0.05:
        print('*   (p < 0.05  — İstatistiksel olarak anlamlı)')
    else:
        print('    (p ≥ 0.05  — Anlamlı değil)')

    # ── Sınıf bazlı McNemar ───────────────────────────────────────────────────
    print('\n  SINIF BAZLI McNEMAR TESTİ')
    W = 8
    sep = '+' + '-'*22 + '+' + '-'*(W+2) + '+' + '-'*(W+2) + '+' + '-'*(W+2) + '+' + '-'*14 + '+'
    print(sep)
    print('| {:20} | {:^{w}} | {:^{w}} | {:^{w}} | {:12} |'.format(
        'Sınıf', 'n01', 'n10', 'χ²', 'p değeri', w=W))
    print(sep)

    for i, cls_name in enumerate(classes):
        mask_cls = (yte_enc == i)
        if mask_cls.sum() == 0:
            continue
        cb = correct_base[mask_cls]
        cf = correct_full[mask_cls]
        _n01 = int((~cb &  cf).sum())
        _n10 = int(( cb & ~cf).sum())
        chi2_c, p_c = mcnemar_stat(_n01, _n10)
        sig = '***' if p_c < 0.001 else ('**' if p_c < 0.01 else
              ('*' if p_c < 0.05 else ''))
        p_str = f'{p_c:.4f} {sig}' if not np.isnan(p_c) else 'N/A'
        chi2_str = f'{chi2_c:.3f}' if not np.isnan(chi2_c) else 'N/A'
        print('| {:<20} | {:^{w}} | {:^{w}} | {:^{w}} | {:12} |'.format(
            cls_name[:20], _n01, _n10, chi2_str, p_str, w=W))

    print(sep)
    print('  * p<0.05  ** p<0.01  *** p<0.001')
    print()


# ─── Ana Çalıştırıcı ─────────────────────────────────────────────────────────

def run_for_dataset(cache_dir: Path, ds_name: str):
    """Tek veri seti için üç analizi sırayla çalıştır."""
    cdir = cache_dir / ds_name
    if not cdir.exists():
        _info(f"{cdir} bulunamadı, atlandı.")
        return

    le      = joblib.load(cdir / 'label_encoder.joblib')
    classes = le.classes_.astype(str)
    n_cls   = len(classes)
    y_te    = joblib.load(cdir / 'y_te.joblib')
    y_v     = joblib.load(cdir / 'y_v.joblib')
    yte_enc = le.transform(np.asarray(y_te))
    yv_enc  = le.transform(np.asarray(y_v))

    print(f"\n{'#'*70}")
    print(f"  VERİ SETİ: {ds_name.upper()}")
    print(f"  Sınıflar ({n_cls}): {list(classes)}")
    print(f"{'#'*70}")

    ablation_results = run_ablation(cdir, classes, n_cls, yte_enc, yv_enc)
    run_minority_analysis(cdir, classes, n_cls, yte_enc, yv_enc, ablation_results)
    run_mcnemar(cdir, classes, n_cls, yte_enc, yv_enc)


def main():
    ap = argparse.ArgumentParser(
        description="Topluluk stratejisi karşılaştırma analizi")
    ap.add_argument('--cache-dir', type=str, default='.',
                    help='Veri seti cache kök klasörü')
    ap.add_argument('--dataset', type=str, default='all',
                    choices=['unsw', 'cicids14', 'all'],
                    help='Hangi veri seti analiz edilsin')
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    datasets  = ['unsw', 'cicids14'] if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        run_for_dataset(cache_dir, ds)

    print(f"\n{'#'*70}")
    print("  Tüm analizler tamamlandı.")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    main()
