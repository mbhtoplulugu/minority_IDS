"""
ablation_ensemble_analysis.py  —  Ensemble Kademeli Ablation Analizi
=====================================================================
4 sütunlu tablo üretir (her dataset icin ayri):
  1. Default Ensemble (Soft Voting) : Esit agirlikli basit soft voting (baseline)
  2. Default Ensemble (Hard Voting): Basit hard voting (baseline)
  3. +Uzman Modeller               : Uzman overrideleri (sabit 0.5 esik, validation-gating YOK)
  4. +Esik & Otonom Karar          : Tam pipeline (optimized thresholds + validation-gating)

Her sütunda sinif bazli Precision, Recall, F1 + genel Accuracy, Macro F1 raporlanir.

Kullanim:
  python ablation_ensemble_analysis.py --cache-dir unsw cicids14
  python ablation_ensemble_analysis.py --cache-dir unsw --output-dir ablation_results/ensemble
"""

import argparse, warnings, time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support
)

warnings.filterwarnings("ignore")

# ── sütun sabitleri ──────────────────────────────────────────────────────────
COL_DEFAULT_SOFT = "Default Ensemble (Soft Voting)"
COL_DEFAULT_HARD = "Default Ensemble (Hard Voting)"
COL_EXPERTS = "+Uzman Modeller"
COL_FULL    = "+Esik & Otonom Karar"
ALL_COLS = [COL_DEFAULT_SOFT, COL_DEFAULT_HARD, COL_EXPERTS, COL_FULL]

# Temel modeller (RF haric)
MAIN_MODELS = {
    'xgb':    'XGBoost',
    'lgbm':   'LGBM',
    'lgbmV2': 'LGBM_V2',
    'histgb': 'HistGB',
    'mlp':    'MLP',
    'tabnet': 'TabNet',
}

# ── yardimci ─────────────────────────────────────────────────────────────────

def _info(m):  print(f"  [Info]  {m}")
def _stage(m): print(f"\n{'='*60}\n[Stage] {m}\n{'='*60}")


def _load_aligned(path, classes):
    """Olasilik matrisini yukle ve sinif sirasini hizala."""
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


def get_f1_arr(y_true, y_pred, n_cls):
    """Sinif bazli F1 dizisi dondurur."""
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_cls), zero_division=0)
    return f1


def metrics_per_class(y_true, y_pred, classes):
    """Her sinif icin Precision, Recall, F1 + genel metrikler."""
    n = len(classes)
    p_arr, r_arr, f_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(n), zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=np.arange(n))

    per_cls = {}
    for i, cls in enumerate(classes):
        per_cls[cls] = {"F1": f_arr[i], "Recall": r_arr[i], "Precision": p_arr[i]}
    overall = {
        "Accuracy":        acc,
        "Macro Precision": mp,
        "Macro Recall":    mr,
        "Macro F1":        mf,
    }
    return per_cls, overall


# ── sütun 1: default ensemble ────────────────────────────────────────────────

def compute_default_ensemble(cdir, classes):
    """
    Esit agirlikli basit soft voting.
    Tum temel modellerin test olasilik ortalamasi.
    Hicbir uzman, esik veya otonom mekanizma yok.
    """
    P_sum = None
    model_count = 0

    for m_key in MAIN_MODELS:
        path = cdir / f'proba_{m_key}_oof_test.npz'
        if path.exists():
            P_m = _load_aligned(path, classes)
            if P_sum is None:
                P_sum = np.zeros_like(P_m, dtype=np.float64)
            P_sum += P_m
            model_count += 1
            _info(f"Default Soft: {m_key} yuklendi")

    if model_count == 0 or P_sum is None:
        _info("HATA: Hicbir model olasilik dosyasi bulunamadi!")
        return None

    P_avg = P_sum / model_count
    _info(f"Default Ensemble (Soft Voting): {model_count} model ile esit agirlikli soft voting")
    return P_avg.argmax(axis=1)


def compute_default_ensemble_hard(cdir, classes):
    """
    Basit hard voting.
    Her modelin tek tahmini kullanilir ve en cok oy alan sinif secilir.
    """
    vote_matrix = None
    model_count = 0

    for m_key in MAIN_MODELS:
        path = cdir / f'proba_{m_key}_oof_test.npz'
        if path.exists():
            P_m = _load_aligned(path, classes)
            pred = P_m.argmax(axis=1)
            if vote_matrix is None:
                vote_matrix = np.zeros((len(pred), len(classes)), dtype=np.int32)
            vote_matrix[np.arange(len(pred)), pred] += 1
            model_count += 1
            _info(f"Default Hard: {m_key} yuklendi")

    if model_count == 0 or vote_matrix is None:
        _info("HATA: Hicbir model olasilik dosyasi bulunamadi!")
        return None

    _info(f"Default Ensemble (Hard Voting): {model_count} model ile basit hard voting")
    return vote_matrix.argmax(axis=1)


# ── sütun 2: +uzman modeller ─────────────────────────────────────────────────

def compute_experts_raw(cdir, classes, base_preds):
    """
    Default ensemble uzerine uzman overrideleri.
    Sabit 0.5 esik, validation-gating YOK.
    Optimize edilmis esikler KULLANILMAZ.
    """
    yhat = base_preds.copy()
    n_overrides = 0

    # Surgical Expert Override (sabit 0.5 esik)
    for expert_proba_path in sorted(cdir.glob('proba_surgical_*_oof_test.npz')):
        cls_name = expert_proba_path.stem.replace('proba_surgical_', '').replace('_oof_test', '')
        if cls_name not in classes:
            continue
        c_idx = int(np.where(classes == cls_name)[0][0])
        P_surg = _load_aligned(expert_proba_path, classes)
        mask = P_surg[:, c_idx] >= 0.5
        count = int(np.sum(mask))
        if count > 0:
            yhat[mask] = c_idx
            n_overrides += count
            _info(f"Surgical: {cls_name} -> {count} override (sabit esik=0.5)")

    # Fast Binary Expert Override (sabit 0.5 esik)
    for expert_proba_path in sorted(cdir.glob('proba_fast_expert_*_oof_test.npz')):
        m_key = expert_proba_path.stem.replace('proba_', '').replace('_oof_test', '')
        cls_name = m_key.replace('fast_expert_', '')
        if cls_name not in classes:
            continue
        c_idx = int(np.where(classes == cls_name)[0][0])
        P_fe = _load_aligned(expert_proba_path, classes)
        mask = P_fe[:, c_idx] >= 0.5
        count = int(np.sum(mask))
        if count > 0:
            yhat[mask] = c_idx
            n_overrides += count
            _info(f"Fast: {cls_name} -> {count} override (sabit esik=0.5)")

    _info(f"+Uzman Modeller: toplam {n_overrides} override uygulandi")
    return yhat


# ── sütun 3: +esik & otonom karar ────────────────────────────────────────────

def compute_full_pipeline(cdir, classes, yte_enc, yv_enc):
    """
    Tam heuristic ensemble pipeline:
      1. Validation F1-weighted class-wise soft voting
      2. Surgical expert override (optimize edilmis esik + validation-gating)
      3. Fast expert override (optimize edilmis esik + validation-gating)
    """
    n_cls = len(classes)

    # --- 1. Validation F1 agirliklari ---
    val_model_results = {}
    for m_key in MAIN_MODELS:
        path = cdir / f'proba_{m_key}_oof_valid.npz'
        if path.exists():
            P_v = _load_aligned(path, classes)
            f1s = get_f1_arr(yv_enc, P_v.argmax(axis=1), n_cls)
            val_model_results[m_key] = f1s

    # --- 2. Validation soft voting (gating baseline) ---
    P_soft_val = np.zeros((len(yv_enc), n_cls), dtype=np.float64)
    weight_sum = np.zeros(n_cls, dtype=np.float64)
    for m_key in MAIN_MODELS:
        if m_key in val_model_results:
            weight_arr = val_model_results[m_key]
            p_path = cdir / f'proba_{m_key}_oof_valid.npz'
            if p_path.exists():
                P_m = _load_aligned(p_path, classes)
                P_soft_val += P_m * weight_arr
                weight_sum += weight_arr
    weight_sum[weight_sum == 0] = 1e-9
    P_soft_val /= weight_sum
    base_val_preds = P_soft_val.argmax(axis=1)
    base_val_f1 = get_f1_arr(yv_enc, base_val_preds, n_cls)
    _info(f"Validation weighted soft voting Macro F1: {np.mean(base_val_f1):.4f}")

    # --- 3. Test soft voting (weighted) ---
    P_soft_te = None
    weight_sum_te = np.zeros(n_cls, dtype=np.float64)
    for m_key in MAIN_MODELS:
        if m_key in val_model_results:
            weight_arr = val_model_results[m_key]
            p_path = cdir / f'proba_{m_key}_oof_test.npz'
            if p_path.exists():
                P_m = _load_aligned(p_path, classes)
                if P_soft_te is None:
                    P_soft_te = np.zeros_like(P_m, dtype=np.float64)
                P_soft_te += P_m * weight_arr
                weight_sum_te += weight_arr

    if P_soft_te is None:
        _info("HATA: Test olasiliklari yuklenemedi!")
        return None

    weight_sum_te[weight_sum_te == 0] = 1e-9
    P_soft_te /= weight_sum_te
    yhat_te = P_soft_te.argmax(axis=1)

    # --- 4. Surgical Expert Override (optimized threshold + validation-gating) ---
    for expert_pkl in sorted(cdir.glob('proba_surgical_*_oof_test.npz')):
        # Correctly parse class name from filename: proba_surgical_<cls>_oof_test.npz
        cls_name = expert_pkl.stem.replace('proba_surgical_', '').replace('_oof_test', '').replace('_expert', '')
        if cls_name not in classes:
            continue

        thresh_path = cdir / f'surgical_{cls_name}_threshold.pkl'
        proba_path  = expert_pkl
        proba_val_path = cdir / f'proba_surgical_{cls_name}_oof_valid.npz'

        c_idx = int(np.where(classes == cls_name)[0][0])

        if thresh_path.exists():
            with open(thresh_path, 'rb') as f:
                opt_thresh = pickle.load(f)
        else:
            opt_thresh = 0.20 if cls_name in ['worms', 'web attack brute', 'web attack xss', 'web attack sql', 'heartbleed', 'infiltration'] else 0.50

        with open(thresh_path, 'rb') as f:
            opt_thresh = pickle.load(f)

        # Validation gating (Precision Safeguard)
        apply = True
        if proba_val_path.exists():
            P_surg_val = _load_aligned(proba_val_path, classes)
            temp_val = base_val_preds.copy()
            mask_val = P_surg_val[:, c_idx] >= opt_thresh
            temp_val[mask_val] = c_idx
            
            prec_base, _, f1_base_arr, _ = precision_recall_fscore_support(yv_enc, base_val_preds, average=None, zero_division=0)
            prec_new, _, f1_new_arr, _ = precision_recall_fscore_support(yv_enc, temp_val, average=None, zero_division=0)
            
            target_improved = f1_new_arr[c_idx] > f1_base_arr[c_idx]
            precision_safeguard = np.all(prec_new >= 0.97 * prec_base)
            
            if not (target_improved and precision_safeguard):
                _info(f"Surgical: {cls_name} BYPASSED (Failed Precision Safeguard or F1 Improvement, esik={opt_thresh:.3f})")
                apply = False

        if apply:
            P_surg = _load_aligned(proba_path, classes)
            mask = P_surg[:, c_idx] >= opt_thresh
            count = int(np.sum(mask))
            if count > 0:
                yhat_te[mask] = c_idx
                _info(f"Surgical: {cls_name} -> {count} override (esik={opt_thresh:.3f}, val-gated)")

    # --- 5. Fast Expert Override (optimized threshold + validation-gating) ---
    for expert_val_path in sorted(cdir.glob('proba_fast_expert_*_oof_valid.npz')):
        cls_name = expert_val_path.stem.replace('proba_fast_expert_', '').replace('_oof_valid', '')
        if cls_name not in classes:
            continue

        expert_test_path = cdir / f'proba_fast_expert_{cls_name}_oof_test.npz'
        if not expert_test_path.exists():
            continue

        c_idx = int(np.where(classes == cls_name)[0][0])

        # Optimize edilmis esik
        thresh_path = cdir / f'fast_threshold_{cls_name}.pkl'
        if thresh_path.exists():
            with open(thresh_path, 'rb') as f:
                opt_thresh = pickle.load(f)
        else:
            opt_thresh = 0.20 if cls_name in ['worms', 'web attack brute', 'web attack xss', 'web attack sql', 'heartbleed', 'infiltration'] else 0.50

        # Validation gating (Precision Safeguard)
        P_fe_val = _load_aligned(expert_val_path, classes)
        temp_val = base_val_preds.copy()
        mask_val = P_fe_val[:, c_idx] >= opt_thresh
        temp_val[mask_val] = c_idx
        
        prec_base, _, f1_base_arr, _ = precision_recall_fscore_support(yv_enc, base_val_preds, average=None, zero_division=0)
        prec_new, _, f1_new_arr, _ = precision_recall_fscore_support(yv_enc, temp_val, average=None, zero_division=0)
        
        target_improved = f1_new_arr[c_idx] > f1_base_arr[c_idx]
        precision_safeguard = np.all(prec_new >= 0.97 * prec_base)

        if not (target_improved and precision_safeguard):
            _info(f"Fast: {cls_name} BYPASSED (Failed Precision Safeguard or F1 Improvement, esik={opt_thresh:.3f})")
        else:
            P_fe_te = _load_aligned(expert_test_path, classes)
            mask = P_fe_te[:, c_idx] >= opt_thresh
            count = int(np.sum(mask))
            if count > 0:
                yhat_te[mask] = c_idx
                _info(f"Fast: {cls_name} -> {count} override (esik={opt_thresh:.3f}, val-gated)")

    return yhat_te


# ── tablo uretici ────────────────────────────────────────────────────────────

def build_ensemble_ablation_table(cache_dir, dataset_label):
    """Ana tablo uretici. 3 sutunlu sinif bazli metrik tablosu dondurur."""
    cdir = Path(cache_dir)
    le = joblib.load(cdir / 'label_encoder.joblib')
    classes = le.classes_.astype(str)
    n_cls = len(classes)

    y_te = joblib.load(cdir / 'y_te.joblib')
    y_v  = joblib.load(cdir / 'y_v.joblib')
    yte_enc = le.transform(np.asarray(y_te))
    yv_enc  = le.transform(np.asarray(y_v))

    # Satir indeksi: her sinif icin 3 satir + 4 genel
    row_index = []
    for cls in classes:
        row_index += [f"{cls} — Precision", f"{cls} — Recall", f"{cls} — F1"]
    row_index += ["── Accuracy", "── Macro Precision", "── Macro Recall", "── Macro F1"]

    df = pd.DataFrame(index=row_index, columns=ALL_COLS, dtype=float)

    def _fill_col(col_label, y_pred):
        per_cls, overall = metrics_per_class(yte_enc, y_pred, classes)
        for cls in classes:
            df.loc[f"{cls} — Precision", col_label] = per_cls[cls]["Precision"]
            df.loc[f"{cls} — Recall",    col_label] = per_cls[cls]["Recall"]
            df.loc[f"{cls} — F1",        col_label] = per_cls[cls]["F1"]
        df.loc["── Accuracy",        col_label] = overall["Accuracy"]
        df.loc["── Macro Precision", col_label] = overall["Macro Precision"]
        df.loc["── Macro Recall",    col_label] = overall["Macro Recall"]
        df.loc["── Macro F1",        col_label] = overall["Macro F1"]

    # ── Sütun 1: Default Ensemble (Soft Voting) ───────────────────────────
    _stage(f"[{dataset_label}] Sutun 1: Default Ensemble (Soft Voting)")
    t0 = time.time()
    y_default_soft = compute_default_ensemble(cdir, classes)
    if y_default_soft is not None:
        _fill_col(COL_DEFAULT_SOFT, y_default_soft)
        macro_f1 = float(df.loc["── Macro F1", COL_DEFAULT_SOFT])
        _info(f"Default Ensemble (Soft Voting) Macro F1: {macro_f1:.4f} ({time.time()-t0:.1f}s)")

    # ── Sütun 2: Default Ensemble (Hard Voting) ───────────────────────────
    _stage(f"[{dataset_label}] Sutun 2: Default Ensemble (Hard Voting)")
    t0 = time.time()
    y_default_hard = compute_default_ensemble_hard(cdir, classes)
    if y_default_hard is not None:
        _fill_col(COL_DEFAULT_HARD, y_default_hard)
        macro_f1 = float(df.loc["── Macro F1", COL_DEFAULT_HARD])
        _info(f"Default Ensemble (Hard Voting) Macro F1: {macro_f1:.4f} ({time.time()-t0:.1f}s)")

    # ── Sütun 3: +Uzman Modeller ──────────────────────────────────────────
    _stage(f"[{dataset_label}] Sutun 3: +Uzman Modeller")
    t0 = time.time()
    if y_default_soft is not None:
        y_experts = compute_experts_raw(cdir, classes, y_default_soft)
        if y_experts is not None:
            _fill_col(COL_EXPERTS, y_experts)
            macro_f1 = float(df.loc["── Macro F1", COL_EXPERTS])
            _info(f"+Uzman Modeller Macro F1: {macro_f1:.4f} ({time.time()-t0:.1f}s)")

    # ── Sütun 4: +Esik & Otonom Karar ─────────────────────────────────────
    _stage(f"[{dataset_label}] Sutun 4: +Esik & Otonom Karar")
    t0 = time.time()
    y_full = compute_full_pipeline(cdir, classes, yte_enc, yv_enc)
    if y_full is not None:
        _fill_col(COL_FULL, y_full)
        macro_f1 = float(df.loc["── Macro F1", COL_FULL])
        _info(f"+Esik & Otonom Karar Macro F1: {macro_f1:.4f} ({time.time()-t0:.1f}s)")

    return df, classes


# ── gorsellestime ─────────────────────────────────────────────────────────────

def plot_ensemble_table(df, dataset_label, out_dir):
    """Tablo gorseli: F1 + genel metrikler."""
    show_rows = [r for r in df.index if "— F1" in r or r.startswith("──")]
    df_show = df.loc[show_rows].astype(float)
    cols = ALL_COLS

    fig, ax = plt.subplots(figsize=(len(cols) * 3.2 + 2, len(show_rows) * 0.42 + 1.8))
    ax.axis("off")

    vals = df_show.values
    cell_colors = []
    for row in vals:
        row_colors = []
        for v in row:
            if np.isnan(v):
                row_colors.append("#EEEEEE")
            else:
                g = int(120 + v * 135)
                r_c = int(255 - v * 180)
                row_colors.append(f"#{r_c:02X}{g:02X}60")
        cell_colors.append(row_colors)

    tbl = ax.table(
        cellText=[[f"{v:.4f}" if not np.isnan(v) else "-" for v in row] for row in vals],
        rowLabels=show_rows,
        colLabels=cols,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)

    ax.set_title(f"Ensemble Ablation — {dataset_label}", fontsize=11, pad=12, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / f"ensemble_ablation_{dataset_label}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    _info(f"Gorsel kaydedildi: {out_path}")


def plot_heatmap(df, dataset_label, out_dir, metric="F1"):
    """Sinif bazli isi haritasi."""
    cls_rows = [r for r in df.index if f"— {metric}" in r]
    if not cls_rows:
        return
    cols = ALL_COLS
    df_sub = df.loc[cls_rows, cols].astype(float)
    ylabels = [r.replace(f" — {metric}", "") for r in cls_rows]

    fig, ax = plt.subplots(figsize=(len(cols) * 2.5, max(3, len(cls_rows) * 0.55)))
    im = ax.imshow(df_sub.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=9, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04)
    for i in range(len(ylabels)):
        for j in range(len(cols)):
            v = df_sub.values[i, j]
            ax.text(j, i, f"{v:.3f}" if not np.isnan(v) else "-",
                    ha="center", va="center", fontsize=7,
                    color="black" if v > 0.3 else "white")
    ax.set_title(f"{dataset_label} — Sinif Bazli {metric} Isi Haritasi", fontsize=10)
    plt.tight_layout()
    out_path = out_dir / f"ensemble_ablation_{dataset_label}_heatmap_{metric}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    _info(f"Isi haritasi kaydedildi: {out_path}")


def plot_bar_macro_f1(df, dataset_label, out_dir):
    """Macro F1 bar chart."""
    if "── Macro F1" not in df.index:
        return
    vals = [float(df.loc["── Macro F1", c]) if not pd.isna(df.loc["── Macro F1", c]) else 0
            for c in ALL_COLS]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [plt.cm.tab10(i % 10) for i in range(len(ALL_COLS))]
    bars = ax.bar(ALL_COLS, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title(f"{dataset_label} — Ensemble Ablation Macro F1", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f"ensemble_ablation_{dataset_label}_bar_MacroF1.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    _info(f"Bar chart kaydedildi: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Ensemble Kademeli Ablation Analizi")
    ap.add_argument("--cache-dir", nargs="+", required=True,
                    help="Cache dizinleri (ornek: unsw cicids14)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Cikti dizini (varsayilan: ablation_results/<dataset>/)")
    args = ap.parse_args()

    print(f"\n{'#'*65}")
    print(f"  ENSEMBLE KADEMELI ABLATION ANALIZI")
    print(f"  Veri setleri: {args.cache_dir}")
    print(f"{'#'*65}")

    for cache_dir_str in args.cache_dir:
        cdir = Path(cache_dir_str)
        dataset_label = cdir.name

        if args.output_dir:
            out_dir = Path(args.output_dir) / dataset_label
        else:
            out_dir = cdir.parent / "ablation_results" / dataset_label
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#'*65}")
        print(f"  DATASET: {dataset_label.upper()}")
        print(f"  Cache : {cdir.resolve()}")
        print(f"  Cikti : {out_dir.resolve()}")
        print(f"{'#'*65}\n")

        # Gerekli dosyalari kontrol et
        if not (cdir / 'label_encoder.joblib').exists():
            print(f"  [Atlandi] {cdir} icinde label_encoder.joblib bulunamadi.")
            continue
        if not (cdir / 'y_te.joblib').exists():
            print(f"  [Atlandi] {cdir} icinde y_te.joblib bulunamadi.")
            continue

        df, classes = build_ensemble_ablation_table(str(cdir), dataset_label)

        # CSV kaydet
        csv_path = out_dir / f"ensemble_ablation_{dataset_label}.csv"
        df.to_csv(csv_path, float_format="%.4f")
        _info(f"CSV kaydedildi: {csv_path}")

        # Konsol ciktisi — genel metrikler
        gen_rows = [r for r in df.index if r.startswith("──")]
        print(f"\n  {dataset_label.upper()} — Genel Metrikler:")
        print(f"  {'Metrik':<22} " + "  ".join(f"{c:>25}" for c in ALL_COLS))
        print(f"  {'-'*(22 + len(ALL_COLS)*27)}")
        for row in gen_rows:
            vals_str = [
                f"{float(df.loc[row, c]):>25.4f}" if not pd.isna(df.loc[row, c]) else f"{'—':>25}"
                for c in ALL_COLS
            ]
            print(f"  {row:<22} {'  '.join(vals_str)}")

        # Konsol ciktisi — sinif bazli F1
        print(f"\n  {dataset_label.upper()} — Sinif Bazli F1:")
        print(f"  {'Sinif':<22} " + "  ".join(f"{c:>25}" for c in ALL_COLS))
        print(f"  {'-'*(22 + len(ALL_COLS)*27)}")
        for cls in classes:
            row_key = f"{cls} — F1"
            if row_key in df.index:
                vals_str = [
                    f"{float(df.loc[row_key, c]):>25.4f}" if not pd.isna(df.loc[row_key, c]) else f"{'—':>25}"
                    for c in ALL_COLS
                ]
                print(f"  {cls:<22} {'  '.join(vals_str)}")

        # Gorsellestime
        plot_ensemble_table(df, dataset_label, out_dir)
        plot_heatmap(df, dataset_label, out_dir, metric="F1")
        plot_heatmap(df, dataset_label, out_dir, metric="Recall")
        plot_bar_macro_f1(df, dataset_label, out_dir)

    print(f"\n{'#'*65}")
    print(f"  Ensemble ablation analizi tamamlandi.")
    print(f"{'#'*65}\n")


if __name__ == "__main__":
    main()
