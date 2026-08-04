import argparse, joblib, pickle, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support, f1_score, classification_report,
    accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
)

def _info(m):  print(f"[Info]  {m}")
def _stage(m): print(f"[Stage] {m}")

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

def get_f1_arr(y_true, y_pred_encoded, n_cls):
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_encoded, labels=range(n_cls), zero_division=0)
    return f1


def _apply_correction_boost(P_base, P_expert, c_idx, threshold, alpha=0.30):
    """Temel ensemble yanlış sınıf seçmişken, uzman yüksek güvenle doğru sınıfı işaret ediyorsa onu düzeltir."""
    if P_base is None or P_expert is None:
        return P_base

    expert_conf = P_expert[:, c_idx]
    base_pred = np.argmax(P_base, axis=1)
    mask = (base_pred != c_idx) & (expert_conf >= threshold)
    if not np.any(mask):
        return P_base

    boosted = np.array(P_base, copy=True, dtype=np.float32)
    boosted[mask, c_idx] = np.maximum(
        boosted[mask, c_idx],
        np.minimum(0.999, (1.0 - alpha) * boosted[mask, c_idx] + alpha * expert_conf[mask])
    )

    row_sums = boosted[mask].sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    boosted[mask] = boosted[mask] / row_sums
    return boosted


def _dynamic_macro_tol(class_freq: float, base_tol: float = 0.001) -> float:
    """
    Sinif frekansina gore kabul edilebilir Macro F1 dusus toleransini hesaplar.
    Nadir siniflar (dusuk frekans) daha buyuk tolerans alir cunku
    o sinifin katkisi Macro F1'e matematiksel olarak sinirlidir.

    freq < 0.001  -> 0.05  (heartbleed, web_attack_sql gibi <1000 ornek)
    freq < 0.010  -> 0.02  (web_attack_brute, web_attack_xss gibi <10k ornek)
    freq >= 0.010 -> base_tol (0.001 — buyuk siniflar icin siki)
    """
    if class_freq < 0.001:
        return 0.05
    elif class_freq < 0.010:
        return 0.02
    else:
        return base_tol


def _adaptive_prec_threshold(prec_base_val: float) -> float:
    """
    Sinifin mevcut precision degerine gore kabul edilebilir minimum
    precision esigini hesaplar.

    Dusuk precision siniflar (< 0.50) : mutlak 0.03 dusus izni
    Orta  precision siniflar (0.50-0.90): %5 goreceli dusus izni
    Yuksek precision siniflar (> 0.90) : %1 goreceli dusus izni
    """
    if prec_base_val < 0.50:
        return max(prec_base_val - 0.03, 0.0)
    elif prec_base_val < 0.90:
        return prec_base_val * 0.95
    else:
        return prec_base_val * 0.99


def _expert_accepted(
    y_true: np.ndarray,
    base_preds: np.ndarray,
    new_preds: np.ndarray,
    c_idx: int,
    prec_base_arr: np.ndarray,
    f1_base_arr: np.ndarray,
    n_cls: int,
    class_freq: float = 0.01,
    label: str = "",
) -> bool:
    """
    Expert katki kabul kriteri — 4 kosul:

    1. Hedef sinif F1 artmali (veya esit kalmali).
    2. Genel Macro F1 dusmesin — tolerans sinif frekansina gore dinamik.
       Nadir siniflar (freq<0.001) icin 0.05, az siniflar icin 0.02,
       buyuk siniflar icin 0.001.
    3. Hedef sinifin precision degeri adaptif esigi saglamali.
    4. Hicbir sinifin F1'i kendi base degerinin
       max(0.03, 0.05 * f1_base[j])'dan fazla dusmesin.
    """
    prec_new, _, f1_new, _ = precision_recall_fscore_support(
        y_true, new_preds, labels=range(n_cls), zero_division=0)

    macro_base = float(np.mean(f1_base_arr))
    macro_new  = float(np.mean(f1_new))
    macro_tol  = _dynamic_macro_tol(class_freq)

    # Kosul 1
    if f1_new[c_idx] < f1_base_arr[c_idx] - 1e-6:
        _info(f"    [{label}] RED — Kosul1: hedef F1 dusus "
              f"{f1_base_arr[c_idx]:.4f} -> {f1_new[c_idx]:.4f}")
        return False

    # Kosul 2 — dinamik tolerans
    if macro_new < macro_base - macro_tol:
        _info(f"    [{label}] RED — Kosul2: Macro F1 dusus "
              f"{macro_base:.4f} -> {macro_new:.4f} (tol={macro_tol:.3f})")
        return False

    # Kosul 3
    min_prec = _adaptive_prec_threshold(float(prec_base_arr[c_idx]))
    if prec_new[c_idx] < min_prec:
        _info(f"    [{label}] RED — Kosul3: hedef precision "
              f"{prec_base_arr[c_idx]:.4f} -> {prec_new[c_idx]:.4f} "
              f"(esik={min_prec:.4f})")
        return False

    # Kosul 4
    for j in range(n_cls):
        delta_tol = max(0.03, 0.05 * float(f1_base_arr[j]))
        if f1_new[j] < f1_base_arr[j] - delta_tol:
            _info(f"    [{label}] RED — Kosul4: sinif[{j}] F1 dusus "
                  f"{f1_base_arr[j]:.4f} -> {f1_new[j]:.4f} "
                  f"(tol={delta_tol:.4f})")
            return False

    _info(f"    [{label}] KABUL — hedef F1 {f1_base_arr[c_idx]:.4f} -> {f1_new[c_idx]:.4f}, "
          f"Macro {macro_base:.4f} -> {macro_new:.4f} (tol={macro_tol:.3f})")
    return True

# ── Yardimci metrik fonksiyonlari ────────────────────────────────────────────

def _model_scalar_metrics(y_true, y_pred):
    """Acc, Macro Prec/Recall/F1, MCC (scalar)."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    return dict(acc=acc, prec=p, recall=r, f1=f, mcc=mcc)


def _print_model_comparison(model_metrics, display_models):
    """Tum modeller: Accuracy | Precision | Recall | F1 | MCC  (Macro, Test)"""
    W = 10
    n = len(display_models)
    sep = "+" + "-" * 16 + ("+" + "-" * (W + 2)) * n + "+"
    print("\n" + "=" * (18 + (W + 3) * n))
    print("  TUM MODELLER — ACC / PRECISION / RECALL / F1 / MCC  (Test, Macro)")
    print(sep)
    h = "| {:14} |".format("Metrik")
    for m in display_models:
        h += " {:^{w}} |".format(m[:W], w=W)
    print(h)
    print(sep)
    for label, key in [("Accuracy",  "acc"),
                       ("Precision", "prec"),
                       ("Recall",    "recall"),
                       ("F1",        "f1"),
                       ("MCC",       "mcc")]:
        row = "| {:<14} |".format(label)
        for m in display_models:
            v = model_metrics.get(m, {}).get(key, float("nan"))
            row += " {:^{w}.4f} |".format(v, w=W)
        print(row)
    print(sep)


def _print_ensemble_extended(yte_enc, yhat_te, P_soft_te, classes, n_cls):
    """Ensemble: Macro + Weighted + ROC-AUC + PR-AUC + MCC"""
    acc = accuracy_score(yte_enc, yhat_te)
    mcc = matthews_corrcoef(yte_enc, yhat_te)
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        yte_enc, yhat_te, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        yte_enc, yhat_te, average="weighted", zero_division=0)

    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(yte_enc, classes=list(range(n_cls)))
        if y_bin.shape[1] > 1:
            roc_auc = roc_auc_score(y_bin, P_soft_te, average="macro", multi_class="ovr")
            ap_scores = [average_precision_score(y_bin[:, i], P_soft_te[:, i])
                         for i in range(n_cls) if y_bin[:, i].sum() > 0]
            pr_auc = float(np.mean(ap_scores)) if ap_scores else float("nan")
        else:
            roc_auc = pr_auc = float("nan")
    except Exception as e:
        _info(f"ROC/PR hesaplanamadi: {e}")
        roc_auc = pr_auc = float("nan")

    W = 12
    sep = "+" + "-" * 26 + "+" + "-" * (W + 2) + "+" + "-" * (W + 2) + "+"
    total_w = 26 + 2 * (W + 3) - 1
    print("\n" + sep)
    print("| {:^{w}} |".format("ENSEMBLE GENISLETILMIS METRIKLER", w=total_w))
    print(sep)
    print("| {:<24} | {:^{w}} | {:^{w}} |".format("Metrik", "Macro", "Weighted", w=W))
    print(sep)
    for label, vm, vw in [("Accuracy",  acc,   acc),
                           ("Precision", p_mac, p_w),
                           ("Recall",    r_mac, r_w),
                           ("F1-Score",  f_mac, f_w)]:
        print("| {:<24} | {:^{w}.4f} | {:^{w}.4f} |".format(label, vm, vw, w=W))
    print(sep)
    print("| {:<24} | {:^{w}.4f} | {:^{w}} |".format("ROC-AUC (OvR macro)", roc_auc, "-", w=W))
    print("| {:<24} | {:^{w}.4f} | {:^{w}} |".format("PR-AUC  (macro avg)", pr_auc,  "-", w=W))
    print("| {:<24} | {:^{w}.4f} | {:^{w}} |".format("MCC",                 mcc,     "-", w=W))
    print(sep)


def _plot_roc_pr(yte_enc, n_cls, model_probas, cache_dir):
    """ROC ve PR egrisi — Ensemble + tum modeller tek grafik."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, precision_recall_curve, auc

        y_bin = label_binarize(yte_enc, classes=list(range(n_cls)))
        if y_bin.shape[1] == 1:
            _info("ROC/PR grafigi: tek sinif, atlandi.")
            return

        COLORS = [
            "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
            "#8338EC", "#3A86FF", "#06D6A0", "#FB5607", "#FFBE0B",
            "#8D99AE", "#B5838D",
        ]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        for ax, curve_type in zip(axes, ["ROC", "PR"]):
            for i, (mname, P) in enumerate(model_probas.items()):
                color = COLORS[i % len(COLORS)]
                lw    = 2.5 if mname == "Ensemble" else 1.5
                ls    = "-"  if mname == "Ensemble" else "--"
                alpha = 1.0  if mname == "Ensemble" else 0.75

                aucs = []
                if curve_type == "ROC":
                    mean_x = np.linspace(0, 1, 200)
                    y_interps = []
                    for j in range(n_cls):
                        if y_bin[:, j].sum() == 0:
                            continue
                        fpr_j, tpr_j, _ = roc_curve(y_bin[:, j], P[:, j])
                        aucs.append(auc(fpr_j, tpr_j))
                        y_interps.append(np.interp(mean_x, fpr_j, tpr_j))
                    if y_interps:
                        ax.plot(mean_x, np.mean(y_interps, axis=0),
                                color=color, lw=lw, ls=ls, alpha=alpha,
                                label=f"{mname} (AUC={np.mean(aucs):.3f})")
                else:
                    mean_x = np.linspace(0, 1, 200)
                    y_interps = []
                    for j in range(n_cls):
                        if y_bin[:, j].sum() == 0:
                            continue
                        prec_j, rec_j, _ = precision_recall_curve(y_bin[:, j], P[:, j])
                        aucs.append(auc(rec_j, prec_j))
                        y_interps.append(np.interp(mean_x, rec_j[::-1], prec_j[::-1]))
                    if y_interps:
                        ax.plot(mean_x, np.mean(y_interps, axis=0),
                                color=color, lw=lw, ls=ls, alpha=alpha,
                                label=f"{mname} (AP={np.mean(aucs):.3f})")

            if curve_type == "ROC":
                ax.plot([0, 1], [0, 1], "k--", lw=1, label="Rastgele (AUC=0.500)")
                ax.set_xlabel("Yanlış Pozitif Oranı", fontsize=12)
                ax.set_ylabel("Doğru Positif Oranı", fontsize=12)
                ax.set_title("ROC Egrisi (Makro Ort.)", fontsize=13, fontweight="bold")
                ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
            else:
                ax.set_xlabel("Duyarlılık", fontsize=12)
                ax.set_ylabel("Kesinlik", fontsize=12)
                ax.set_title("PR Egrisi (Makro Ort.)", fontsize=13, fontweight="bold")
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])
            ax.grid(True, linestyle="--", alpha=0.4)

        plt.suptitle("Ensemble + Modeller — ROC & PR Egrisi",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        out = Path(cache_dir) / "ensemble_roc_pr_curves.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        _info(f"ROC/PR grafigi kaydedildi: {out}")

    except Exception as e:
        _info(f"Grafik olusturulamadi: {e}")


def heuristic_ensemble(cache_dir: str, exclude_weak: bool = False):
    """
    Unified Hybrid Ensemble Architecture
    Base: Class-Wise Validation-Weighted Soft Voting
    Overrides: Autonomous Surgical & Fast Binary Experts
    """
    cdir = Path(cache_dir)
    le = joblib.load(cdir / 'label_encoder.joblib')
    classes = le.classes_.astype(str)
    n_cls = len(classes)

    _stage("Loading Ground Truth and Model Predictions")
    y_te = joblib.load(cdir / 'y_te.joblib')
    y_v  = joblib.load(cdir / 'y_v.joblib')
    yte_enc = le.transform(np.asarray(y_te))
    yv_enc  = le.transform(np.asarray(y_v))

    # --- 1. Comparative Analysis ---
    _stage("Performing Comparative Analysis of Individual Models")

    model_results = {}
    main_models = {
        'xgb':    'XGBoost',
        'lgbm':   'LGBM_2',
        'lgbmV2': 'LGBM_1',
        'histgb': 'HistGB',
        'mlp':    'MLP',
        'tabnet': 'TabNet',
        'rf':     'RF',
    }

    for m_key, m_name in main_models.items():
        path = cdir / f'proba_{m_key}_oof_test.npz'
        if path.exists():
            P = _load_aligned(path, classes)
            model_results[m_name] = get_f1_arr(yte_enc, P.argmax(axis=1), n_cls)

    for p in list(cdir.glob('proba_*_oof_test.npz')):
        name = p.stem.replace('proba_', '').replace('_oof_test', '')
        if name in main_models:
            continue
        if 'aecnn' in name.lower() or 'ae_cnn' in name.lower():
            continue
        f_name = (name.replace('fast_expert_', 'FE_')
                      .replace('expert_', 'E_')
                      .replace('bb_', 'BB_')
                      .replace('tabnet_', 'TN_').title())
        P = _load_aligned(p, classes)
        f1s = get_f1_arr(yte_enc, P.argmax(axis=1), n_cls)
        if np.max(f1s) > 0.01:
            model_results[f_name] = f1s

    sorted_models  = sorted(model_results.keys(),
                            key=lambda x: np.mean(model_results[x]), reverse=True)
    display_models = sorted_models[:15]

    # Sinif bazli F1 tablosu (orijinal)
    print("\n" + "=" * 110)
    print(f"| {'DETAILED COMPARATIVE PERFORMANCE ANALYSIS (Test Set F1-Scores)':^106} |")
    print("+" + "-" * 15 + "+" + ("-" * 10 + "+") * len(display_models))
    header = f"| {'Class':<13} |"
    for m in display_models:
        header += f" {m[:8]:^8} |"
    print(header)
    print("+" + "-" * 15 + "+" + ("-" * 10 + "+") * len(display_models))
    for i, cls in enumerate(classes):
        row = f"| {cls:<13} |"
        for m in display_models:
            row += f" {model_results[m][i]:^8.4f} |"
        print(row)
    print("+" + "-" * 15 + "+" + ("-" * 10 + "+") * len(display_models))
    macro_row = f"| {'Macro F1':<13} |"
    for m in display_models:
        macro_row += f" {np.mean(model_results[m]):^8.4f} |"
    print(macro_row)
    print("+" + "-" * 15 + "+" + ("-" * 10 + "+") * len(display_models))

    # Acc / Precision / Recall / F1 / MCC tablosu
    model_metrics_dict = {}
    # ROC/PR grafigi icin: sadece ana modeller (surgical/fast expert haric)
    main_model_names = set(main_models.values())
    model_probas_dict  = {}

    for m_name in display_models:
        ppath_key = None
        for mk, mn in main_models.items():
            if mn == m_name:
                ppath_key = mk
                break
        if ppath_key:
            p = cdir / f'proba_{ppath_key}_oof_test.npz'
        else:
            slug = (m_name.lower()
                    .replace('fe_', 'fast_expert_')
                    .replace('e_', 'expert_'))
            p = cdir / f'proba_{slug}_oof_test.npz'
            if not p.exists():
                candidates = list(cdir.glob(
                    f'proba_*{slug.split("_")[0]}*_oof_test.npz'))
                p = candidates[0] if candidates else None
        if p and Path(p).exists():
            P = _load_aligned(str(p), classes)
            model_metrics_dict[m_name] = _model_scalar_metrics(yte_enc, P.argmax(axis=1))
            # Grafige sadece ana modelleri ekle
            if m_name in main_model_names:
                model_probas_dict[m_name] = P

    _print_model_comparison(model_metrics_dict, display_models)

    # --- 2. Base Consensus: Class-Wise Validation-Weighted Soft Voting ---
    _stage("Establishing Base Consensus: Class-Wise Validation-Weighted Soft Voting")

    val_model_results = {}
    for m_key in main_models:
        path = cdir / f'proba_{m_key}_oof_valid.npz'
        if path.exists():
            P_v = _load_aligned(path, classes)
            val_model_results[m_key] = get_f1_arr(yv_enc, P_v.argmax(axis=1), n_cls)

    P_soft_val = np.zeros((len(yv_enc), n_cls), dtype=np.float32)
    weight_sum = np.zeros(n_cls)
    for m_key in main_models:
        if m_key in val_model_results:
            w = val_model_results[m_key]
            p_path = cdir / f'proba_{m_key}_oof_valid.npz'
            if p_path.exists():
                P_soft_val += _load_aligned(p_path, classes) * w
                weight_sum += w
    weight_sum[weight_sum == 0] = 1e-9
    P_soft_val /= weight_sum

    base_val_preds = P_soft_val.argmax(axis=1)
    _info(f"Consensus Base Model Validation Macro F1: "
          f"{np.mean(get_f1_arr(yv_enc, base_val_preds, n_cls)):.4f}")

    # Test soft voting
    dummy_path = cdir / 'proba_xgb_oof_test.npz'
    if not dummy_path.exists():
        for m in main_models:
            c = cdir / f'proba_{m}_oof_test.npz'
            if c.exists():
                dummy_path = c
                break
        else:
            existing = [f.name for f in cdir.iterdir()] if cdir.exists() else []
            raise RuntimeError(
                f"No proba_*_oof_test.npz found in '{cdir}'.\n"
                "Files: " + "\n  ".join(existing or ["(empty)"]))

    P_soft_te = np.zeros_like(_load_aligned(dummy_path, classes))
    weight_sum_te = np.zeros(n_cls)
    for m_key in main_models:
        if m_key in val_model_results:
            w = val_model_results[m_key]
            p_path = cdir / f'proba_{m_key}_oof_test.npz'
            if p_path.exists():
                P_soft_te += _load_aligned(p_path, classes) * w
                weight_sum_te += w
    weight_sum_te[weight_sum_te == 0] = 1e-9
    P_soft_te /= weight_sum_te

    # --- 2b. Uncertainty-aware expert boost (class-wise) ---
    _stage("Applying Uncertainty-Aware Expert Boosts")
    base_val_preds = P_soft_val.argmax(axis=1)
    prec_base_arr, _, f1_base_arr, _ = precision_recall_fscore_support(
        yv_enc, base_val_preds, labels=range(n_cls), zero_division=0)

    # Sinif frekanslarini hesapla (dinamik tolerans icin)
    vc = np.bincount(yv_enc, minlength=n_cls)
    freq_arr = vc / max(vc.sum(), 1)

    for expert_proba_path in sorted(cdir.glob('proba_surgical_*_oof_test.npz')):
        cls_name = (expert_proba_path.stem
                    .replace('proba_surgical_', '')
                    .replace('_oof_test', '')
                    .replace('_expert', ''))
        if cls_name not in classes:
            continue
        proba_val_path = cdir / f'proba_surgical_{cls_name}_oof_valid.npz'
        if not proba_val_path.exists():
            continue
        c_idx = int(np.where(classes == cls_name)[0][0])
        thresh_path = cdir / f'surgical_{cls_name}_threshold.pkl'
        opt_thresh = pickle.load(open(thresh_path, 'rb')) if thresh_path.exists() else 0.5
        threshold = float(opt_thresh)

        P_surg_val = _load_aligned(proba_val_path, classes)
        temp_val = _apply_correction_boost(P_soft_val, P_surg_val, c_idx, threshold, alpha=0.35)
        temp_val_preds = temp_val.argmax(axis=1)

        if _expert_accepted(yv_enc, base_val_preds, temp_val_preds, c_idx,
                            prec_base_arr, f1_base_arr, n_cls,
                            class_freq=float(freq_arr[c_idx]),
                            label=f"BOOST surgical/{cls_name}"):
            P_soft_val    = temp_val
            base_val_preds = P_soft_val.argmax(axis=1)
            prec_base_arr, _, f1_base_arr, _ = precision_recall_fscore_support(
                yv_enc, base_val_preds, labels=range(n_cls), zero_division=0)
            P_soft_te = _apply_correction_boost(
                P_soft_te,
                _load_aligned(expert_proba_path, classes),
                c_idx, threshold, alpha=0.35)

    for expert_val_path in sorted(cdir.glob('proba_fast_expert_*_oof_valid.npz')):
        cls_name = (expert_val_path.stem
                    .replace('proba_fast_expert_', '')
                    .replace('_oof_valid', ''))
        if cls_name not in classes:
            continue
        expert_test_path = cdir / f'proba_fast_expert_{cls_name}_oof_test.npz'
        if not expert_test_path.exists():
            continue
        c_idx = int(np.where(classes == cls_name)[0][0])
        thresh_path = cdir / f'fast_threshold_{cls_name}.pkl'
        opt_thresh = pickle.load(open(thresh_path, 'rb')) if thresh_path.exists() else 0.5
        threshold = float(opt_thresh)

        P_fe_val = _load_aligned(expert_val_path, classes)
        temp_val = _apply_correction_boost(P_soft_val, P_fe_val, c_idx, threshold, alpha=0.20)
        temp_val_preds = temp_val.argmax(axis=1)

        if _expert_accepted(yv_enc, base_val_preds, temp_val_preds, c_idx,
                            prec_base_arr, f1_base_arr, n_cls,
                            class_freq=float(freq_arr[c_idx]),
                            label=f"BOOST fast/{cls_name}"):
            P_soft_val    = temp_val
            base_val_preds = P_soft_val.argmax(axis=1)
            prec_base_arr, _, f1_base_arr, _ = precision_recall_fscore_support(
                yv_enc, base_val_preds, labels=range(n_cls), zero_division=0)
            P_soft_te = _apply_correction_boost(
                P_soft_te,
                _load_aligned(expert_test_path, classes),
                c_idx, threshold, alpha=0.20)

    base_val_preds = P_soft_val.argmax(axis=1)
    yhat_te = P_soft_te.argmax(axis=1)

    # --- 3. Surgical & Fast Expert Overrides ---
    _stage("Applying Autonomous Surgical Overrides (Validation-Gated)")
    base_val_preds = P_soft_val.argmax(axis=1)
    prec_base_arr, _, f1_base_arr, _ = precision_recall_fscore_support(
        yv_enc, base_val_preds, labels=range(n_cls), zero_division=0)

    for expert_proba_path in sorted(cdir.glob('proba_surgical_*_oof_test.npz')):
        cls_name = (expert_proba_path.stem
                    .replace('proba_surgical_', '')
                    .replace('_oof_test', '')
                    .replace('_expert', ''))
        if cls_name not in classes:
            continue
        thresh_path    = cdir / f'surgical_{cls_name}_threshold.pkl'
        proba_val_path = cdir / f'proba_surgical_{cls_name}_oof_valid.npz'
        if not proba_val_path.exists():
            continue
        c_idx = int(np.where(classes == cls_name)[0][0])

        with open(thresh_path, 'rb') as f:
            opt_thresh = pickle.load(f)

        P_surg_val = _load_aligned(proba_val_path, classes)
        temp_val = base_val_preds.copy()
        temp_val[P_surg_val[:, c_idx] >= opt_thresh] = c_idx

        if _expert_accepted(yv_enc, base_val_preds, temp_val, c_idx,
                            prec_base_arr, f1_base_arr, n_cls,
                            class_freq=float(freq_arr[c_idx]),
                            label=f"OVERRIDE surgical/{cls_name}"):
            P_surg_te = _load_aligned(expert_proba_path, classes)
            mask = P_surg_te[:, c_idx] >= opt_thresh
            if mask.sum() > 0:
                yhat_te[mask] = c_idx
                _info(f"  -> SURGICAL OVERRIDE: {cls_name} "
                      f"{int(mask.sum())} ornek (thresh={opt_thresh:.3f})")
            base_val_preds = temp_val
            prec_base_arr, _, f1_base_arr, _ = precision_recall_fscore_support(
                yv_enc, base_val_preds, labels=range(n_cls), zero_division=0)

    for expert_val_path in sorted(cdir.glob('proba_fast_expert_*_oof_valid.npz')):
        cls_name = (expert_val_path.stem
                    .replace('proba_fast_expert_', '')
                    .replace('_oof_valid', ''))
        if cls_name not in classes:
            continue
        expert_test_path = cdir / f'proba_fast_expert_{cls_name}_oof_test.npz'
        if not expert_test_path.exists():
            continue
        c_idx = int(np.where(classes == cls_name)[0][0])
        thresh_path = cdir / f'fast_threshold_{cls_name}.pkl'
        opt_thresh = pickle.load(open(thresh_path, 'rb')) if thresh_path.exists() else 0.5
        _info(f"  Degerlendir: {cls_name} (thresh={opt_thresh:.3f})")

        P_fe_val = _load_aligned(expert_val_path, classes)
        temp_val = base_val_preds.copy()
        temp_val[P_fe_val[:, c_idx] >= opt_thresh] = c_idx

        if _expert_accepted(yv_enc, base_val_preds, temp_val, c_idx,
                            prec_base_arr, f1_base_arr, n_cls,
                            class_freq=float(freq_arr[c_idx]),
                            label=f"OVERRIDE fast/{cls_name}"):
            P_fe_te = _load_aligned(expert_test_path, classes)
            mask = P_fe_te[:, c_idx] >= opt_thresh
            if mask.sum() > 0:
                yhat_te[mask] = c_idx
                _info(f"  -> FAST OVERRIDE: {cls_name} "
                      f"{int(mask.sum())} ornek (thresh={opt_thresh:.3f})")
            base_val_preds = temp_val
            prec_base_arr, _, f1_base_arr, _ = precision_recall_fscore_support(
                yv_enc, base_val_preds, labels=range(n_cls), zero_division=0)

    # --- FINAL ---
    f1s_ens  = get_f1_arr(yte_enc, yhat_te, n_cls)
    macro_ens = np.mean(f1s_ens)
    acc_ens   = accuracy_score(yte_enc, yhat_te)

    _stage("Hybrid Ensemble Final Performance")
    print("\n" + "=" * 50)
    print(f"| {'FINAL ENSEMBLE RESULTS (AUTO-ROUTING)':^46} |")
    print("+" + "-" * 20 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    print(f"| {'Class':<18} | {'F1-Score':^10} | {'Status':^10} |")
    print("+" + "-" * 20 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    median_f1 = float(np.median(f1s_ens[f1s_ens > 0])) if np.any(f1s_ens > 0) else 0.5
    for i, cls in enumerate(classes):
        v = f1s_ens[i]
        status = "OK" if v >= median_f1 else "LOW" if v > 0 else "-"
        print(f"| {cls:<18} | {v:^10.4f} | {status:^10} |")
    print("+" + "-" * 20 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    print(f"| {'MACRO F1':<18} | {macro_ens:^10.4f} | "
          f"{'SUCCESS' if macro_ens >= 0.75 else 'IN-PROG':^10} |")
    print(f"| {'ACCURACY':<18} | {acc_ens:^10.4f} | {'-':^10} |")
    print("=" * 50)
    _info(f"Hybrid Ensemble Macro F1: {macro_ens:.4f}")
    _info(f"Overall Accuracy: {acc_ens:.4f}")

    # Genisletilmis metrikler: Macro/Weighted + ROC-AUC + PR-AUC + MCC
    _print_ensemble_extended(yte_enc, yhat_te, P_soft_te, classes, n_cls)

    # ROC ve PR egrisi grafigi
    model_probas_dict["Ensemble"] = P_soft_te
    _plot_roc_pr(yte_enc, n_cls, model_probas_dict, str(cdir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--unsw',    action='store_true', default=True)
    mode_group.add_argument('--cicids',  action='store_true')
    mode_group.add_argument('--cicids14',action='store_true')
    parser.add_argument('--cache-dir', type=str, default='.')
    parser.add_argument('--exclude-weak-classes', action='store_true')
    args = parser.parse_args()

    dataset_mode    = 'cicids14' if args.cicids14 else ('cicids' if args.cicids else 'unsw')
    cache_mode_name = dataset_mode + "_excluded" if args.exclude_weak_classes else dataset_mode
    heuristic_ensemble(str(Path(args.cache_dir) / cache_mode_name),
                       args.exclude_weak_classes)
