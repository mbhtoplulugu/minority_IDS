"""
ablation_study.py  —  IDS Ablasyon Analizi
===========================================
Her model icin ayri tablo uretilir.
Tablo formati:
  Satirlar : sinif isimleri + genel satirlar (Accuracy, Macro Recall, Macro F1)
  Sutunlar : model_raw | model+oznitelik | model+oznitelik+RUS |
             model+oznitelik+RUS+Tomek | model+oznitelik+RUS+Tomek+SMOTE

Kullanim:
  python ablation_study.py --cache-dir .
  python ablation_study.py --cache-dir . --output-dir ablation_results --skip-resampling
"""

import argparse, warnings, pickle, time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_fscore_support, classification_report
)
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax

warnings.filterwarnings("ignore")

# ── sabitleri ────────────────────────────────────────────────────────────────

BASE_MODEL_KEYS = {
    "xgb":    "XGBoost",
    "lgbm":   "LGBM",
    "lgbmV2": "LGBM_V2",
    "histgb": "HistGB",
    "mlp":    "MLP",
    "tabnet": "TabNet",
    "rf":     "RandomForest",
}

# Sutun baslikları — kümülatif ekleme sırası
COL_RAW       = "Raw"
COL_ENH       = "+Oznitelik"
COL_RUS       = "+Oznitelik+RUS"
COL_TOMEK     = "+Oznitelik+RUS+Tomek"
COL_SMOTE     = "+Oznitelik+RUS+Tomek+SMOTE"
COL_STD_LOSS  = "+SMOTE+StdLoss"    # Sadece XGB ve LGBM: standart cross-entropy
COL_FOCAL     = "+SMOTE+FocalLoss"  # Sadece XGB ve LGBM: focal loss

ALL_COLS         = [COL_RAW, COL_ENH, COL_RUS, COL_TOMEK, COL_SMOTE]
FOCAL_LOSS_COLS  = [COL_STD_LOSS, COL_FOCAL]   # Sadece XGB/LGBM
FOCAL_LOSS_KEYS  = {"xgb", "lgbm", "lgbmV2"}   # Bu modeller için ek sütunlar üretilir

# ── yardimci ─────────────────────────────────────────────────────────────────

def _info(m):  print(f"  [Info]  {m}")
def _stage(m): print(f"\n{'='*60}\n[Stage] {m}\n{'='*60}")


def load_base_data(cdir):
    Xt_tr = joblib.load(cdir / "Xt_tr.joblib")
    Xt_v  = joblib.load(cdir / "Xt_v.joblib")
    Xt_te = joblib.load(cdir / "Xt_te.joblib")
    y_tr  = joblib.load(cdir / "y_tr.joblib")
    y_v   = joblib.load(cdir / "y_v.joblib")
    y_te  = joblib.load(cdir / "y_te.joblib")
    le    = joblib.load(cdir / "label_encoder.joblib")
    return Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te, le


def load_enhanced(cdir):
    """En buyuk enhanced feature setini yukle. Yoksa None."""
    candidates = sorted(cdir.glob("Xt_tr_enhanced*.joblib"))
    if not candidates:
        return None, None, None
    sfx = candidates[-1].stem.replace("Xt_tr_", "")
    vf  = cdir / f"Xt_v_{sfx}.joblib"
    tef = cdir / f"Xt_te_{sfx}.joblib"
    if not (vf.exists() and tef.exists()):
        return None, None, None
    return (joblib.load(candidates[-1]),
            joblib.load(vf),
            joblib.load(tef))


def load_proba(path, classes):
    data = np.load(path, allow_pickle=True)
    P = data["proba"];  src = data["classes"].astype(str)
    out = np.zeros((P.shape[0], len(classes)), dtype=np.float32)
    idx = {c: i for i, c in enumerate(src)}
    for j, c in enumerate(classes):
        if c in idx: out[:, j] = P[:, idx[c]]
    return out


def metrics_per_class(y_true, y_pred, classes):
    """
    Her sinif icin f1, recall, precision + genel satirlar
    dondurur: dict {sinif_adi: {F1, Recall, Precision}}
    ve genel satirlar: Accuracy, Macro F1, Macro Recall, Macro Precision
    """
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
        "Macro Recall":    mr,
        "Macro F1":        mf,
        "Macro Precision": mp,
    }
    return per_cls, overall


# ── focal loss fonksiyonlari ─────────────────────────────────────────────────

def _focal_loss_xgb(y_true, y_pred):
    """Multiclass Focal Loss for XGBoost (unsw_nb15_pipeline.py ile ayni)."""
    gamma = 2.0
    y = y_true if not hasattr(y_true, 'get_label') else y_true.get_label()
    y = y.astype(int)
    is_1d = (y_pred.ndim == 1)
    if is_1d:
        n_classes = len(y_pred) // max(len(y), 1)
        n_classes = max(n_classes, 2)
    else:
        n_classes = y_pred.shape[1]
    alpha = np.full(n_classes, 2.0, dtype=np.float32)
    preds = y_pred.reshape(-1, n_classes) if is_1d else y_pred
    p = softmax(preds, axis=1)
    y_oh = np.eye(n_classes)[y]
    grad = (p - y_oh)
    hess = p * (1.0 - p)
    for i in range(n_classes):
        wf = alpha[i] * np.power(1.0 - p[:, i], gamma)
        grad[:, i] *= wf
        hess[:, i] *= wf
    return (grad.flatten(), hess.flatten()) if is_1d else (grad, hess)


def _focal_loss_lgbm(y_true, y_pred):
    """Multiclass Focal Loss for LightGBM (lightgbm_model.py ile ayni)."""
    gamma = 2.0
    y = y_true.astype(int)
    is_1d = (y_pred.ndim == 1)
    if is_1d:
        n_classes = len(y_pred) // max(len(y), 1)
        n_classes = max(n_classes, 2)
        preds = y_pred.reshape(n_classes, -1).T
    else:
        n_classes = y_pred.shape[1]
        preds = y_pred
    alpha = np.full(n_classes, 2.0, dtype=np.float32)
    p = softmax(preds, axis=1)
    y_oh = np.eye(n_classes)[y]
    grad = (p - y_oh)
    hess = p * (1.0 - p)
    for i in range(n_classes):
        wf = alpha[i] * np.power(1.0 - p[:, i], gamma)
        grad[:, i] *= wf
        hess[:, i] *= wf
    return (grad.T.flatten(), hess.T.flatten()) if is_1d else (grad, hess)


def _quick_model_with_focal(model_key, X_tr, y_enc, n_classes):
    """XGB/LGBM icin focal loss ile egitim. predict_proba softmax gerektirir."""
    if model_key == "xgb":
        try:
            from xgboost import XGBClassifier
            m = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                              n_jobs=-1, random_state=42, verbosity=0,
                              tree_method="hist", num_class=n_classes,
                              objective=_focal_loss_xgb)
            m.fit(X_tr, y_enc)
            return m, True   # True = softmax gerekiyor
        except Exception as e:
            _info(f"XGBoost focal hatasi: {e}")
            return None, False

    if model_key in ("lgbm", "lgbmV2"):
        try:
            from lightgbm import LGBMClassifier
            num_leaves = 31 if model_key == "lgbm" else 63
            depth     = 6  if model_key == "lgbm" else 8
            m = LGBMClassifier(n_estimators=150, max_depth=depth,
                               num_leaves=num_leaves, learning_rate=0.1,
                               n_jobs=-1, random_state=42, verbose=-1,
                               objective=_focal_loss_lgbm)
            m.fit(X_tr, y_enc)
            return m, True
        except Exception as e:
            _info(f"LGBM focal hatasi: {e}")
            return None, False

    return None, False


def _predict_focal(model, X, needs_softmax):
    """Focal loss ile egitilmis modelden olasilik tahminleri al."""
    raw = model.predict_proba(X)
    if needs_softmax:
        raw = softmax(raw, axis=1)
    return raw.argmax(axis=1)


# ── hizli model egitimi (resampling ablasyonu icin) ──────────────────────────

def _quick_model(model_key, X_tr, y_enc, use_class_weight=True):
    """Her model icin hafif hiperparametreli egitim."""
    cw = "balanced" if use_class_weight else None

    if model_key == "xgb":
        try:
            from xgboost import XGBClassifier
            m = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                              n_jobs=-1, random_state=42, verbosity=0,
                              tree_method="hist")
            m.fit(X_tr, y_enc)
            return m
        except Exception as e:
            _info(f"XGBoost hatasi: {e}")

    if model_key == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(n_estimators=150, max_depth=6, num_leaves=31,
                               learning_rate=0.1, n_jobs=-1, random_state=42,
                               verbose=-1, class_weight=cw)
            m.fit(X_tr, y_enc)
            return m
        except Exception as e:
            _info(f"LGBM hatasi: {e}")

    if model_key == "lgbmV2":
        try:
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(n_estimators=200, max_depth=8, num_leaves=63,
                               learning_rate=0.05, n_jobs=-1, random_state=42,
                               verbose=-1, class_weight=cw,
                               subsample=0.8, colsample_bytree=0.8)
            m.fit(X_tr, y_enc)
            return m
        except Exception as e:
            _info(f"LGBM_V2 hatasi: {e}")

    if model_key == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=100, max_depth=6,
                                           random_state=42)
        m.fit(X_tr, y_enc)
        return m

    if model_key == "tabnet":
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=100,
                          random_state=42, early_stopping=True,
                          learning_rate_init=5e-4)
        m.fit(X_tr, y_enc)
        return m

    if model_key == "mlp":
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50,
                          random_state=42, early_stopping=True)
        m.fit(X_tr, y_enc)
        return m

    if model_key == "rf":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1,
                                   random_state=42, class_weight=cw)
        m.fit(X_tr, y_enc)
        return m

    _info(f"Bilinmeyen model_key='{model_key}', sklearn MLP ile devam ediliyor")
    from sklearn.neural_network import MLPClassifier
    m = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50,
                      random_state=42, early_stopping=True)
    m.fit(X_tr, y_enc)
    return m


def _predict(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X).argmax(axis=1)
    return model.predict(X)


# ── resampling konfigurasyonlari uret ────────────────────────────────────────

def build_resampling_configs(Xt_tr, y_tr_enc):
    """
    Geri donus: list of (col_label, X_train, y_train_enc)
    Sirasi: Raw, +ENH (dis), +RUS, +Tomek, +SMOTE
    ENH bu fonksiyonun disinda eklenir.
    """
    configs = [("__RAW__", Xt_tr, y_tr_enc)]   # raw placeholder

    X_prev, y_prev = Xt_tr, y_tr_enc

    # +RUS — majority siniflari 200k ile sinirla
    try:
        from imblearn.under_sampling import RandomUnderSampler
        uniq, cnts = np.unique(y_prev, return_counts=True)
        # Sadece 200k'yi aşan sınıfları kırp; küçük sınıflara dokunma
        strat = {c: min(cnt, 200_000) for c, cnt in zip(uniq, cnts)}
        if all(cnt <= 200_000 for cnt in cnts):
            # Hicbir sinif 200k'yi gecmiyor — yine de dengeli bir cap uygula
            max_cnt = int(max(cnts))
            cap = max(max_cnt // 3, int(min(cnts)))  # en buyugu 1/3'e indir
            strat = {c: min(cnt, cap) for c, cnt in zip(uniq, cnts)}
            _info(f"RUS: tum siniflar 200k altinda, dinamik cap={cap} uygulanıyor")
        rus = RandomUnderSampler(sampling_strategy=strat, random_state=42)
        X_rus, y_rus = rus.fit_resample(X_prev, y_prev)
        configs.append((COL_RUS, X_rus, y_rus))
        X_prev, y_prev = X_rus, y_rus
    except Exception as e:
        _info(f"RUS hatasi: {e}")
        X_rus, y_rus = X_prev, y_prev
        configs.append((COL_RUS, X_rus, y_rus))

    # +Tomek — sinir orneklerini temizle
    # Guvenlik: cok kucuk siniflar (< 20 ornek) olan veri setlerinde
    # Tomek sinir analizi yanlis ciftler olusturabilir — bu durumda atla.
    try:
        from imblearn.under_sampling import TomekLinks
        vc_check = pd.Series(X_prev if False else y_prev).value_counts()
        min_cls_count = int(vc_check.min())
        if min_cls_count < 20:
            _info(f"TomekLinks: min sinif cok kucuk ({min_cls_count}<20), Tomek atlaniyor — RUS verisi kopyalanıyor")
            configs.append((COL_TOMEK, X_prev, y_prev))
        else:
            tl = TomekLinks(sampling_strategy="not minority", n_jobs=-1)
            X_tl, y_tl = tl.fit_resample(X_prev, y_prev)
            configs.append((COL_TOMEK, X_tl, y_tl))
            X_prev, y_prev = X_tl, y_tl
    except Exception as e:
        _info(f"TomekLinks hatasi: {e}")
        X_tl, y_tl = X_prev, y_prev
        configs.append((COL_TOMEK, X_tl, y_tl))

    # +SMOTE — sadece kucuk siniflari makul seviyeye cek
    try:
        from imblearn.over_sampling import SMOTE
        vc = pd.Series(y_prev).value_counts()
        median_cnt = int(vc.median())

        # Sadece median'in altindaki siniflari oversample et, en fazla 2x'ine cek
        over_dict = {}
        for c, cnt in vc.items():
            if cnt < median_cnt:
                over_dict[c] = min(cnt * 2, median_cnt)

        if not over_dict:
            _info("SMOTE: tum siniflar dengeli, atlaniyor")
            configs.append((COL_SMOTE, X_prev, y_prev))
        else:
            # k_neighbors: en kucuk sinif boyutuna gore guvenli deger
            min_cnt = int(vc.min())
            k = max(1, min(5, min_cnt - 1))
            if k < 1:
                _info(f"SMOTE: min sinif cok kucuk ({min_cnt}), atlaniyor")
                configs.append((COL_SMOTE, X_prev, y_prev))
            else:
                smote = SMOTE(sampling_strategy=over_dict, k_neighbors=k,
                              random_state=42)
                X_sm, y_sm = smote.fit_resample(X_prev, y_prev)
                configs.append((COL_SMOTE, X_sm, y_sm))
    except Exception as e:
        _info(f"SMOTE hatasi: {e} — onceki adimin verisi kullanilacak")
        configs.append((COL_SMOTE, X_prev, y_prev))

    return configs


# ── ana tablo uretici ─────────────────────────────────────────────────────────

def build_model_table(model_key, model_name, cdir, classes, yte_enc,
                      Xt_tr_raw, y_tr_enc, Xt_te_raw,
                      Xt_tr_enh, Xt_te_enh,
                      skip_resampling=False):
    """
    Bir model icin tam ablasyon tablosu uret.
    Satirlar: sinif isimleri (F1 / Recall / Precision) + genel satirlar
    Sutunlar: Raw | +Oznitelik | +RUS | +Tomek | +SMOTE
    """
    n_cls = len(classes)

    # Hangi sütunlar bu model için geçerli
    model_cols = ALL_COLS + (FOCAL_LOSS_COLS if model_key in FOCAL_LOSS_KEYS else [])

    # Satir indeksi: her sinif icin 3 satir + 4 genel
    row_index = []
    for cls in classes:
        row_index += [f"{cls} — F1", f"{cls} — Recall", f"{cls} — Precision"]
    row_index += ["── Accuracy", "── Macro Recall", "── Macro F1", "── Macro Precision"]

    df = pd.DataFrame(index=row_index, columns=model_cols, dtype=float)

    def _fill_col(col_label, y_pred):
        per_cls, overall = metrics_per_class(yte_enc, y_pred, classes)
        for cls in classes:
            df.loc[f"{cls} — F1",        col_label] = per_cls[cls]["F1"]
            df.loc[f"{cls} — Recall",    col_label] = per_cls[cls]["Recall"]
            df.loc[f"{cls} — Precision", col_label] = per_cls[cls]["Precision"]
        df.loc["── Accuracy",        col_label] = overall["Accuracy"]
        df.loc["── Macro Recall",    col_label] = overall["Macro Recall"]
        df.loc["── Macro F1",        col_label] = overall["Macro F1"]
        df.loc["── Macro Precision", col_label] = overall["Macro Precision"]

    # ── Sutun 1: RAW — ham features, hizli model ile ────────────
    t0 = time.time()
    m_raw = _quick_model(model_key, Xt_tr_raw, y_tr_enc, use_class_weight=True)
    if m_raw is None:
        _info(f"{model_name} RAW → model uretilemedi, sutun bos kalacak")
    else:
        yhat_raw = _predict(m_raw, Xt_te_raw)
        _fill_col(COL_RAW, yhat_raw)
        _, _, mf1_raw, _ = precision_recall_fscore_support(
            yte_enc, yhat_raw, average="macro", zero_division=0)
        _info(f"{model_name} RAW  → {Xt_tr_raw.shape[1]} ozellik, "
              f"MacroF1={mf1_raw:.4f} ({time.time()-t0:.1f}s)")

    # ── Sutun 2: +OZNITELIK — enhanced features, hizli model ────
    if Xt_tr_enh is not None and Xt_te_enh is not None:
        t0 = time.time()
        m_enh = _quick_model(model_key, Xt_tr_enh, y_tr_enc, use_class_weight=True)
        if m_enh is None:
            df[COL_ENH] = df[COL_RAW]
            _info(f"{model_name} +ENH → model uretilemedi, RAW kopyalandi")
        else:
            yhat_enh = _predict(m_enh, Xt_te_enh)
            _fill_col(COL_ENH, yhat_enh)
            _, _, mf1_enh, _ = precision_recall_fscore_support(
                yte_enc, yhat_enh, average="macro", zero_division=0)
            _info(f"{model_name} +ENH → {Xt_tr_enh.shape[1]} ozellik, "
                  f"MacroF1={mf1_enh:.4f} ({time.time()-t0:.1f}s)")
    else:
        df[COL_ENH] = df[COL_RAW]
        _info(f"{model_name} +ENH → enhanced yok, RAW kopyalandi")

    # ── Sutunlar 3-5: Resampling (enhanced X ile) ────────────────
    X_base = Xt_tr_enh if Xt_tr_enh is not None else Xt_tr_raw
    X_te   = Xt_te_enh if Xt_te_enh is not None else Xt_te_raw

    if skip_resampling:
        for col in [COL_RUS, COL_TOMEK, COL_SMOTE]:
            df[col] = df[COL_ENH]
        _info(f"{model_name}: resampling atlandi (--skip-resampling)")
    else:
        configs = build_resampling_configs(X_base, y_tr_enc)
        filled_cols = set()
        for col_label, X_tr_cfg, y_tr_cfg in configs:
            if col_label == "__RAW__":
                continue
            if col_label not in {COL_RUS, COL_TOMEK, COL_SMOTE}:
                continue
            t0 = time.time()
            # Tomek zaten sinir orneklerini temizledi — class_weight gereksiz ve celiskiyor
            use_cw = (col_label != COL_TOMEK)
            m = _quick_model(model_key, X_tr_cfg, y_tr_cfg, use_class_weight=use_cw)
            if m is None:
                _info(f"{model_name} {col_label} → model uretilemedi, onceki sutun kopyalanacak")
                continue
            yhat = _predict(m, X_te)
            _fill_col(col_label, yhat)
            filled_cols.add(col_label)
            _, _, mf1, _ = precision_recall_fscore_support(
                yte_enc, yhat, average="macro", zero_division=0)
            _info(f"{model_name} {col_label:35s} → MacroF1={mf1:.4f}  ({time.time()-t0:.1f}s)")

        # Doldurulamayan sutunlari onceki adimla doldur
        prev = COL_ENH
        for col in [COL_RUS, COL_TOMEK, COL_SMOTE]:
            if col not in filled_cols or df[col].isna().all():
                df[col] = df[prev]
                _info(f"{model_name} {col} → bos, '{prev}' ile dolduruldu")
            prev = col

    # ── Focal Loss sütunlari (sadece XGB ve LGBM) ────────────────
    if model_key in FOCAL_LOSS_KEYS:
        # SMOTE sonrasi veriyi yeniden uret (en son resampling adimi)
        X_base = Xt_tr_enh if Xt_tr_enh is not None else Xt_tr_raw
        X_te   = Xt_te_enh if Xt_te_enh is not None else Xt_te_raw

        if not skip_resampling:
            configs_fl = build_resampling_configs(X_base, y_tr_enc)
            # Son config SMOTE sonrasi veri
            X_smote, y_smote = X_base, y_tr_enc
            for lbl, Xc, yc in configs_fl:
                if lbl == COL_SMOTE:
                    X_smote, y_smote = Xc, yc
        else:
            X_smote, y_smote = X_base, y_tr_enc

        n_cls_count = len(classes)

        # +SMOTE + Standart Loss
        t0 = time.time()
        m_std = _quick_model(model_key, X_smote, y_smote)
        if m_std is not None:
            yhat_std = _predict(m_std, X_te)
            _fill_col(COL_STD_LOSS, yhat_std)
            _, _, mf1, _ = precision_recall_fscore_support(
                yte_enc, yhat_std, average="macro", zero_division=0)
            _info(f"{model_name} {COL_STD_LOSS:30s} → MacroF1={mf1:.4f}  ({time.time()-t0:.1f}s)")

        # +SMOTE + Focal Loss
        t0 = time.time()
        m_fl, needs_softmax = _quick_model_with_focal(model_key, X_smote, y_smote, n_cls_count)
        if m_fl is not None:
            yhat_fl = _predict_focal(m_fl, X_te, needs_softmax)
            _fill_col(COL_FOCAL, yhat_fl)
            _, _, mf1, _ = precision_recall_fscore_support(
                yte_enc, yhat_fl, average="macro", zero_division=0)
            _info(f"{model_name} {COL_FOCAL:30s} → MacroF1={mf1:.4f}  ({time.time()-t0:.1f}s)")

    return df


# ── gorsellestime ─────────────────────────────────────────────────────────────

def plot_model_table(df, model_name, out_dir):
    """Bir modelin ablasyon tablosunu PNG olarak kaydet."""
    show_rows = [r for r in df.index if "— F1" in r or r.startswith("──")]
    df_show = df.loc[show_rows].astype(float)
    cols = list(df_show.columns)  # modele gore degisen sutun listesi

    fig, ax = plt.subplots(figsize=(len(cols) * 2.2 + 1, len(show_rows) * 0.42 + 1.5))
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
                r = int(255 - v * 180)
                row_colors.append(f"#{r:02X}{g:02X}60")
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

    ax.set_title(f"Ablasyon Analizi — {model_name}", fontsize=11, pad=12, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / f"ablation_{model_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    _info(f"Gorsek kaydedildi: {out_path}")


def plot_macro_f1_comparison(all_dfs, out_dir):
    """Tum modellerin Macro F1 sutunlarini tek grafige topla."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
    x = np.arange(len(ALL_COLS))
    width = 0.12
    offsets = np.linspace(-(len(all_dfs)-1)/2, (len(all_dfs)-1)/2, len(all_dfs)) * width

    for i, (mname, df) in enumerate(all_dfs.items()):
        try:
            vals = [float(df.loc["── Macro F1", col]) if col in df.columns else 0
                    for col in ALL_COLS]
        except Exception:
            continue
        ax.bar(x + offsets[i], vals, width * 0.9, label=mname,
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_COLS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title("Tum Modeller — Ablasyon Sutunlari Macro F1 Karsilastirmasi")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = out_dir / "ablation_all_models_macro_f1.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    _info(f"Tum model karsilastirma grafigi: {out_path}")


def plot_accuracy_comparison(all_dfs, out_dir):
    """Tum modellerin Accuracy sutunlarini tek grafige topla."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
    x = np.arange(len(ALL_COLS))
    width = 0.12
    offsets = np.linspace(-(len(all_dfs)-1)/2, (len(all_dfs)-1)/2, len(all_dfs)) * width

    for i, (mname, df) in enumerate(all_dfs.items()):
        try:
            vals = [float(df.loc["── Accuracy", col]) if col in df.columns else 0
                    for col in ALL_COLS]
        except Exception:
            continue
        ax.bar(x + offsets[i], vals, width * 0.9, label=mname,
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_COLS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Tum Modeller — Ablasyon Sutunlari Accuracy Karsilastirmasi")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = out_dir / "ablation_all_models_accuracy.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    _info(f"Tum model accuracy karsilastirma grafigi: {out_path}")


def plot_per_class_heatmap(df, model_name, out_dir, metric="F1"):
    """Bir model icin sinif x sutun isi haritasi (secili metrik)."""
    cls_rows = [r for r in df.index if f"— {metric}" in r]
    if not cls_rows:
        return
    cols = list(df.columns)  # modele gore degisen sutun listesi
    df_sub = df.loc[cls_rows, cols].astype(float)
    ylabels = [r.replace(f" — {metric}", "") for r in cls_rows]

    fig, ax = plt.subplots(figsize=(len(cols) * 1.6, max(3, len(cls_rows) * 0.55)))
    im = ax.imshow(df_sub.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04)
    for i in range(len(ylabels)):
        for j in range(len(cols)):
            v = df_sub.values[i, j]
            ax.text(j, i, f"{v:.3f}" if not np.isnan(v) else "-",
                    ha="center", va="center", fontsize=7,
                    color="black" if v > 0.3 else "white")
    ax.set_title(f"{model_name} — Sinif Bazli {metric} Isi Haritasi", fontsize=10)
    plt.tight_layout()
    out_path = out_dir / f"ablation_{model_name}_heatmap_{metric}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    _info(f"Isi haritasi kaydedildi: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

# Bilinen veri seti cache klasorleri
DATASET_DIRS = {
    "unsw":           "UNSW-NB15",
    "unsw_excluded":  "UNSW-NB15 (zayif siniflar haric)",
    "cicids":         "CICIDS17",
    "cicids_excluded":"CICIDS17 (zayif siniflar haric)",
}


def run_ablation_for_dataset(cdir, out_dir, model_keys, skip_resampling, dataset_label):
    """Tek bir veri seti icin ablasyon calistir, sonuclari dondur."""
    print(f"\n{'#'*65}")
    print(f"  VERİ SETİ: {dataset_label}")
    print(f"  Cache  : {cdir}")
    print(f"  Cikti  : {out_dir}")
    print(f"{'#'*65}\n")

    if not (cdir / "Xt_tr.joblib").exists():
        print(f"  [Atlandi] {cdir} icinde Xt_tr.joblib bulunamadi.")
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)

    Xt_tr_raw, y_tr, Xt_v, y_v, Xt_te_raw, y_te, le = load_base_data(cdir)
    classes  = le.classes_.astype(str)
    yte_enc  = le.transform(np.asarray(y_te))
    y_tr_enc = le.transform(np.asarray(y_tr))

    Xt_tr_enh, _, Xt_te_enh = load_enhanced(cdir)
    if Xt_tr_enh is not None:
        _info(f"Enhanced features: {Xt_tr_enh.shape[1]} ozellik")
    else:
        _info("Enhanced features bulunamadi — ENH sutunu RAW ile ayni olacak")

    all_dfs = {}

    for key in model_keys:
        name = BASE_MODEL_KEYS.get(key, key)
        _stage(f"[{dataset_label}] Model: {name}")

        df = build_model_table(
            model_key       = key,
            model_name      = name,
            cdir            = cdir,
            classes         = classes,
            yte_enc         = yte_enc,
            Xt_tr_raw       = Xt_tr_raw,
            y_tr_enc        = y_tr_enc,
            Xt_te_raw       = Xt_te_raw,
            Xt_tr_enh       = Xt_tr_enh,
            Xt_te_enh       = Xt_te_enh,
            skip_resampling = skip_resampling,
        )

        all_dfs[name] = df

        csv_path = out_dir / f"ablation_{name}.csv"
        df.to_csv(csv_path, float_format="%.4f")
        _info(f"CSV kaydedildi: {csv_path}")

        # Konsola ozet
        gen_rows = [r for r in df.index if r.startswith("──")]
        model_cols = list(df.columns)
        print(f"\n  {name} — Genel Metrikler:")
        print(f"  {'Metrik':<22} " + "  ".join(f"{c:>28}" for c in model_cols))
        print(f"  {'-'*( 22 + len(model_cols)*30 )}")
        for row in gen_rows:
            vals = [f"{float(df.loc[row, c]):>28.4f}" if not pd.isna(df.loc[row, c]) else f"{'—':>28}"
                    for c in model_cols]
            print(f"  {row:<22} {'  '.join(vals)}")

        plot_model_table(df, name, out_dir)
        plot_per_class_heatmap(df, name, out_dir, metric="F1")
        plot_per_class_heatmap(df, name, out_dir, metric="Recall")

    if len(all_dfs) > 1:
        plot_macro_f1_comparison(all_dfs, out_dir)
        plot_accuracy_comparison(all_dfs, out_dir)

        summary_rows = []
        for mname, df in all_dfs.items():
            for metric in ["── Macro F1", "── Accuracy", "── Macro Recall", "── Macro Precision"]:
                if metric in df.index:
                    row = {"Model": mname, "Metrik": metric.replace("── ", "")}
                    for col in list(df.columns):   # modele gore degisken sutunlar
                        row[col] = float(df.loc[metric, col]) if not pd.isna(df.loc[metric, col]) else None
                    summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = out_dir / "ablation_summary.csv"
        summary_df.to_csv(summary_path, index=False, float_format="%.4f")
        _info(f"Ozet CSV kaydedildi: {summary_path}")

        _stage(f"OZET — Macro F1 ({dataset_label})")
        piv = summary_df[summary_df["Metrik"] == "Macro F1"].set_index("Model")[ALL_COLS]
        print(piv.to_string(float_format=lambda x: f"{x:.4f}"))

    return all_dfs


def main():
    ap = argparse.ArgumentParser(description="IDS Ablation Study — Sinif Bazli Tablo (Cok Veri Seti)")
    ap.add_argument("--cache-dir",       type=str, default=".",
                    help="Kok klasor (unsw/, cicids/ alt klasorlerini icerir)")
    ap.add_argument("--output-dir",      type=str, default=None)
    ap.add_argument("--skip-resampling", action="store_true")
    ap.add_argument("--models",          nargs="*", default=None,
                    help="Sadece belirli modeller: xgb lgbm lgbmV2 histgb mlp tabnet rf")
    ap.add_argument("--datasets",        nargs="*", default=None,
                    help="Sadece belirli veri setleri: unsw cicids unsw_excluded cicids_excluded")
    args = ap.parse_args()

    root    = Path(args.cache_dir)
    out_root = Path(args.output_dir) if args.output_dir else root / "ablation_results"
    model_keys = args.models if args.models else list(BASE_MODEL_KEYS.keys())

    # Hangi veri setleri kullanilacak
    datasets_to_run = args.datasets if args.datasets else list(DATASET_DIRS.keys())

    print(f"\n{'#'*65}")
    print(f"  IDS ABLASYON ANALIZI — COK VERİ SETİ")
    print(f"  Kok    : {root}")
    print(f"  Cikti  : {out_root}")
    print(f"  Veri   : {datasets_to_run}")
    print(f"{'#'*65}")

    for ds_key in datasets_to_run:
        cdir    = root / ds_key
        out_dir = out_root / ds_key
        label   = DATASET_DIRS.get(ds_key, ds_key)
        run_ablation_for_dataset(cdir, out_dir, model_keys, args.skip_resampling, label)

    print(f"\n{'#'*65}")
    print(f"  Tum veri setleri tamamlandi. Ciktilar: {out_root}")
    print(f"{'#'*65}\n")


if __name__ == "__main__":
    main()
