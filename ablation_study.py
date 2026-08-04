"""
ablation_study.py  —  IDS Ablasyon Analizi
===========================================
Her model icin ayri tablo uretilir.
Tablo formati:
  Satirlar : sinif isimleri + genel satirlar (Accuracy, Macro Recall, Macro F1)
  Sutunlar : Raw | +Oznitelik | +Oznitelik+RUS | +Oznitelik+RUS+Tomek | +Oznitelik+RUS+Tomek+SMOTE

Kullanim:
  # Hazır stage verileriyle (prepare_ablation_data.py çıktısı):
  python ablation_study.py --ablation-data ablation_data --datasets cicids14 unsw

  # Eski pipeline cache'iyle (geriye dönük uyumluluk):
  python ablation_study.py --cache-dir . --output-dir ablation_results

  # Sadece belirli modeller:
  python ablation_study.py --ablation-data ablation_data --datasets cicids14 --models xgb lgbm
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
}

# Sutun baslikları — kümülatif ekleme sırası
COL_RAW       = "Raw"
COL_ENH       = "+Oznitelik"
COL_RUS       = "+Oznitelik+RUS"
COL_TOMEK     = "+Oznitelik+RUS+Tomek"
COL_SMOTE     = "+Oznitelik+RUS+Tomek+SMOTE"
COL_COST_ADJ  = "+Focal_ClassWeight"
COL_CV5       = "+5Fold_CV"

STAGE_COLS       = [COL_RAW, COL_ENH, COL_RUS, COL_TOMEK, COL_SMOTE]
ALL_COLS         = STAGE_COLS + [COL_COST_ADJ, COL_CV5]
FOCAL_LOSS_KEYS  = {"xgb", "lgbm", "lgbmV2"}   # Focal Loss uygulanacak modeller
CLASS_WEIGHT_KEYS= {"histgb", "tabnet", "mlp", "rf"} # Class Weight uygulanacak modeller

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


# ── Stage tabanlı veri yükleme (prepare_ablation_data.py çıktısı) ────────────

def _load_stage(stage_dir: Path):
    """
    Bir ablasyon aşamasının X_train, y_train, X_test, y_test verilerini yükler.
    Geri dönüş: (X_train, y_train, X_test, y_test) — hepsi np.ndarray
    Başarısız olursa None döner.
    """
    required = ["X_train.joblib", "X_test.joblib", "y_train.joblib", "y_test.joblib"]
    if not all((stage_dir / f).exists() for f in required):
        return None
    X_tr = np.asarray(joblib.load(stage_dir / "X_train.joblib"), dtype=np.float32)
    X_te = np.asarray(joblib.load(stage_dir / "X_test.joblib"),  dtype=np.float32)
    y_tr = np.asarray(joblib.load(stage_dir / "y_train.joblib"))
    y_te = np.asarray(joblib.load(stage_dir / "y_test.joblib"))
    # Boyut garantisi — stage A/B/C/D/E'nin test seti aynı boyutta olmalı
    assert X_tr.shape[1] == X_te.shape[1], (
        f"{stage_dir.name}: train={X_tr.shape[1]} != test={X_te.shape[1]} sütun"
    )
    return X_tr, y_tr, X_te, y_te


def load_ablation_stages(ds_dir: Path):
    """
    prepare_ablation_data.py çıktısından tüm 5 aşamayı yükler.
    ds_dir: ablation_data/cicids14  veya  ablation_data/unsw  gibi bir yol.

    Geri dönüş:
        stages  : dict{col_label -> (X_tr, y_tr, X_te, y_te)}
        le      : LabelEncoder
        classes : np.ndarray[str]

    Başarısız olan aşamalar None olarak işaretlenir.
    """
    stage_map = {
        COL_RAW:   "stage_A_normalized",
        COL_ENH:   "stage_B_feature_eng",
        COL_RUS:   "stage_C_rus",
        COL_TOMEK: "stage_D_tomek",
        COL_SMOTE: "stage_E_smote",
    }

    # LabelEncoder: herhangi bir stage'den oku
    le = None
    for sname in stage_map.values():
        le_path = ds_dir / sname / "label_encoder.joblib"
        if le_path.exists():
            le = joblib.load(le_path)
            break
    if le is None:
        return None, None, None

    stages = {}
    for col_label, sname in stage_map.items():
        sdir = ds_dir / sname
        result = _load_stage(sdir)
        if result is None:
            _info(f"  [Uyarı] Aşama bulunamadı: {sdir}")
        stages[col_label] = result

    classes = le.classes_.astype(str)
    return stages, le, classes


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
            m.set_params(num_class=n_classes)
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

def _quick_model(model_key, X_tr, y_enc, use_class_weight=False):
    """
    Her model icin pipeline'a yakin parametreli egitim.
    Ablasyonun amaci her stage'i AYNI gucte modelle karsilastirmaktir,
    bu nedenle parametreler pipeline (unsw_nb15_pipeline.py) ile uyumludur.
    OOF yerine single-fit kullanilir (hiz-kalite dengesi).
    """
    cw = "balanced" if use_class_weight else None
    n_samples = len(y_enc)

    # Buyuk veri setlerinde (>300k) estimator sayisini otomatik azalt
    def _n_est(full, mini=150):
        return full if n_samples <= 300_000 else mini

    if model_key == "xgb":
        try:
            from xgboost import XGBClassifier
            m = XGBClassifier(
                n_estimators=_n_est(400), max_depth=9, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=2, gamma=1,
                n_jobs=-1, random_state=42, verbosity=0,
                tree_method="hist",
            )
            m.fit(X_tr, y_enc)
            return m
        except Exception as e:
            _info(f"XGBoost hatasi: {e}")

    if model_key == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(
                n_estimators=_n_est(400), max_depth=6, num_leaves=63,
                learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20,
                n_jobs=-1, random_state=42, verbose=-1,
                class_weight=cw,
            )
            m.fit(X_tr, y_enc)
            return m
        except Exception as e:
            _info(f"LGBM hatasi: {e}")
            

    if model_key == "lgbmV2":
        try:
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(
                n_estimators=_n_est(400), max_depth=8, num_leaves=127,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20,
                n_jobs=-1, random_state=42, verbose=-1,
                class_weight=cw,
            )
            m.fit(X_tr, y_enc)
            return m
        except Exception as e:
            _info(f"LGBM_V2 hatasi: {e}")

    if model_key == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(
            max_iter=_n_est(300), max_depth=8, learning_rate=0.08,
            min_samples_leaf=20, l2_regularization=0.1,
            random_state=42,
            class_weight=cw,
        )
        m.fit(X_tr, y_enc)
        return m

    if model_key == "tabnet":
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128), max_iter=200,
            random_state=42, early_stopping=True,
            learning_rate_init=1e-3, batch_size=min(1024, n_samples),
            n_iter_no_change=15,
        )
        m.fit(X_tr, y_enc)
        return m

    if model_key == "mlp":
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=200,
            random_state=42, early_stopping=True,
            learning_rate_init=1e-3, batch_size=min(1024, n_samples),
            n_iter_no_change=15,
        )
        m.fit(X_tr, y_enc)
        return m

    if model_key == "rf":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(
            n_estimators=_n_est(300), max_depth=None,
            min_samples_leaf=2, n_jobs=-1,
            random_state=42, class_weight=cw,
        )
        m.fit(X_tr, y_enc)
        return m

    _info(f"Bilinmeyen model_key='{model_key}', sklearn MLP ile devam ediliyor")
    from sklearn.neural_network import MLPClassifier
    m = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200,
                      random_state=42, early_stopping=True)
    m.fit(X_tr, y_enc)
    return m


def _predict(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X).argmax(axis=1)
    return model.predict(X)


def _quick_model_cv5(model_key, X_train, y_train, X_test, n_folds=5, use_focal=False, use_class_weight=False):
    """
    5-Fold Stratified CV ile model egitip test seti uzerinde
    olasilik ortalamasi ile tahmin uret.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    n_classes = len(np.unique(y_train))
    n_test = X_test.shape[0]
    proba_sum = np.zeros((n_test, n_classes), dtype=np.float64)
    fold_count = 0

    MAX_FOLD_TRAIN = 300_000

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_tr = X_train[tr_idx]
        y_fold_tr = y_train[tr_idx]

        # Fold basina stratified subsample
        if len(y_fold_tr) > MAX_FOLD_TRAIN:
            rng = np.random.default_rng(42 + fold_idx)
            cls_u, cnts = np.unique(y_fold_tr, return_counts=True)
            ratio = MAX_FOLD_TRAIN / len(y_fold_tr)
            idx_list = []
            for ci, cnt in zip(cls_u, cnts):
                cls_idx = np.where(y_fold_tr == ci)[0]
                n_keep = max(1, int(np.ceil(cnt * ratio)))
                chosen = rng.choice(cls_idx, min(n_keep, len(cls_idx)), replace=False)
                idx_list.append(chosen)
            sub_idx = np.concatenate(idx_list)
            rng.shuffle(sub_idx)
            X_fold_tr = X_fold_tr[sub_idx]
            y_fold_tr = y_fold_tr[sub_idx]

        if use_focal and model_key in FOCAL_LOSS_KEYS:
            m, needs_sm = _quick_model_with_focal(model_key, X_fold_tr, y_fold_tr, n_classes)
        else:
            m = _quick_model(model_key, X_fold_tr, y_fold_tr, use_class_weight=use_class_weight)
            needs_sm = False

        if m is None:
            continue

        if hasattr(m, 'predict_proba'):
            fold_proba = m.predict_proba(X_test)
            if use_focal and model_key in FOCAL_LOSS_KEYS and needs_sm:
                from scipy.special import softmax
                fold_proba = softmax(fold_proba, axis=1)
            if fold_proba.shape[1] == n_classes:
                proba_sum += fold_proba
            else:
                # Model tum siniflari gormemis olabilir — kolon hizalama
                if hasattr(m, 'classes_'):
                    for j, c in enumerate(m.classes_):
                        c_int = int(c)
                        if 0 <= c_int < n_classes:
                            proba_sum[:, c_int] += fold_proba[:, j]
                else:
                    for j in range(min(fold_proba.shape[1], n_classes)):
                        proba_sum[:, j] += fold_proba[:, j]
        else:
            pred = m.predict(X_test)
            for i, p in enumerate(pred):
                if int(p) < n_classes:
                    proba_sum[i, int(p)] += 1.0

        fold_count += 1
        _info(f"    Fold {fold_idx+1}/{n_folds} tamamlandi")

    if fold_count == 0:
        return None

    proba_avg = proba_sum / fold_count
    return proba_avg.argmax(axis=1)


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

    # +Tomek — KALDIRILDI
    # Ablasyon analizi Tomek'in nadir sinif orneklerini silerek sistematik
    # zarar verdigini gostermistir. Stage D = Stage C kopyasi.
    X_tl, y_tl = X_prev, y_prev
    configs.append((COL_TOMEK, X_tl, y_tl))
    _info("Tomek atlandi (kalici olarak kaldirildi) — RUS verisi kopyalandi")

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

def build_model_table_from_stages(model_key, model_name, stages, classes, cdir=None):
    """
    Stage tabanlı ablasyon tablosu uret.
    stages: dict{col_label -> (X_tr, y_tr, X_te, y_te) veya None}
    Her sütun için o aşamanın X_train'i ile model eğitilir, X_test ile tahmin yapılır.
    Kolon uyumsuzluğu imkânsız çünkü her aşama kendi numpy array'iyle gelir.
    """
    # Test için referans y_te ve X_te: tüm aşamalarda test seti aynı (stage_B'den)
    # Önce geçerli bir aşamadan y_te'yi al
    y_te_enc = None
    for col in [COL_SMOTE, COL_TOMEK, COL_RUS, COL_ENH, COL_RAW]:
        if stages.get(col) is not None:
            y_te_enc = stages[col][3]  # y_te
            break
    if y_te_enc is None:
        _info(f"{model_name}: Hiçbir aşamada test verisi bulunamadı, tablo boş.")
        return None

    n_cls = len(classes)
    model_cols = ALL_COLS

    row_index = []
    for cls in classes:
        row_index += [f"{cls} — F1", f"{cls} — Recall", f"{cls} — Precision"]
    row_index += ["── Accuracy", "── Macro Recall", "── Macro F1", "── Macro Precision"]

    df = pd.DataFrame(index=row_index, columns=model_cols, dtype=float)

    def _fill_col(col_label, y_pred):
        per_cls, overall = metrics_per_class(y_te_enc, y_pred, classes)
        for cls in classes:
            df.loc[f"{cls} — F1",        col_label] = per_cls[cls]["F1"]
            df.loc[f"{cls} — Recall",    col_label] = per_cls[cls]["Recall"]
            df.loc[f"{cls} — Precision", col_label] = per_cls[cls]["Precision"]
        df.loc["── Accuracy",        col_label] = overall["Accuracy"]
        df.loc["── Macro Recall",    col_label] = overall["Macro Recall"]
        df.loc["── Macro F1",        col_label] = overall["Macro F1"]
        df.loc["── Macro Precision", col_label] = overall["Macro Precision"]

    def _train_and_eval(col_label):
        """
        Verilen aşama için model eğit ve tahmin et.
        Stage A ve B büyük olabilir (2M+); sınıf dengeli stratified subsample ile
        yönetilebilir boyuta indirilir (max 500k). Stage C/D/E zaten RUS ile küçük.
        """
        stage_data = stages.get(col_label)
        if stage_data is None:
            return None
        X_tr, y_tr, X_te, y_te = stage_data

        if X_tr.shape[1] != X_te.shape[1]:
            _info(f"  {model_name} {col_label}: kolon uyumsuzluğu "
                  f"train={X_tr.shape[1]} vs test={X_te.shape[1]} — atlandı")
            return None

        t0 = time.time()

        # RAW sütunu için mevcut proba dosyası varsa kullan
        if col_label == COL_RAW and cdir is not None:
            proba_path = Path(cdir) / f"proba_{model_key}_oof_test.npz"
            if proba_path.exists():
                P = load_proba(proba_path, classes)
                y_pred = P.argmax(axis=1)
                _, _, mf1, _ = precision_recall_fscore_support(
                    y_te_enc, y_pred, average="macro", zero_division=0)
                _info(f"  {model_name} {col_label:30s} → proba.npz ({mf1:.4f})")
                return y_pred

        # Büyük veri setlerinde stratified subsample (sınıf oranlarını koru)
        # Eşik: 500k — bu üzerindeyse her sınıftan en fazla proportional örnekle
        MAX_TRAIN = 500_000
        if len(y_tr) > MAX_TRAIN:
            rng = np.random.default_rng(42)
            classes_u, counts = np.unique(y_tr, return_counts=True)
            # Her sınıftan en fazla proportional pay al
            ratio = MAX_TRAIN / len(y_tr)
            idx_list = []
            for cls_i, cnt in zip(classes_u, counts):
                cls_idx = np.where(y_tr == cls_i)[0]
                n_keep = max(1, min(cnt, int(np.ceil(cnt * ratio))))
                chosen = rng.choice(cls_idx, n_keep, replace=False)
                idx_list.append(chosen)
            sub_idx = np.concatenate(idx_list)
            rng.shuffle(sub_idx)
            X_tr = X_tr[sub_idx]
            y_tr = y_tr[sub_idx]
            _info(f"  {model_name} {col_label}: subsample {len(sub_idx):,} / {len(stage_data[1]):,}")

        m = _quick_model(model_key, X_tr, y_tr)
        if m is None:
            _info(f"  {model_name} {col_label}: model oluşturulamadı")
            return None
        y_pred = _predict(m, X_te)
        _, _, mf1, _ = precision_recall_fscore_support(
            y_te_enc, y_pred, average="macro", zero_division=0)
        _info(f"  {model_name} {col_label:30s} → train={X_tr.shape}, "
              f"MacroF1={mf1:.4f} ({time.time()-t0:.1f}s)")
        return y_pred

    # ── Her sütun için eğit ve doldur ──────────────────────────────────────
    prev_col = None
    for col in STAGE_COLS:
        y_pred = _train_and_eval(col)
        if y_pred is not None:
            _fill_col(col, y_pred)
            prev_col = col
        elif prev_col is not None:
            df[col] = df[prev_col]
            _info(f"  {model_name} {col}: başarısız, '{prev_col}' kopyalandı")

    # ── 6. Focal / Class Weight Sütunu ────────────────────────────────────
    smote_data = stages.get(COL_SMOTE)
    if smote_data is not None:
        X_sm, y_sm, X_te_cost, _ = smote_data
        n_cls_count = len(classes)
        t0 = time.time()
        
        if model_key in FOCAL_LOSS_KEYS:
            m_cost, needs_sm = _quick_model_with_focal(model_key, X_sm, y_sm, n_cls_count)
            if m_cost is not None:
                yhat_cost = _predict_focal(m_cost, X_te_cost, needs_sm)
                _fill_col(COL_COST_ADJ, yhat_cost)
                _, _, mf1, _ = precision_recall_fscore_support(
                    y_te_enc, yhat_cost, average="macro", zero_division=0)
                _info(f"  {model_name} {COL_COST_ADJ:30s} → Focal Loss, MacroF1={mf1:.4f} ({time.time()-t0:.1f}s)")
            else:
                df[COL_COST_ADJ] = df[COL_SMOTE]
                _info(f"  {model_name} {COL_COST_ADJ}: Focal Loss basarisiz, kopyalandi")
        else:
            m_cost = _quick_model(model_key, X_sm, y_sm, use_class_weight=True)
            if m_cost is not None:
                yhat_cost = _predict(m_cost, X_te_cost)
                _fill_col(COL_COST_ADJ, yhat_cost)
                _, _, mf1, _ = precision_recall_fscore_support(
                    y_te_enc, yhat_cost, average="macro", zero_division=0)
                _info(f"  {model_name} {COL_COST_ADJ:30s} → Class Weight, MacroF1={mf1:.4f} ({time.time()-t0:.1f}s)")
            else:
                df[COL_COST_ADJ] = df[COL_SMOTE]
                _info(f"  {model_name} {COL_COST_ADJ}: Class Weight basarisiz, kopyalandi")
    elif prev_col is not None:
        df[COL_COST_ADJ] = df[prev_col]

    # ── 7. 5-Fold CV sütunu ───────────────────────────────────────────────
    if smote_data is not None:
        X_tr_cv, y_tr_cv, X_te_cv, _ = smote_data
        t0 = time.time()
        
        use_focal = model_key in FOCAL_LOSS_KEYS
        use_cw = model_key not in FOCAL_LOSS_KEYS
        
        y_pred_cv = _quick_model_cv5(model_key, X_tr_cv, y_tr_cv, X_te_cv, 
                                     use_focal=use_focal, use_class_weight=use_cw)
        if y_pred_cv is not None:
            _fill_col(COL_CV5, y_pred_cv)
            _, _, mf1_cv, _ = precision_recall_fscore_support(
                y_te_enc, y_pred_cv, average="macro", zero_division=0)
            _info(f"  {model_name} {COL_CV5:30s} → 5-Fold CV, MacroF1={mf1_cv:.4f} ({time.time()-t0:.1f}s)")
        else:
            df[COL_CV5] = df[COL_COST_ADJ]
            _info(f"  {model_name} {COL_CV5}: 5-Fold CV başarısız")
    elif prev_col is not None:
        df[COL_CV5] = df[COL_COST_ADJ]
        _info(f"  {model_name} {COL_CV5}: SMOTE verisi yok, '{COL_COST_ADJ}' kopyalandı")

    return df


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
    model_cols = ALL_COLS

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

    # ── Sutun 1: RAW — ham features ──────────────────────────────
    # Oncelik: mevcut proba_*.npz dosyasini kullan (pipeline'in tam egitimiyle uyumlu)
    # Yoksa: ham veri uzerinde hafif subsample + hizli model
    t0 = time.time()
    proba_path = cdir / f"proba_{model_key}_oof_test.npz"
    if proba_path.exists():
        P = load_proba(proba_path, classes)
        _fill_col(COL_RAW, P.argmax(axis=1))
        _, _, mf1_raw, _ = precision_recall_fscore_support(
            yte_enc, P.argmax(axis=1), average="macro", zero_division=0)
        _info(f"{model_name} RAW  → proba dosyasindan ({mf1_raw:.4f})")
    else:
        # Proba yok — subsampling ile hizli egitim (max 100k ornek)
        rng = np.random.default_rng(42)
        n_sub = min(len(y_tr_enc), 100_000)
        idx = rng.choice(len(y_tr_enc), n_sub, replace=False)
        X_sub = Xt_tr_raw[idx]
        y_sub = y_tr_enc[idx]
        m_raw = _quick_model(model_key, X_sub, y_sub)
        if m_raw is None:
            _info(f"{model_name} RAW → model uretilemedi, sutun bos kalacak")
        else:
            yhat_raw = _predict(m_raw, Xt_te_raw)
            _fill_col(COL_RAW, yhat_raw)
            _, _, mf1_raw, _ = precision_recall_fscore_support(
                yte_enc, yhat_raw, average="macro", zero_division=0)
            _info(f"{model_name} RAW  → subsample {n_sub}, MacroF1={mf1_raw:.4f} ({time.time()-t0:.1f}s)")

    # ── Sutun 2: +OZNITELIK — enhanced features, hizli model ────
    if Xt_tr_enh is not None and Xt_te_enh is not None:
        t0 = time.time()
        m_enh = _quick_model(model_key, Xt_tr_enh, y_tr_enc)
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

    X_cv_data, y_cv_data = X_base, y_tr_enc  # CV5 icin fallback

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
            if col_label == COL_SMOTE:
                X_cv_data, y_cv_data = X_tr_cfg, y_tr_cfg
            t0 = time.time()
            m = _quick_model(model_key, X_tr_cfg, y_tr_cfg)
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

    # ── 6. Focal / Class Weight Sütunu ────────────────────────────
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
    t0 = time.time()

    if model_key in FOCAL_LOSS_KEYS:
        m_fl, needs_softmax = _quick_model_with_focal(model_key, X_smote, y_smote, n_cls_count)
        if m_fl is not None:
            yhat_fl = _predict_focal(m_fl, X_te, needs_softmax)
            _fill_col(COL_COST_ADJ, yhat_fl)
            _, _, mf1, _ = precision_recall_fscore_support(
                yte_enc, yhat_fl, average="macro", zero_division=0)
            _info(f"{model_name} {COL_COST_ADJ:30s} → Focal Loss, MacroF1={mf1:.4f}  ({time.time()-t0:.1f}s)")
        else:
            df[COL_COST_ADJ] = df[COL_SMOTE]
            _info(f"{model_name} {COL_COST_ADJ}: Focal Loss basarisiz, kopyalandi")
    else:
        m_cw = _quick_model(model_key, X_smote, y_smote, use_class_weight=True)
        if m_cw is not None:
            yhat_cw = _predict(m_cw, X_te)
            _fill_col(COL_COST_ADJ, yhat_cw)
            _, _, mf1, _ = precision_recall_fscore_support(
                yte_enc, yhat_cw, average="macro", zero_division=0)
            _info(f"{model_name} {COL_COST_ADJ:30s} → Class Weight, MacroF1={mf1:.4f}  ({time.time()-t0:.1f}s)")
        else:
            df[COL_COST_ADJ] = df[COL_SMOTE]
            _info(f"{model_name} {COL_COST_ADJ}: Class Weight basarisiz, kopyalandi")

    # ── 7. 5-Fold CV sütunu ──────────────────────────────────────────
    t0 = time.time()
    use_focal = model_key in FOCAL_LOSS_KEYS
    use_cw = model_key not in FOCAL_LOSS_KEYS
    
    y_pred_cv5 = _quick_model_cv5(model_key, X_cv_data, y_cv_data, X_te,
                                  use_focal=use_focal, use_class_weight=use_cw)
    if y_pred_cv5 is not None:
        _fill_col(COL_CV5, y_pred_cv5)
        _, _, mf1_cv, _ = precision_recall_fscore_support(
            yte_enc, y_pred_cv5, average="macro", zero_division=0)
        _info(f"{model_name} {COL_CV5:30s} → 5-Fold CV, MacroF1={mf1_cv:.4f} ({time.time()-t0:.1f}s)")
    else:
        df[COL_CV5] = df[COL_COST_ADJ]
        _info(f"{model_name} {COL_CV5}: 5-Fold CV başarısız, '{COL_COST_ADJ}' kopyalandı")

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

# Bilinen veri seti cache klasorleri (eski pipeline cache modu)
DATASET_DIRS = {
    "unsw":           "UNSW-NB15",
    "unsw_excluded":  "UNSW-NB15 (zayif siniflar haric)",
    "cicids":         "CICIDS17",
    "cicids_excluded":"CICIDS17 (zayif siniflar haric)",
    "cicids14":       "CICIDS17 (14 sinif)",
}


def run_ablation_stage_based(ds_dir: Path, out_dir: Path, model_keys: list,
                              dataset_label: str):
    """
    prepare_ablation_data.py çıktısından stage tabanlı ablasyon çalıştırır.
    ds_dir: ablation_data/cicids14  veya  ablation_data/unsw
    Her aşama kendi numpy array'iyle gelir — kolon uyumsuzluğu imkânsız.
    """
    print(f"\n{'#'*65}")
    print(f"  VERİ SETİ (Stage Modu): {dataset_label}")
    print(f"  Stage Dir : {ds_dir}")
    print(f"  Çıktı     : {out_dir}")
    print(f"{'#'*65}\n")

    # Stage A varlığı kontrol et
    if not (ds_dir / "stage_A_normalized" / "X_train.joblib").exists():
        print(f"  [Atlandi] {ds_dir} icinde stage_A_normalized/X_train.joblib bulunamadi.")
        print(f"  Once 'python prepare_ablation_data.py' calistirin.")
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)

    stages, le, classes = load_ablation_stages(ds_dir)
    if stages is None:
        print(f"  [Hata] Aşamalar yüklenemedi: {ds_dir}")
        return {}

    _info(f"Sınıflar ({len(classes)}): {list(classes)}")

    all_dfs = {}

    for key in model_keys:
        name = BASE_MODEL_KEYS.get(key, key)
        _stage(f"[{dataset_label}] Model: {name}")

        df = build_model_table_from_stages(
            model_key=key,
            model_name=name,
            stages=stages,
            classes=classes,
            cdir=None,  # proba.npz dosyası yok stage modunda
        )
        if df is None:
            continue

        all_dfs[name] = df

        csv_path = out_dir / f"ablation_{name}.csv"
        df.to_csv(csv_path, float_format="%.4f")
        _info(f"CSV kaydedildi: {csv_path}")

        gen_rows = [r for r in df.index if r.startswith("──")]
        model_cols = list(df.columns)
        print(f"\n  {name} — Genel Metrikler:")
        print(f"  {'Metrik':<22} " + "  ".join(f"{c:>28}" for c in model_cols))
        print(f"  {'-'*(22 + len(model_cols)*30)}")
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
                    for col in list(df.columns):
                        row[col] = float(df.loc[metric, col]) if not pd.isna(df.loc[metric, col]) else None
                    summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = out_dir / "ablation_summary.csv"
        summary_df.to_csv(summary_path, index=False, float_format="%.4f")
        _info(f"Özet CSV kaydedildi: {summary_path}")

    return all_dfs


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
    ap.add_argument("--cache-dir",       type=str, default=None,
                    help="Eski pipeline cache kök klasörü (unsw/, cicids14/ alt klasörlerini içerir)")
    ap.add_argument("--ablation-data",   type=str, default=None,
                    help="prepare_ablation_data.py çıktısı kök klasörü (önerilen mod)")
    ap.add_argument("--output-dir",      type=str, default=None)
    ap.add_argument("--skip-resampling", action="store_true",
                    help="Sadece eski cache modunda geçerli")
    ap.add_argument("--models",          nargs="*", default=None,
                    help="Sadece belirli modeller: xgb lgbm lgbmV2 histgb mlp tabnet rf")
    ap.add_argument("--datasets", "--dataset", nargs="*", default=None,
                    dest="datasets",
                    help="Sadece belirli veri setleri: unsw cicids14 (stage modu) veya "
                         "unsw cicids unsw_excluded cicids_excluded (cache modu)")
    args = ap.parse_args()

    model_keys = args.models if args.models else list(BASE_MODEL_KEYS.keys())

    # ── Stage tabanlı mod (önerilen) ──────────────────────────────────────────
    if args.ablation_data:
        ablation_root = Path(args.ablation_data)
        out_root = Path(args.output_dir) if args.output_dir else ablation_root / ".." / "ablation_results"
        out_root = out_root.resolve()

        # Hangi veri setleri
        if args.datasets:
            datasets_to_run = args.datasets
        else:
            # ablation_data klasöründeki alt klasörleri otomatik bul
            datasets_to_run = [
                d.name for d in ablation_root.iterdir()
                if d.is_dir() and (d / "stage_A_normalized").exists()
            ]
            if not datasets_to_run:
                datasets_to_run = ["unsw", "cicids14"]

        print(f"\n{'#'*65}")
        print(f"  IDS ABLASYON ANALİZİ — STAGE MODU")
        print(f"  Kaynak : {ablation_root}")
        print(f"  Çıktı  : {out_root}")
        print(f"  Veri   : {datasets_to_run}")
        print(f"  Modeller: {model_keys}")
        print(f"{'#'*65}")

        for ds_key in datasets_to_run:
            ds_dir  = ablation_root / ds_key
            out_dir = out_root / ds_key
            label   = DATASET_DIRS.get(ds_key, ds_key)
            run_ablation_stage_based(ds_dir, out_dir, model_keys, label)

    # ── Eski pipeline cache modu (geriye dönük uyumluluk) ────────────────────
    else:
        cache_root = Path(args.cache_dir) if args.cache_dir else Path(".")
        out_root = Path(args.output_dir) if args.output_dir else cache_root / "ablation_results"

        datasets_to_run = args.datasets if args.datasets else list(DATASET_DIRS.keys())

        print(f"\n{'#'*65}")
        print(f"  IDS ABLASYON ANALİZİ — CACHE MODU (Eski)")
        print(f"  Kök    : {cache_root}")
        print(f"  Çıktı  : {out_root}")
        print(f"  Veri   : {datasets_to_run}")
        print(f"{'#'*65}")

        for ds_key in datasets_to_run:
            cdir    = cache_root / ds_key
            out_dir = out_root / ds_key
            label   = DATASET_DIRS.get(ds_key, ds_key)
            run_ablation_for_dataset(cdir, out_dir, model_keys, args.skip_resampling, label)

    print(f"\n{'#'*65}")
    print(f"  Tüm veri setleri tamamlandı.")
    print(f"{'#'*65}\n")


if __name__ == "__main__":
    main()
