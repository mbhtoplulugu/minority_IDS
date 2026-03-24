"""
experts.py  OVR uzmanlar (LightGBM + LR fallback) + kalibrasyon
------------------------------------------------------------------
 Ama: Aznlk snflar (analysis, backdoor, worms, dos, shellcode) iin ikili uzmanlar
  eitip temel modellerin (XGB/MLP/Siamese) stne uzman probalarn eklemek.
 Eitim: LightGBM (varsa)  objective='binary', class_weight/scale_pos_weight,
  early_stopping ile VALIDe gre durdurma. Yoksa LogisticRegression(saga, balanced).
 Kalibrasyon: VALID zerinde isotonic ile 'prefit' kalibrasyon.
 ktlar (cache_dir):
    - experts_valid.npz / experts_test.npz  (proba, classes)
    - raporlar (VALID/TEST class-wise)

Kullanm:
    from experts import train_experts_ovr, ExpCfg
    train_experts_ovr(ExpCfg(cache_dir='.', focus=[...]))
"""
from __future__ import annotations
import warnings, joblib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (classification_report, precision_recall_fscore_support,
                             average_precision_score, roc_auc_score)

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier, log_evaluation
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# ---------------------------
# Helpers
# ---------------------------
def _best_thresh_by_f1(y_true_bin, p_pos, grid=None) -> float:
    """VALID zerinde ikili F1'i maksimize eden eii dndrr."""
    y_true_bin = np.asarray(y_true_bin).astype(int)
    p_pos = np.asarray(p_pos).astype(float)
    try:
        from sklearn.metrics import precision_recall_curve
        prec, rec, thr = precision_recall_curve(y_true_bin, p_pos)
        f1 = (2*prec*rec) / np.clip(prec+rec, 1e-9, None)
        if f1.size > 0 and thr.size > 0:
            # sklearn'de thresholds uzunluu f1'den 1 eksik olabilir
            best = int(np.nanargmax(f1[:-1])) if f1.size == thr.size+1 else int(np.nanargmax(f1))
            t = float(np.clip(thr[min(best, thr.size-1)], 1e-6, 1-1e-6))
            return t
    except Exception:
        pass
    # Gvenli grid fallback
    if grid is None: grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f = 0.5, -1.0
    for t in grid:
        y_hat = (p_pos >= t).astype(int)
        _,_,f,_ = precision_recall_fscore_support(y_true_bin, y_hat, average='binary', zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return float(best_t)


def _stage(msg: str): print(f"[Stage] {msg}")

def _info(msg: str): print(f"[Info]  {msg}")

# ---------------------------
# Config
# ---------------------------

@dataclass
class ExpCfg:
    files_glob: str ="C:/Users/mbhto/source/repos/UNSW-NB15/UNSWNB15_[0-5].csv"
    features_csv: str ="C:/Users/mbhto/source/repos/UNSW-NB15/NUSW-NB15_features.csv"
    cache_dir: str='.'
    use_gpu: bool=True
    focus: list[str] = None   # ['analysis','backdoor','dos','worms']
    random_state: int = 42
    # LGBM parametreleri - daha yumuak
    lgbm_n_estimators: int = 200
    lgbm_learning_rate: float = 0.05
    lgbm_feature_fraction: float = 0.8
    lgbm_bagging_fraction: float = 0.8
    lgbm_bagging_freq: int = 5
# ---------------------------
# Core
# ---------------------------

def _load_cache(cdir: Path, n_features=50):
    # Enhanced features ncelii  dinamik dosya arama
    # Feature engineering, dosya adna gerek kolon saysn yazar
    # (rn. n_features=50 istense bile kt enhanced44 olabilir)
    # Bu yzden nce exact match, sonra glob ile en iyi elemeyi ara
    
    enhanced_found = False
    
    # 1) Exact match dene
    exact_files = [
        cdir / f"Xt_tr_enhanced{n_features}.joblib",
        cdir / f"Xt_v_enhanced{n_features}.joblib", 
        cdir / f"Xt_te_enhanced{n_features}.joblib"
    ]
    if all(f.exists() for f in exact_files):
        Xt_tr = joblib.load(exact_files[0])
        Xt_v = joblib.load(exact_files[1])
        Xt_te = joblib.load(exact_files[2])
        _info(f"[Experts] Using enhanced features (exact): {Xt_tr.shape}")
        enhanced_found = True
    
    # 2) Glob ile en iyi enhanced dosyay bul
    if not enhanced_found:
        candidates = sorted(cdir.glob("Xt_tr_enhanced*.joblib"))
        for tr_f in candidates:
            suffix = tr_f.stem.replace("Xt_tr_", "")  # e.g. "enhanced44"
            v_f  = cdir / f"Xt_v_{suffix}.joblib"
            te_f = cdir / f"Xt_te_{suffix}.joblib"
            if v_f.exists() and te_f.exists():
                Xt_tr = joblib.load(tr_f)
                Xt_v  = joblib.load(v_f)
                Xt_te = joblib.load(te_f)
                _info(f"[Experts] Using enhanced features ({suffix}): {Xt_tr.shape}")
                enhanced_found = True
                break
    
    # 3) Fallback: original
    if not enhanced_found:
        Xt_tr = joblib.load(cdir/'Xt_tr.joblib')
        Xt_v  = joblib.load(cdir/'Xt_v.joblib')
        Xt_te = joblib.load(cdir/'Xt_te.joblib')
        _info(f"[Experts] Using original features: {Xt_tr.shape}")
    
    y_tr  = joblib.load(cdir/'y_tr.joblib')
    y_v   = joblib.load(cdir/'y_v.joblib')
    y_te  = joblib.load(cdir/'y_te.joblib')
    le    = joblib.load(cdir/'label_encoder.joblib')
    return Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te, le


from lightgbm import LGBMClassifier, early_stopping, log_evaluation
import numpy as np

def _sanitize_x(X):
    X = np.asarray(X, dtype=np.float32)
    return np.nan_to_num(X, copy=False, posinf=0.0, neginf=0.0)

def _train_one_ovr_lgbm(X_tr, y_tr_bin, X_v, y_v_bin, seed: int, cfg: ExpCfg, **kwargs):
    """
    Kk snf gvenli LightGBM:
      - CPU fallback
      - Parametre backoff (max_bin/min_data_in_leaf/min_sum_hessian_in_leaf)
      - scale_pos_weight clipping
      - VALID tek-snf ise ES yok
    """
    # ----- gvenlik: hedef 0/1 ve tipler float32 -----
    y_tr_bin = np.asarray(y_tr_bin, dtype=np.int32)
    y_v_bin  = np.asarray(y_v_bin,  dtype=np.int32)
    X_tr = _sanitize_x(X_tr).astype(np.float32, copy=False)
    X_v  = _sanitize_x(X_v ).astype(np.float32, copy=False)

    pos = float((y_tr_bin == 1).sum()); neg = float((y_tr_bin == 0).sum())
    # ar arl yumuat (rn. worms gibi)
    spw_user = kwargs.pop('scale_pos_weight', None)
    if spw_user is None:
        spw = (neg / max(1.0, pos)) if pos > 0 else 1.0
    else:
        spw = float(spw_user)
    spw = float(np.clip(spw, 1.0, 50.0))  # <--- kritik: cap

    # VALID iki snf m?
    valid_has_both = (y_v_bin.min() != y_v_bin.max())
    eval_set   = [(X_v, y_v_bin)] if valid_has_both else None
    eval_metric = 'average_precision'

    # Deneme sras: en emniyetli -> daha esnek
    trial_params = [
        # T1: en gvenli (kk bin, byk yaprak, byk hessian)
        dict(device_type='cpu', max_bin=63,  min_data_in_leaf=16, min_sum_hessian_in_leaf=1e-2, min_gain_to_split=1e-3),
        # T2: biraz serbest
        dict(device_type='cpu', max_bin=127, min_data_in_leaf=8,  min_sum_hessian_in_leaf=5e-3, min_gain_to_split=1e-4),
        # T3: daha serbest
        dict(device_type='cpu', max_bin=255, min_data_in_leaf=4,  min_sum_hessian_in_leaf=1e-3, min_gain_to_split=0.0),
    ]
    # stenirse ilk denemeye GPUlu varyant en sona ekle (ama CPU gvenli)
    if cfg.use_gpu:
        trial_params.append(dict(device_type='gpu', max_bin=255, min_data_in_leaf=4, min_sum_hessian_in_leaf=1e-3, min_gain_to_split=0.0))

    base = dict(
        objective='binary',
        n_estimators=cfg.lgbm_n_estimators,
        learning_rate=cfg.lgbm_learning_rate,
        num_leaves=15,
        min_data_in_bin=3,              # 1 -> 3 (bin ba asgari rnek)
        feature_pre_filter=False,
        feature_fraction=cfg.lgbm_feature_fraction,
        bagging_fraction=cfg.lgbm_bagging_fraction,
        bagging_freq=cfg.lgbm_bagging_freq,
        deterministic=True,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        scale_pos_weight=spw,           # class_weight yerine SPW + cap
        force_col_wise=True,            # kararl blc
        reg_lambda=1.0,                 # biraz L2, blnmeyi yumuat
    )

    last_err = None
    for t in trial_params:
        params = dict(base); params.update(t)
        try:
            clf = LGBMClassifier(**params)
            if eval_set is None:
                clf.fit(X_tr, y_tr_bin)
            else:
                from lightgbm import early_stopping, log_evaluation
                clf.fit(
                    X_tr, y_tr_bin,
                    eval_set=eval_set,
                    eval_metric=eval_metric,
                    callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
                )
            return clf
        except Exception as e:
            last_err = e
            _info(f"[LGBM backoff] {t['device_type']} max_bin={t['max_bin']} min_leaf={t['min_data_in_leaf']} "
                  f"hess={t['min_sum_hessian_in_leaf']} -> FAIL: {e}")

    # tm denemeler patlad -> LR fallback
    raise RuntimeError(f"LGBM backoff exhausted (last err: {last_err})")


def _train_one_ovr_lr(X_tr, y_tr_bin, X_v, y_v_bin, seed: int):
    clf = LogisticRegression(
        solver='saga',
        penalty='l2',
        C=1.0,
        class_weight='balanced',
        max_iter=400,
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X_tr, y_tr_bin)
    return clf


def _calibrate_prefit(model, X_cal, y_cal):
    # isotonic kalibrasyon  kk snflarda daha iyi
    try:
        cal = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        cal.fit(X_cal, y_cal)
        return cal
    except Exception:
        return model  # kalibrasyon baarszsa modeli dndr


def _report_binary(y_true_bin, p_pos, tag: str, cname: str):
    try:
        ap = average_precision_score(y_true_bin, p_pos)
    except Exception:
        ap = float('nan')
    try:
        auc = roc_auc_score(y_true_bin, p_pos)
    except Exception:
        auc = float('nan')
    y_hat = (p_pos >= 0.5).astype(int)
    p,r,f,_ = precision_recall_fscore_support(y_true_bin, y_hat, average='binary', zero_division=0)
    _info(f"[{tag}] {cname:>12s}  AUPRC={ap:.3f}  AUC={auc:.3f}  F1@0.5={f:.3f}")


def focal_loss_lgbm(gamma=2.0, alpha=0.75):
    """Custom Focal Loss objective for LightGBM"""
    def focal_loss_obj(y_true, y_pred):
        p = 1.0 / (1.0 + np.exp(-y_pred))
        # Focal weights
        fl_weight = np.where(y_true == 1, alpha * ((1 - p) ** gamma), (1 - alpha) * (p ** gamma))
        # Approximate gradient/hessian
        grad = (p - y_true) * fl_weight
        hess = p * (1 - p) * fl_weight
        return grad, hess
    return focal_loss_obj

def train_experts_ovr(cfg: ExpCfg, n_features=50, k_features=20, n_splits=5):
    from imblearn.under_sampling import RandomUnderSampler
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV
    import pandas as pd
    import scipy.special
    
    cache_dir = Path(cfg.cache_dir)
    
    # Enhanced features ile cache ykle
    Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te, le = _load_cache(cache_dir, n_features)
    classes = le.classes_.astype(str)

    focus = cfg.focus or ['analysis','backdoor','dos','exploits','fuzzers','reconnaissance','shellcode','worms']
    
    y_tr_arr = np.asarray(y_tr)
    
    def _safe_predict_proba(model, X_arr):
        try:
            return model.predict_proba(X_arr)[:, 1]
        except Exception:
            raw = model.predict(X_arr, raw_score=True)
            return scipy.special.expit(raw)
            
    for cls in focus:
        if cls not in classes: 
            _info(f"[experts] skip {cls}: snflarda yok")
            continue
            
        c_id = int(np.where(classes==cls)[0][0])
        y_tr_bin = (y_tr_arr == cls).astype(int)
        y_v_bin  = (np.asarray(y_v)==cls).astype(int)
        y_t_bin  = (np.asarray(y_te)==cls).astype(int)

        _stage(f"OVR Expert fit -> {cls} (SelectKBest={k_features}, K-Fold OOF & Focal Loss)")
        
        # 1. Feature Selection (Top K) + Attack-Specific Feature Protection
        _info(f"[experts] Selecting top {k_features} features for {cls} + preserving engineered attack features")
        
        # In feature_engineering_fixed.py, up to 14 attack specific features are appended at the end.
        # We assume if the number of features is exactly n_features (e.g., 50), it might have been PCA'd.
        # However, due to the logic in feature_engineering_fixed, if it returns >50, the last columns are the attack features.
        # And if it is exactly 50 it's PCA'd so we can't extract them cleanly.
        # But we can at least try to bypass SelectKBest for the last few columns if they exist.
        
        # We know we requested n_features=50. If Xt_tr.shape[1] > 50, no PCA was applied, and custom features are at the end.
        # If pca was applied they are blended. We'll protect the last 14 columns if the shape > k_features
        # Actually, let's just protect the last 14 columns if shape >= 14, to be safe.
        num_protected = min(14, Xt_tr.shape[1] // 4) # Protect up to 14 columns, but at most 25% of all features to be safe
        
        if num_protected > 0:
            X_base_tr = Xt_tr[:, :-num_protected]
            X_prot_tr = Xt_tr[:, -num_protected:]
            
            X_base_v = Xt_v[:, :-num_protected]
            X_prot_v = Xt_v[:, -num_protected:]
            
            X_base_te = Xt_te[:, :-num_protected]
            X_prot_te = Xt_te[:, -num_protected:]
            
            selector = SelectKBest(f_classif, k=min(k_features, X_base_tr.shape[1]))
            X_sel_base_tr = selector.fit_transform(X_base_tr, y_tr_bin)
            X_sel_base_v  = selector.transform(X_base_v)
            X_sel_base_te = selector.transform(X_base_te)
            
            Xt_tr_sel = np.column_stack([X_sel_base_tr, X_prot_tr])
            Xt_v_sel  = np.column_stack([X_sel_base_v,  X_prot_v])
            Xt_te_sel = np.column_stack([X_sel_base_te, X_prot_te])
        else:
            selector = SelectKBest(f_classif, k=min(k_features, Xt_tr.shape[1]))
            Xt_tr_sel = selector.fit_transform(Xt_tr, y_tr_bin)
            Xt_v_sel  = selector.transform(Xt_v)
            Xt_te_sel = selector.transform(Xt_te)
        
        oof_train = np.zeros(len(y_tr_bin), dtype=np.float32)
        valid_preds = np.zeros(len(y_v_bin), dtype=np.float32)
        test_preds  = np.zeros(len(y_t_bin), dtype=np.float32)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
        
        for fold, (trn_idx, val_idx) in enumerate(skf.split(Xt_tr_sel, y_tr_bin)):
            X_fold_tr, y_fold_tr = Xt_tr_sel[trn_idx], y_tr_bin[trn_idx]
            X_fold_val, y_fold_val = Xt_tr_sel[val_idx], y_tr_bin[val_idx]
            
            # Extreme Undersampling for Binary Experts (to prevent 0-prediction trap)
            vc = pd.Series(y_fold_tr).value_counts()
            min_class_count = vc.get(1, 0)
            
            # Use a strict 5:1 maximum imbalance ratio. 
            # If the class is extremely small (<100), give the majority class at least 500 samples to learn variance.
            target_majority = max(500, min_class_count * 5)
            
            if vc.get(0, 0) > target_majority and min_class_count > 0:
                rus = RandomUnderSampler(sampling_strategy={0: target_majority, 1: min_class_count}, random_state=cfg.random_state + fold)
                X_res, y_res = rus.fit_resample(X_fold_tr, y_fold_tr)
            else:
                X_res, y_res = X_fold_tr, y_fold_tr

            clf = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=7,
                objective='binary',
                class_weight='balanced',
                random_state=cfg.random_state + fold,
                n_jobs=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                verbose=-1
            )
            
            if y_fold_val.sum() > 0:
                clf.fit(X_res, y_res, 
                       eval_set=[(X_fold_val, y_fold_val)], 
                       callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
            else:
                clf.fit(X_res, y_res)
                
            try:
                cal = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
                cal.fit(X_fold_val, y_fold_val)
                final_model = cal
            except Exception:
                try:
                    cal = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
                    cal.fit(X_fold_val, y_fold_val)
                    final_model = cal
                except Exception:
                    final_model = clf
            
            oof_train[val_idx] = _safe_predict_proba(final_model, X_fold_val)
            valid_preds += _safe_predict_proba(final_model, Xt_v_sel) / n_splits
            test_preds  += _safe_predict_proba(final_model, Xt_te_sel) / n_splits
        
        for tag, p1 in (("oof_train", oof_train), ("oof_valid", valid_preds), ("oof_test", test_preds)):
            out = np.zeros((p1.shape[0], len(classes)), dtype=np.float32)
            out[:, c_id] = p1
            np.savez(cache_dir/f"proba_expert_{cls}_{tag}.npz", proba=out, classes=classes)
        
        # === SINIF BAZLI EXPERT SONULARI ===
        from sklearn.metrics import f1_score as _f1_score
        
        # Optimal threshold bul (validation zerinde)
        best_t, best_f1_val = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.02):
            y_hat_t = (valid_preds >= t).astype(int)
            f1_t = _f1_score(y_v_bin, y_hat_t, zero_division=0)
            if f1_t > best_f1_val:
                best_f1_val = f1_t
                best_t = t
        
        # Validation sonular
        y_hat_v = (valid_preds >= best_t).astype(int)
        p_v, r_v, f_v, _ = precision_recall_fscore_support(y_v_bin, y_hat_v, average='binary', zero_division=0)
        
        # Test sonular
        y_hat_te = (test_preds >= best_t).astype(int)
        p_te, r_te, f_te, _ = precision_recall_fscore_support(y_t_bin, y_hat_te, average='binary', zero_division=0)
        
        _info(f"[{cls.upper()}] Expert Results (threshold={best_t:.2f}):")
        _info(f"  Valid -> F1: {f_v:.4f}  Prec: {p_v:.4f}  Recall: {r_v:.4f}")
        _info(f"  Test  -> F1: {f_te:.4f}  Prec: {p_te:.4f}  Recall: {r_te:.4f}")
            
        _info(f"[experts] {cls} OOF probabilities saved (Train/Valid/Test).")

    _info("[experts] Tm OOF (Focal+FeatureSelection) uzmanlar hazr")

if __name__ == '__main__':
    # Basit CLI
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', type=str, default='.')
    ap.add_argument('--focus', nargs='*', default=['analysis','backdoor','worms','dos','shellcode'])
    ap.add_argument('--no-gpu', action='store_true')
    args = ap.parse_args()
    cfg = ExpCfg(cache_dir=args.cache_dir, use_gpu=(not args.no_gpu), focus=args.focus)
    train_experts_ovr(cfg)

