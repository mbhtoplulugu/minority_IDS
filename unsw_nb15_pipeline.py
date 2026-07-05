import os, glob, joblib, warnings, argparse, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import StandardScaler, OrdinalEncoder,FunctionTransformer, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, log_loss
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.special import expit, softmax

from data_preprocessing import DPConfig, ensure_preprocessed
from smote_enn import Config as PipelineConfig
# Experts & stacking with experts (optional imports)
try:
    from experts import train_experts_ovr, ExpCfg as ExpertsCfg
    from stack_experts import stack_with_experts
    from feature_engineering_fixed import run_feature_engineering_pipeline
    from tabnet_model import run_tabnet_oof
except Exception:
    train_experts_ovr = None
    ExpertsCfg = None
    stack_with_experts = None
    run_feature_engineering_pipeline = None
    run_tabnet_oof = None

warnings.filterwarnings("ignore")

def _stage(msg: str): print(f"[Stage] {msg}")
def _info(msg: str): print(f"[Info]  {msg}")

# ================= CONFIG =================
@dataclass
class Config:
    files_glob:str
    features_csv:str
    cache_dir:str='.'
    data_cache:str="cache_all_data.pkl"
    xy_cache:str="cache_xy.pkl"
    target:str="attack_cat"
    random_state:int=42
    valid_size:float=0.15
    test_size:float=0.15
    smote_ratio:float=0.6
    smote_min_floor:int=500
    rus_frac:float=0.03
    use_gpu:bool=True
    verbose:bool=True
    n_jobs:int=-1
    batch_size:int=1024
    lr:float=1e-3
    epochs:int=10
    # ===== DUAL-MODE =====
    dataset_mode:str='unsw'      # 'unsw' veya 'cicids'
    train_csv:str|None=None      # CICIDS modu icin
    test_csv:str|None=None       # CICIDS modu icin
    exclude_weak:bool=False      # Zayif siniflari verisetinden at

# ================= IO + Preprocess =================
def detect_gpu_tree_method(prefer_gpu: bool = True) -> str:
    if not prefer_gpu:
        return 'hist'
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES', None) == '':
            return 'hist'
        return 'gpu_hist'
    except Exception:
        return 'hist'

def get_le(cache_dir:str='.'):
    from sklearn.preprocessing import LabelEncoder
    return joblib.load(Path(cache_dir)/'label_encoder.joblib')

def save_proba_npz(path:str, proba:np.ndarray, le=None, classes=None):
    if classes is None:
        if le is not None and hasattr(le, "classes_"):
            classes = le.classes_
        else:
            raise ValueError("LabelEncoder veya explicit classes gerekli")
    np.savez(path, proba=proba, classes=np.asarray(classes, dtype=str))
    _info(f"proba saved: {path} with classes={list(classes)}")

def load_proba_aligned(path:str, target_classes:np.ndarray)->np.ndarray:
    p = Path(path)
    assert p.exists(), f"proba file not found: {path}"
    data = np.load(p, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        P = data['proba']
        src_classes = data['classes'].astype(str)
    else:
        P = data
        src_classes = target_classes
    C = len(target_classes)
    assert P.ndim == 2, f"Proba shape beklenmeyen: {P.shape}"
    assert len(src_classes) == P.shape[1], f"classes len {len(src_classes)} != proba C {P.shape[1]}"
    out = np.zeros((P.shape[0], C), dtype=np.float32)
    idx_map = {c:i for i,c in enumerate(src_classes)}
    for j,c in enumerate(target_classes):
        if c in idx_map and idx_map[c] < P.shape[1]:
            out[:,j] = P[:, idx_map[c]]
        else:
            out[:,j] = 0.0
    return out

# --- Tip yardmclar (ekleyin) ---
def _as_series(y) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    return pd.Series(np.asarray(y))

def _as_array(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy()
    return np.asarray(x)

def _resolve_pair(vf:str, tf:str):
    if not Path(vf).exists() and Path(vf.replace('.npz','.npy')).exists():
        vf = vf.replace('.npz','.npy')
    if not Path(tf).exists() and Path(tf.replace('.npz','.npy')).exists():
        tf = tf.replace('.npz','.npy')
    return vf, tf

def preprocess_and_cache(cfg: Config):
    """
    DPConfig ile cache'i garanti eder, Xt_* ve y_*'yi dndrr.
    Downstream uyumu iin X: np.ndarray (float32, C-contig), y: pd.Series.
    Dual-mode: cfg.dataset_mode'a gore UNSW veya CICIDS verisini yukler.
    """
    from data_preprocessing import DPConfig, ensure_preprocessed

    dp = DPConfig(
        files_glob=cfg.files_glob,
        features_csv=cfg.features_csv,
        cache_dir=cfg.cache_dir,
        target='attack_cat',
        valid_size=cfg.valid_size,
        test_size=cfg.test_size,
        scale_categoricals=True,
        use_minmax_for_cat=True,
        scale_numeric=True,
        strict_feature_list=False,
        dataset_mode=getattr(cfg, 'dataset_mode', 'unsw'),
        train_csv=getattr(cfg, 'train_csv', None),
        test_csv=getattr(cfg, 'test_csv', None),
        exclude_weak=getattr(cfg, 'exclude_weak', False),
    )
    _info(f"Dataset mode: {dp.dataset_mode}")
    Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te, _ = ensure_preprocessed(dp)

    # ---> Tipleri normalize et (downstreamde srtnmeyi azaltr)
    Xt_tr = _as_array(Xt_tr); Xt_v = _as_array(Xt_v); Xt_te = _as_array(Xt_te)
    y_tr  = _as_series(y_tr); y_v  = _as_series(y_v); y_te = _as_series(y_te)

    return Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te

def ensure_tail_features(cfg):
    cdir = Path(cfg.cache_dir)
    mask_f = cdir/'tail_feature_mask.npy'
    w_f    = cdir/'tail_feature_weight.npy'
    if not (mask_f.exists() and w_f.exists()):
        _stage("Tail feature importance (OVR) hesaplanyor")
        script = Path(__file__).with_name("tail_features.py")
        cmd = [sys.executable, str(script), '--cache-dir', str(cdir), '--k-top', '180']
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"tail_features.py failed: rc={res.returncode}")
    else:
        _info("Tail feature cache bulundu  tail_feature_mask.npy / tail_feature_weight.npy")

# ================= Resampling =================

def _print_dist(tag: str, y, logger=None):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    vc = y.value_counts()
    ratios = (vc / vc.sum()).sort_values(ascending=False)
    s = ", ".join([f"{k}:{p*100:.2f}%" for k, p in ratios.items()])
    msg = f"[Dist] {tag}: {s}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

def apply_smart_resampling(X: np.ndarray, y, cfg: Config):
    """
    Akll Resampling: Dev snflar RAM snrlarna (max 300k) gre undersample,
    Ar kk snflar ise SMOTE ile makul bir boyuta (min 5k) oversample yapar.
    Yeni: Tomek Links ile Normal vs Fuzzers snrn temizler.
    """
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.over_sampling import SMOTE
    y = _as_series(y)
    X = _as_array(X)
    _print_dist('train.before_resample', y)

    vc = y.value_counts()
    
    # 1. Undersample (Yksek snflar baskla)
    max_samples = 200_000
    under_dict = {c: min(count, max_samples) for c, count in vc.items()}
    rus = RandomUnderSampler(sampling_strategy=under_dict, random_state=cfg.random_state)
    X_res, y_res = rus.fit_resample(X, y)
    y_res = _as_series(y_res)
    _print_dist('train.after_undersample', y_res)

    # 2. Tomek Links (Boundary Cleaning - Normal vs Fuzzers/Exploits)
    # Bu adm akan rnekleri (noise) temizleyerek daha net snrlar oluturur.
    _info("Tomek Links ile snr temizlii yaplyor (Normal vs others)...")
    tl = TomekLinks(sampling_strategy='majority', n_jobs=-1)
    X_res, y_res = tl.fit_resample(X_res, y_res)
    y_res = _as_series(y_res)
    _print_dist('train.after_tomek', y_res)

    # 3. Oversample (Ar kkleri ykselt)
    min_samples = 8000
    vc2 = y_res.value_counts()
    over_dict = {c: max(count, min_samples) for c, count in vc2.items()}
    
    curr_min_count = vc2.min()
    k_neigh = min(3, curr_min_count - 1) if curr_min_count > 1 else 1

    smote = SMOTE(sampling_strategy=over_dict, random_state=cfg.random_state, k_neighbors=k_neigh)
    X_final, y_final = smote.fit_resample(X_res, y_res)
    
    _print_dist('train.after_smote (final)', y_final)

    return X_final, _as_series(y_final).to_numpy()

# ================= MODELS =================
def focal_loss_multiclass_xgb(y_true, y_pred):
    """Custom Multiclass Focal Loss for XGBoost - DYNAMIC SCALE"""
    gamma = 2.0
    
    y = y_true if not hasattr(y_true, 'get_label') else y_true.get_label()
    y = y.astype(int)
    
    # Kacinci sinif oldugunu otomatik tespit et
    is_1d = (y_pred.ndim == 1)
    if is_1d:
        # XGBoost flattened output verirse
        # (n_samples * n_classes) boyutundadir. 
        # n_classes'i y'nin max degerinden bulmaya calisalim (en garantisi static n_classes gecirmektir ama XGB buna izin vermez kolayca)
        # O yuzden global veya class_count uzerinden gitmeliyiz.
        pass

    # Not: Bu fonksiyonun icinde LabelEncoder'a erismek icin global kullanmali veya 
    # predict_proba'dan sonra cagirmaliyiz. Ancak XGB training sirasinda cagirdigi icin:
    # Simdilik varsayilan 10, eger y_pred seklinden anlasiliyorsa onu kullan:
    n_classes = 10 
    if not is_1d:
        n_classes = y_pred.shape[1]
    else:
        # XGB flatten yapinca n_samples * n_classes olur. 
        # y.shape[0] * n_classes = y_pred.shape[0]
        n_classes = len(y_pred) // len(y)

    # Dinamik alpha (azinlik siniflara daha fazla agirlik)
    # Varsayilan olarak 1.0, ama bazi siniflar icin manuel yukseltilebilir
    alpha = np.ones(n_classes, dtype=np.float32)
    # Azinlik siniflar (UNSW-NB15 icin 0-9 arasi bazi indeksler)
    # Gercek IDS'lerde genellikle ilk veya son sinif normaldir, digerleri saldiridir.
    for i in range(n_classes):
        alpha[i] = 2.0 # Genel saldiri agirligi
    
    if is_1d:
        preds = y_pred.reshape(-1, n_classes)
    else:
        preds = y_pred
        
    p = softmax(preds, axis=1)
    y_true_oh = np.eye(n_classes)[y]
    
    grad = (p - y_true_oh)
    hess = p * (1.0 - p)
    
    for i in range(n_classes):
        weight_factor = alpha[i] * np.power(1.0 - p[:, i], gamma)
        grad[:, i] *= weight_factor
        hess[:, i] *= weight_factor
        
    if is_1d:
        return grad.flatten(), hess.flatten()
    return grad, hess

def run_xgb_oof(cfg: Config, cached_data=None):
    """Smart Resampling ve K-Fold OOF ile XGB eitir, Meta-model iin OOF matrix kaydeder."""
    from sklearn.model_selection import StratifiedKFold
    
    if cached_data is not None:
        Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te = cached_data
        _stage(f"Using pre-loaded cached data for XGB ({Xt_tr.shape[1]} features, {Xt_tr.shape[0]} samples)")
    else:
        _stage("Preprocess & cache")
        Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te = preprocess_and_cache(cfg)

    cache_dir = Path(getattr(cfg,'cache_dir','.'))
    from sklearn.preprocessing import LabelEncoder
    le:LabelEncoder = joblib.load(cache_dir/'label_encoder.joblib')

    yv_enc  = le.transform(np.asarray(y_v))
    yte_enc = le.transform(np.asarray(y_te))

    tree_method = detect_gpu_tree_method(cfg.use_gpu)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
    
    oof_train = np.zeros((len(y_tr), len(le.classes_)), dtype=np.float32)
    test_preds = np.zeros((len(y_te), len(le.classes_)), dtype=np.float32)
    valid_preds = np.zeros((len(y_v), len(le.classes_)), dtype=np.float32)

    _stage(f"XGBoost 5-Fold OOF Training (tree_method={tree_method})")
    y_tr_arr = np.asarray(y_tr)
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(Xt_tr, y_tr_arr)):
        _info(f"Fold {fold+1}/5 starting...")
        X_fold_tr, y_fold_tr = Xt_tr[trn_idx], y_tr_arr[trn_idx]
        X_fold_val = Xt_tr[val_idx]
        
        X_res, y_res = apply_smart_resampling(X_fold_tr, y_fold_tr, cfg)
        yres_enc = le.transform(np.asarray(y_res))
        
        model = XGBClassifier(
            tree_method=tree_method,
            n_estimators=400, max_depth=9, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=2, gamma=1,
            random_state=cfg.random_state + fold,
            n_jobs=cfg.n_jobs, verbosity=0,
            objective=focal_loss_multiclass_xgb
        )
        
        # We need to manually tell XGBoost the number of classes when using a custom objective
        model.set_params(num_class=len(le.classes_))
        
        model.fit(X_res, yres_enc)
        
        oof_train[val_idx] = model.predict_proba(X_fold_val)
        valid_preds += model.predict_proba(Xt_v) / skf.n_splits
        test_preds += model.predict_proba(Xt_te) / skf.n_splits
        
    save_proba_npz(str(cache_dir/"proba_xgb_oof_train.npz"), oof_train, le=le)
    save_proba_npz(str(cache_dir/"proba_xgb_oof_valid.npz"), valid_preds, le=le)
    save_proba_npz(str(cache_dir/"proba_xgb_oof_test.npz"), test_preds, le=le)
    
    _stage("Training Final XGB Model on all data with Smart Resampling")
    X_res_all, y_res_all = apply_smart_resampling(Xt_tr, y_tr_arr, cfg)
    yres_all_enc = le.transform(np.asarray(y_res_all))
    
    final_model = XGBClassifier(
        tree_method=tree_method,
        n_estimators=600, max_depth=9, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, gamma=1,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs, verbosity=0,
        objective=focal_loss_multiclass_xgb
    )
    final_model.set_params(num_class=len(le.classes_))
    final_model.fit(X_res_all, yres_all_enc)
    
    xgb_model_path = cache_dir / "model_xgb_final.json"
    final_model.save_model(str(xgb_model_path))
    _info(f"Final XGB modeli kaydedildi: {xgb_model_path}")
    
    y_pred_te = test_preds.argmax(1)
    acc = accuracy_score(yte_enc, y_pred_te)
    _,_,f,_ = precision_recall_fscore_support(yte_enc, y_pred_te, average='macro', zero_division=0)
    _info(f"OOF XGB TEST -> Acc:{acc:.4f} MacroF1:{f:.4f}")
    print(classification_report(yte_enc, y_pred_te, target_names=le.classes_, digits=3))

# ================= AUTOENCODER WRAPPER =================
def run_autoencoder(cfg, n_features=50):
    from autoencoder import train_autoencoder
    from pathlib import Path
    import torch

    _stage("Autoencoder training with enhanced features")
    cache_dir = Path(cfg.cache_dir)

    # Enhanced features ncelii
    enhanced_files = [
        cache_dir / f"Xt_tr_enhanced{n_features}.joblib",
        cache_dir / f"Xt_v_enhanced{n_features}.joblib", 
        cache_dir / f"Xt_te_enhanced{n_features}.joblib"
    ]
    
    if all(f.exists() for f in enhanced_files):
        Xt_tr = joblib.load(enhanced_files[0])
        Xt_v = joblib.load(enhanced_files[1])
        Xt_te = joblib.load(enhanced_files[2])
        _info(f"[AE] Using enhanced features: {Xt_tr.shape}")
    else:
        # Fallback to original
        Xt_tr = joblib.load(cache_dir/"Xt_tr.joblib")
        Xt_v = joblib.load(cache_dir/"Xt_v.joblib")
        Xt_te = joblib.load(cache_dir/"Xt_te.joblib")
        _info(f"[AE] Using original features: {Xt_tr.shape}")

    input_dim = Xt_tr.shape[1]

    # model eit
    model = train_autoencoder(
        Xt_tr,
        input_dim=input_dim,
        latent_dim=64,
        epochs=20,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        cache_dir=cfg.cache_dir
    )
    torch.save(model.state_dict(), Path(cfg.cache_dir)/"autoencoder_model.pt")
    _info("Autoencoder model saved with tail features")


# ================= MAIN =================
# --- CLI ---

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Dataset-Agnostic IDS Pipeline (OOF Level-2 Stacking)")
    
    # ===== DATASET MODE =====
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument('--unsw', action='store_true', default=True, help='UNSW-NB15 modu (default)')
    mode_group.add_argument('--cicids', action='store_true', help='CICIDS17 modu')
    mode_group.add_argument('--cicids14', action='store_true', help='CICIDS17 14 sinifli modu')
    
    ap.add_argument('--files-glob', type=str, default=None)
    ap.add_argument('--features-csv', type=str, default=None)
    ap.add_argument('--cache-dir', type=str, default='.')
    # CICIDS specific
    ap.add_argument('--train-csv', type=str, default=None, help='CICIDS train CSV yolu')
    ap.add_argument('--test-csv', type=str, default=None, help='CICIDS test CSV yolu')

    # Core Execution Flags
    ap.add_argument('--prep-only',    action='store_true', help='Sadece preprocess + cache uret')
    ap.add_argument('--run-xgb',      action='store_true', help='XGBoost OOF P_train uret')
    ap.add_argument('--run-lgbm',     action='store_true', help='LightGBM OOF P_train uret')
    ap.add_argument('--run-lgbm-v2',  action='store_true', help='LightGBM V2 OOF P_train uret')
    ap.add_argument('--run-histgb',   action='store_true', help='HistGradientBoosting (Low RAM CPU) OOF P_train uret')
    ap.add_argument('--run-mlp',      action='store_true', help='MLP OOF P_train uret')
    ap.add_argument('--run-tabnet',   action='store_true', help='TabNet Multi-class OOF P_train uret')
    ap.add_argument('--run-autoencoder', action='store_true', help='Autoencoder modelini egit')
    ap.add_argument('--train-experts', action='store_true', help='(DEPRECATED) Eski OVR Uzmanlari egit')
    ap.add_argument('--run-fast-experts', action='store_true', help='(DEPRECATED) Eski hizli uzman')
    ap.add_argument('--run-autonomous-experts', action='store_true', help='F1 skoruna gore otonom olarak Fast (<0.85) ve Surgical (<0.75) uzman atar')
    ap.add_argument('--run-stack-experts', action='store_true', help='Level-2 Meta-Model egit ve degerlendir')
    ap.add_argument('--run-heuristic-ensemble', action='store_true', help='Kural Tabanli (Heuristic) Akilli Melezleme Yap')
    ap.add_argument('--run-all',      action='store_true', help='Tum boru hattini sirayla calistir')
    ap.add_argument('--exclude-weak-classes', action='store_true', help='Ensemble basarisini hesaplarken en dusuk performansli siniflari(unsw: analysis,backdoor / cicids: xss,infiltration) hesaplamadan cikar')
    
    # Options
    ap.add_argument('--no-gpu',       action='store_true', help='GPU kullanimini devre disi birak')
    ap.add_argument('--focus', nargs='*', default=None, help='Uzmanlar icin odak siniflar (orn: analysis backdoor)')
    ap.add_argument('--n-features', type=int, default=75, help='Tum modeller icin temel feature sayisi')

    args = ap.parse_args()
    
    # ===== DATASET MODE RESOLVE =====
    dataset_mode = 'cicids14' if args.cicids14 else ('cicids' if args.cicids else 'unsw')
    
    if dataset_mode in ('cicids', 'cicids14'):
        FILES_GLOB = ''  # CICIDS modunda kullanilmaz
        FEATURES_CSV = ''  # CICIDS modunda kullanilmaz
        _stage(f"=== {'CICIDS14' if dataset_mode == 'cicids14' else 'CICIDS17'} MODU ===")
    else:
        FILES_GLOB = r"C:/Users/mbhto/source/repos/UNSW-NB15/UNSWNB15_[0-5].csv" if args.files_glob is None else args.files_glob
        FEATURES_CSV = r"C:/Users/mbhto/source/repos/UNSW-NB15/NUSW-NB15_features.csv" if args.features_csv is None else args.features_csv
        _stage("=== UNSW-NB15 MODU ===")
    
    cache_mode_name = dataset_mode + "_excluded" if args.exclude_weak_classes else dataset_mode
    CACHEDIR = Path(args.cache_dir) / cache_mode_name
    CACHEDIR.mkdir(parents=True, exist_ok=True)
    
    cfg = Config(
        files_glob=FILES_GLOB, features_csv=FEATURES_CSV,
        use_gpu=(not getattr(args,'no_gpu',False)),
        dataset_mode=dataset_mode,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        exclude_weak=args.exclude_weak_classes,
    )
    cfg.cache_dir = str(CACHEDIR)

    # Hi bayrak gelmezse veya run-all gelirse
    run_all = args.run_all or not any([
        args.prep_only, args.run_xgb, args.run_lgbm, args.run_lgbm_v2,
        args.run_histgb, args.run_mlp, args.run_tabnet, args.run_autoencoder, 
        args.train_experts, args.run_fast_experts, args.run_autonomous_experts,
        args.run_stack_experts, args.run_heuristic_ensemble
    ])
    if not args.run_all and run_all:
        _info("Belirli bir bayrak verilmedi, --run-all varsaylan uygulanyor")

    # 1. Veri Hazrl (bir kez yap, sonular payla)
    _cached_data = None
    _cached_data_orig = None  # Balangta None - scope sorunu nlenir
    _cached_data_enh  = None  # Balangta None - scope sorunu nlenir

    if run_all or args.prep_only or args.run_xgb:
        _stage("Running preprocessing (once for all models)")
        _cached_data = preprocess_and_cache(cfg)

        # Dataset-Agnostic Feature Engine (Dinamik Ozellik Sentezi)
        if run_feature_engineering_pipeline:
            actual_n = run_feature_engineering_pipeline(cfg, n_features=args.n_features, method='enhanced')
            if actual_n:
                args.n_features = actual_n

        # Use the original features from the preprocessed data.
        _cached_data_orig = _cached_data

    # === MODELLER ICIN FARKLI VERI KELERI HAZIRLA ===
    # Bu blok her zaman alr: model admlar iin orig/enh verisi gerekiyor.
    def _ensure_data_loaded():
        """Orijinal ve Enhanced cache verilerini (varsa) ykler."""
        nonlocal _cached_data_orig, _cached_data_enh
        if _cached_data_orig is None:
            _stage("Lazily loading original preprocessed data for model steps")
            _cached_data_orig = preprocess_and_cache(cfg)
        if _cached_data_enh is None:
            import re as _re
            target_enh = CACHEDIR / f'Xt_tr_enhanced{args.n_features}.joblib'
            if target_enh.exists():
                actual_n_feat = args.n_features
                _stage(f"Loading enhanced features (n={actual_n_feat}) from exact match.")
                Xt_tr_enh = joblib.load(CACHEDIR / f'Xt_tr_enhanced{actual_n_feat}.joblib')
                Xt_v_enh  = joblib.load(CACHEDIR / f'Xt_v_enhanced{actual_n_feat}.joblib')
                Xt_te_enh = joblib.load(CACHEDIR / f'Xt_te_enhanced{actual_n_feat}.joblib')
                _, y_tr, _, y_v, _, y_te = _cached_data_orig
                _cached_data_enh = (Xt_tr_enh, y_tr, Xt_v_enh, y_v, Xt_te_enh, y_te)
                _info(f"Enhanced data ready: {Xt_tr_enh.shape[1]} features.")
            else:
                enhanced_pattern = str(CACHEDIR / "Xt_tr_enhanced*.joblib")
                enhanced_files_found = glob.glob(enhanced_pattern)
                if enhanced_files_found:
                    def _get_n(file_path):
                        m = _re.search(r'enhanced(\d+)', file_path)
                        return int(m.group(1)) if m else 0
                    latest = sorted(enhanced_files_found, key=_get_n)[-1]
                    actual_n_feat = _get_n(latest) or args.n_features
                    _stage(f"Loading enhanced features (n={actual_n_feat}) for LGBM/MLP/Experts")
                    Xt_tr_enh = joblib.load(CACHEDIR / f'Xt_tr_enhanced{actual_n_feat}.joblib')
                    Xt_v_enh  = joblib.load(CACHEDIR / f'Xt_v_enhanced{actual_n_feat}.joblib')
                    Xt_te_enh = joblib.load(CACHEDIR / f'Xt_te_enhanced{actual_n_feat}.joblib')
                    _, y_tr, _, y_v, _, y_te = _cached_data_orig
                    _cached_data_enh = (Xt_tr_enh, y_tr, Xt_v_enh, y_v, Xt_te_enh, y_te)
                    _info(f"Enhanced data ready: {Xt_tr_enh.shape[1]} features.")
                else:
                    _info("Enhanced features not found! Falling back to original for all models.")
                    _cached_data_enh = _cached_data_orig

    # 2. Level-1 / XGBoost (OOF) - ENHANCED FEATURES
    if run_all or args.run_xgb:
        _ensure_data_loaded()
        _data_to_use = _cached_data_enh if _cached_data_enh is not None else _cached_data_orig
        _info(f"Running XGBoost on ({_data_to_use[0].shape[1]}) features...")
        run_xgb_oof(cfg, cached_data=_data_to_use)

    # 2.5 Level-1 / LightGBM (OOF) - ENHANCED (58) FEATURES
    if run_all or args.run_lgbm:
        _ensure_data_loaded()
        try:
            from lightgbm_model import run_lgbm_oof
            _info("Running LightGBM on ENHANCED features...")
            run_lgbm_oof(cfg, n_features=args.n_features, cached_data=_cached_data_enh)
        except Exception as e:
            _info(f"LGBM model run failed: {e}")

    # 2.6 Level-1 / LightGBM V2 (OOF) - ENHANCED FEATURES
    if run_all or args.run_lgbm_v2:
        _ensure_data_loaded()
        try:
            from lightgbm_model_v2 import run_lgbm_oof_v2
            _data_to_use = _cached_data_enh if _cached_data_enh is not None else _cached_data_orig
            _info(f"Running LightGBM V2 on ({_data_to_use[0].shape[1]}) features...")
            run_lgbm_oof_v2(cfg, n_features=args.n_features, cached_data=_data_to_use)
        except Exception as e:
            _info(f"LGBM V2 model run failed: {e}")

    # 2.7 Level-1 / HistGradientBoosting (Low-RAM CPU) - ENHANCED (58) FEATURES (auto-load)
    if run_all or args.run_histgb:
        try:
            from hist_gb_model import run_hist_gb_oof
            _info("Running HistGB on ENHANCED (auto-load) features...")
            run_hist_gb_oof(cfg, n_features=args.n_features)
        except Exception as e:
            _info(f"HistGB model run failed: {e}")

    # 2.8 Level-1 / DOS RF Expert - NEW specialized model
    if run_all or args.run_fast_experts: 
        _info("Skipping old DOS RF Expert (RandomForest was too heavy). Handled by Fast Binary Experts.")
        pass
            
    # 3. Level-1 / MLP (OOF) - ENHANCED (58) FEATURES
    if run_all or args.run_mlp:
        try:
            import mlp_model
            _info("Running MLP on ENHANCED features...")
            mlp_model.run_mlp_oof(cfg, n_features=args.n_features) # mlp_model.py iinden enhanced arar
        except Exception as e:
            _info(f"MLP model run failed: {e}")

    # 3.5 Level-1 / TabNet (Sequential Attention Multi-class)
    if run_all or args.run_tabnet:
        try:
            if run_tabnet_oof:
                _info("Running TabNet Multi-class on ENHANCED features...")
                run_tabnet_oof(cfg, n_features=args.n_features, cached_data=_cached_data_enh)
            else:
                _info("run_tabnet_oof not imported!")
        except Exception as e:
            _info(f"TabNet model run failed: {e}")
            import traceback; traceback.print_exc()

    # 4. Level-1 / Autoencoder
    if run_all or args.run_autoencoder:
        run_autoencoder(cfg, n_features=args.n_features)

    # 5. Level-1 / Uzman Modeller (Focal Loss + SelectKBest + OOF) (Eski Algoritma)
    if args.train_experts:
        _info("Skipping old legacy experts. Handled by Fast Binary Experts.")
        pass

    # 5.1 Level-1 / Autonomous Experts (Fast & Surgical)
    if run_all or args.run_autonomous_experts or args.run_fast_experts:
        from fast_binary_expert import fast_binary_experts
        from threshold_optimization import optimize_thresholds_fast
        from autonomous_surgical_expert import autonomous_surgical_experts
        import traceback
        from sklearn.metrics import f1_score, precision_recall_fscore_support
        
        try:
            if args.focus:
                fast_classes = args.focus
                surgical_classes = args.focus
                _info(f"Manual focus classes: {fast_classes}")
            else:
                _info("Auto-detecting weak classes from Base Model validation performance...")
                le_auto = joblib.load(CACHEDIR / 'label_encoder.joblib')
                cls_names = le_auto.classes_.astype(str)
                
                xgb_val_path = CACHEDIR / 'proba_xgb_oof_valid.npz'
                if xgb_val_path.exists():
                    y_v_auto = joblib.load(CACHEDIR / 'y_v.joblib')
                    yv_enc = le_auto.transform(np.asarray(y_v_auto))
                    data_val = np.load(xgb_val_path, allow_pickle=True)
                    P_val = data_val['proba']
                    yhat_val = P_val.argmax(axis=1)
                    _, _, f1_per_class, _ = precision_recall_fscore_support(
                        yv_enc, yhat_val, labels=range(len(cls_names)), zero_division=0)
                    
                    FAST_THRESHOLD = 0.85
                    SURGICAL_THRESHOLD = 0.75
                    fast_classes = []
                    surgical_classes = []
                    
                    for i, c in enumerate(cls_names):
                        if c == 'normal' or c == 'generic':
                            continue
                        
                        f1_val = f1_per_class[i]
                        samples = np.sum(yv_enc == i)
                        
                        if samples >= 10:
                            if f1_val < SURGICAL_THRESHOLD:
                                fast_classes.append(c)
                                surgical_classes.append(c)
                                _info(f"  -> [AUTONOMOUS] {c}: Val F1={f1_val:.4f} < {SURGICAL_THRESHOLD} => FAST + SURGICAL EXPERT ATANACAK")
                            elif f1_val < FAST_THRESHOLD:
                                fast_classes.append(c)
                                _info(f"  -> [AUTONOMOUS] {c}: Val F1={f1_val:.4f} < {FAST_THRESHOLD} => SADECE FAST EXPERT ATANACAK")
                            else:
                                _info(f"  -> [AUTONOMOUS] {c}: Val F1={f1_val:.4f} >= {FAST_THRESHOLD} => SAGLIKLI, UZMAN GEREKMIYOR")
                        else:
                            _info(f"  -> [AUTONOMOUS] {c}: Val F1={f1_val:.4f} ama ornek ({samples}) < 10, ATLANDI")
                else:
                    _info("XGBoost OOF valid proba bulunamadi, fallback: tum azinlik siniflar.")
                    fast_classes = [c for c in cls_names if c not in ('normal', 'generic')]
                    surgical_classes = fast_classes.copy()
            
            if len(fast_classes) > 0:
                _info(f"Starting FAST EXPERT training for: {fast_classes}")
                fast_binary_experts(cache_dir=cfg.cache_dir, minority_classes=fast_classes, n_features=args.n_features)
                optimize_thresholds_fast(cache_dir=cfg.cache_dir, minority_classes=fast_classes, n_features=args.n_features)
            else:
                _info("Hicbir sinif Fast Expert esiginin altinda degil.")
                
            if len(surgical_classes) > 0:
                _info(f"Starting SURGICAL EXPERT training for: {surgical_classes}")
                autonomous_surgical_experts(cache_dir=cfg.cache_dir, target_classes=surgical_classes, n_features=args.n_features)
            else:
                _info("Hicbir sinif Surgical Expert esiginin altinda degil.")
                
        except Exception as e:
            _info(f"Autonomous Experts run failed: {e}")
            traceback.print_exc()

    # 6. Level-2 / K-Fold Meta Model Stacking
    if run_all or args.run_stack_experts:
        if stack_with_experts is None:
            raise RuntimeError("stack_experts.py import edilemedi")
        focus = args.focus or ['analysis','backdoor','dos','exploits','fuzzers','reconnaissance','shellcode','worms']
        stack_with_experts(cache_dir=cfg.cache_dir, focus=focus)
        
    # 7. Level-2 / Rule-Based (Heuristic) Intelligent Ensembling
    if run_all or args.run_heuristic_ensemble:
        try:
            from heuristic_ensemble import heuristic_ensemble
            heuristic_ensemble(cache_dir=cfg.cache_dir, exclude_weak=args.exclude_weak_classes)
        except Exception as e:
            _info(f"Heuristic Ensemble run failed: {e}")
            
    _info("Pipeline execution finished.")
    
if __name__=='__main__':
    main()


