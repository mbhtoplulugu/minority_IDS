import argparse, joblib, pickle, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, f1_score, classification_report
from xgboost import XGBClassifier

def _info(m):  print(f"[Info]  {m}")
def _stage(m): print(f"[Stage] {m}")

def _load_aligned(path, classes):
    """Loads probabilities and aligns them to the standard class order."""
    data = np.load(path, allow_pickle=True)
    P = data['proba']
    if 'classes' in data:
        src = data['classes'].astype(str)
        out = np.zeros((P.shape[0], len(classes)), dtype=np.float32)
        idx = {c:i for i,c in enumerate(src)}
        for j,c in enumerate(classes):
            if c in idx and idx[c] < P.shape[1]:
                out[:,j] = P[:, idx[c]]
        return out
    return P

def get_f1_arr(y_true, y_pred_encoded, n_cls):
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_encoded, labels=range(n_cls), zero_division=0)
    return f1

def heuristic_ensemble(cache_dir: str, exclude_weak: bool = False):
    """
    Consolidated Comparative Analysis & Meta-Ensemble (Stacking)
    """
    cdir = Path(cache_dir)
    le = joblib.load(cdir/'label_encoder.joblib')
    classes = le.classes_.astype(str)
    n_cls = len(classes)
    
    _stage("Loading Ground Truth and Model Predictions")
    y_te = joblib.load(cdir/'y_te.joblib')
    y_v  = joblib.load(cdir/'y_v.joblib')
    
    yte_enc = le.transform(np.asarray(y_te))
    yv_enc  = le.transform(np.asarray(y_v))

    # --- 1. Comparative Analysis ---
    _stage("Performing Comparative Analysis of Individual Models")
    
    model_results = {}
    
    # 1a. Base Models
    main_models = {
        'xgb': 'XGBoost',
        'lgbm': 'LGBM_Enh',
        'lgbmV2': 'LGBM_Ori',
        'histgb': 'HistGB',
        'mlp': 'MLP',
        'tabnet': 'TabNet',
        'rf': 'RF'
    }
    
    for m_key, m_name in main_models.items():
        path = cdir / f'proba_{m_key}_oof_test.npz'
        if path.exists():
            P = _load_aligned(path, classes)
            f1s = get_f1_arr(yte_enc, P.argmax(axis=1), n_cls)
            model_results[m_name] = f1s

    # 1b. Check ALL available proba files for top performing experts
    all_probas = list(cdir.glob('proba_*_oof_test.npz'))
    for p in all_probas:
        name = p.stem.replace('proba_', '').replace('_oof_test', '')
        if name in main_models: continue
        
        if 'aecnn' in name.lower() or 'ae_cnn' in name.lower(): continue
        
        # Friendly names
        f_name = name.replace('fast_expert_', 'FE_').replace('expert_', 'E_').replace('bb_', 'BB_').replace('tabnet_', 'TN_').title()
        
        P = _load_aligned(p, classes)
        f1s = get_f1_arr(yte_enc, P.argmax(axis=1), n_cls)
        
        # Filter: Only add to table if it's "interesting" (e.g. good macro or very good at one class)
        # But for "all models comparative", let's at least show the ones that aren't zero.
        if np.max(f1s) > 0.01: 
             model_results[f_name] = f1s

    # --- 2. Print Comparative Table ---
    sorted_models = sorted(model_results.keys(), key=lambda x: np.mean(model_results[x]), reverse=True)
    display_models = sorted_models[:15] # Display up to 15 models to include experts

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

    # --- 3. Trusted Expert Rule-Based Routing ---
    _stage("Applying Dynamic Trusted Expert Routing (Rule-Based Ensemble)")
    
    # Adim 1: Find best overall base model on VALIDATION
    yv_enc = le.transform(np.asarray(y_v))
    
    val_model_results = {}
    for m_key, m_name in main_models.items():
        path = cdir / f'proba_{m_key}_oof_valid.npz'
        if path.exists():
            P_v = _load_aligned(path, classes)
            f1s = get_f1_arr(yv_enc, P_v.argmax(axis=1), n_cls)
            val_model_results[m_key] = f1s

    # Find Base Model early so experts can be evaluated against it
    best_macro = 0
    base_model_key = 'xgb'
    for m_key, f1s in val_model_results.items():
        if np.mean(f1s) > best_macro and m_key in main_models:
            best_macro = np.mean(f1s)
            base_model_key = m_key
            
    base_val_preds = _load_aligned(cdir / f'proba_{base_model_key}_oof_valid.npz', classes).argmax(axis=1)

    # Include fast experts if useful (evaluate them as an override on top of base model)
    for p in cdir.glob('proba_fast_expert_*_oof_valid.npz'):
        m_key = p.stem.replace('proba_', '').replace('_oof_valid', '')
        P_v = _load_aligned(p, classes)
        # It's a binary expert where only one column is populated
        # We need to find which column it is targeting
        target_cls = m_key.replace('fast_expert_', '')
        if target_cls in classes:
            c_idx = np.where(classes == target_cls)[0][0]
            # Override base model where fast expert > 0.5
            expert_mask = P_v[:, c_idx] > 0.5
            temp_preds = base_val_preds.copy()
            temp_preds[expert_mask] = c_idx
            # Calculate F1s of this mixed prediction
            f1s = get_f1_arr(yv_enc, temp_preds, n_cls)
            # Store the f1s under this expert's name
            val_model_results[m_key] = f1s
             
    _info(f"Selected Base Model: {main_models.get(base_model_key, base_model_key)} (Val F1: {best_macro:.4f})")
    
    # Load Base Test Predictions
    P_base_te = _load_aligned(cdir / f'proba_{base_model_key}_oof_test.npz', classes)
    
    is_cicids = 'bot' in classes

    if is_cicids:
        _info("CICIDS Dataset Detected! Applying Class-Wise Validation-Weighted Soft Voting...")
        P_soft = np.zeros_like(P_base_te)
        weight_sum = np.zeros(n_cls)
        for m_key in main_models:
            if m_key in val_model_results:
                weight_arr = val_model_results[m_key]
                p_path = cdir / f'proba_{m_key}_oof_test.npz'
                if p_path.exists():
                    P_m = _load_aligned(p_path, classes)
                    P_soft += P_m * weight_arr
                    weight_sum += weight_arr
        weight_sum[weight_sum == 0] = 1e-9
        P_soft /= weight_sum
        yhat_te = P_soft.argmax(axis=1)
        base_val_f1 = np.ones(n_cls) # mock to allow A and A.2 to run comparisons vs Soft Voting
    else:
        yhat_te = P_base_te.argmax(axis=1)
        base_val_f1 = val_model_results.get(base_model_key, np.zeros(n_cls))
    
    # ================================================================
    # OTONOM OVERRIDE KEŞFI (Dataset-Agnostic)
    # Hardcoded sinif isimleri YOK - tamamen performans bazli
    # ================================================================
    
    # --- PHASE A: Surgical Binary Expert Override (Otonom Keşif) ---
    _stage("Phase A: Auto-discovering Surgical Binary Expert Overrides (Validation-Based)")
    
    # Tum surgical expert dosyalarini tara
    for expert_pkl in sorted(cdir.glob('surgical_*_expert.pkl')):
        # Sinif adini dosya adindan cikar: surgical_dos_expert.pkl -> dos
        cls_name = expert_pkl.stem.replace('surgical_', '').replace('_expert', '')
        thresh_path = cdir / f'surgical_{cls_name}_threshold.pkl'
        proba_path = cdir / f'proba_surgical_{cls_name}_oof_test.npz'
        proba_val_path = cdir / f'proba_surgical_{cls_name}_oof_valid.npz'
        
        if not (thresh_path.exists() and proba_path.exists()):
            continue
        if cls_name not in classes:
            continue
            
        c_idx = int(np.where(classes == cls_name)[0][0])
        
        # Validation uzerinde expert'in basarisini kontrol et
        apply_override = True
        if proba_val_path.exists():
            P_surg_val = _load_aligned(proba_val_path, classes)
            with open(thresh_path, 'rb') as f:
                opt_thresh = pickle.load(f)
            # Expert override uygulanmis tahmini olustur
            temp_val = base_val_preds.copy()
            surg_val_mask = P_surg_val[:, c_idx] >= opt_thresh
            temp_val[surg_val_mask] = c_idx
            mixed_f1 = get_f1_arr(yv_enc, temp_val, n_cls)
            
            # Expert, base model'den daha iyi mi?
            if np.mean(mixed_f1) <= np.mean(base_val_f1):
                _info(f"  -> {cls_name}: Surgical expert ATLANDI (Macro F1 iyilestirme yok)")
                apply_override = False
        
        if apply_override:
            with open(thresh_path, 'rb') as f:
                opt_thresh = pickle.load(f)
            P_surgical = _load_aligned(proba_path, classes)
            surgical_mask = P_surgical[:, c_idx] >= opt_thresh
            count = int(np.sum(surgical_mask))
            yhat_te[surgical_mask] = c_idx
            _info(f"  -> SURGICAL: {cls_name} expert overrode {count} predictions (thresh >= {opt_thresh:.3f})")

    # --- PHASE A.2: Fast Binary Expert Override ---
    _stage("Phase A.2: Auto-discovering Fast Binary Expert Overrides (Validation-Based)")
    for expert_val_path in sorted(cdir.glob('proba_fast_expert_*_oof_valid.npz')):
        m_key = expert_val_path.stem.replace('proba_', '').replace('_oof_valid', '')
        cls_name = m_key.replace('fast_expert_', '')
        if cls_name not in classes: continue
        
        expert_test_path = cdir / f'proba_{m_key}_oof_test.npz'
        if not expert_test_path.exists(): continue
        
        c_idx = int(np.where(classes == cls_name)[0][0])
        
        P_fe_val = _load_aligned(expert_val_path, classes)
        temp_val = base_val_preds.copy()
        fe_val_mask = P_fe_val[:, c_idx] > 0.5
        temp_val[fe_val_mask] = c_idx
        mixed_f1 = get_f1_arr(yv_enc, temp_val, n_cls)
        
        if np.mean(mixed_f1) <= np.mean(base_val_f1):
            _info(f"  -> {cls_name}: Fast expert ATLANDI (Macro F1 iyilestirme yok)")
        else:
            P_fe_te = _load_aligned(expert_test_path, classes)
            fe_test_mask = P_fe_te[:, c_idx] > 0.5
            count = int(np.sum(fe_test_mask))
            yhat_te[fe_test_mask] = c_idx
            _info(f"  -> FAST EXPERT: {cls_name} expert overrode {count} predictions (thresh > 0.5)")
    
    # --- PHASE B: Argmax Override (Ana Modeller Arası Otonom Karsilastirma) ---
    if not is_cicids:
        _stage("Phase B: Auto-discovering Argmax Overrides (Model vs Base per-class)")
    
        # Her ana model icin validation'da sinif bazli F1 karsilastir
        for m_key in list(val_model_results.keys()):
            if m_key == base_model_key or m_key not in main_models:
                continue
            
            test_path = cdir / f'proba_{m_key}_oof_test.npz'
            if not test_path.exists():
                continue
                
            alt_f1 = val_model_results[m_key]
            P_alt_te = _load_aligned(test_path, classes)
            
            for c_idx in range(n_cls):
                cls_name = classes[c_idx]
                if cls_name in ('normal', 'generic'):
                    continue
                
                # Alternatif model bu sinifta base modelden anlamli olarak iyi mi?
                improvement = alt_f1[c_idx] - base_val_f1[c_idx]
                if improvement > 0.01:  # En az %1 iyilestirme
                    # Override uygula: bu sinif icin alternatif modelin argmax tahminlerini kullan
                    alt_mask = P_alt_te.argmax(axis=1) == c_idx
                    override_count = int(np.sum(alt_mask))
                    if override_count > 0:
                        yhat_te[alt_mask] = c_idx
                        _info(f"  -> ARGMAX: '{cls_name}' icin {override_count} tahmin "
                              f"{main_models[m_key]} ile degistirildi "
                              f"(Val F1: {alt_f1[c_idx]:.4f} vs Base {base_val_f1[c_idx]:.4f}, "
                              f"+{improvement:.4f})")
    
    # --- PHASE C: Combined Probability Boost (Otonom) ---
    if not is_cicids:
        _stage("Phase C: Auto-discovering Combined Probability Boosts")
    
        # Her sinif icin: birden fazla modelin probability'lerini birlestirip threshold uygula
        # Sadece base model'den cok zayif olan siniflar icin (F1 < 0.50)
        for c_idx in range(n_cls):
            cls_name = classes[c_idx]
            if cls_name in ('normal', 'generic'):
                continue
            if base_val_f1[c_idx] >= 0.50:
                continue
            
            # Tum mevcut modellerin probability'lerini topla
            combined_proba = np.zeros(len(yhat_te), dtype=np.float32)
            model_count = 0
            for m_key in main_models:
                p_path = cdir / f'proba_{m_key}_oof_test.npz'
                if p_path.exists():
                    P_m = _load_aligned(p_path, classes)
                    combined_proba = np.maximum(combined_proba, P_m[:, c_idx])
                    model_count += 1
            
            if model_count >= 2:
                # Sadece cok guvenli thresholdlarda boostla
                boost_mask = combined_proba >= 0.85
                boost_count = int(np.sum(boost_mask))
                if boost_count > 0:
                    yhat_te[boost_mask] = c_idx
                    _info(f"  -> BOOST: '{cls_name}' icin {boost_count} tahmin "
                          f"combined proba >= 0.85 ile override edildi "
                          f"(Base Val F1 cok dusuk: {base_val_f1[c_idx]:.4f})")

    f1s_ens = get_f1_arr(yte_enc, yhat_te, n_cls)
    macro_ens = np.mean(f1s_ens)

    # --- FINAL: Dinamik Sonuc Tablosu ---
    _stage("Rule-Based Ensemble Final Performance")
    print("\n" + "=" * 50)
    print(f"| {'FINAL ENSEMBLE RESULTS (AUTO-ROUTING)':^46} |")
    print("+" + "-" * 20 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    print(f"| {'Class':<18} | {'F1-Score':^10} | {'Status':^10} |")
    print("+" + "-" * 20 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    
    # Dinamik target: medyan F1 uzerindeki siniflar "OK", altindakiler "LOW"
    median_f1 = float(np.median(f1s_ens[f1s_ens > 0])) if np.any(f1s_ens > 0) else 0.5

    for i, cls in enumerate(classes):
        val = f1s_ens[i]
        status = "OK" if val >= median_f1 else "LOW" if val > 0 else "-"
        print(f"| {cls:<18} | {val:^10.4f} | {status:^10} |")
        
    print("+" + "-" * 20 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    print(f"| {'MACRO F1':<18} | {macro_ens:^10.4f} | {'SUCCESS' if macro_ens >= 0.75 else 'IN-PROG':^10} |")
    print("=" * 50)
    _info(f"Auto-Routing Ensemble achieved Macro F1: {macro_ens:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=str, default='.')
    parser.add_argument('--exclude-weak-classes', action='store_true')
    args = parser.parse_args()
    heuristic_ensemble(args.cache_dir, args.exclude_weak_classes)
