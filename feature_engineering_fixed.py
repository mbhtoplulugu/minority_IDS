import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import joblib
from pathlib import Path

def create_smaller_input_features(X, method='pca', n_features=50, cache_dir='.'):
    cache_path = Path(cache_dir)
    
    if method == 'pca':
        # Available features'dan fazla component isteme
        max_components = min(n_features, X.shape[1], X.shape[0])
        print(f"PCA: Requesting {n_features} components, using {max_components} (available: {X.shape[1]})")
        
        pca = PCA(n_components=max_components, random_state=42)
        X_reduced = pca.fit_transform(X)
        joblib.dump(pca, cache_path / f'pca_{max_components}.joblib')
        return X_reduced

def apply_feature_reduction(X, method='pca', n_features=50, cache_dir='.'):
    cache_path = Path(cache_dir)
    
    if method == 'pca':
        # Gerek component saysn bul
        max_components = min(n_features, X.shape[1])
        pca_file = cache_path / f'pca_{max_components}.joblib'
        
        # Eer exact file yoksa, mevcut PCA files' kontrol et
        if not pca_file.exists():
            pca_files = list(cache_path.glob('pca_*.joblib'))
            if pca_files:
                pca_file = pca_files[0]  # lk bulduunu kullan
                print(f"Using existing PCA file: {pca_file.name}")
        
        pca = joblib.load(pca_file)
        return pca.transform(X)

def calculate_zscore_for_each_feature(X):
    return np.abs(stats.zscore(X, axis=0))

def count_features_with_zscore_above_3(X):
    zscore = calculate_zscore_for_each_feature(X)
    return (zscore > 3).sum(axis=1)

def mark_values_in_top_bottom_1_percent(X):
    p1 = np.percentile(X, 1, axis=0)
    p99 = np.percentile(X, 99, axis=0)
    return (X <= p1) | (X >= p99)

def create_core_statistical_features(X):
    """Sadece core statistical moments"""
    features = []
    features.append(np.mean(X, axis=1))
    features.append(np.std(X, axis=1))
    features.append(stats.skew(X, axis=1))
    features.append(stats.kurtosis(X, axis=1))
    return np.column_stack(features)

def create_attack_specific_features(X, original_feature_names):
    """Literatr temelli attack-specific features - TM SINIFLAR N"""
    df = pd.DataFrame(X, columns=original_feature_names)
    attack_features = []
    feature_names = []
    
    # 1. WORMS-SPECIFIC FEATURES (Updated with Fan-Out Ratio)
    if 'srcip' in df.columns and 'dstip' in df.columns:
        # IP diversity (basitletirilmi entropy) - map ile tm satrlara yay
        ip_diversity_map = df.groupby('srcip')['dstip'].nunique() / df.groupby('srcip').size()
        ip_diversity = df['srcip'].map(ip_diversity_map).fillna(0).values
        attack_features.append(ip_diversity)
        feature_names.append('ip_diversity_ratio')
        
        # NEW: Fan-Out Ratio (Spreading behavior)
        # Ratio of unique destinations to the total number of connections for that source
        fan_out = df.groupby('srcip')['dstip'].transform('nunique') / (df.groupby('srcip')['srcip'].transform('count') + 1e-10)
        attack_features.append(fan_out.values)
        feature_names.append('fan_out_ratio')

    if 'srcip' in df.columns and 'dsport' in df.columns:
        # Port diversity - map ile tm satrlara yay
        port_diversity_map = df.groupby('srcip')['dsport'].nunique() / df.groupby('srcip').size()
        port_diversity = df['srcip'].map(port_diversity_map).fillna(0).values
        attack_features.append(port_diversity)
        feature_names.append('port_diversity_ratio')

    # 2. ANALYSIS-SPECIFIC FEATURES  
    if 'srcip' in df.columns and 'proto' in df.columns:
        # Protocol concentration - map ile tm satrlara yay
        proto_concentration_map = df.groupby('srcip')['proto'].apply(
            lambda x: x.value_counts().max() / len(x)
        )
        proto_concentration = df['srcip'].map(proto_concentration_map).fillna(0).values
        attack_features.append(proto_concentration)
        feature_names.append('protocol_concentration')

    # 3. BACKDOOR-SPECIFIC FEATURES
    if 'dur' in df.columns:
        # Long duration indicator
        duration_threshold = df['dur'].quantile(0.9)
        long_duration = (df['dur'] > duration_threshold).astype(float)
        attack_features.append(long_duration)
        feature_names.append('is_long_duration')

    if 'dsport' in df.columns:
        # Non-standard port usage
        standard_ports = [80, 443, 22, 21, 25, 53, 110, 143, 993, 995]
        non_standard_port = (~df['dsport'].isin(standard_ports)).astype(float)
        attack_features.append(non_standard_port)
        feature_names.append('uses_non_standard_port')

    # 4. DOS-SPECIFIC FEATURES
    if 'spkts' in df.columns and 'dur' in df.columns:
        # Packet rate
        packet_rate = df['spkts'] / (df['dur'] + 1e-10)
        attack_features.append(packet_rate)
        feature_names.append('packet_rate')

    if 'sbytes' in df.columns and 'dur' in df.columns:
        # Byte rate  
        byte_rate = df['sbytes'] / (df['dur'] + 1e-10)
        attack_features.append(byte_rate)
        feature_names.append('byte_rate')

    # 5. EXPLOITS-SPECIFIC FEATURES
    if 'sbytes' in df.columns and 'dbytes' in df.columns:
        # Payload size asymmetry (exploits often have small requests, large responses)
        payload_asymmetry = np.abs(df['sbytes'] - df['dbytes']) / (df['sbytes'] + df['dbytes'] + 1e-10)
        attack_features.append(payload_asymmetry)
        feature_names.append('payload_asymmetry')
    
    if 'spkts' in df.columns and 'dpkts' in df.columns:
        # Packet count asymmetry
        packet_asymmetry = np.abs(df['spkts'] - df['dpkts']) / (df['spkts'] + df['dpkts'] + 1e-10)
        attack_features.append(packet_asymmetry)
        feature_names.append('packet_asymmetry')

    # 6. FUZZERS-SPECIFIC FEATURES
    if 'sbytes' in df.columns and 'spkts' in df.columns:
        # Average packet size (fuzzers often send many small packets)
        avg_packet_size = df['sbytes'] / (df['spkts'] + 1e-10)
        attack_features.append(avg_packet_size)
        feature_names.append('avg_packet_size')
    
    if 'dur' in df.columns:
        # Very short duration indicator (fuzzers often quick)
        short_duration_threshold = df['dur'].quantile(0.1)
        very_short_duration = (df['dur'] <= short_duration_threshold).astype(float)
        attack_features.append(very_short_duration)
        feature_names.append('is_very_short_duration')

    # 7. RECONNAISSANCE-SPECIFIC FEATURES (Updated with Port Entropy)
    if 'sport' in df.columns:
        # Source port entropy (randomization check)
        # Using transform with nunique over a rolling-like group or global if srcip is not enough
        sport_entropy = df.groupby('srcip')['sport'].transform('nunique') / (df.groupby('srcip')['sport'].transform('count') + 1e-10)
        attack_features.append(sport_entropy.values)
        feature_names.append('src_port_entropy')

    if 'dsport' in df.columns:
        # Port scanning indicator (common ports)
        common_scan_ports = [21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 993, 995, 1433, 3389]
        scans_common_port = df['dsport'].isin(common_scan_ports).astype(float)
        attack_features.append(scans_common_port)
        feature_names.append('scans_common_port')
    
    if 'spkts' in df.columns and 'sbytes' in df.columns:
        # Small packet indicator (reconnaissance often uses small packets)
        small_packet_ratio = (df['sbytes'] / (df['spkts'] + 1e-10) < 100).astype(float)
        attack_features.append(small_packet_ratio)
        feature_names.append('uses_small_packets')

    # 8. SHELLCODE-SPECIFIC FEATURES (Updated with Payload Density)
    if 'sbytes' in df.columns and 'spkts' in df.columns:
        # Payload density (Byte/Packet ratio variance)
        payload_density = df['sbytes'] / (df['spkts'] + 1e-10)
        attack_features.append(payload_density.values)
        feature_names.append('payload_density')

    if 'sbytes' in df.columns:
        # Specific payload size ranges (shellcode often has characteristic sizes)
        shellcode_size_range = ((df['sbytes'] >= 100) & (df['sbytes'] <= 2000)).astype(float)
        attack_features.append(shellcode_size_range)
        feature_names.append('shellcode_size_range')
    
    if 'dsport' in df.columns:
        # High port usage (shellcode often targets high ports)
        high_port_usage = (df['dsport'] > 1024).astype(float)
        attack_features.append(high_port_usage)
        feature_names.append('uses_high_port')

    # 9. GENERIC-SPECIFIC FEATURES (catch-all patterns)
    if 'dur' in df.columns and 'sbytes' in df.columns:
        # Throughput (generic attacks often have moderate throughput)
        throughput = df['sbytes'] / (df['dur'] + 1e-10)
        moderate_throughput = ((throughput > np.percentile(throughput, 25)) & 
                              (throughput < np.percentile(throughput, 75))).astype(float)
        attack_features.append(moderate_throughput)
        feature_names.append('moderate_throughput')

    # 10. CROSS-CLASS FEATURES (useful for all attack types)
    if 'spkts' in df.columns and 'dpkts' in df.columns and 'sbytes' in df.columns and 'dbytes' in df.columns:
        # Communication efficiency
        total_packets = df['spkts'] + df['dpkts']
        total_bytes = df['sbytes'] + df['dbytes']
        comm_efficiency = total_bytes / (total_packets + 1e-10)
        attack_features.append(comm_efficiency)
        feature_names.append('communication_efficiency')
    
    if 'dur' in df.columns and 'spkts' in df.columns:
        # Activity intensity
        activity_intensity = df['spkts'] / (df['dur'] + 1e-10)
        attack_features.append(activity_intensity)
        feature_names.append('activity_intensity')

    # 11. DEEP CYBERSECURITY FEATURES (PHASE 3)
    
    # A. Packet Loss Ratios & Asymmetry (Analysis & Fuzzers)
    if 'sloss' in df.columns and 'spkts' in df.columns and 'dloss' in df.columns and 'dpkts' in df.columns:
        s_loss_ratio = df['sloss'] / (df['spkts'] + 1e-10)
        d_loss_ratio = df['dloss'] / (df['dpkts'] + 1e-10)
        loss_asymmetry = np.abs(s_loss_ratio - d_loss_ratio)
        
        attack_features.extend([s_loss_ratio, d_loss_ratio, loss_asymmetry])
        feature_names.extend(['s_loss_ratio', 'd_loss_ratio', 'loss_asymmetry'])

    # B. TTL Asymmetry (Spoofing, Backdoors, Exploits)
    if 'sttl' in df.columns and 'dttl' in df.columns:
        ttl_asymmetry = np.abs(df['sttl'] - df['dttl']) / (df['sttl'] + df['dttl'] + 1e-10)
        attack_features.append(ttl_asymmetry)
        feature_names.append('ttl_asymmetry')

    # C. TCP Setup Inefficiency (Fuzzers, Slowloris/Recon)
    if 'tcprtt' in df.columns and 'dur' in df.columns:
        tcp_setup_inefficiency = df['tcprtt'] / (df['dur'] + 1e-10)
        attack_features.append(tcp_setup_inefficiency)
        feature_names.append('tcp_setup_inefficiency')
        
    if 'ackdat' in df.columns and 'synack' in df.columns:
        tcp_ack_syn_ratio = df['ackdat'] / (df['synack'] + 1e-10)
        attack_features.append(tcp_ack_syn_ratio)
        feature_names.append('tcp_ack_syn_ratio')

    # D. Jitter Asymmetry (DoS, DDoS, Fuzzers)
    if 'sjit' in df.columns and 'djit' in df.columns:
        jitter_asymmetry = np.abs(df['sjit'] - df['djit']) / (df['sjit'] + df['djit'] + 1e-10)
        attack_features.append(jitter_asymmetry)
        feature_names.append('jitter_asymmetry')

    if attack_features:
        return np.column_stack(attack_features), feature_names
    else:
        return np.array([]).reshape(len(X), 0), []

def select_best_features_with_y(X, y, k=100):
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector

def check_cache_consistency(cache_dir):
    cache_path = Path(cache_dir)
    base_files = ['Xt_tr.joblib', 'Xt_v.joblib', 'Xt_te.joblib']
    missing = [f for f in base_files if not (cache_path / f).exists()]
    if missing:
        print(f"Missing base cache files: {missing}")
        return False
    return True

def run_feature_engineering_pipeline(cfg, n_features=30, method='enhanced'):
    """Optimized feature engineering pipeline - TM SINIFLAR N"""
    cache_dir = Path(cfg.cache_dir)
    
    if not check_cache_consistency(cache_dir):
        raise RuntimeError("Cache files missing. Run preprocessing first.")
    
    # Veri ykle
    Xt_tr = joblib.load(cache_dir / "Xt_tr.joblib")
    Xt_v = joblib.load(cache_dir / "Xt_v.joblib") 
    Xt_te = joblib.load(cache_dir / "Xt_te.joblib")
    y_tr = joblib.load(cache_dir / "y_tr.joblib")
    
    print(f"Original shape: {Xt_tr.shape}")
    
    if method == 'simple':
        # Simple method - sadece PCA
        Xt_tr_final = create_smaller_input_features(Xt_tr, 'pca', n_features, cache_dir)
        Xt_v_final = apply_feature_reduction(Xt_v, 'pca', n_features, cache_dir)
        Xt_te_final = apply_feature_reduction(Xt_te, 'pca', n_features, cache_dir)
        
    elif method == 'enhanced':
        print("Creating OPTIMIZED enhanced features...")
        
        # Original feature names from preprocess meta
        try:
            meta = joblib.load(cache_dir / "preprocess_meta.joblib")
            original_names = meta["num_cols"] + meta["cat_cols"]
        except:
            original_names = [f"feature_{i}" for i in range(Xt_tr.shape[1])]
        
        # Z-score outlier detection
        zscore_count_tr = count_features_with_zscore_above_3(Xt_tr)
        zscore_count_v = count_features_with_zscore_above_3(Xt_v)
        zscore_count_te = count_features_with_zscore_above_3(Xt_te)
        
        extreme_tr = mark_values_in_top_bottom_1_percent(Xt_tr).sum(axis=1)
        extreme_v = mark_values_in_top_bottom_1_percent(Xt_v).sum(axis=1)
        extreme_te = mark_values_in_top_bottom_1_percent(Xt_te).sum(axis=1)
        
        # Core statistical moments
        stat_tr = create_core_statistical_features(Xt_tr)
        stat_v = create_core_statistical_features(Xt_v)
        stat_te = create_core_statistical_features(Xt_te)
        
        # Anomaly scores
        iso_forest = IsolationForest(random_state=42, n_jobs=-1, contamination=0.1)
        iso_scores_tr = iso_forest.fit(Xt_tr).decision_function(Xt_tr)
        iso_scores_v = iso_forest.decision_function(Xt_v)
        iso_scores_te = iso_forest.decision_function(Xt_te)
        
        # Attack-specific features
        attack_tr, attack_names = create_attack_specific_features(Xt_tr, original_names)
        attack_v, _ = create_attack_specific_features(Xt_v, original_names)
        attack_te, _ = create_attack_specific_features(Xt_te, original_names)
        
        # Sadece base features iin SelectKBest (custom features korunur)
        Xt_tr_base = np.column_stack([
            Xt_tr, zscore_count_tr, extreme_tr, stat_tr,
            iso_scores_tr
        ])
        Xt_v_base = np.column_stack([
            Xt_v, zscore_count_v, extreme_v, stat_v,
            iso_scores_v
        ])
        Xt_te_base = np.column_stack([
            Xt_te, zscore_count_te, extreme_te, stat_te,
            iso_scores_te
        ])
        
        num_attack_features = attack_tr.shape[1] if attack_tr.shape[1] > 0 else 0
        features_to_select = max(1, n_features - num_attack_features)
        
        current_base_features = Xt_tr_base.shape[1]
        
        if current_base_features > features_to_select:
            # Sadece base zellikler zerinde seim yap
            Xt_tr_selected_base, selector = select_best_features_with_y(Xt_tr_base, y_tr, k=features_to_select)
            Xt_v_selected_base = selector.transform(Xt_v_base)
            Xt_te_selected_base = selector.transform(Xt_te_base)
            joblib.dump(selector, cache_dir / f'feature_selector_{features_to_select}.joblib')
            print(f"SelectKBest (Base): {current_base_features} -> {features_to_select} features")
        else:
            Xt_tr_selected_base, Xt_v_selected_base, Xt_te_selected_base = Xt_tr_base, Xt_v_base, Xt_te_base
            print(f"No base feature selection needed: {current_base_features} <= {features_to_select}")
        
        # Seilmi zelliklerle korunan attack feature'lar birletir
        if num_attack_features > 0:
            Xt_tr_selected = np.column_stack([Xt_tr_selected_base, attack_tr])
            Xt_v_selected = np.column_stack([Xt_v_selected_base, attack_v])
            Xt_te_selected = np.column_stack([Xt_te_selected_base, attack_te])
            print(f"Protected {num_attack_features} custom attack features. Total: {Xt_tr_selected.shape[1]}")
        else:
            Xt_tr_selected, Xt_v_selected, Xt_te_selected = Xt_tr_selected_base, Xt_v_selected_base, Xt_te_selected_base
            print(f"No custom attack features generated. Total: {Xt_tr_selected.shape[1]}")
        
        # Final features (PCA sadece gerekirse)
        if Xt_tr_selected.shape[1] > n_features:
            Xt_tr_final = create_smaller_input_features(Xt_tr_selected, 'pca', n_features, cache_dir)
            Xt_v_final = apply_feature_reduction(Xt_v_selected, 'pca', n_features, cache_dir)
            Xt_te_final = apply_feature_reduction(Xt_te_selected, 'pca', n_features, cache_dir)
        else:
            # PCA'ya gerek yok, direkt kullan
            Xt_tr_final, Xt_v_final, Xt_te_final = Xt_tr_selected, Xt_v_selected, Xt_te_selected
            print(f"No PCA needed: {Xt_tr_selected.shape[1]} features")
        
        # === CRITICAL: Scale final features ===
        # Baz engineered zellikler (rate gibi) ok byk olabilir ve MLP'yi bozabilir.
        print("Scaling final engineered features...")
        scaler = StandardScaler()
        Xt_tr_final = scaler.fit_transform(Xt_tr_final)
        Xt_v_final = scaler.transform(Xt_v_final)
        Xt_te_final = scaler.transform(Xt_te_final)
        joblib.dump(scaler, cache_dir / f"scaler_enhanced{Xt_tr_final.shape[1]}.joblib")
    
    print(f"Final shape: {Xt_tr_final.shape}")
    
    # Cache kaydet
    actual_features = Xt_tr_final.shape[1]
    suffix = f"_{method}{actual_features}"
    joblib.dump(Xt_tr_final, cache_dir / f"Xt_tr{suffix}.joblib")
    joblib.dump(Xt_v_final, cache_dir / f"Xt_v{suffix}.joblib")
    joblib.dump(Xt_te_final, cache_dir / f"Xt_te{suffix}.joblib")
    
    print(f"Enhanced features saved with suffix: {suffix} (actual features: {actual_features})")
    
    return Xt_tr_final, Xt_v_final, Xt_te_final
