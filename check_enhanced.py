import joblib, numpy as np
from pathlib import Path

required = {
    'communication_efficiency': {'sbytes','dbytes','spkts','dpkts'},
    'activity_intensity':       {'spkts','dpkts','dur'},
    'is_long_duration':         {'dur'},
    'uses_non_standard_port':   {'dsport'},
    'packet_rate':              {'spkts','dur'},
    'byte_rate':                {'sbytes','dur'},
    'payload_asymmetry':        {'sbytes','dbytes'},
    'packet_asymmetry':         {'spkts','dpkts'},
    'avg_packet_size':          {'sbytes','spkts'},
    'is_very_short_duration':   {'dur'},
    'src_port_entropy':         {'sport','srcip'},
    'scans_common_port':        {'dsport'},
    'uses_small_packets':       {'spkts','sbytes'},
    'payload_density':          {'sbytes','spkts'},
    'shellcode_size_range':     {'sbytes'},
    'uses_high_port':           {'dsport'},
    'moderate_throughput':      {'dur','sbytes'},
    's_loss_ratio':             {'sloss','spkts'},
    'd_loss_ratio':             {'dloss','dpkts'},
    'loss_asymmetry':           {'sloss','spkts','dloss','dpkts'},
    'ttl_asymmetry':            {'sttl','dttl'},
    'tcp_setup_inefficiency':   {'tcprtt','dur'},
    'tcp_ack_syn_ratio':        {'ackdat','synack'},
    'jitter_asymmetry':         {'sjit','djit'},
}

for ds in ['unsw', 'cicids']:
    p = Path(ds)
    if not p.exists():
        print(f'{ds}: klasor yok\n'); continue

    meta = joblib.load(p / 'preprocess_meta.joblib')
    raw_cols = set(meta['num_cols'] + meta['cat_cols'])

    # En buyuk enhanced
    candidates = sorted(p.glob('Xt_tr_enhanced*.joblib'))
    enh = joblib.load(candidates[-1]) if candidates else None
    raw = joblib.load(p / 'Xt_tr.joblib')

    print(f'{"="*55}')
    print(f'  {ds.upper()}')
    print(f'{"="*55}')
    print(f'  Raw shape      : {raw.shape}')
    print(f'  Enhanced shape : {enh.shape if enh is not None else "YOK"}')
    print(f'  Raw NaN/Inf    : {np.isnan(raw).sum()} / {np.isinf(raw).sum()}')
    if enh is not None:
        print(f'  Enh NaN/Inf    : {np.isnan(enh).sum()} / {np.isinf(enh).sum()}')
        col_means = np.abs(enh.mean(axis=0))
        print(f'  Scaling OK     : {(col_means < 0.5).sum()}/{enh.shape[1]} sutun |mean|<0.5')

    print(f'\n  Ozniteligin ham kolonlara gore uretim durumu:')
    ok, missing_list = [], []
    for feat, deps in required.items():
        if deps.issubset(raw_cols):
            ok.append(feat)
        else:
            missing_list.append((feat, deps - raw_cols))

    print(f'  Uretilecek   : {len(ok)}/{len(required)}')
    print(f'  Uretilemeyecek: {len(missing_list)}/{len(required)}')
    if missing_list:
        for f, m in missing_list:
            print(f'    - {f}: eksik = {m}')

    # Stat + iso + attack ozellikleri = beklenen pool
    stat_count = 7  # zscore_count, extreme_count, mean, std, skew, kurt, iso_forest
    expected_pool = len(meta['num_cols']) + len(meta['cat_cols']) + stat_count + len(ok)
    actual_enh = enh.shape[1] if enh is not None else 0
    target_n = int(candidates[-1].stem.replace('Xt_tr_enhanced','')) if candidates else 0
    print(f'\n  Beklenen pool  : {expected_pool}  (raw={len(raw_cols)}, stat={stat_count}, attack={len(ok)})')
    print(f'  SelectKBest k  : {target_n}  -> gercekte {min(target_n, expected_pool)} secilmeli')
    print(f'  Gercek enhanced: {actual_enh}  {"OK" if actual_enh == min(target_n, expected_pool) else "UYUMSUZ!"}')
    print()
