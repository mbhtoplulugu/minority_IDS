"""
SMOTE-ENN Configuration
Pipeline configuration for SMOTE + ENN resampling
"""
from dataclasses import dataclass

@dataclass
class Config:
    """Pipeline configuration for SMOTE-ENN resampling"""
    files_glob: str
    features_csv: str
    cache_dir: str = '.'
    data_cache: str = "cache_all_data.pkl"
    xy_cache: str = "cache_xy.pkl"
    target: str = "attack_cat"
    random_state: int = 42
    valid_size: float = 0.15
    test_size: float = 0.15
    smote_ratio: float = 0.6
    smote_min_floor: int = 500
    rus_frac: float = 0.03
    use_gpu: bool = True
    verbose: bool = True
    n_jobs: int = -1
    batch_size: int = 1024
    lr: float = 1e-3
    epochs: int = 10