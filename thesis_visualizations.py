import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

# Kendi projenizdeki metodları dahil ediyoruz
from unsw_nb15_pipeline import Config, preprocess_and_cache, apply_smart_resampling

def plot_class_distribution(y_before, y_after, le, save_path="class_dist_comparison.png"):
    plt.figure(figsize=(14, 6))
    
    # Gerçek isimlere çevir
    y_before_labels = le.inverse_transform(y_before)
    y_after_labels = le.inverse_transform(y_after)
    
    # Sınıf sayılarını hesapla
    val_counts_before = pd.Series(y_before_labels).value_counts()
    val_counts_after = pd.Series(y_after_labels).value_counts()
    
    df_plot = pd.DataFrame({
        'Orijinal Veri': val_counts_before,
        'SMOTE + TomekLinks': val_counts_after
    }).fillna(0)
    
    # Grafiği çizdir
    df_plot.plot(kind='bar', figsize=(14,6), colormap='viridis', edgecolor='black')
    plt.title("Sınıf Dağılımı Karşılaştırması: İşlem Öncesi ve Sonrası", fontsize=16)
    plt.xlabel("Saldırı Sınıfları", fontsize=12)
    plt.ylabel("Örnek Sayısı (Log Skalası)", fontsize=12)
    plt.yscale('log') # Azınlık sınıflarını daha iyi görmek için log scale
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Grafik kaydedildi: {save_path}")
    plt.close()

def plot_pca_before_after(X_before, y_before, X_after, y_after, le, save_path="pca_comparison.png", sample_size=10000):
    print("PCA hesaplanıyor, bu işlem birkaç dakika sürebilir...")
    
    # PCA'in çok uzun sürmemesi için alt örneklem alıyoruz
    if len(X_before) > sample_size:
        idx = np.random.choice(len(X_before), sample_size, replace=False)
        X_before, y_before = X_before[idx], y_before[idx]
        
    if len(X_after) > sample_size:
        idx = np.random.choice(len(X_after), sample_size, replace=False)
        X_after, y_after = X_after[idx], y_after[idx]
        
    from sklearn.preprocessing import StandardScaler
    
    # PCA öncesi veriyi standartlaştırmak grafiğin tek bir noktaya sıkışmasını önler
    scaler_before = StandardScaler()
    X_before_scaled = scaler_before.fit_transform(X_before)
    pca_before = PCA(n_components=2, random_state=42)
    X_before_pca = pca_before.fit_transform(X_before_scaled)
    
    scaler_after = StandardScaler()
    X_after_scaled = scaler_after.fit_transform(X_after)
    pca_after = PCA(n_components=2, random_state=42)
    X_after_pca = pca_after.fit_transform(X_after_scaled)
    
    y_before_labels = le.inverse_transform(y_before)
    y_after_labels = le.inverse_transform(y_after)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    sns.scatterplot(x=X_before_pca[:,0], y=X_before_pca[:,1], hue=y_before_labels, ax=axes[0], palette='tab10', s=15, alpha=0.7)
    axes[0].set_title("Orijinal Veri Dağılımı (PCA)", fontsize=14)
    axes[0].legend(loc='best', fontsize='small')
    
    sns.scatterplot(x=X_after_pca[:,0], y=X_after_pca[:,1], hue=y_after_labels, ax=axes[1], palette='tab10', s=15, alpha=0.7)
    axes[1].set_title("SMOTE + TomekLinks ve Öznitelik Mühendisliği Sonrası (PCA)", fontsize=14)
    axes[1].legend(loc='best', fontsize='small')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"PCA Grafiği kaydedildi: {save_path}")
    plt.close()

def plot_confusion_matrix_compare(y_true, y_pred_baseline, y_pred_proposed, le, save_path="cm_comparison.png"):
    cm_base = confusion_matrix(y_true, y_pred_baseline)
    cm_prop = confusion_matrix(y_true, y_pred_proposed)
    
    # Oransal (Normalize) matrise çevirmek için
    cm_base_norm = cm_base.astype('float') / cm_base.sum(axis=1)[:, np.newaxis]
    cm_prop_norm = cm_prop.astype('float') / cm_prop.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.heatmap(cm_base_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[0], 
                xticklabels=le.classes_, yticklabels=le.classes_)
    axes[0].set_title('Baseline Model (Orijinal Dengesiz Veri)', fontsize=14)
    axes[0].set_ylabel('Gerçek Sınıf')
    axes[0].set_xlabel('Tahmin Edilen Sınıf')
    
    sns.heatmap(cm_prop_norm, annot=True, fmt='.2f', cmap='Greens', ax=axes[1], 
                xticklabels=le.classes_, yticklabels=le.classes_)
    axes[1].set_title('Önerilen Yöntem (SMOTE+Tomek+OÖ)', fontsize=14)
    axes[1].set_ylabel('Gerçek Sınıf')
    axes[1].set_xlabel('Tahmin Edilen Sınıf')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Confusion Matrix Grafiği kaydedildi: {save_path}")
    plt.close()

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Tez Görselleştirme Aracı")
    
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument('--unsw', action='store_true', default=True, help='UNSW-NB15 modu (default)')
    mode_group.add_argument('--cicids', action='store_true', help='CICIDS17 modu')
    
    ap.add_argument('--files-glob', type=str, default=r"C:/Users/mbhto/source/repos/UNSW-NB15/UNSWNB15_[0-5].csv")
    ap.add_argument('--features-csv', type=str, default=r"C:/Users/mbhto/source/repos/UNSW-NB15/NUSW-NB15_features.csv")
    ap.add_argument('--cache-dir', type=str, default='.')
    ap.add_argument('--train-csv', type=str, default=None, help='CICIDS train CSV yolu')
    ap.add_argument('--test-csv', type=str, default=None, help='CICIDS test CSV yolu')
    
    args = ap.parse_args()
    
    dataset_mode = 'cicids' if args.cicids else 'unsw'
    cache_mode_name = dataset_mode
    CACHEDIR = Path(args.cache_dir) / cache_mode_name
    
    print(f"Veriler yükleniyor... Mod: {dataset_mode.upper()}")
    
    cfg = Config(
        files_glob='' if args.cicids else args.files_glob,
        features_csv='' if args.cicids else args.features_csv,
        dataset_mode=dataset_mode,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
    )
    cfg.cache_dir = str(CACHEDIR)
    
    # 1. Önceden işlenmiş orijinal verileri ve LabelEncoder'ı çek
    Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te = preprocess_and_cache(cfg)
    le = joblib.load(Path(cfg.cache_dir) / 'label_encoder.joblib')
    
    # y_tr zaten pd.Series veya ndarray olarak geliyor, numpy dizisine çeviriyoruz
    y_tr_arr = np.asarray(y_tr)
    y_te_arr = np.asarray(y_te)
    
    # String labelları integer encoder'a çevir
    y_tr_enc = le.transform(y_tr_arr)
    y_te_enc = le.transform(y_te_arr)
    
    # 1.5. Enhanced (Öznitelik Eklenmiş ve Seçilmiş) Veriyi Yükleme Denemesi
    import glob, re
    enhanced_pattern = str(CACHEDIR / "Xt_tr_enhanced*.joblib")
    enhanced_files = glob.glob(enhanced_pattern)
    
    if enhanced_files:
        def _get_n(file_path):
            m = re.search(r'enhanced(\d+)', file_path)
            return int(m.group(1)) if m else 0
        latest = sorted(enhanced_files, key=_get_n)[-1]
        n_feat = _get_n(latest)
        
        print(f"Öznitelik mühendisliği (Feature Engineering) uygulanmış veri bulundu! (n_features={n_feat})")
        Xt_tr_prop = joblib.load(CACHEDIR / f'Xt_tr_enhanced{n_feat}.joblib')
        Xt_te_prop = joblib.load(CACHEDIR / f'Xt_te_enhanced{n_feat}.joblib')
    else:
        print("Öznitelik mühendisliği verisi bulunamadı. Sadece orijinal özelliklerle SMOTE uygulanacak.")
        Xt_tr_prop = Xt_tr
        Xt_te_prop = Xt_te

    # 2. Resampling (TomekLinks + SMOTE) Uygulanmış Veri
    # Dikkat: Akıllı yeniden örneklemeyi öznitelik mühendisliği UYGULANMIŞ veri (Xt_tr_prop) üzerinde yapıyoruz.
    print("Akıllı Yeniden Örnekleme (TomekLinks + SMOTE) uygulanıyor...")
    X_resampled, y_resampled = apply_smart_resampling(Xt_tr_prop, y_tr_arr, cfg)
    y_resampled_enc = le.transform(y_resampled)
    
    # Görseller için dosya ön eklentisi
    prefix = f"{dataset_mode}_"
    
    # Görsel 1: Sınıf Dağılımı (Bar Chart)
    plot_class_distribution(y_tr_enc, y_resampled_enc, le, save_path=f"{prefix}class_dist_comparison.png")
    
    # Görsel 2: Veri Dağılımı (PCA)
    # PCA'da orijinal Xt_tr ile yeni üretilen X_resampled verilerini kıyaslıyoruz.
    plot_pca_before_after(Xt_tr, y_tr_enc, X_resampled, y_resampled_enc, le, save_path=f"{prefix}pca_comparison.png")
    
    # Görsel 3: Karmaşıklık Matrisi (Confusion Matrix)
    # Hızlıca iki basit XGB modeli eğitip test verisi üzerinde karşılaştırıyoruz
    print("Baseline (Dengesiz, Orijinal Özellikler) XGBoost modeli eğitiliyor...")
    model_base = XGBClassifier(tree_method='hist', n_estimators=50, max_depth=6, random_state=42)
    model_base.fit(Xt_tr, y_tr_enc)
    y_pred_base = model_base.predict(Xt_te) # Orijinal test verisi ile tahmin
    
    print("Önerilen (SMOTE+Tomek + Öznitelik Mühendisliği) XGBoost modeli eğitiliyor...")
    model_prop = XGBClassifier(tree_method='hist', n_estimators=50, max_depth=6, random_state=42)
    model_prop.fit(X_resampled, y_resampled_enc)
    y_pred_prop = model_prop.predict(Xt_te_prop) # Öznitelik eklenmiş test verisi ile tahmin
    
    plot_confusion_matrix_compare(y_te_enc, y_pred_base, y_pred_prop, le, save_path=f"{prefix}cm_comparison.png")
    
    print("Tüm işlemler tamamlandı. Grafikler proje dizininize kaydedildi.")
