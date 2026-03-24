import argparse, joblib, os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def _info(m): print(f"[Info]  {m}")
def _stage(m): print(f"[Stage] {m}")

class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def run_autoencoder_anomaly_scorer(cache_dir, n_features=50, epochs=20, batch_size=512):
    _stage("Autoencoder Anomaly Scorer Training (PyTorch)")
    
    cdir = Path(cache_dir)
    le = joblib.load(cdir / 'label_encoder.joblib')
    classes = le.classes_.astype(str)
    
    try:
        Xt_tr = joblib.load(cdir / f'Xt_tr_enhanced{n_features}.joblib')
        Xt_v  = joblib.load(cdir / f'Xt_v_enhanced{n_features}.joblib')
        Xt_te = joblib.load(cdir / f'Xt_te_enhanced{n_features}.joblib')
        _info("Enhanced features yuklendi.")
    except Exception:
        Xt_tr = joblib.load(cdir / 'Xt_tr.joblib')
        Xt_v  = joblib.load(cdir / 'Xt_v.joblib')
        Xt_te = joblib.load(cdir / 'Xt_te.joblib')
        _info("Orijinal features yuklendi.")

    y_tr = joblib.load(cdir / 'y_tr.joblib')
    
    normal_cls_name = 'normal'
    if normal_cls_name not in classes:
        raise ValueError("Normal class bulunamadi!")
        
    normal_encoded = le.transform([normal_cls_name])[0]
    y_tr_enc = le.transform(np.asarray(y_tr))
    
    normal_indices = np.where(y_tr_enc == normal_encoded)[0]
    np.random.seed(42)
    np.random.shuffle(normal_indices)
    
    # Validation split
    val_split = int(0.1 * len(normal_indices))
    val_idx = normal_indices[:val_split]
    train_idx = normal_indices[val_split:]
    
    X_train_normal = Xt_tr[train_idx].astype(np.float32)
    X_val_normal = Xt_tr[val_idx].astype(np.float32)
    
    _info(f"Sadece NORMAL Egitim Verisi: {X_train_normal.shape[0]}")
    _info(f"NORMAL Validation Verisi: {X_val_normal.shape[0]}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _info(f"Kullanilan cihaz: {device}")
    
    model = AnomalyAutoencoder(Xt_tr.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_normal)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val_normal)), batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cdir / 'autoencoder_anomaly_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                _info(f"Early stopping at epoch {epoch+1}")
                break
                
    _info("En iyi Autoencoder modeli yukleniyor.")
    model.load_state_dict(torch.load(cdir / 'autoencoder_anomaly_model.pth', weights_only=True))
    model.eval()

    _stage("Generating Anomaly Scores (MSE) for all sets")
    
    def get_mse(data, batch_sz=4096):
        loader = DataLoader(TensorDataset(torch.tensor(data.astype(np.float32))), batch_size=batch_sz, shuffle=False)
        mses = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                mse = torch.mean((batch_x - outputs)**2, dim=1)
                mses.append(mse.cpu().numpy())
        return np.concatenate(mses)
        
    mse_tr = get_mse(Xt_tr)
    mse_v  = get_mse(Xt_v)
    mse_te = get_mse(Xt_te)
    
    _info(f"MSE Train Range: {mse_tr.min():.4f} - {mse_tr.max():.4f}")
    
    # === SINIF BAZLI ANOMALY SCORE STATSTKLER ===
    _stage("Autoencoder Per-Class Anomaly Score Statistics (Train Set)")
    y_tr_arr = np.asarray(y_tr)
    print(f"\n  {'Class':<20} {'Mean MSE':>12} {'Median MSE':>12} {'Std MSE':>12} {'Count':>8}")
    print(f"  {'-'*66}")
    
    for cls_name in sorted(classes):
        cls_mask = (y_tr_arr == cls_name)
        if cls_mask.sum() == 0:
            continue
        cls_mse = mse_tr[cls_mask]
        print(f"  {cls_name:<20} {cls_mse.mean():>12.6f} {np.median(cls_mse):>12.6f} {cls_mse.std():>12.6f} {cls_mask.sum():>8}")
    
    # Normal vs Attack ayrm
    normal_mask = (y_tr_arr == 'normal')
    attack_mask = ~normal_mask
    if normal_mask.sum() > 0 and attack_mask.sum() > 0:
        normal_mse_mean = mse_tr[normal_mask].mean()
        attack_mse_mean = mse_tr[attack_mask].mean()
        _info(f"Normal MSE Mean: {normal_mse_mean:.6f} | Attack MSE Mean: {attack_mse_mean:.6f} | Separation Ratio: {attack_mse_mean/max(normal_mse_mean, 1e-10):.2f}x")
    
    Xt_tr_aug = np.column_stack((Xt_tr, mse_tr))
    Xt_v_aug  = np.column_stack((Xt_v, mse_v))
    Xt_te_aug = np.column_stack((Xt_te, mse_te))
    
    aug_name = f"_augmented{n_features+1}"
    joblib.dump(Xt_tr_aug, cdir / f"Xt_tr{aug_name}.joblib")
    joblib.dump(Xt_v_aug,  cdir / f"Xt_v{aug_name}.joblib")
    joblib.dump(Xt_te_aug, cdir / f"Xt_te{aug_name}.joblib")
    
    _info(f"Datasetler kaydedildi: Xt_*_augmented{n_features+1}.joblib")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', type=str, default='.')
    ap.add_argument('--n-features', type=int, default=50)
    args = ap.parse_args()
    
    run_autoencoder_anomaly_scorer(args.cache_dir, n_features=args.n_features)
