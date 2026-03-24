"""
Tail-aware Autoencoder for UNSW-NB15 (minority-focused)
-------------------------------------------------------
 AE tail_features mask'i ile secilmis kolonlarda egitilir.
 Egitim sonrasi model (ae.pt) + meta (ae_meta.npz) kaydedilir.
 encode_with_autoencoder() ile latent embedding cikarilabilir.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import argparse, joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def _info(msg): print(f"[Info]  {msg}")

class MinorityAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, dropout=0.2):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

def train_autoencoder(
    X_train,
    *,
    input_dim: int,
    latent_dim: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 512,
    cache_dir: str = ".",
    dropout: float = 0.2,
):
    """X_train: numpy array (float32); input_dim: feature sayisi"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    model = MinorityAutoencoder(input_dim=input_dim, latent_dim=latent_dim, dropout=dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # veriyi tensora cevir
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    X_train = X_train.astype(np.float32, copy=False)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # egitim
    for ep in range(1, epochs+1):
        model.train(); total = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            xhat, _ = model(xb)
            loss = loss_fn(xhat, xb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        _info(f"[AE] epoch {ep}/{epochs} loss={total/len(ds):.4f}")

    # kaydet
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cdir / "ae.pt")
    np.savez(cdir / "ae_meta.npz", latent_dim=int(latent_dim), input_dim=int(input_dim))
    _info("Autoencoder model + meta kaydedildi")

    return model

# ---------------------------
# Encode helper
# ---------------------------
def encode_with_autoencoder(X, cache_dir="."):
    cdir = Path(cache_dir)
    meta = np.load(cdir/"ae_meta.npz", allow_pickle=True)
    latent_dim = int(meta["latent_dim"])
    input_dim = int(meta["input_dim"])
    model = MinorityAutoencoder(input_dim, latent_dim)
    model.load_state_dict(torch.load(cdir/"ae.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        z = model.encoder(torch.tensor(X, dtype=torch.float32))
    return z.numpy()

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', type=str, default='.')
    ap.add_argument('--latent-dim', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--n-features', type=int, default=50)
    args = ap.parse_args()

    cdir = Path(args.cache_dir)
    # Enhanced features ncelii
    enhanced_file = cdir / f"Xt_tr_enhanced{args.n_features}.joblib"
    if enhanced_file.exists():
        Xt_tr = joblib.load(enhanced_file)
        _info(f"Using enhanced features: {Xt_tr.shape}")
    else:
        Xt_tr = joblib.load(cdir/"Xt_tr.joblib")
        _info(f"Using original features: {Xt_tr.shape}")
    input_dim = Xt_tr.shape[1]

    train_autoencoder(
        Xt_tr,
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.bs,
        cache_dir=str(cdir)
    )


if __name__ == '__main__':
    main()
