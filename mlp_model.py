import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def get_device(use_gpu=True):
    return torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

class EnhancedMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        return self.net(x)

def run_mlp_oof(cfg, n_features=50):
    from unsw_nb15_pipeline import apply_smart_resampling, _info, _stage, save_proba_npz
    from experts import _load_cache
    
    _stage("MLP 5-Fold OOF Training")
    cache_dir = Path(cfg.cache_dir)
    
    Xt_tr, y_tr, Xt_v, y_v, Xt_te, y_te, le = _load_cache(cache_dir, n_features)
    
    device = get_device(getattr(cfg, 'use_gpu', True))
    _info(f"MLP using device: {device}")
    
    y_tr_arr = np.asarray(y_tr)
    y_v_arr = np.asarray(y_v)
    y_te_arr = np.asarray(y_te)
    
    yv_enc = le.transform(y_v_arr)
    yte_enc = le.transform(y_te_arr)
    
    num_classes = len(le.classes_)
    oof_train = np.zeros((len(y_tr), num_classes), dtype=np.float32)
    valid_preds = np.zeros((len(y_v), num_classes), dtype=np.float32)
    test_preds = np.zeros((len(y_te), num_classes), dtype=np.float32)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
    
    batch_size = getattr(cfg, 'batch_size', 1024)
    epochs = 40 # Up to 40 epochs with early stopping
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(Xt_tr, y_tr_arr)):
        _info(f"Fold {fold+1}/5 MLP training...")
        X_fold_tr, y_fold_tr = Xt_tr[trn_idx], y_tr_arr[trn_idx]
        X_fold_val, y_fold_val = Xt_tr[val_idx], y_tr_arr[val_idx]
        
        X_res, y_res = apply_smart_resampling(X_fold_tr, y_fold_tr, cfg)
        yres_enc = le.transform(np.asarray(y_res))
        yval_enc = le.transform(np.asarray(y_fold_val))
        
        t_X_tr = torch.tensor(X_res, dtype=torch.float32)
        t_y_tr = torch.tensor(yres_enc, dtype=torch.long)
        t_X_val = torch.tensor(X_fold_val, dtype=torch.float32)
        t_y_val = torch.tensor(yval_enc, dtype=torch.long)
        
        train_loader = DataLoader(TensorDataset(t_X_tr, t_y_tr), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(t_X_val, t_y_val), batch_size=batch_size*2, shuffle=False)
        
        model = EnhancedMLP(Xt_tr.shape[1], num_classes).to(device)
        
        # Calculate class weights for imbalanced data
        counts = pd.Series(yres_enc).value_counts().sort_index().values
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * num_classes
        t_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=t_weights)
        optimizer = optim.AdamW(model.parameters(), lr=getattr(cfg, 'lr', 1e-3), weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        early_stop_patience = 7
        
        for ep in range(epochs):
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * bx.size(0)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits = model(bx)
                    loss = criterion(logits, by)
                    val_loss += loss.item() * bx.size(0)
            
            val_loss /= len(t_y_val)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break
        
        model.load_state_dict(best_model_state)
        model.eval()
        
        batch_size_inf = batch_size * 2
        def predict_loader(X_np):
            ds = TensorDataset(torch.tensor(X_np, dtype=torch.float32))
            dl = DataLoader(ds, batch_size=batch_size_inf, shuffle=False)
            preds = []
            with torch.no_grad():
                for (bx,) in dl:
                    bx = bx.to(device)
                    probs = torch.softmax(model(bx), dim=1)
                    preds.append(probs.cpu().numpy())
            return np.vstack(preds)
        
        oof_train[val_idx] = predict_loader(X_fold_val)
        valid_preds += predict_loader(Xt_v) / skf.n_splits
        test_preds += predict_loader(Xt_te) / skf.n_splits
        
    save_proba_npz(str(cache_dir/"proba_mlp_oof_train.npz"), oof_train, le=le)
    save_proba_npz(str(cache_dir/"proba_mlp_oof_valid.npz"), valid_preds, le=le)
    save_proba_npz(str(cache_dir/"proba_mlp_oof_test.npz"), test_preds, le=le)
    
    # === SINIF BAZLI SONULARI YAZDIR ===
    from sklearn.metrics import classification_report
    _stage("MLP Per-Class Results (Validation Set)")
    yhat_v = valid_preds.argmax(axis=1)
    print(classification_report(yv_enc, yhat_v, target_names=le.classes_, digits=4, zero_division=0))
    
    _stage("MLP Per-Class Results (Test Set)")
    yhat_te = test_preds.argmax(axis=1)
    print(classification_report(yte_enc, yhat_te, target_names=le.classes_, digits=4, zero_division=0))
    
    # Macro F1 zet
    _, _, f1_val, _ = precision_recall_fscore_support(yv_enc, yhat_v, average='macro', zero_division=0)
    _, _, f1_te, _ = precision_recall_fscore_support(yte_enc, yhat_te, average='macro', zero_division=0)
    _info(f"MLP Macro F1 -> Valid: {f1_val:.4f} | Test: {f1_te:.4f}")
    
    _info("MLP OOF probabilities successfully saved.")
