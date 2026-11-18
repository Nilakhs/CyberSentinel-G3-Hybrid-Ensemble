"""
G3 Autoencoder — CyberSentinel AI Hybrid IDS
=============================================
Dense Autoencoder for anomaly detection and latent feature extraction.

Architecture:
  Encoder: 66 → 32 → 16 → 8 (latent)
  Decoder: 8  → 16 → 32 → 66 (reconstruction)

Role in the G3 pipeline:
  - Trained ONLY on NORMAL training flows
  - Produces two types of meta-features:
    (a) latent vector (8-dim): compressed representation of each flow
    (b) reconstruction error (1-dim scalar): high error = anomalous
  - Both are fed to the stacking meta-classifier

Design rules:
  - Scaler is fitted on TRAIN data in data_loader.py — not here
  - Training on OOF normal-only sub-splits avoids leakage
  - Anomaly threshold set from TRAIN reconstruction errors only
  - Uses PyTorch (pure Python, no TF/Keras dependency)

Dependencies: torch, numpy, scikit-learn
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import config as C


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class _AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dims: list, dropout_rate: float):
        super().__init__()
        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout_rate)]
            prev = h
        enc_layers += [nn.Linear(prev, latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (symmetric)
        dec_layers = [nn.Linear(latent_dim, hidden_dims[-1]), nn.ReLU(), nn.Dropout(dropout_rate)]
        for h in reversed(hidden_dims[:-1]):
            dec_layers += [nn.Linear(hidden_dims[hidden_dims.index(h) + 1] if False else
                            (dec_layers[-3].out_features if hasattr(dec_layers[-3], 'out_features') else hidden_dims[-1]),
                            h), nn.ReLU(), nn.Dropout(dropout_rate)]
        # Build decoder cleanly
        dec_dims = [latent_dim] + list(reversed(hidden_dims)) + [input_dim]
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.ReLU())
                dec_layers.append(nn.Dropout(dropout_rate))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class G3Autoencoder:
    """
    Wrapper around _AutoencoderNet with fit/transform/reconstruction_error API.
    """

    def __init__(self, params: dict = None):
        p = params or C.AE_PARAMS.copy()
        self.input_dim   = p["input_dim"]
        self.latent_dim  = p["latent_dim"]
        self.hidden_dims = p["hidden_dims"]
        self.dropout     = p["dropout_rate"]
        self.lr          = p["learning_rate"]
        self.batch_size  = p["batch_size"]
        self.epochs      = p["epochs"]
        self.patience    = p["early_stop_patience"]

        self.net = None
        self.device = torch.device("cpu")  # CPU-only for portability
        self.threshold_ = None             # Set from training recon errors

    def _build_net(self):
        return _AutoencoderNet(
            self.input_dim, self.latent_dim,
            self.hidden_dims, self.dropout
        ).to(self.device)

    def fit(self, X_normal: np.ndarray, verbose: bool = True) -> "G3Autoencoder":
        """
        Train on NORMAL flows only (X_normal already scaled).
        Sets self.threshold_ from 95th percentile of training reconstruction errors.
        """
        torch.manual_seed(C.SEED)
        self.net = self._build_net()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn   = nn.MSELoss()

        X_t = torch.tensor(X_normal, dtype=torch.float32)
        ds  = TensorDataset(X_t, X_t)
        dl  = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                         generator=torch.Generator().manual_seed(C.SEED))

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, _ in dl:
                xb = xb.to(self.device)
                recon, _ = self.net(xb)
                loss = loss_fn(recon, xb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(X_normal)

            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                patience_counter += 1

            if verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(f"  [AE] Epoch {epoch+1:3d}/{self.epochs} | "
                      f"loss = {epoch_loss:.6f} | patience {patience_counter}/{self.patience}")

            if patience_counter >= self.patience:
                if verbose:
                    print(f"  [AE] Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        # Set threshold from TRAINING reconstruction errors (percentile)
        recon_errors = self.reconstruction_error(X_normal)
        self.threshold_ = float(np.percentile(recon_errors, 95))
        print(f"  [AE] Anomaly threshold (95th pct of normal train errors): {self.threshold_:.6f}")
        return self

    @torch.no_grad()
    def _forward_batch(self, X: np.ndarray):
        """Run forward pass, return (reconstructions, latent_vecs) as numpy."""
        assert self.net is not None, "Call fit() first"
        self.net.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        recon, latent = self.net(X_t)
        return recon.cpu().numpy(), latent.cpu().numpy()

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Per-sample MSE reconstruction error. Shape: (n_samples,)."""
        recon, _ = self._forward_batch(X)
        return np.mean((X - recon) ** 2, axis=1)

    def latent_features(self, X: np.ndarray) -> np.ndarray:
        """Bottleneck latent vector. Shape: (n_samples, latent_dim)."""
        _, latent = self._forward_batch(X)
        return latent

    def meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        Combined meta-features for stacking: latent (8) + recon_error (1) = 9 features.
        Shape: (n_samples, 9)
        """
        recon, latent = self._forward_batch(X)
        recon_err = np.mean((X - recon) ** 2, axis=1, keepdims=True)
        return np.concatenate([latent, recon_err], axis=1)

    def predict_anomaly(self, X: np.ndarray) -> np.ndarray:
        """
        Binary anomaly flag: True if reconstruction error > threshold_.
        Not used in stacking — only for standalone E1 evaluation.
        """
        assert self.threshold_ is not None, "Call fit() first"
        return self.reconstruction_error(X) > self.threshold_

    def save(self, path: str = None):
        path = path or os.path.join(C.MODELS_DIR, "g3_autoencoder.pt")
        checkpoint = {
            "state_dict": self.net.state_dict(),
            "params": {
                "input_dim": self.input_dim, "latent_dim": self.latent_dim,
                "hidden_dims": self.hidden_dims, "dropout_rate": self.dropout,
                "learning_rate": self.lr, "batch_size": self.batch_size,
                "epochs": self.epochs, "early_stop_patience": self.patience,
            },
            "threshold": self.threshold_,
        }
        torch.save(checkpoint, path)
        print(f"[AE] Model saved to {path}")

    def load(self, path: str = None) -> "G3Autoencoder":
        path = path or os.path.join(C.MODELS_DIR, "g3_autoencoder.pt")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.net = self._build_net()
        self.net.load_state_dict(ckpt["state_dict"])
        self.threshold_ = ckpt.get("threshold")
        print(f"[AE] Model loaded from {path}, threshold = {self.threshold_:.6f}")
        return self


# ---------------------------------------------------------------------------
# OOF Autoencoder meta-features for stacking
# ---------------------------------------------------------------------------

def generate_oof_ae_meta(
    X_train_ae: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = C.OOF_N_SPLITS,
    seed: int = C.SEED,
) -> tuple[np.ndarray, list]:
    """
    Generate out-of-fold Autoencoder meta-features for stacking.

    LEAKAGE-FREE:
    - Autoencoder is trained on NORMAL flows in the TRAINING folds only.
    - Meta-features are extracted for the VALIDATION fold.
    - The meta-classifier sees no test data during this stage.

    Returns:
        oof_meta  : np.ndarray (n_train, 9) — latent(8) + recon_error(1)
        fold_aes  : list of fitted G3Autoencoder (one per fold)
    """
    n = len(X_train_ae)
    oof_meta  = np.zeros((n, C.AE_LATENT_DIM + C.AE_RECON_DIM), dtype=float)
    fold_aes  = []

    # Use same folds as XGBoost OOF (same seed + n_splits)
    y_int = np.array([C.CLASS_TO_IDX[c] for c in y_train], dtype=int)
    skf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_ae, y_int)):
        y_train_fold = y_train[train_idx]
        normal_mask  = y_train_fold == C.NORMAL_CLASS
        X_ae_normal  = X_train_ae[train_idx][normal_mask]

        print(f"  [AE OOF] Fold {fold_idx+1}/{n_splits}: "
              f"training on {normal_mask.sum()} normal flows ...")

        ae = G3Autoencoder()
        ae.fit(X_ae_normal, verbose=False)
        meta = ae.meta_features(X_train_ae[val_idx])
        oof_meta[val_idx] = meta
        fold_aes.append(ae)

        # Quick sanity
        recon_err_val = meta[:, -1]
        print(f"  [AE OOF] Fold {fold_idx+1}: val recon_error mean = {recon_err_val.mean():.6f} "
              f"(normal rows mean = {recon_err_val[y_train[val_idx] == C.NORMAL_CLASS].mean():.6f}, "
              f"attack rows mean = {recon_err_val[y_train[val_idx] != C.NORMAL_CLASS].mean():.6f})")

    return oof_meta, fold_aes


if __name__ == "__main__":
    from data_loader import load_splits, get_normal_mask
    _, y_tr, _, y_te, X_tr_ae, X_te_ae, _ = load_splits()

    normal_mask = get_normal_mask(y_tr)
    X_normal_train = X_tr_ae[normal_mask]
    print(f"\nTraining AE on {X_normal_train.shape[0]} normal flows, {X_normal_train.shape[1]} features ...")

    ae = G3Autoencoder()
    ae.fit(X_normal_train)

    meta_tr = ae.meta_features(X_tr_ae)
    meta_te = ae.meta_features(X_te_ae)
    print(f"\nMeta-features — Train: {meta_tr.shape}, Test: {meta_te.shape}")

    normal_err   = ae.reconstruction_error(X_te_ae[y_te == C.NORMAL_CLASS]).mean()
    attack_err   = ae.reconstruction_error(X_te_ae[y_te != C.NORMAL_CLASS]).mean()
    print(f"Test recon error — normal: {normal_err:.6f} | attacks: {attack_err:.6f}")
    print(f"Separation ratio: {attack_err / (normal_err + 1e-9):.2f}x")
