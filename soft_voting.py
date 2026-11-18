"""
G3 Soft Voting Ensemble (E3) — CyberSentinel AI Hybrid IDS
==========================================================
Implements the weighted soft-voting scheme adapted from Almuhanna & Dardouri (2025):
  - Combines XGBoost multi-class probabilities with Autoencoder anomaly probability estimates.
  - Weights are assigned based on validation performance on training folds.
  - Evaluated as Experiment E3 for direct empirical comparison against Stacking (E2).
"""

import numpy as np
import config as C
from xgboost_model import G3XGBoostModel
from autoencoder import G3Autoencoder


class G3SoftVotingEnsemble:
    """
    Weighted Soft-Voting ensemble combining XGBoost and Autoencoder.
    """

    def __init__(self, w_xgb: float = 0.8, w_ae: float = 0.2):
        self.w_xgb = w_xgb
        self.w_ae = w_ae
        self.xgb_model: G3XGBoostModel = None
        self.ae_model: G3Autoencoder = None
        self.normal_mse_mean = 0.0
        self.normal_mse_std = 1.0

    def fit(self, X_train_xgb: np.ndarray, X_train_ae: np.ndarray, y_train: np.ndarray):
        """Fit both base models on the training set."""
        # 1. Fit XGBoost
        self.xgb_model = G3XGBoostModel()
        self.xgb_model.fit(X_train_xgb, y_train)

        # 2. Fit Autoencoder on normal flows only
        normal_mask = (y_train == C.NORMAL_CLASS)
        self.ae_model = G3Autoencoder()
        self.ae_model.fit(X_train_ae[normal_mask], verbose=False)

        # Compute normal reconstruction statistics for probability calibration
        train_normal_errors = self.ae_model.reconstruction_error(X_train_ae[normal_mask])
        self.normal_mse_mean = float(np.mean(train_normal_errors))
        self.normal_mse_std = float(np.std(train_normal_errors)) + 1e-6
        return self

    def _ae_to_proba(self, X_ae: np.ndarray) -> np.ndarray:
        """
        Converts Autoencoder reconstruction error into a 4-class pseudo-probability.
        High error -> low P(normal), high P(attack).
        Attack probability is distributed across the 3 attack classes (bruteforce, scan, synflood).
        """
        errors = self.ae_model.reconstruction_error(X_ae)
        # Z-score relative to normal training errors
        z = (errors - self.normal_mse_mean) / self.normal_mse_std
        # Normal probability using sigmoid-like decay
        p_normal = 1.0 / (1.0 + np.exp(np.clip(z - 2.0, -10, 10)))
        p_attack = 1.0 - p_normal

        # Distribute attack probability across 3 attack classes
        # Order: [bruteforce, normal, scan, synflood]
        n_samples = len(X_ae)
        ae_proba = np.zeros((n_samples, 4), dtype=float)

        ae_proba[:, C.CLASS_TO_IDX["normal"]] = p_normal
        for atk in ["bruteforce", "scan", "synflood"]:
            ae_proba[:, C.CLASS_TO_IDX[atk]] = p_attack / 3.0

        return ae_proba

    def predict_proba(self, X_xgb: np.ndarray, X_ae: np.ndarray) -> np.ndarray:
        """Weighted average of predicted probabilities."""
        p_xgb = self.xgb_model.predict_proba(X_xgb)
        p_ae = self._ae_to_proba(X_ae)

        # Normalize weights
        total_w = self.w_xgb + self.w_ae
        w1 = self.w_xgb / total_w
        w2 = self.w_ae / total_w

        return (w1 * p_xgb) + (w2 * p_ae)

    def predict(self, X_xgb: np.ndarray, X_ae: np.ndarray) -> np.ndarray:
        """Class prediction via argmax of ensemble probabilities."""
        proba = self.predict_proba(X_xgb, X_ae)
        best_indices = np.argmax(proba, axis=1)
        return np.array([C.IDX_TO_CLASS[i] for i in best_indices])
