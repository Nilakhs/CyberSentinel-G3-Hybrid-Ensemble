"""
G3 XGBoost Model — CyberSentinel AI Hybrid IDS
===============================================
Wraps the XGBoost classifier for the G3 stacking pipeline.

Role in the architecture:
  - Branch 1 of the hybrid system
  - Trained with out-of-fold (OOF) strategy to generate LEAKAGE-FREE probability
    vectors for the stacking meta-classifier
  - Refitted on full training data for final inference

Uses: 22 selected features (config.FEATURES_XGB = SELECTED_22_FEATURES)
"""

import os
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import config as C


class G3XGBoostModel:
    """
    XGBoost classifier for the G3 hybrid pipeline.

    Key design decisions:
    - Uses integer class encoding (XGBoost requirement)
    - class_weight is handled via scale_pos_weight or sample_weight for
      bruteforce minority class
    - Provides both predict() and predict_proba() methods
    """

    def __init__(self, params: dict = None):
        self.params = params or C.XGB_PARAMS.copy()
        self.model = None
        self.classes_ = C.CLASSES  # alphabetical: bruteforce, normal, scan, synflood

    def _make_model(self):
        return XGBClassifier(**self.params)

    def _encode(self, y: np.ndarray) -> np.ndarray:
        return np.array([C.CLASS_TO_IDX[c] for c in y], dtype=int)

    def _decode(self, y_int: np.ndarray) -> np.ndarray:
        return np.array([C.IDX_TO_CLASS[i] for i in y_int])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit on (X, y_strings). Encodes labels internally."""
        y_int = self._encode(y)
        self.model = self._make_model()
        self.model.fit(X, y_int)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return string class labels."""
        assert self.model is not None, "Call fit() first"
        return self._decode(self.model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return (n_samples, 4) probability matrix.
        Column order matches C.CLASSES (alphabetical).
        """
        assert self.model is not None, "Call fit() first"
        return self.model.predict_proba(X)

    def save(self, path: str = None):
        path = path or os.path.join(C.MODELS_DIR, "g3_xgb.json")
        self.model.save_model(path)
        print(f"[XGBoost] Model saved to {path}")

    def load(self, path: str = None):
        path = path or os.path.join(C.MODELS_DIR, "g3_xgb.json")
        self.model = self._make_model()
        self.model.load_model(path)
        return self


def generate_oof_xgb_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = C.OOF_N_SPLITS,
    seed: int = C.SEED,
) -> tuple[np.ndarray, list]:
    """
    Generate out-of-fold XGBoost probability predictions for stacking.

    LEAKAGE-FREE by design:
    - Each fold predicts on data NOT seen during that fold's training.
    - The meta-classifier is later trained on these OOF predictions.

    Returns:
        oof_proba : np.ndarray (n_train, 4) — OOF probability matrix
        fold_models : list of fitted G3XGBoostModel (one per fold)
    """
    n = len(X_train)
    oof_proba = np.zeros((n, len(C.CLASSES)), dtype=float)
    fold_models = []

    # Encode labels for stratification
    y_int = np.array([C.CLASS_TO_IDX[c] for c in y_train], dtype=int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_int)):
        X_tr_fold = X_train[train_idx]
        y_tr_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]

        m = G3XGBoostModel()
        m.fit(X_tr_fold, y_tr_fold)
        proba = m.predict_proba(X_val_fold)
        oof_proba[val_idx] = proba
        fold_models.append(m)

        val_labels = y_train[val_idx]
        val_pred   = m.predict(X_val_fold)
        fold_acc   = float((val_pred == val_labels).mean())
        print(f"  [XGB OOF] Fold {fold_idx+1}/{n_splits}: val acc = {fold_acc:.4f}")

    overall_acc = float(
        (np.argmax(oof_proba, axis=1) ==
         np.array([C.CLASS_TO_IDX[c] for c in y_train])).mean()
    )
    print(f"  [XGB OOF] Overall OOF accuracy = {overall_acc:.4f}")
    return oof_proba, fold_models


if __name__ == "__main__":
    # Quick smoke test — requires data_loader
    from data_loader import load_splits
    X_tr_xgb, y_tr, X_te_xgb, y_te, _, _, _ = load_splits()

    print("Testing XGBoost fit + predict...")
    m = G3XGBoostModel()
    m.fit(X_tr_xgb, y_tr)
    y_pred = m.predict(X_te_xgb)
    acc = float((y_pred == y_te).mean())
    print(f"Test accuracy: {acc:.4f}")

    proba = m.predict_proba(X_te_xgb)
    print(f"Proba shape: {proba.shape}, sum check (row 0): {proba[0].sum():.4f}")
