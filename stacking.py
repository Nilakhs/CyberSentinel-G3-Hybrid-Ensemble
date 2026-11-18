"""
G3 Stacking Meta-Classifier — CyberSentinel AI Hybrid IDS
==========================================================
Implements the leakage-free stacking pipeline:

  Phase A (Training):
    1. Generate OOF XGBoost probability predictions (3-fold CV on rounds 3,4,5)
    2. Generate OOF Autoencoder meta-features (latent + recon_error) on same folds
    3. Concatenate: 4 (XGB probs) + 8 (AE latent) + 1 (AE recon) = 13 meta-features
    4. Train Logistic Regression meta-classifier on the 13-dim meta-features

  Phase B (Refit on full training data):
    5. Refit XGBoost on all 7941 training flows
    6. Refit Autoencoder on all normal training flows
    (Meta-classifier stays as trained in Phase A — it already saw all training meta-features via OOF)

  Phase C (Test-time inference, evaluated ONCE on round 6):
    7. XGBoost: predict_proba on round 6 → 4-dim
    8. Autoencoder: meta_features on round 6 → 9-dim
    9. Concatenate → 13-dim → meta-classifier → final 4-class prediction

Leakage safeguards:
  - Base models never predict on data they were trained on (OOF)
  - Autoencoder trained only on normal flows in TRAIN folds (not val folds)
  - Meta-classifier never sees test data
  - Anomaly threshold set from train recon errors only
  - Scaler fitted on train data only (in data_loader)
"""

import os
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config as C
from xgboost_model import G3XGBoostModel, generate_oof_xgb_proba
from autoencoder import G3Autoencoder, generate_oof_ae_meta


class G3StackingClassifier:
    """
    Two-level stacking: XGBoost + Autoencoder → Logistic Regression meta-classifier.
    """

    def __init__(self):
        # Base models (refitted on full train data in Phase B)
        self.xgb_model: G3XGBoostModel = None
        self.ae_model: G3Autoencoder   = None
        # Meta-classifier (trained in Phase A on OOF meta-features)
        self.meta_clf: LogisticRegression = None
        self.meta_scaler: StandardScaler  = None  # Optional scaling of meta-features
        # Metadata for saving/loading
        self.classes_ = C.CLASSES

    # ------------------------------------------------------------------
    # Phase A — OOF meta-feature generation + meta-classifier training
    # ------------------------------------------------------------------

    def fit_meta(
        self,
        X_train_xgb: np.ndarray,
        X_train_ae:  np.ndarray,
        y_train:     np.ndarray,
        n_splits:    int = C.OOF_N_SPLITS,
    ) -> dict:
        """
        Generate OOF meta-features and train the meta-classifier.

        Args:
            X_train_xgb  : (n_train, 22) XGBoost features (unscaled)
            X_train_ae   : (n_train, 66) Autoencoder features (scaled by data_loader)
            y_train      : (n_train,) string labels
            n_splits     : number of CV folds

        Returns:
            info dict with OOF accuracy statistics
        """
        print("\n=== Phase A: Generating OOF meta-features ===")

        # --- OOF XGBoost probabilities ---
        print("\n[Stacking] Step 1: OOF XGBoost probability predictions ...")
        oof_xgb_proba, _ = generate_oof_xgb_proba(X_train_xgb, y_train, n_splits=n_splits)

        # --- OOF Autoencoder meta-features ---
        print("\n[Stacking] Step 2: OOF Autoencoder meta-features ...")
        oof_ae_meta, _   = generate_oof_ae_meta(X_train_ae, y_train, n_splits=n_splits)

        # --- Build 13-dim meta-feature matrix ---
        # Columns: [xgb_p_bruteforce, xgb_p_normal, xgb_p_scan, xgb_p_synflood,
        #           ae_latent_0..7, ae_recon_error]
        meta_X = np.concatenate([oof_xgb_proba, oof_ae_meta], axis=1)
        assert meta_X.shape == (len(y_train), C.META_FEATURE_DIM), \
            f"Expected meta shape ({len(y_train)}, {C.META_FEATURE_DIM}), got {meta_X.shape}"

        print(f"\n[Stacking] Meta-feature matrix: {meta_X.shape}")
        print(f"           Columns: {C.XGB_PROBA_DIM} XGB proba + "
              f"{C.AE_LATENT_DIM} AE latent + {C.AE_RECON_DIM} recon_err")

        # --- Train meta-classifier ---
        print("\n[Stacking] Step 3: Training meta-classifier (Logistic Regression) ...")
        self.meta_scaler = StandardScaler()
        meta_X_scaled = self.meta_scaler.fit_transform(meta_X)

        self.meta_clf = LogisticRegression(**C.META_CLF_PARAMS)
        self.meta_clf.fit(meta_X_scaled, y_train)

        # OOF performance of the meta-classifier
        oof_pred = self.meta_clf.predict(meta_X_scaled)
        oof_acc  = float((oof_pred == y_train).mean())

        from sklearn.metrics import f1_score
        oof_f1 = float(f1_score(y_train, oof_pred, average="macro", zero_division=0))
        print(f"\n[Stacking] OOF meta-classifier — accuracy: {oof_acc:.4f}, macro-F1: {oof_f1:.4f}")
        print("  NOTE: OOF accuracy is on training data with meta-scaler fitted on train — "
              "this is an optimistic estimate; true performance is on round 6.")

        return {"oof_acc": oof_acc, "oof_macro_f1": oof_f1,
                "meta_shape": meta_X.shape}

    # ------------------------------------------------------------------
    # Phase B — Refit base models on FULL training data
    # ------------------------------------------------------------------

    def refit_base_models(
        self,
        X_train_xgb: np.ndarray,
        X_train_ae:  np.ndarray,
        y_train:     np.ndarray,
    ):
        """
        Refit XGBoost and Autoencoder on the FULL training set.
        Meta-classifier is NOT changed here.
        """
        print("\n=== Phase B: Refitting base models on full training data ===")

        # Refit XGBoost
        print("[Stacking] Refitting XGBoost on all training flows ...")
        self.xgb_model = G3XGBoostModel()
        self.xgb_model.fit(X_train_xgb, y_train)
        print("[Stacking] XGBoost refit complete.")

        # Refit Autoencoder on NORMAL flows only
        normal_mask = y_train == C.NORMAL_CLASS
        X_ae_normal = X_train_ae[normal_mask]
        print(f"[Stacking] Refitting Autoencoder on {normal_mask.sum()} normal flows ...")
        self.ae_model = G3Autoencoder()
        self.ae_model.fit(X_ae_normal, verbose=True)
        print("[Stacking] Autoencoder refit complete.")

    # ------------------------------------------------------------------
    # Phase C — Inference (test time)
    # ------------------------------------------------------------------

    def _build_meta_features(
        self,
        X_xgb: np.ndarray,
        X_ae:  np.ndarray,
    ) -> np.ndarray:
        """Build 13-dim meta-feature matrix from base model outputs."""
        xgb_proba = self.xgb_model.predict_proba(X_xgb)   # (n, 4)
        ae_meta   = self.ae_model.meta_features(X_ae)       # (n, 9)
        return np.concatenate([xgb_proba, ae_meta], axis=1) # (n, 13)

    def predict(
        self,
        X_xgb: np.ndarray,
        X_ae:  np.ndarray,
    ) -> np.ndarray:
        """Return string class predictions via stacking pipeline."""
        meta_X        = self._build_meta_features(X_xgb, X_ae)
        meta_X_scaled = self.meta_scaler.transform(meta_X)
        return self.meta_clf.predict(meta_X_scaled)

    def predict_proba(
        self,
        X_xgb: np.ndarray,
        X_ae:  np.ndarray,
    ) -> np.ndarray:
        """Return (n, 4) class probabilities from meta-classifier."""
        meta_X        = self._build_meta_features(X_xgb, X_ae)
        meta_X_scaled = self.meta_scaler.transform(meta_X)
        return self.meta_clf.predict_proba(meta_X_scaled)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, dirpath: str = None):
        dirpath = dirpath or C.MODELS_DIR
        self.xgb_model.save(os.path.join(dirpath, "g3_xgb_full.json"))
        self.ae_model.save(os.path.join(dirpath, "g3_ae_full.pt"))
        pkl = {
            "meta_clf":    self.meta_clf,
            "meta_scaler": self.meta_scaler,
            "classes":     self.classes_,
        }
        with open(os.path.join(dirpath, "g3_stacking_meta.pkl"), "wb") as f:
            pickle.dump(pkl, f)
        print(f"[Stacking] All components saved to {dirpath}/")

    def load(self, dirpath: str = None) -> "G3StackingClassifier":
        dirpath = dirpath or C.MODELS_DIR
        self.xgb_model = G3XGBoostModel()
        self.xgb_model.load(os.path.join(dirpath, "g3_xgb_full.json"))
        self.ae_model = G3Autoencoder()
        self.ae_model.load(os.path.join(dirpath, "g3_ae_full.pt"))
        with open(os.path.join(dirpath, "g3_stacking_meta.pkl"), "rb") as f:
            pkl = pickle.load(f)
        self.meta_clf    = pkl["meta_clf"]
        self.meta_scaler = pkl["meta_scaler"]
        self.classes_    = pkl["classes"]
        print("[Stacking] All components loaded.")
        return self
