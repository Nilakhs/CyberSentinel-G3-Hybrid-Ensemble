"""
G3 Data Loader — CyberSentinel AI Hybrid IDS
=============================================
Loads and splits the dataset according to the strict rules:
  - TRAIN: rounds 3, 4, 5
  - TEST: round 6
  - Features: use config.FEATURES_XGB or config.FEATURES_AE
  - NEVER use ENVELOPE columns as model inputs
  - NEVER use test-round data for fitting scalers, imputers, or thresholds

Usage:
    from data_loader import load_splits
    (X_train, y_train, X_test, y_test,
     X_train_ae, X_test_ae) = load_splits()
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import config as C


def load_raw() -> pd.DataFrame:
    """Load all_flows.csv and validate its structure."""
    df = pd.read_csv(C.ALL_FLOWS_CSV)

    # Validate shape
    assert ROUND_COL in df.columns, f"Missing column: {C.ROUND_COL}"
    assert LABEL_COL in df.columns, f"Missing column: {C.LABEL_COL}"

    # All 66 features must be present
    missing = [f for f in C.ALL_66_FEATURES if f not in df.columns]
    assert not missing, f"Missing feature columns: {missing}"

    # Envelope columns should exist (but are NOT used as features)
    for col in C.ENVELOPE:
        if col not in df.columns:
            print(f"[WARNING] Envelope column '{col}' not found — that's OK if data was pre-stripped.")

    print(f"[data_loader] Loaded {len(df):,} rows × {len(df.columns)} cols from all_flows.csv")
    print(f"[data_loader] Label distribution:\n{df[C.LABEL_COL].value_counts().to_string()}")
    print(f"[data_loader] Round distribution:\n{df[C.ROUND_COL].value_counts().sort_index().to_string()}")
    return df

ROUND_COL = C.ROUND_COL
LABEL_COL = C.LABEL_COL


def split_df(df: pd.DataFrame):
    """Split into train (rounds 3,4,5) and test (round 6) DataFrames."""
    train_df = df[df[C.ROUND_COL].isin(C.TRAIN_ROUNDS)].reset_index(drop=True)
    test_df  = df[df[C.ROUND_COL] == C.TEST_ROUND].reset_index(drop=True)
    print(f"[data_loader] Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    print(f"[data_loader] Train label dist: {dict(train_df[C.LABEL_COL].value_counts())}")
    print(f"[data_loader] Test  label dist: {dict(test_df[C.LABEL_COL].value_counts())}")
    return train_df, test_df


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit StandardScaler on TRAINING data only. Never fit on test data."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def load_splits(scale: bool = True):
    """
    Main entry point.

    Returns:
        X_train_xgb  : np.ndarray (n_train, 22) — for XGBoost (selected features)
        y_train      : np.ndarray (n_train,)    — string labels
        X_test_xgb   : np.ndarray (n_test, 22)  — for XGBoost
        y_test       : np.ndarray (n_test,)     — string labels
        X_train_ae   : np.ndarray (n_train, 66) — scaled, for Autoencoder
        X_test_ae    : np.ndarray (n_test, 66)  — scaled with TRAIN scaler

    NOTE: Scaler is fitted on TRAIN only and applied to TEST. No test leakage.
    """
    df = load_raw()
    train_df, test_df = split_df(df)

    # --- XGBoost features (22 selected) ---
    X_train_xgb = train_df[C.FEATURES_XGB].to_numpy(dtype=float)
    X_test_xgb  = test_df[C.FEATURES_XGB].to_numpy(dtype=float)

    # --- Autoencoder features (all 66) ---
    X_train_ae_raw = train_df[C.FEATURES_AE].to_numpy(dtype=float)
    X_test_ae_raw  = test_df[C.FEATURES_AE].to_numpy(dtype=float)

    # --- Labels ---
    y_train = train_df[C.LABEL_COL].to_numpy()
    y_test  = test_df[C.LABEL_COL].to_numpy()

    # --- Scale for Autoencoder (TRAIN scaler only) ---
    if scale:
        ae_scaler = fit_scaler(X_train_ae_raw)
        X_train_ae = ae_scaler.transform(X_train_ae_raw)
        X_test_ae  = ae_scaler.transform(X_test_ae_raw)
    else:
        ae_scaler  = None
        X_train_ae = X_train_ae_raw
        X_test_ae  = X_test_ae_raw

    return (X_train_xgb, y_train, X_test_xgb, y_test,
            X_train_ae, X_test_ae, ae_scaler)


def get_normal_mask(y: np.ndarray) -> np.ndarray:
    """Boolean mask: True where label is 'normal'."""
    return y == C.NORMAL_CLASS


def encode_labels(y: np.ndarray) -> np.ndarray:
    """String labels → integer indices (for XGBoost)."""
    return np.array([C.CLASS_TO_IDX[c] for c in y], dtype=int)


def decode_labels(y_int: np.ndarray) -> np.ndarray:
    """Integer indices → string labels."""
    return np.array([C.IDX_TO_CLASS[i] for i in y_int])


if __name__ == "__main__":
    result = load_splits()
    X_tr_xgb, y_tr, X_te_xgb, y_te, X_tr_ae, X_te_ae, scaler = result
    print(f"\nXGBoost features — Train: {X_tr_xgb.shape}, Test: {X_te_xgb.shape}")
    print(f"AE features      — Train: {X_tr_ae.shape},  Test: {X_te_ae.shape}")
    print(f"Labels           — Train: {y_tr.shape}, Test: {y_te.shape}")
    print(f"NaN in AE train: {np.isnan(X_tr_ae).sum()}")
    print(f"Inf in AE train: {np.isinf(X_tr_ae).sum()}")
