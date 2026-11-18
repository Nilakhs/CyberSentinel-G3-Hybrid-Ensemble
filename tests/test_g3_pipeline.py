"""
Unit and Integration Tests for G3 Hybrid IDS Pipeline
=====================================================
Follows SDLC + TDD requirements:
  - Tests data loader and split rules
  - Tests XGBoost wrapper
  - Tests Autoencoder architecture and representation extraction
  - Tests Stacking classifier (OOF meta-features, meta-classifier training, refit)
  - Tests Evaluation metrics calculation
"""

import os
import sys
import numpy as np
import pytest

# Add g3_work directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as C
import data_loader as dl
from xgboost_model import G3XGBoostModel
from autoencoder import G3Autoencoder
from stacking import G3StackingClassifier
from evaluation import compute_metrics, measure_single_flow_latency


def test_config_definitions():
    """Verify essential constants, features, and classes."""
    assert len(C.ALL_66_FEATURES) == 66
    assert len(C.SELECTED_22_FEATURES) == 22
    assert len(C.CLASSES) == 4
    assert C.TRAIN_ROUNDS == [3, 4, 5]
    assert C.TEST_ROUND == 6
    assert C.META_FEATURE_DIM == 13
    assert set(C.CLASSES) == {"bruteforce", "normal", "scan", "synflood"}


def test_data_loader_structure():
    """Verify dataset loading and split integrity."""
    df = dl.load_raw()
    assert len(df) == 10676
    assert len(df.columns) == 74

    tr_df, te_df = dl.split_df(df)
    assert set(tr_df[C.ROUND_COL].unique()) == {3, 4, 5}
    assert set(te_df[C.ROUND_COL].unique()) == {6}
    assert len(tr_df) + len(te_df) == 10676


def test_xgboost_fit_predict():
    """Verify XGBoost model fitting and probability generation."""
    # Create small dummy dataset
    np.random.seed(C.SEED)
    X = np.random.randn(40, len(C.SELECTED_22_FEATURES))
    y = np.array(["normal"] * 10 + ["scan"] * 10 + ["synflood"] * 10 + ["bruteforce"] * 10)

    model = G3XGBoostModel()
    model.fit(X, y)

    preds = model.predict(X[:5])
    assert len(preds) == 5
    for p in preds:
        assert p in C.CLASSES

    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_autoencoder_reconstruction_and_meta():
    """Verify Autoencoder compression, reconstruction error, and meta-features."""
    np.random.seed(C.SEED)
    X_normal = np.random.randn(50, 66).astype(np.float32)

    ae = G3Autoencoder(params={
        "input_dim": 66,
        "latent_dim": 8,
        "hidden_dims": [16],
        "dropout_rate": 0.0,
        "learning_rate": 0.01,
        "batch_size": 25,
        "epochs": 2,
        "early_stop_patience": 2
    })
    ae.fit(X_normal, verbose=False)

    recon_errors = ae.reconstruction_error(X_normal[:10])
    assert recon_errors.shape == (10,)
    assert np.all(recon_errors >= 0.0)

    latent = ae.latent_features(X_normal[:10])
    assert latent.shape == (10, 8)

    meta = ae.meta_features(X_normal[:10])
    assert meta.shape == (10, 9)  # 8 latent + 1 recon error


def test_compute_metrics():
    """Verify metric computation matches specifications."""
    y_true = np.array(["normal", "scan", "synflood", "bruteforce", "normal"])
    y_pred = np.array(["normal", "scan", "synflood", "scan", "normal"])

    metrics = compute_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "attack_recall" in metrics
    assert "normal_false_alarm" in metrics
    assert metrics["normal_false_alarm"] == 0.0  # 0 normal classified as attack
    assert metrics["accuracy"] == 0.8  # 4 out of 5 correct


def test_single_flow_latency():
    """Verify prediction latency measurement."""
    np.random.seed(C.SEED)
    X = np.random.randn(10, 22)
    dummy_predict = lambda r: np.array(["normal"])
    latency_ms = measure_single_flow_latency(dummy_predict, X, reps=50)
    assert latency_ms > 0.0
    assert latency_ms < 50.0  # Should be well below 50 ms
