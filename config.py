"""
G3 Configuration — CyberSentinel AI Hybrid IDS
================================================
Single source of truth for all paths, constants, seeds and hyperparameters.

Track: G3 — Hybrid / Ensemble Learning
Architecture: XGBoost + Autoencoder + Stacking Meta-Classifier (E2 primary)

Rules:
  - TRAIN on rounds 3, 4, 5 only.
  - TEST on round 6 only. Never touch test data until final evaluation.
  - Use exactly the 66 FEATURE columns (not envelope, not round, not label).
  - ENVELOPE columns are NEVER model inputs.
  - Reproducible random seed everywhere.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.environ.get("CYBERSENTINEL_UPSTREAM", os.path.join(_HERE, "..", "upstream", "Network_Intrusion_AISD"))
DATA_DIR = os.environ.get("CYBERSENTINEL_DATA_DIR", os.path.join(UPSTREAM, "cybersentinel_code", "v2", "data", "flows"))
UPSTREAM_MODELS_DIR = os.path.join(UPSTREAM, "cybersentinel_code", "v2", "models")
UPSTREAM_FEATURES_V2 = os.path.join(UPSTREAM, "cybersentinel_code", "v2", "features_v2.py")

RESULTS_DIR = os.path.join(_HERE, "results")
MODELS_DIR  = os.path.join(_HERE, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset rules (MUST NOT change)
# ---------------------------------------------------------------------------
ALL_FLOWS_CSV  = os.path.join(DATA_DIR, "all_flows.csv")
TRAIN_ROUNDS   = [3, 4, 5]
TEST_ROUND     = 6
LABEL_COL      = "label"
ROUND_COL      = "round"

# Classes — in alphabetical order (matches XGBoost integer encoding)
CLASSES = ["bruteforce", "normal", "scan", "synflood"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

NORMAL_CLASS = "normal"

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
# 66 features from features_v2.py — copied here so g3_work has no import
# dependency on the upstream code (which is READ-ONLY).
# Source: upstream/cybersentinel_code/v2/features_v2.py  FEATURES list.

_DIRS = ["bidirectional", "src2dst", "dst2src"]
NFSTREAM_FEATURES = (
    ["protocol"]
    + [f"{d}_duration_ms" for d in _DIRS]
    + [f"{d}_{m}" for d in _DIRS for m in ("packets", "bytes")]
    + [f"{d}_{s}_ps" for d in _DIRS for s in ("min", "mean", "stddev", "max")]
    + [f"{d}_{s}_piat_ms" for d in _DIRS for s in ("min", "mean", "stddev", "max")]
    + [f"{d}_{fl}_packets" for d in _DIRS for fl in ("syn", "cwr", "ece", "urg", "ack", "psh", "rst", "fin")]
)
ENGINEERED_FEATURES = ["bytes_per_s", "packets_per_s", "down_up_ratio"]
WINDOW_FEATURES     = ["win_flows_10s", "win_dst_ports_10s", "win_dst_ips_10s",
                        "win_rst_ratio_10s", "win_half_open_10s"]
ALL_66_FEATURES     = NFSTREAM_FEATURES + ENGINEERED_FEATURES + WINDOW_FEATURES

assert len(ALL_66_FEATURES) == 66, f"Expected 66 features, got {len(ALL_66_FEATURES)}"

# 22 selected features from upstream/models/best_meta.json
# (Phase 3 feature selection result — usable in G3 experiments)
SELECTED_22_FEATURES = [
    "dst2src_max_ps", "src2dst_bytes", "bidirectional_bytes",
    "bidirectional_min_ps", "dst2src_min_ps", "src2dst_min_ps",
    "bytes_per_s", "src2dst_duration_ms", "packets_per_s",
    "src2dst_max_piat_ms", "win_dst_ports_10s", "src2dst_rst_packets",
    "bidirectional_syn_packets", "src2dst_min_piat_ms", "win_half_open_10s",
    "dst2src_mean_piat_ms", "bidirectional_fin_packets", "dst2src_min_piat_ms",
    "bidirectional_min_piat_ms", "src2dst_syn_packets", "protocol",
    "win_dst_ips_10s",
]
assert len(SELECTED_22_FEATURES) == 22

# Envelope columns — NEVER model inputs
ENVELOPE = [
    "src_ip", "src_port", "dst_ip", "dst_port",
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
]

# Which feature set to use for G3 training (default: 66 for Autoencoder richness,
# 22 for XGBoost branch to match baseline)
FEATURES_XGB = SELECTED_22_FEATURES   # Match existing baseline
FEATURES_AE  = ALL_66_FEATURES        # Autoencoder uses all 66 for richer reconstruction

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# XGBoost hyperparameters
# ---------------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators   = 200,
    max_depth      = 4,
    learning_rate  = 0.1,
    tree_method    = "hist",
    n_jobs         = 1,
    verbosity      = 0,
    random_state   = SEED,
)

# ---------------------------------------------------------------------------
# Autoencoder hyperparameters
# ---------------------------------------------------------------------------
AE_PARAMS = dict(
    input_dim      = 66,           # Must match len(FEATURES_AE)
    # Encoder: 66 → 32 → 16 → latent(8)
    # Decoder: 8  → 16 → 32 → 66
    latent_dim     = 8,
    hidden_dims    = [32, 16],     # encoder layers (decoder is symmetric)
    dropout_rate   = 0.1,
    learning_rate  = 0.001,
    batch_size     = 128,
    epochs         = 50,           # with early stopping patience=5
    early_stop_patience = 5,
)

# Meta-features dimension = XGB_PROBA_DIM + AE_LATENT_DIM + AE_RECON_DIM
XGB_PROBA_DIM  = len(CLASSES)   # 4
AE_LATENT_DIM  = AE_PARAMS["latent_dim"]  # 8
AE_RECON_DIM   = 1              # scalar reconstruction error
META_FEATURE_DIM = XGB_PROBA_DIM + AE_LATENT_DIM + AE_RECON_DIM  # = 13

# ---------------------------------------------------------------------------
# Stacking cross-validation
# ---------------------------------------------------------------------------
OOF_N_SPLITS  = 3     # 3-fold OOF on training rounds (keeps brute-force class viable)

# ---------------------------------------------------------------------------
# Meta-classifier
# ---------------------------------------------------------------------------
META_CLF_PARAMS = dict(
    # Logistic Regression — interpretable, fast, appropriate for 13 meta-features
    max_iter     = 1000,
    class_weight = "balanced",
    random_state = SEED,
    C            = 1.0,
)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
LATENCY_REPS = 200   # Number of prediction calls for timing measurement

# ---------------------------------------------------------------------------
# Experiment identifiers
# ---------------------------------------------------------------------------
EXPERIMENT_IDS = {
    "E0": "XGBoost_Baseline",
    "E1": "Autoencoder_Standalone",
    "E2": "XGBoost_AE_Stacking",
    "E3": "XGBoost_AE_SoftVoting",
    # E4 (LSTM) only if justified by Agent 4 findings
}
