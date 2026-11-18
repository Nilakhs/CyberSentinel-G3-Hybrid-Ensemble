"""
G3 Experiments Orchestrator — CyberSentinel AI Hybrid IDS
=========================================================
Runs all required experiments:
  E0: Existing XGBoost Baseline (shipped 22 features)
  E1: Standalone Autoencoder Anomaly Detection (66 features, trained on normal only)
  E2: XGBoost + Autoencoder + Stacking Meta-Classifier (PRIMARY G3 DELIVERABLE)
  E3: XGBoost + Autoencoder + Soft Voting (Comparison from Almuhanna et al., 2025)

Records all metrics specified in Phase 7:
  - Accuracy
  - Macro-F1
  - Per-class recall (normal, scan, synflood, bruteforce)
  - Attack recall
  - False-alarm rate on normal
  - Single-flow prediction time in milliseconds (latency)
  - Model size in KB

Outputs:
  - results/g3_comparison.csv
  - results/confusion_E0_XGBoost.png
  - results/confusion_E2_Stacking.png
  - results/confusion_E3_SoftVoting.png
  - results/model_comparison_bar.png
  - models/ (trained artifacts)
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import data_loader as dl
from xgboost_model import G3XGBoostModel
from autoencoder import G3Autoencoder
from stacking import G3StackingClassifier
from soft_voting import G3SoftVotingEnsemble
from evaluation import compute_metrics, measure_single_flow_latency, get_model_size_kb, plot_confusion_matrix


def run_all():
    print("=" * 70)
    print(" CYBERSENTINEL AI — G3 HYBRID / ENSEMBLE LEARNING EXPERIMENTS")
    print("=" * 70)

    # 1. Load Data
    print("\n[Step 1] Loading and preprocessing dataset...")
    X_tr_xgb, y_tr, X_te_xgb, y_te, X_tr_ae, X_te_ae, scaler = dl.load_splits(scale=True)
    normal_mask_tr = (y_tr == C.NORMAL_CLASS)
    normal_mask_te = (y_te == C.NORMAL_CLASS)

    print(f"Train samples: {len(y_tr):,} (Normal: {normal_mask_tr.sum():,}, Attack: {(~normal_mask_tr).sum():,})")
    print(f"Test  samples: {len(y_te):,} (Normal: {normal_mask_te.sum():,}, Attack: {(~normal_mask_te).sum():,})")

    results = []

    # -----------------------------------------------------------------------
    # E0: XGBoost Baseline
    # -----------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(" [E0] Running XGBoost Baseline (22 Selected Features)...")
    print("-" * 60)
    xgb_base = G3XGBoostModel()
    xgb_base.fit(X_tr_xgb, y_tr)
    y_pred_e0 = xgb_base.predict(X_te_xgb)

    m_e0 = compute_metrics(y_te, y_pred_e0)
    m_e0["latency_ms"] = measure_single_flow_latency(xgb_base.predict, X_te_xgb)
    m_e0["size_kb"] = get_model_size_kb(xgb_base)
    m_e0["experiment"] = "E0: XGBoost Baseline"
    m_e0["type"] = "Supervised ML"
    results.append(m_e0)

    plot_confusion_matrix(y_te, y_pred_e0, "E0: XGBoost Baseline",
                          os.path.join(C.RESULTS_DIR, "confusion_E0_XGBoost.png"))
    print(f"E0 Results -> Acc: {m_e0['accuracy']:.4f} | Macro-F1: {m_e0['macro_f1']:.4f} | "
          f"False Alarm: {m_e0['normal_false_alarm']:.4f} | Latency: {m_e0['latency_ms']:.3f} ms")

    # -----------------------------------------------------------------------
    # E1: Standalone Autoencoder Anomaly Detector
    # -----------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(" [E1] Running Standalone Autoencoder (Unsupervised on Normal Flows)...")
    print("-" * 60)
    ae_standalone = G3Autoencoder()
    # Train only on normal training flows
    ae_standalone.fit(X_tr_ae[normal_mask_tr], verbose=False)

    # Anomaly prediction: True = Attack, False = Normal
    is_anomaly = ae_standalone.predict_anomaly(X_te_ae)
    y_pred_e1_binary = np.where(is_anomaly, "attack", "normal")
    y_te_binary = np.where(normal_mask_te, "normal", "attack")

    from sklearn.metrics import recall_score, roc_auc_score
    ae_attack_recall = float(recall_score(y_te_binary == "attack", y_pred_e1_binary == "attack"))
    ae_false_alarm = float((is_anomaly & normal_mask_te).sum() / normal_mask_te.sum())
    ae_auc = float(roc_auc_score(y_te_binary == "attack", ae_standalone.reconstruction_error(X_te_ae)))

    m_e1 = {
        "experiment": "E1: Standalone Autoencoder",
        "type": "Unsupervised DL Anomaly",
        "accuracy": round(float((y_pred_e1_binary == y_te_binary).mean()), 4),
        "macro_f1": round(float(ae_attack_recall), 4),  # binary proxy
        "attack_recall": round(ae_attack_recall, 4),
        "normal_false_alarm": round(ae_false_alarm, 4),
        "recall_normal": round(1.0 - ae_false_alarm, 4),
        "recall_scan": round(float(ae_standalone.predict_anomaly(X_te_ae[y_te == 'scan']).mean()), 4),
        "recall_synflood": round(float(ae_standalone.predict_anomaly(X_te_ae[y_te == 'synflood']).mean()), 4),
        "recall_bruteforce": round(float(ae_standalone.predict_anomaly(X_te_ae[y_te == 'bruteforce']).mean()), 4),
        "latency_ms": measure_single_flow_latency(ae_standalone.predict_anomaly, X_te_ae),
        "size_kb": get_model_size_kb(ae_standalone),
    }
    results.append(m_e1)
    print(f"E1 Results -> Binary Acc: {m_e1['accuracy']:.4f} | Attack Recall: {m_e1['attack_recall']:.4f} | "
          f"False Alarm: {m_e1['normal_false_alarm']:.4f} | ROC-AUC: {ae_auc:.4f} | Latency: {m_e1['latency_ms']:.3f} ms")

    # -----------------------------------------------------------------------
    # E2: G3 Stacking Meta-Classifier (PRIMARY DELIVERABLE)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(" [E2] Running XGBoost + Autoencoder + Stacking Meta-Classifier...")
    print("-" * 60)
    stacker = G3StackingClassifier()
    # Phase A: OOF meta-features & meta-classifier training
    meta_info = stacker.fit_meta(X_tr_xgb, X_tr_ae, y_tr, n_splits=C.OOF_N_SPLITS)
    # Phase B: Refit base models on full training rounds
    stacker.refit_base_models(X_tr_xgb, X_tr_ae, y_tr)
    # Phase C: Inference on held-out round 6
    y_pred_e2 = stacker.predict(X_te_xgb, X_te_ae)

    m_e2 = compute_metrics(y_te, y_pred_e2)
    m_e2["latency_ms"] = measure_single_flow_latency(stacker.predict, (X_te_xgb, X_te_ae))
    m_e2["size_kb"] = get_model_size_kb(stacker)
    m_e2["experiment"] = "E2: G3 Hybrid Stacking"
    m_e2["type"] = "Hybrid Ensemble (ML+DL)"
    results.append(m_e2)

    plot_confusion_matrix(y_te, y_pred_e2, "E2: G3 Stacking Meta-Classifier",
                          os.path.join(C.RESULTS_DIR, "confusion_E2_Stacking.png"))
    stacker.save(C.MODELS_DIR)
    print(f"E2 Results -> Acc: {m_e2['accuracy']:.4f} | Macro-F1: {m_e2['macro_f1']:.4f} | "
          f"False Alarm: {m_e2['normal_false_alarm']:.4f} | Latency: {m_e2['latency_ms']:.3f} ms")

    # -----------------------------------------------------------------------
    # E3: Soft Voting Ensemble (Almuhanna 2025 approach)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(" [E3] Running XGBoost + Autoencoder + Soft Voting (Paper 1)...")
    print("-" * 60)
    soft_voter = G3SoftVotingEnsemble(w_xgb=0.85, w_ae=0.15)
    soft_voter.fit(X_tr_xgb, X_tr_ae, y_tr)
    y_pred_e3 = soft_voter.predict(X_te_xgb, X_te_ae)

    m_e3 = compute_metrics(y_te, y_pred_e3)
    m_e3["latency_ms"] = measure_single_flow_latency(soft_voter.predict, (X_te_xgb, X_te_ae))
    m_e3["size_kb"] = get_model_size_kb(soft_voter)
    m_e3["experiment"] = "E3: G3 Soft Voting"
    m_e3["type"] = "Voting Ensemble (ML+DL)"
    results.append(m_e3)

    plot_confusion_matrix(y_te, y_pred_e3, "E3: G3 Soft Voting",
                          os.path.join(C.RESULTS_DIR, "confusion_E3_SoftVoting.png"))
    print(f"E3 Results -> Acc: {m_e3['accuracy']:.4f} | Macro-F1: {m_e3['macro_f1']:.4f} | "
          f"False Alarm: {m_e3['normal_false_alarm']:.4f} | Latency: {m_e3['latency_ms']:.3f} ms")

    # -----------------------------------------------------------------------
    # Summary Table & Charts
    # -----------------------------------------------------------------------
    df_res = pd.DataFrame(results)
    csv_path = os.path.join(C.RESULTS_DIR, "g3_comparison.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\n[Summary] All experiment metrics saved to {csv_path}")

    # Plot comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Macro-F1 & Accuracy
    models = [r["experiment"].split(":")[0] for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    x = np.arange(len(models))
    width = 0.35

    axes[0].bar(x - width/2, accs, width, label='Accuracy', color='#2b5c8f')
    axes[0].bar(x + width/2, f1s, width, label='Macro-F1', color='#4ba3e3')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Detection Performance (Round 6)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # 2. Prediction Latency (ms)
    latencies = [r["latency_ms"] for r in results]
    axes[1].bar(models, latencies, color='#e67e22', width=0.5)
    axes[1].set_ylabel('Milliseconds (ms)')
    axes[1].set_title('Single-Flow Prediction Latency')
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(latencies):
        axes[1].text(i, v + max(latencies)*0.02, f"{v:.2f}ms", ha='center', fontsize=9)

    # 3. False Alarm Rate on Normal Traffic
    fa_rates = [r["normal_false_alarm"] * 100 for r in results]
    axes[2].bar(models, fa_rates, color='#c0392b', width=0.5)
    axes[2].set_ylabel('False Alarm Rate (%)')
    axes[2].set_title('False Alarms on Normal Traffic')
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(fa_rates):
        axes[2].text(i, v + 0.1, f"{v:.2f}%", ha='center', fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(C.RESULTS_DIR, "model_comparison_bar.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[Summary] Saved comparison visualization to {chart_path}")

    print("\n" + "=" * 70)
    print(" FINAL G3 EXPERIMENTAL RESULTS TABLE:")
    print("=" * 70)
    cols_to_print = ["experiment", "accuracy", "macro_f1", "recall_bruteforce", "normal_false_alarm", "latency_ms", "size_kb"]
    print(df_res[cols_to_print].to_string(index=False))
    print("=" * 70)

    return df_res


if __name__ == "__main__":
    run_all()
