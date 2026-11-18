"""
G3 Evaluation Suite — CyberSentinel AI Hybrid IDS
=================================================
Calculates standard metrics required by Phase 7:
  1. Accuracy
  2. Macro-F1
  3. Per-class recall (normal, scan, synflood, bruteforce)
  4. Attack recall and False-alarm rate on normal
  5. Single-flow prediction latency in milliseconds (with warmup + timing)
  6. Model size (KB)

Also handles:
  - Confusion matrix generation and saving
  - Model comparison visualization
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, roc_auc_score

import config as C


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list = None) -> dict:
    """
    Computes all classification metrics for multi-class IDS.
    """
    classes = classes or C.CLASSES
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class recall
    recalls = {}
    for c in classes:
        recalls[f"recall_{c}"] = float(recall_score(y_true == c, y_pred == c, zero_division=0))

    # Binary metrics: Attack vs Normal
    true_attack = y_true != C.NORMAL_CLASS
    pred_attack = y_pred != C.NORMAL_CLASS
    n_attack = int(true_attack.sum())
    n_normal = int((~true_attack).sum())

    attack_recall = float((pred_attack & true_attack).sum() / n_attack) if n_attack else 0.0
    false_alarm = float((pred_attack & ~true_attack).sum() / n_normal) if n_normal else 0.0

    metrics = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "attack_recall": round(attack_recall, 4),
        "normal_false_alarm": round(false_alarm, 4),
    }
    for k, v in recalls.items():
        metrics[k] = round(v, 4)

    return metrics


def measure_single_flow_latency(predict_fn, sample_input, reps: int = C.LATENCY_REPS) -> float:
    """
    Measures average single-flow prediction time in milliseconds with warmup.
    sample_input can be a single array or tuple of arrays (for multi-branch models).
    """
    # Warm-up pass
    if isinstance(sample_input, tuple):
        predict_fn(*(x[:1] for x in sample_input))
    else:
        predict_fn(sample_input[:1])

    # Repeated timing
    start = time.perf_counter()
    if isinstance(sample_input, tuple):
        single_in = tuple(x[:1] for x in sample_input)
        for _ in range(reps):
            predict_fn(*single_in)
    else:
        single_in = sample_input[:1]
        for _ in range(reps):
            predict_fn(single_in)
    end = time.perf_counter()

    avg_ms = ((end - start) / reps) * 1000.0
    return round(avg_ms, 3)


def get_model_size_kb(obj) -> float:
    """Estimates in-memory serialized size in KB."""
    try:
        data = pickle.dumps(obj)
        return round(len(data) / 1024.0, 1)
    except Exception:
        return 0.0


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str):
    """Plots and saves confusion matrix."""
    classes = C.CLASSES
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name} (Round 6 Test)")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[evaluation] Saved confusion matrix to {save_path}")
