"""
SignTalk AI: Confusion Matrix Analysis and High-Resolution Visualization.

Features:
  - Generates raw and normalized confusion matrices
  - Identifies top confused class pairs (Class A -> Class B)
  - Identifies potential contributing factors (kinematic similarity, landmark noise, class frequency)
  - Exports publication-ready confusion matrix plots and analysis tables
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Known anatomical / kinematic similarities in ISL vocabulary
KINEMATIC_SIMILARITY_MAP = {
    ("hello", "thankyou"): "Both signs involve open palm orientation near the head/face region.",
    ("thankyou", "hello"): "Both signs share palm-forward orientation and initial touch near chin/temple.",
    ("good", "teacher"): "Both signs utilize thumb / index orientation near torso/chest.",
    ("teacher", "good"): "Shared upper-chest vertical articulation and fist/hand shape.",
    ("monday", "time"): "Both involve index finger pointing or circular temporal motion.",
    ("time", "monday"): "Shared wrist contact / wrist pointing articulation.",
    ("car", "house"): "Both signs involve two hands interacting in front of the torso.",
    ("house", "car"): "Two-handed symmetric configuration in lower chest zone.",
    ("happy", "good"): "Both are positive valence signs with upward chest sweep movements.",
    ("bird", "teacher"): "Rapid finger movement / pinch gesture in upper quadrant."
}


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes raw and normalized confusion matrices.
    
    Returns:
        (cm_raw, cm_norm)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(np.float64),
        np.maximum(row_sums, 1),
        out=np.zeros_like(cm, dtype=np.float64)
    )
    return cm, cm_norm


def analyze_top_confusions(
    cm: np.ndarray,
    class_names: List[str],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Identifies the top misclassified pairs (True Class -> Predicted Class).
    """
    num_classes = len(class_names)
    confusions = []

    for t in range(num_classes):
        for p in range(num_classes):
            if t != p and cm[t, p] > 0:
                true_label = class_names[t]
                pred_label = class_names[p]
                cause = KINEMATIC_SIMILARITY_MAP.get(
                    (true_label.lower(), pred_label.lower()),
                    "Shared spatial coordinate trajectory or MediaPipe landmark tracking variance."
                )
                confusions.append({
                    "true_class_id": int(t),
                    "true_label": true_label,
                    "pred_class_id": int(p),
                    "pred_label": pred_label,
                    "count": int(cm[t, p]),
                    "probable_cause": cause
                })

    confusions.sort(key=lambda x: x["count"], reverse=True)
    return confusions[:top_k]


def plot_confusion_matrix(
    cm_norm: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Normalized Confusion Matrix",
    cmap: str = "Blues"
) -> str:
    """
    Renders and saves a high-resolution confusion matrix plot.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Frequency (Recall)", rotation=270, labelpad=15, fontsize=10)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=9)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.set_ylabel("True Sign Class", fontsize=10, fontweight="semibold")
    ax.set_xlabel("Predicted Sign Class", fontsize=10, fontweight="semibold")

    # Annotate cells
    thresh = cm_norm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            color = "white" if val > thresh else "black"
            text = f"{val:.2f}" if val > 0 else "0"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
