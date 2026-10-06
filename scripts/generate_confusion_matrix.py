#!/usr/bin/env python3
"""
SignTalk AI: Confusion Matrix Visualizer.

Generates heatmap visualizations of the raw and row-normalized confusion
matrix from evaluation metrics JSON. Saves output to experiments/baseline/plots/.
"""

import sys
import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="Generate confusion matrix heatmap plot.")
    parser.add_argument(
        "--metrics-json",
        type=str,
        default="experiments/baseline/metrics/evaluation_test.json",
        help="Path to evaluation metrics JSON."
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="experiments/baseline/plots/confusion_matrix.png",
        help="Output image path for confusion matrix."
    )
    args = parser.parse_args()

    if not os.path.exists(args.metrics_json):
        print(f"[ERROR] Metrics JSON not found: {args.metrics_json}")
        sys.exit(1)

    with open(args.metrics_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cm = np.array(metrics["confusion_matrix"])
    cm_norm = np.array(metrics["confusion_matrix_normalized"])
    class_names = list(metrics["per_class"].keys())
    K = len(class_names)

    fig, ax = plt.subplots(figsize=(9, 8))
    cax = ax.matshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(cax)

    # Set ticks and labels
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(class_names, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # Annotate cells with normalized percentage and raw count
    for i in range(K):
        for j in range(K):
            val_norm = cm_norm[i, j]
            val_raw = cm[i, j]
            color = "white" if val_norm > 0.5 else "black"
            text = f"{val_norm*100:.0f}%\n({val_raw})" if val_raw > 0 else "0"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Normalized Confusion Matrix (Test Split — Acc: {metrics['accuracy']*100:.1f}%)",
        fontsize=12,
        fontweight="bold",
        pad=20
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    plt.savefig(args.out_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved confusion matrix heatmap to: {args.out_path}")


if __name__ == "__main__":
    main()
