#!/usr/bin/env python3
"""
SignTalk AI: ST-GCN Confusion Matrix Visualizer.

Generates annotated raw and row-normalized confusion matrix heatmaps
from evaluation metrics JSON and saves to experiments/stgcn/plots/confusion_matrix.png.
"""

import sys
import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Generate ST-GCN confusion matrix heatmaps.")
    parser.add_argument(
        "--metrics-json",
        type=str,
        default="experiments/stgcn/metrics/evaluation_test.json",
        help="Path to evaluation JSON containing confusion matrix."
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="experiments/stgcn/plots/confusion_matrix.png",
        help="Output image path for heatmap."
    )
    args = parser.parse_args()

    if not os.path.exists(args.metrics_json):
        print(f"[ERROR] Metrics JSON not found: {args.metrics_json}")
        sys.exit(1)

    with open(args.metrics_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    cm_raw = np.array(metrics.get("confusion_matrix", metrics.get("confusion_matrix_raw", [])))
    cm_norm = np.array(metrics.get("confusion_matrix_norm", metrics.get("confusion_matrix_normalized", [])))
    class_names = metrics.get("class_names", [f"Class_{i}" for i in range(len(cm_raw))])

    if len(cm_raw) == 0:
        print("[ERROR] Confusion matrix data missing from metrics JSON.")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 1. Raw Count Heatmap
    im1 = axes[0].imshow(cm_raw, interpolation="nearest", cmap="Blues")
    axes[0].set_title(f"ST-GCN Test Confusion Matrix (Counts) - Split: {data.get('split', 'test')}", fontsize=12, fontweight="bold")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Normalized Percentage Heatmap
    im2 = axes[1].imshow(cm_norm, interpolation="nearest", cmap="Greens")
    axes[1].set_title(f"ST-GCN Test Confusion Matrix (Normalized %) - Split: {data.get('split', 'test')}", fontsize=12, fontweight="bold")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    for idx, ax in enumerate(axes):
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted Gloss", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Gloss", fontsize=11, fontweight="bold")

        matrix = cm_raw if idx == 0 else cm_norm
        fmt = "d" if idx == 0 else ".1f"
        thresh = matrix.max() / 2.0 if matrix.max() > 0 else 1.0

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = int(matrix[i, j]) if idx == 0 else float(matrix[i, j])
                color = "white" if val > thresh else "black"
                text = f"{val:{fmt}}" if idx == 0 else f"{val:.1f}%"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    plt.savefig(args.out_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved ST-GCN confusion matrix heatmap to: {args.out_path}")


if __name__ == "__main__":
    main()
