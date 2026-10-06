#!/usr/bin/env python3
"""
SignTalk AI: ST-GCN Training Curves and Diagnostics Plotter.

Generates 4-panel diagnostic training plots:
  - Cross-Entropy Loss (Train vs. Validation)
  - Top-1 Accuracy (Train vs. Validation)
  - Macro F1-Score (Train vs. Validation)
  - Learning Rate decay trajectory
Saves artifact to experiments/stgcn/plots/training_curves.png.
"""

import sys
import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot ST-GCN training and validation history curves.")
    parser.add_argument(
        "--history-csv",
        type=str,
        default="experiments/stgcn/metrics/training_history.csv",
        help="Path to training history CSV."
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="experiments/stgcn/plots/training_curves.png",
        help="Output image path for curves."
    )
    args = parser.parse_args()

    if not os.path.exists(args.history_csv):
        print(f"[ERROR] History CSV not found: {args.history_csv}")
        sys.exit(1)

    df = pd.read_csv(args.history_csv)
    epochs = df["epoch"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Loss Curve
    axes[0].plot(epochs, df["train_loss"], label="Train Loss", color="tab:blue", linewidth=2)
    axes[0].plot(epochs, df["val_loss"], label="Val Loss", color="tab:orange", linewidth=2, linestyle="--")
    axes[0].set_title("Cross-Entropy Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # 2. Accuracy Curve
    axes[1].plot(epochs, df["train_acc"], label="Train Acc", color="tab:green", linewidth=2)
    axes[1].plot(epochs, df["val_acc"], label="Val Acc", color="tab:red", linewidth=2, linestyle="--")
    axes[1].set_title("Top-1 Accuracy", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # 3. Macro F1 Curve
    axes[2].plot(epochs, df["train_macro_f1"], label="Train Macro F1", color="tab:purple", linewidth=2)
    axes[2].plot(epochs, df["val_macro_f1"], label="Val Macro F1", color="tab:brown", linewidth=2, linestyle="--")
    axes[2].set_title("Macro F1-Score", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    # 4. Learning Rate Curve
    if "lr" in df.columns:
        axes[3].plot(epochs, df["lr"], label="Learning Rate", color="tab:cyan", linewidth=2)
        axes[3].set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("LR")
        axes[3].grid(True, linestyle="--", alpha=0.5)
        axes[3].legend()
    else:
        axes[3].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    plt.savefig(args.out_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved ST-GCN training curves to: {args.out_path}")


if __name__ == "__main__":
    main()
