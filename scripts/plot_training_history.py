#!/usr/bin/env python3
"""
SignTalk AI: Training Curves and Metric Plotter.

Generates diagnostic training plots:
  - Loss curves (Train Loss vs. Validation Loss)
  - Accuracy curves (Train Accuracy vs. Validation Accuracy)
  - F1 curves (Train Macro F1 vs. Validation Macro F1)
Saves plot artifact to experiments/baseline/plots/training_curves.png.
"""

import sys
import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot training and validation history curves.")
    parser.add_argument(
        "--history-csv",
        type=str,
        default="experiments/baseline/metrics/training_history.csv",
        help="Path to training history CSV."
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="experiments/baseline/plots/training_curves.png",
        help="Output image path for curves."
    )
    args = parser.parse_args()

    if not os.path.exists(args.history_csv):
        print(f"[ERROR] History CSV not found: {args.history_csv}")
        sys.exit(1)

    df = pd.read_csv(args.history_csv)
    epochs = df["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

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

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    plt.savefig(args.out_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved training curves to: {args.out_path}")


if __name__ == "__main__":
    main()
