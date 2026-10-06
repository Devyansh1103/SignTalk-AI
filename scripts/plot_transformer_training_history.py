"""
SignTalk AI: Plot Transformer Training History.

Generates 4-panel training trajectory figures:
  1. Train and Validation Cross-Entropy Loss
  2. Train and Validation Token Accuracy
  3. Validation Sequence Exact Match (Teacher-Forced vs Autoregressive Generation)
  4. Cosine Annealing Learning Rate Schedule
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def plot_transformer_history(
    csv_path: str = "experiments/transformer/metrics/metrics.csv",
    output_path: str = "experiments/transformer/plots/training_curves.png"
):
    if not os.path.exists(csv_path):
        print(f"Metrics CSV not found at {csv_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(csv_path)

    epochs = df["epoch"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss
    axes[0, 0].plot(epochs, df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    axes[0, 0].plot(epochs, df["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    axes[0, 0].set_title("Cross-Entropy Loss (Token Level)", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # 2. Token Accuracy
    axes[0, 1].plot(epochs, df["train_token_acc"] * 100, label="Train Token Acc", color="#2ca02c", lw=2)
    axes[0, 1].plot(epochs, df["val_token_acc"] * 100, label="Val Token Acc", color="#d62728", lw=2, linestyle="--")
    axes[0, 1].set_title("Token Accuracy (%)", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()

    # 3. Sequence Exact Match Accuracy
    axes[1, 0].plot(epochs, df["val_teacher_seq_acc"] * 100, label="Val Seq Acc (Teacher Forced)", color="#9467bd", lw=2)
    axes[1, 0].plot(epochs, df["val_gen_seq_acc"] * 100, label="Val Seq Acc (Autoregressive)", color="#8c564b", lw=2, linestyle="-.")
    axes[1, 0].set_title("Sequence Exact Match Accuracy (%)", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Exact Match (%)")
    axes[1, 0].legend()

    # 4. Learning Rate Schedule
    axes[1, 1].plot(epochs, df["lr"].astype(float), label="Learning Rate", color="#e377c2", lw=2)
    axes[1, 1].set_title("Cosine Annealing Learning Rate", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].legend()

    plt.suptitle("SignTalk AI — Sign Language Transformer Training Trajectory", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training curve plot to: {output_path}")


if __name__ == "__main__":
    plot_transformer_history()
