#!/usr/bin/env python3
"""
SignTalk AI: Sequence Visualization and Temporal Trajectory Plotter.

Generates diagnostic plots from canonical sequence archives:
  - Multi-articulatory temporal trajectories X(t), Y(t), Z(t)
  - Joint velocity magnitude curves ||v(t)||_2
  - Comparative analysis of GOOD vs REJECT sequences
Saves outputs to data/interim/visualizations/.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def plot_sequence_trajectories(npz_path: str, out_path: str, title: str):
    with np.load(npz_path) as arc:
        data = arc["data"]  # (3, 45, 93)
        mask = arc["mask"]  # (1, 45, 93)
        lbl = str(arc["label"]) if "label" in arc else "N/A"
        seq_id = str(arc["sequence_id"]) if "sequence_id" in arc else os.path.basename(npz_path)

    # data: (3, T, V)
    T = data.shape[1]
    time_axis = np.arange(T) * (1.0 / 25.0)

    # Key articulator nodes:
    # 0: Left Wrist, 21: Right Wrist, 42: Nose, 47: Left Shoulder, 48: Right Shoulder
    nodes = {
        "Right Wrist (Node 21)": (21, "red"),
        "Left Wrist (Node 0)": (0, "blue"),
        "Right Index (Node 29)": (29, "orange"),
        "Left Index (Node 8)": (8, "cyan"),
        "Nose Anchor (Node 42)": (42, "green")
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    coords_names = ["X (Horizontal)", "Y (Vertical)", "Z (Depth)"]

    for c in range(3):
        ax = axes[c]
        for name, (node_idx, color) in nodes.items():
            traj = data[c, :, node_idx]
            v_mask = mask[0, :, node_idx] > 0
            ax.plot(time_axis, traj, label=name, color=color, linewidth=1.5, alpha=0.85)
            # Scatter unmasked points
            ax.scatter(time_axis[~v_mask], traj[~v_mask], color="gray", s=10, marker="x", alpha=0.5)

        ax.set_ylabel(f"Norm {coords_names[c]}")
        ax.grid(True, linestyle="--", alpha=0.5)
        if c == 0:
            ax.set_title(f"{title} — {seq_id} (Label ID: {lbl})", fontsize=12, fontweight="bold")
            ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[2].set_xlabel("Time (seconds @ 25 FPS)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved trajectory plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic sequence visualizations.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/sequence_manifest.csv",
        help="Path to sequence manifest."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/interim/visualizations",
        help="Output directory for plots."
    )
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"[ERROR] Manifest not found: {args.manifest}")
        sys.exit(1)

    df = pd.read_csv(args.manifest)

    # Select representative samples
    # 1. Valid sample (GOOD / ACCEPTABLE)
    valid_samples = df[df["quality_status"].isin(["GOOD", "ACCEPTABLE"])]
    if not valid_samples.empty:
        val_row = valid_samples.iloc[0]
        plot_sequence_trajectories(
            val_row["file_path"],
            os.path.join(args.out_dir, f"{val_row['sequence_id']}_valid_trajectory.png"),
            f"Valid Articulation Trajectory ({val_row['label'].upper()})"
        )

    # 2. Rejected / Stress sample
    reject_samples = df[df["quality_status"] == "REJECT"]
    if not reject_samples.empty:
        rej_row = reject_samples.iloc[0]
        plot_sequence_trajectories(
            rej_row["file_path"],
            os.path.join(args.out_dir, f"{rej_row['sequence_id']}_rejected_trajectory.png"),
            f"Low-Confidence / Rejected Articulation ({rej_row['label'].upper()})"
        )

    # 3. Two-handed sign (e.g. house or car)
    two_hand = df[df["label"].isin(["house", "car"])]
    if not two_hand.empty:
        th_row = two_hand.iloc[0]
        plot_sequence_trajectories(
            th_row["file_path"],
            os.path.join(args.out_dir, f"{th_row['sequence_id']}_bilateral_trajectory.png"),
            f"Bilateral Two-Handed Sign Trajectory ({th_row['label'].upper()})"
        )


if __name__ == "__main__":
    main()
