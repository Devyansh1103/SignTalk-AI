"""
SignTalk AI - Visualization Generator
Generates spatial skeleton overlays, coordinate trajectory plots, and
comparison visuals between raw frames and normalized skeletons.
Saved to: data/interim/visualizations/
"""

import os
import sys
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Anatomical bone connections for 93-node skeleton visualization
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

POSE_BONES = [
    (47, 48),  # Left Shoulder - Right Shoulder
    (47, 49), (49, 51),  # Left Arm
    (48, 50), (50, 52),  # Right Arm
    (42, 43), (43, 45),  # Left Eye/Ear
    (42, 44), (44, 46)   # Right Eye/Ear
]


def draw_skeleton_2d(
    coords: np.ndarray,      # (93, 3) or (93, 2)
    mask: np.ndarray,        # (93,)
    canvas_size: tuple = (480, 640),  # (H, W)
    is_normalized: bool = False
) -> np.ndarray:
    """
    Draws the 93-node skeleton on a black RGB canvas.
    """
    H, W = canvas_size
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Convert coordinates to pixel space
    pts_2d = np.zeros((93, 2), dtype=int)
    for i in range(93):
        if mask[i]:
            if is_normalized:
                # Normalized coordinates centered around (0, 0) with scale ~1.0
                px = int(np.clip((coords[i, 0] * 0.35 + 0.5) * W, 0, W - 1))
                py = int(np.clip((coords[i, 1] * 0.35 + 0.5) * H, 0, H - 1))
            else:
                # Raw [0, 1] normalized pixel coordinates
                px = int(np.clip(coords[i, 0] * W, 0, W - 1))
                py = int(np.clip(coords[i, 1] * H, 0, H - 1))
            pts_2d[i] = [px, py]

    # 1. Draw Pose Bones (Green)
    for p1, p2 in POSE_BONES:
        if mask[p1] and mask[p2]:
            cv2.line(canvas, tuple(pts_2d[p1]), tuple(pts_2d[p2]), (0, 255, 0), 2)

    # 2. Draw Left Hand Bones (Red)
    for b1, b2 in HAND_BONES:
        if mask[b1] and mask[b2]:
            cv2.line(canvas, tuple(pts_2d[b1]), tuple(pts_2d[b2]), (0, 0, 255), 1)

    # 3. Draw Right Hand Bones (Cyan / Blue)
    for b1, b2 in HAND_BONES:
        r1, r2 = b1 + 21, b2 + 21
        if mask[r1] and mask[r2]:
            cv2.line(canvas, tuple(pts_2d[r1]), tuple(pts_2d[r2]), (255, 255, 0), 1)

    # 4. Draw Joint Nodes
    for i in range(93):
        if mask[i]:
            pt = tuple(pts_2d[i])
            if i < 21:
                color = (0, 0, 255)  # Left hand: Red
                radius = 3
            elif i < 42:
                color = (255, 255, 0)  # Right hand: Cyan
                radius = 3
            elif i < 53:
                color = (0, 255, 0)  # Pose: Green
                radius = 4
            else:
                color = (255, 0, 255)  # Face: Magenta
                radius = 2
            cv2.circle(canvas, pt, radius, color, -1)

    return canvas


def plot_temporal_trajectory(
    coords_seq: np.ndarray,  # (T, 93, 3)
    mask_seq: np.ndarray,    # (T, 93)
    out_path: str,
    title: str = "Sign Language Articulator Trajectories Across Time"
):
    """
    Plots the X and Y movement trajectories of key articulators over temporal frames T.
    """
    T = coords_seq.shape[0]
    frames = np.arange(T)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.title(title, fontsize=12, fontweight="bold")

    # Trace Right Wrist (21) and Right Index Tip (29)
    if np.any(mask_seq[:, 21]):
        plt.plot(frames, coords_seq[:, 21, 0], label="Right Wrist X", color="blue", linewidth=1.5)
        plt.plot(frames, coords_seq[:, 21, 1], label="Right Wrist Y", color="navy", linestyle="--", linewidth=1.5)

    if np.any(mask_seq[:, 29]):
        plt.plot(frames, coords_seq[:, 29, 0], label="Right Index Tip X", color="orange", linewidth=1.5)
        plt.plot(frames, coords_seq[:, 29, 1], label="Right Index Tip Y", color="red", linestyle="--", linewidth=1.5)

    # Trace Left Wrist (0)
    if np.any(mask_seq[:, 0]):
        plt.plot(frames, coords_seq[:, 0, 0], label="Left Wrist X", color="green", linewidth=1.2, alpha=0.7)
        plt.plot(frames, coords_seq[:, 0, 1], label="Left Wrist Y", color="darkgreen", linestyle="--", linewidth=1.2, alpha=0.7)

    plt.xlabel("Temporal Frame Index (t)", fontsize=10)
    plt.ylabel("Normalized Coordinate Value", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def generate_sample_visualizations(
    processed_dir: str = "data/processed/landmarks",
    output_dir: str = "data/interim/visualizations",
    num_samples: int = 5
):
    """
    Selects representative processed samples and creates side-by-side validation panels.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(processed_dir, "*", "*.npz")))

    if not files:
        print(f"No processed files found in {processed_dir}")
        return

    print(f"Generating visual validation artifacts for {min(num_samples, len(files))} samples...")

    for idx, f in enumerate(files[:num_samples], start=1):
        with np.load(f, allow_pickle=True) as npz:
            data = npz["data"]  # (3, 45, 93)
            mask = npz["mask"]  # (1, 45, 93)
            sample_id = str(npz["sample_id"])
            class_label = str(npz["class_label"])
            quality = float(npz["quality_score"])

        # Transpose model tensor back to (T, V, C) for plotting
        norm_coords = np.transpose(data, (1, 2, 0))  # (45, 93, 3)
        valid_mask = mask[0]                         # (45, 93)

        # 1. Trajectory plot
        traj_path = os.path.join(output_dir, f"{sample_id}_trajectory.png")
        plot_temporal_trajectory(
            norm_coords,
            valid_mask,
            traj_path,
            title=f"Sample: {sample_id} | Class: {class_label} | Quality: {quality:.2f}"
        )

        # 2. Skeleton Progression Panel (Frames at t=5, 15, 25, 35, 44)
        sample_frames = [5, 15, 25, 35, 44]
        panels = []
        for t in sample_frames:
            t_clamp = min(t, norm_coords.shape[0] - 1)
            skel_img = draw_skeleton_2d(
                norm_coords[t_clamp],
                valid_mask[t_clamp],
                canvas_size=(240, 240),
                is_normalized=True
            )
            # Add frame label text
            cv2.putText(skel_img, f"t={t_clamp}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            panels.append(skel_img)

        strip_img = np.hstack(panels)
        strip_path = os.path.join(output_dir, f"{sample_id}_skeleton_strip.png")
        cv2.imwrite(strip_path, cv2.cvtColor(strip_img, cv2.COLOR_RGB2BGR))

        print(f"Generated: {traj_path} and {strip_path}")

    print("Visual validation artifacts generated successfully in", output_dir)


if __name__ == "__main__":
    generate_sample_visualizations()
