#!/usr/bin/env python3
"""
SignTalk AI: draw live sign segments next to a training clip of the expected sign.

Live segments are saved by the backend when configs/realtime.yaml has debug.save_segments: true
(experiments/realtime_debug/segments/*.npz). Each row of the output image is one clip,
showing 8 skeleton frames in the model's normalized space (mid-shoulder centered, shoulder-width scaled).

Usage:
  python scripts/plot_segments.py --expected hello
  python scripts/plot_segments.py --expected hello --segments experiments/realtime_debug/segments/1530*_*.npz
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import numpy as np
import pandas as pd

from src.preprocessing.sequence_utils import handle_missing_landmarks
from src.preprocessing.normalizer import CoordinateNormalizer

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
CELL = 160


def draw(coords, mask, label):
    """coords (T, 93, 3) normalized; returns one row image of 8 frames."""
    cells = []
    for t in np.linspace(0, len(coords) - 1, 8).astype(int):
        im = np.full((CELL, CELL, 3), 255, np.uint8)
        px = lambda p: (int(CELL / 2 + p[0] * CELL / 5), int(CELL * 0.3 + p[1] * CELL / 5))
        c, m = coords[t], mask[t]
        for i in range(42, 53):
            if m[i]:
                cv2.circle(im, px(c[i]), 2, (150, 150, 150), -1)
        for off, col in ((0, (200, 60, 0)), (21, (0, 60, 200))):
            for a, b in HAND_EDGES:
                if m[off + a] and m[off + b]:
                    cv2.line(im, px(c[off + a]), px(c[off + b]), col, 1)
        cv2.putText(im, f"t{t}", (3, CELL - 5), 0, 0.35, (0, 0, 0), 1)
        cells.append(im)
    row = np.hstack(cells)
    cv2.putText(row, label, (5, 14), 0, 0.45, (0, 0, 180), 1)
    return row


def live_row(path):
    z = np.load(path)
    raw = z["raw_coords"].copy()
    raw[:, :, 1] *= z["aspect_y_scale"][:, None]
    mask = z["mask"].astype(bool)
    vis = mask.astype(np.float32)
    raw, mask = handle_missing_landmarks(raw, mask, strategy="interpolate_and_mask", max_consecutive_missing=10, visibility=vis)
    coords = CoordinateNormalizer(method="torso_scale").normalize_sequence(raw, mask)[0]  # same as live model input
    hand_rate = mask[:, :42].any(axis=1).mean()
    return draw(coords, mask, f"LIVE {os.path.basename(path)} | hands in {hand_rate:.0%} frames | {' '.join(z['top_k'])}")


def train_row(label, k):
    d = pd.read_csv("data/manifests/train.csv")
    rows = []
    for p in d[d["label"] == label]["file_path"].head(k):
        with np.load(p) as z:
            data, mask = z["data"], z["mask"][0].astype(bool)  # (3, 45, 93), (45, 93)
        rows.append(draw(np.transpose(data, (1, 2, 0)), mask, f"TRAIN {label} {os.path.basename(p)}"))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected", required=True, help="label the user meant to sign, e.g. hello")
    ap.add_argument("--segments", nargs="*", help="segment .npz files (default: 6 newest)")
    ap.add_argument("--train-clips", type=int, default=2)
    ap.add_argument("--out", default="experiments/realtime_debug/compare.png")
    a = ap.parse_args()
    files = a.segments or sorted(glob.glob("experiments/realtime_debug/segments/*.npz"), key=os.path.getmtime)[-6:]
    rows = train_row(a.expected, a.train_clips) + [live_row(f) for f in files]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cv2.imwrite(a.out, np.vstack(rows))
    print(f"[OK] {len(rows)} rows -> {a.out}")
