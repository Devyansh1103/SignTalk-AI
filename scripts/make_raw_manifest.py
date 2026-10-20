#!/usr/bin/env python3
"""
SignTalk AI: Build data/metadata/raw_dataset_manifest.csv from the videos in data/raw/videos.

Only files named {label}_signer_{id}_rep{n}.mp4 with a label from the 10-class vocab are used.
Split: stratified per class (reps are shuffled with a fixed seed): ~70% train / 15% val / 15% test,
with at least 1 val and 1 test clip per class when the class has >= 5 clips.
NOTE: with a single signer this measures same-person accuracy only.

Usage: python scripts/make_raw_manifest.py
"""
import argparse, os, re, sys, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2, pandas as pd
from src.data.label_map import LABEL_TO_ID

PAT = re.compile(r"^(?P<label>[a-z]+)_signer_(?P<signer>\d+)_rep(?P<rep>\d+)\.(mp4|mov|avi|mkv)$", re.I)
ap = argparse.ArgumentParser()
ap.add_argument("--videos", default="data/raw/videos")
ap.add_argument("--out", default="data/metadata/raw_dataset_manifest.csv")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

by = {}
for f in sorted(os.listdir(a.videos)):
    m = PAT.match(f)
    if m and m.group("label").lower() in LABEL_TO_ID:
        by.setdefault(m.group("label").lower(), []).append(f)
rng = random.Random(a.seed)
rows, i = [], 1
for label, files in by.items():
    rng.shuffle(files)
    n = len(files)
    n_val = max(1, round(n * 0.15)) if n >= 5 else 0
    n_test = max(1, round(n * 0.15)) if n >= 5 else 0
    for j, f in enumerate(files):
        split = "val" if j < n_val else "test" if j < n_val + n_test else "train"
        p = os.path.join(a.videos, f)
        cap = cv2.VideoCapture(p)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
        rows.append(dict(sample_id=f"raw_{i:04d}", video_file=f, video_path=p.replace("/", os.sep),
                         class_label=label, class_id=LABEL_TO_ID[label],
                         signer_id=f"signer_{PAT.match(f).group('signer')}", split=split,
                         frame_count=fc, fps=fps, duration_sec=round(fc / fps, 2), resolution=f"{w}x{h}"))
        i += 1
df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
df.to_csv(a.out, index=False)
print(df.groupby(["class_label", "split"]).size().unstack(fill_value=0))
print("written:", a.out, len(df), "clips")
