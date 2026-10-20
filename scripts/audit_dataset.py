#!/usr/bin/env python3
"""
SignTalk AI: Dataset audit (run this BEFORE training).

1. Video audit  : counts clips per class in data/raw/videos, flags bad names / low counts,
                  and writes one contact-sheet PNG per class so you can EYEBALL that the
                  video really shows that sign (and has no on-screen text overlay).
2. Manifest audit: checks train/val/test manifests for non-vocab labels, class_id collisions,
                  out-of-range ids and per-class sample counts.

Usage:
  python scripts/audit_dataset.py
  python scripts/audit_dataset.py --videos data/raw/videos --min-clips 15
"""
import argparse, os, re, sys, glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import numpy as np
import pandas as pd
from src.data.label_map import LABEL_TO_ID

PAT = re.compile(r"^(?P<label>[a-z]+)_signer_(?P<signer>\d+)_rep(?P<rep>\d+)\.(mp4|mov|avi|mkv)$", re.I)


def contact_sheet(paths, out_path, n_frames=5, size=(160, 120)):
    rows = []
    for p in paths[:6]:
        cap = cv2.VideoCapture(p)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        row = []
        for i in np.linspace(0, max(total - 1, 0), n_frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            fr = cv2.resize(fr, size) if ok else np.zeros((size[1], size[0], 3), np.uint8)
            row.append(fr)
        cap.release()
        rows.append(np.hstack(row))
    if rows:
        cv2.imwrite(out_path, np.vstack(rows))


def video_audit(video_dir, min_clips, out_dir):
    print(f"\n=== VIDEO AUDIT: {video_dir} ===")
    if str(video_dir).upper() == "NONE":
        print("skipped (--videos NONE)")
        return True
    if not os.path.isdir(video_dir):
        print(f"[ERROR] folder not found: {video_dir}")
        return False
    files = sorted(f for f in os.listdir(video_dir) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")))
    by_label, bad = {}, []
    for f in files:
        m = PAT.match(f)
        if not m or m.group("label").lower() not in LABEL_TO_ID:
            bad.append(f)
            continue
        by_label.setdefault(m.group("label").lower(), []).append(os.path.join(video_dir, f))
    ok = True
    os.makedirs(out_dir, exist_ok=True)
    for label in LABEL_TO_ID:
        clips = by_label.get(label, [])
        signers = {PAT.match(os.path.basename(c)).group("signer") for c in clips}
        flag = "" if len(clips) >= min_clips else f"  <-- kam hai (min {min_clips})"
        if flag: ok = False
        print(f"{label:10s} clips={len(clips):3d} signers={len(signers)}{flag}")
        if clips:
            contact_sheet(clips, os.path.join(out_dir, f"{label}.png"))
    if bad:
        ok = False
        print(f"\n[WARN] {len(bad)} files naming pattern ya vocab se match nahi karti, e.g. {bad[:5]}")
    print(f"\nContact sheets: {out_dir}/<label>.png  -> inhe KHUD dekho. Har sheet me wahi sign dikhna chahiye,"
          " aur frame pe koi text overlay (jaise 'how are you : 99%') nahi hona chahiye.")
    return ok


def manifest_audit(manifest_dir):
    print(f"\n=== MANIFEST AUDIT: {manifest_dir} ===")
    ok = True
    for split in ("train", "val", "test"):
        p = os.path.join(manifest_dir, f"{split}.csv")
        if not os.path.exists(p):
            print(f"[skip] {p} missing"); continue
        d = pd.read_csv(p)
        non = d[~d["label"].isin(LABEL_TO_ID)]
        mism = d[d["label"].isin(LABEL_TO_ID) & (d["class_id"] != d["label"].map(LABEL_TO_ID))]
        counts = d[d["label"].isin(LABEL_TO_ID)]["label"].value_counts().to_dict()
        print(f"{split}: rows={len(d)} non-vocab={len(non)} class_id-mismatch={len(mism)} per-class={counts}")
        if len(non) or len(mism): print("   -> train_stgcn ab restrict_to_vocab se inhe handle karta hai, par manifest rebuild better hai.")
        missing = [l for l in LABEL_TO_ID if l not in counts]
        if missing:
            ok = False; print(f"   [ERROR] {split} me ye classes hi nahi hain: {missing}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/raw/videos")
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--min-clips", type=int, default=15)
    ap.add_argument("--out", default="data/interim/audit")
    a = ap.parse_args()
    v = video_audit(a.videos, a.min_clips, a.out)
    m = manifest_audit(a.manifests)
    print("\nRESULT:", "OK" if (v and m) else "ISSUES FOUND (upar dekho)")
    sys.exit(0 if (v and m) else 1)
