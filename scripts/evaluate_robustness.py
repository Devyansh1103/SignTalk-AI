#!/usr/bin/env python3
"""
SignTalk AI: checks beyond plain test accuracy, for deciding whether a new checkpoint goes live.

For each checkpoint given:
  - INCLUDE test top-1, as recorded and mirrored left <-> right (left-handed signers)
  - top-1 on the held-out own recordings (USER rows of experiments/stgcn/train_data/val.csv,
    written by train_stgcn.py when user recordings exist), per word

Usage:
  python scripts/evaluate_robustness.py experiments/stgcn/checkpoints/best_checkpoint.pt [backups/.../best_checkpoint.pt]
"""
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch

from src.data.label_map import LABEL_TO_ID
from src.data.sign_sequence_dataset import MIRROR_PERM
from src.realtime.model_runner import STGCNRunner


def load(path):
    with np.load(path) as d:
        return d["data"].astype(np.float32), d["mask"].astype(np.float32)


def mirror(x, m):
    x, m = x[:, :, MIRROR_PERM].copy(), m[:, :, MIRROR_PERM].copy()
    x[0] *= -1.0
    return x, m


@torch.no_grad()
def top1(model, samples):
    x = torch.from_numpy(np.stack([s[0] for s in samples]))
    m = torch.from_numpy(np.stack([s[1] for s in samples]))
    return np.concatenate([model(x[i:i + 64], m[i:i + 64]).argmax(1).numpy() for i in range(0, len(x), 64)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--test-manifest", default="data/manifests/test.csv")
    ap.add_argument("--user-val", default="experiments/stgcn/train_data/val.csv")
    a = ap.parse_args()

    test = pd.read_csv(a.test_manifest)
    test = test[test["label"].isin(LABEL_TO_ID)]
    y = test["label"].map(LABEL_TO_ID).to_numpy()
    orig = [load(p) for p in test["file_path"]]
    mirr = [mirror(*s) for s in orig]

    user = None
    if os.path.exists(a.user_val):
        user = pd.read_csv(a.user_val)
        user = user[user["source_dataset"] == "USER"]
        user = user if len(user) else None

    for ckpt in a.checkpoints:
        model = STGCNRunner(checkpoint_path=ckpt, device="cpu", warmup_iterations=0).model
        print(f"\n== {ckpt}")
        print(f"INCLUDE test  top-1 {np.mean(top1(model, orig) == y):6.1%}   mirrored {np.mean(top1(model, mirr) == y):6.1%}   ({len(y)} clips)")
        if user is not None:
            uy = user["label"].map(LABEL_TO_ID).to_numpy()
            ok = top1(model, [load(p) for p in user["file_path"]]) == uy
            per = pd.Series(ok, index=user["label"].to_numpy()).groupby(level=0).agg(["sum", "count"])
            words = " ".join(f"{w}:{int(r['sum'])}/{int(r['count'])}" for w, r in per.iterrows())
            print(f"Own held-out  top-1 {ok.mean():6.1%}   [{words}]")


if __name__ == "__main__":
    main()
