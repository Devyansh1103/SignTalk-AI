#!/usr/bin/env python3
"""
SignTalk AI: fine-tune the deployed ST-GCN on your own recordings (scripts/record_signs.py).

Every 5th recording of each word is held out to measure how well the model knows YOUR signing.
Each epoch mixes all your training recordings (repeated) with a random slice of the INCLUDE
training set, so the other 141 words are not forgotten; INCLUDE val accuracy is tracked for that.
Runs on CPU in ~15-30 min. Does NOT replace the live model; it prints the copy command.

Usage:
  python -u scripts/finetune_user.py
  python -u scripts/finetune_user.py --epochs 12 --include-per-epoch 800
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.data.sign_sequence_dataset import SignSequenceDataset
from src.data.label_map import NUM_CLASSES, ID_TO_LABEL
from src.models.stgcn import SignSTGCN
from src.utils.reproducibility import set_seed

OUT_DIR = "experiments/stgcn_user"


def build_model(ckpt):
    cfg = ckpt.get("config", {})
    m = cfg.get("model", {})
    model = SignSTGCN(
        in_channels=3, num_classes=NUM_CLASSES, num_nodes=93, sequence_length=45,
        graph_strategy=cfg.get("graph", {}).get("strategy", "spatial"),
        block_channels=m.get("block_channels", [64, 64, 128, 128, 256, 256]),
        block_strides=m.get("block_strides", [1, 1, 2, 1, 2, 1]),
        temporal_kernel_size=m.get("temporal_kernel_size", 9),
        dropout=m.get("dropout", 0.3), residual=m.get("residual", True),
        use_learnable_edge_weights=m.get("use_learnable_edge_weights", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total, per_word = 0, 0, {}
    for b in loader:
        pred = model(b["x"], b["mask"]).argmax(1)
        for p, y in zip(pred.tolist(), b["label"].tolist()):
            w = per_word.setdefault(ID_TO_LABEL[y], [0, 0])
            w[0] += int(p == y); w[1] += 1
        correct += int((pred == b["label"]).sum()); total += len(pred)
    return correct / max(total, 1), per_word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="experiments/stgcn/checkpoints/best_checkpoint.pt")
    ap.add_argument("--user-manifest", default="data/manifests/user_recordings.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--user-repeat", type=int, default=4, help="times each own recording appears per epoch")
    ap.add_argument("--include-per-epoch", type=int, default=800, help="INCLUDE train clips mixed in per epoch")
    a = ap.parse_args()
    set_seed(42)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    os.makedirs(OUT_DIR, exist_ok=True)

    # Split own recordings: every 5th take of each word -> val
    user = pd.read_csv(a.user_manifest)
    user["take"] = user.groupby("label").cumcount()
    user["split"] = np.where(user["take"] % 5 == 4, "val", "train")
    user.drop(columns="take").to_csv(f"{OUT_DIR}/user_split.csv", index=False)
    print("[DATA] own recordings (train/val):")
    print(user.groupby("label")["split"].value_counts().unstack(fill_value=0).to_string())

    user_train = SignSequenceDataset(f"{OUT_DIR}/user_split.csv", split="train", augment=True)
    user_val = SignSequenceDataset(f"{OUT_DIR}/user_split.csv", split="val")
    inc_train = SignSequenceDataset("data/manifests/train.csv", split="train", augment=True, restrict_to_vocab=True)
    inc_val = SignSequenceDataset("data/manifests/val.csv", split="val", restrict_to_vocab=True)
    user_val_loader = DataLoader(user_val, batch_size=64)
    inc_val_loader = DataLoader(inc_val, batch_size=64)

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr / 10)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def report(tag):
        u_acc, u_words = evaluate(model, user_val_loader) if len(user_val) else (float("nan"), {})
        i_acc, _ = evaluate(model, inc_val_loader)
        words = " ".join(f"{w}:{c}/{n}" for w, (c, n) in sorted(u_words.items()))
        print(f"{tag}  own-val {u_acc:6.1%} [{words}]  INCLUDE-val {i_acc:6.1%}")
        return u_acc, i_acc

    base_u, base_i = report("before    ")
    best = (-1.0, None)
    rng = np.random.default_rng(0)
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        inc_idx = rng.choice(len(inc_train), size=min(a.include_per_epoch, len(inc_train)), replace=False)
        ds = ConcatDataset([user_train] * a.user_repeat + [Subset(inc_train, inc_idx.tolist())])
        model.train()
        total = 0.0
        for b in DataLoader(ds, batch_size=a.batch_size, shuffle=True):
            opt.zero_grad()
            loss = loss_fn(model(b["x"], b["mask"]), b["label"])
            loss.backward()
            opt.step()
            total += float(loss) * len(b["label"])
        sched.step()
        u_acc, i_acc = report(f"epoch {epoch:2d}  ")
        print(f"           loss {total / len(ds):.3f}  ({time.time() - t0:.0f}s)")
        # Prefer own-signing accuracy, but never at the cost of >3 points on the INCLUDE words
        score = (u_acc if len(user_val) else 0) + i_acc if i_acc >= base_i - 0.03 else -1
        if score > best[0]:
            best = (score, epoch)
            torch.save({**ckpt, "epoch": f"{ckpt.get('epoch')}+user{epoch}",
                        "model_state_dict": model.state_dict(), "is_best": True,
                        "val_metrics": {**ckpt.get("val_metrics", {}), "accuracy": i_acc, "user_val_accuracy": u_acc}},
                       f"{OUT_DIR}/best_checkpoint.pt")

    print(f"\nBefore: own-val {base_u:.1%}, INCLUDE-val {base_i:.1%}")
    if best[1] is None:
        print("[WARN] No epoch kept INCLUDE accuracy within 3 points; nothing saved. Try --lr 1e-4.")
        return
    print(f"[OK] Best epoch {best[1]} saved to {OUT_DIR}/best_checkpoint.pt")
    print("To use it live (backend band karke):")
    print(f"  copy {OUT_DIR.replace('/', chr(92))}\\best_checkpoint.pt experiments\\stgcn\\checkpoints\\best_checkpoint.pt")


if __name__ == "__main__":
    main()
