#!/usr/bin/env python3
"""
SignTalk AI: Small Subset Overfit Test.

Trains ST-GCN on a tiny subset of 4 distinct training samples for 30 epochs
to confirm capacity to achieve ~100% training accuracy and verify gradient
propagation without optimization bottlenecks.
"""

import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.device import get_device
from src.utils.reproducibility import set_seed
from src.models.stgcn import SignSTGCN
from src.data.sign_sequence_dataset import SignSequenceDataset


def run_overfit_test():
    print("\n============================================================")
    print("      SIGNTALK AI: ST-GCN SMALL SUBSET OVERFIT TEST")
    print("============================================================")

    set_seed(42)
    device, dev_info = get_device("auto")
    print(f"[OVERFIT] Device: {device} ({dev_info.get('name', 'CPU')})")

    full_dataset = SignSequenceDataset(
        manifest_path="data/manifests/train.csv",
        split="train",
        filter_rejects=False,
        augment=False
    )

    # Select 4 distinct samples
    subset_indices = [0, 4, 8, 12]  # seq_0001 (hello), seq_0005 (thankyou), seq_0009 (good), seq_0013 (happy)
    subset_dataset = Subset(full_dataset, subset_indices)
    loader = DataLoader(subset_dataset, batch_size=4, shuffle=False)

    model = SignSTGCN(
        in_channels=3,
        num_classes=10,
        num_nodes=93,
        sequence_length=45,
        block_channels=[32, 64, 128],
        block_strides=[1, 1, 1],
        temporal_kernel_size=5,
        dropout=0.0  # Zero dropout for pure fitting capacity
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    print("\nEpoch   Loss       Accuracy   Correct/Total")
    print("-------------------------------------------")

    model.train()
    for epoch in range(1, 31):
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["label"].to(device)
            mask = batch["mask"].to(device) if "mask" in batch else None

            optimizer.zero_grad()
            logits = model(x, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=-1)
            correct = (preds == y).sum().item()
            acc = correct / len(y)

            history.append({
                "epoch": epoch,
                "loss": float(loss.item()),
                "accuracy": float(acc),
                "correct": correct,
                "total": len(y)
            })

            if epoch % 5 == 0 or epoch == 1 or acc == 1.0:
                print(f"{epoch:<7} {loss.item():<10.4f} {acc*100:<10.1f}% {correct}/{len(y)}")

    final_loss = history[-1]["loss"]
    final_acc = history[-1]["accuracy"]
    passed = (final_acc >= 1.0 or final_loss < 0.1)

    print("-------------------------------------------")
    print(f"Final Loss:     {final_loss:.4f}")
    print(f"Final Accuracy: {final_acc*100:.1f}%")
    print(f"Overfit Status: {'PASSED (100% capacity verified)' if passed else 'FAILED'}")
    print("============================================================\n")

    out_dir = "experiments/stgcn/overfit_test"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "overfit_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "test_status": "PASSED" if passed else "FAILED",
            "samples_tested": len(subset_indices),
            "final_loss": final_loss,
            "final_accuracy": final_acc,
            "history": history
        }, f, indent=2)

    return passed


if __name__ == "__main__":
    run_overfit_test()
