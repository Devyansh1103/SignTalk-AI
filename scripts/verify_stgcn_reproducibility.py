#!/usr/bin/env python3
"""
SignTalk AI: ST-GCN Reproducibility Verification Script.

Executes two identical short training runs with seed 42 on the ST-GCN model
and compares losses and validation metrics to verify deterministic execution.

Outputs:
  - experiments/stgcn/metrics/reproducibility_report.json
"""

import sys
import os
import json
import yaml
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.device import get_device
from src.utils.reproducibility import set_seed
from src.models.stgcn import SignSTGCN
from src.data.dataloader import create_sequence_dataloader


def run_reproducibility_trial(seed=42, epochs=3):
    set_seed(seed=seed)
    device, _ = get_device("cpu")

    train_loader = create_sequence_dataloader(
        manifest_path="data/manifests/train.csv",
        split="train",
        batch_size=8,
        shuffle=True,
        seed=seed,
        filter_rejects=False,
        augment=False
    )
    val_loader = create_sequence_dataloader(
        manifest_path="data/manifests/val.csv",
        split="val",
        batch_size=8,
        shuffle=False,
        filter_rejects=False,
        augment=False
    )

    model = SignSTGCN(
        in_channels=3,
        num_classes=10,
        num_nodes=93,
        sequence_length=45,
        block_channels=[32, 64],
        block_strides=[1, 1],
        temporal_kernel_size=5,
        dropout=0.1
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["label"].to(device)
            mask = batch["mask"].to(device) if "mask" in batch else None

            optimizer.zero_grad()
            out = model(x, mask)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            n_train += len(y)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        n_val = 0
        preds_list = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["label"].to(device)
                mask = batch["mask"].to(device) if "mask" in batch else None

                out = model(x, mask)
                loss = loss_fn(out, y)
                val_loss += loss.item() * len(y)
                n_val += len(y)
                preds = torch.argmax(out, dim=-1).cpu().numpy()
                preds_list.extend(preds.tolist())
        val_loss /= n_val

        history.append({
            "epoch": ep,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_predictions": preds_list
        })

    return history


def main():
    print("\n============================================================")
    print("      SIGNTALK AI: ST-GCN REPRODUCIBILITY VERIFICATION")
    print("============================================================")
    print("Executing Trial 1 (seed=42, 3 epochs)...")
    trial_1 = run_reproducibility_trial(seed=42, epochs=3)

    print("Executing Trial 2 (seed=42, 3 epochs)...")
    trial_2 = run_reproducibility_trial(seed=42, epochs=3)

    print("\nComparing Trial 1 vs. Trial 2:")
    print(f"{'Epoch':<8} {'TrainLoss 1':<14} {'TrainLoss 2':<14} {'Diff':<12} {'ValLoss 1':<12} {'ValLoss 2':<12} {'Diff':<12} {'Match'}")
    print("-" * 95)

    all_matched = True
    max_diff = 0.0
    records = []

    for t1, t2 in zip(trial_1, trial_2):
        ep = t1["epoch"]
        t_diff = abs(t1["train_loss"] - t2["train_loss"])
        v_diff = abs(t1["val_loss"] - t2["val_loss"])
        preds_match = (t1["val_predictions"] == t2["val_predictions"])
        if t_diff > 1e-6 or v_diff > 1e-6 or not preds_match:
            all_matched = False
        max_diff = max(max_diff, t_diff, v_diff)

        records.append({
            "epoch": ep,
            "train_loss_1": t1["train_loss"],
            "train_loss_2": t2["train_loss"],
            "train_loss_diff": t_diff,
            "val_loss_1": t1["val_loss"],
            "val_loss_2": t2["val_loss"],
            "val_loss_diff": v_diff,
            "preds_match": preds_match
        })

        print(f"{ep:<8} {t1['train_loss']:<14.6f} {t2['train_loss']:<14.6f} {t_diff:<12.2e} {t1['val_loss']:<12.6f} {t2['val_loss']:<12.6f} {v_diff:<12.2e} {str(preds_match):<10}")

    print("------------------------------------------------------------")
    print(f"Max Float Discrepancy: {max_diff:.2e}")
    print(f"Deterministic Match:   {'PASSED' if all_matched else 'FAILED'}")
    print("============================================================\n")

    out_file = "experiments/stgcn/metrics/reproducibility_report.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_status": "PASSED" if all_matched else "FAILED",
            "seed": 42,
            "epochs_tested": len(trial_1),
            "max_discrepancy": max_diff,
            "trials": records
        }, f, indent=2)

    print(f"[SUCCESS] Saved reproducibility audit report to: {out_file}")


if __name__ == "__main__":
    main()
