#!/usr/bin/env python3
"""
SignTalk AI: Baseline Reproducibility Verification Script.

Executes two identical short training runs using the exact same random seed (42)
and configuration, then verifies whether losses, metrics, and predictions match
bit-for-bit or within numerical float32 precision limits.

Outputs:
  - experiments/baseline/metrics/reproducibility_report.json
"""

import sys
import os
import copy
import json
import yaml
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.device import get_device
from src.utils.reproducibility import set_seed
from src.models.baseline import SignBaselineModel
from src.data.dataloader import create_sequence_dataloader
from src.training.trainer import ModelTrainer


def run_test_epochs(seed=42, epochs=5):
    set_seed(seed=seed)
    device, _ = get_device("cpu")

    # Load baseline config
    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Recreate dataloaders
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

    model_cfg = config["model"]
    model = SignBaselineModel(
        in_channels=model_cfg.get("in_channels", 3),
        num_nodes=model_cfg.get("num_nodes", 93),
        sequence_length=model_cfg.get("sequence_length", 45),
        proj_dim=model_cfg.get("proj_dim", 128),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.3),
        bidirectional=model_cfg.get("bidirectional", True),
        num_classes=model_cfg.get("num_classes", 10),
        pooling_type=model_cfg.get("pooling_type", "mean_max")
    )
    model.to(device)

    opt_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("learning_rate", 0.001)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4))
    )
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for ep in range(1, epochs + 1):
        # Train 1 epoch
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

        # Eval 1 epoch
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
    print("      SIGNTALK AI: REPRODUCIBILITY VERIFICATION TEST")
    print("============================================================")
    print("Running Run A (seed=42, 5 epochs)...")
    hist_a = run_test_epochs(seed=42, epochs=5)

    print("Running Run B (seed=42, 5 epochs)...")
    hist_b = run_test_epochs(seed=42, epochs=5)

    print("\nComparing Run A and Run B outputs:")
    print(f"{'Epoch':<8} {'TrainLoss A':<14} {'TrainLoss B':<14} {'Diff':<12} {'ValLoss A':<12} {'ValLoss B':<12} {'Diff':<12} {'Pred Match'}")
    print("-" * 95)

    all_matched = True
    max_loss_diff = 0.0
    records = []

    for a, b in zip(hist_a, hist_b):
        ep = a["epoch"]
        t_diff = abs(a["train_loss"] - b["train_loss"])
        v_diff = abs(a["val_loss"] - b["val_loss"])
        preds_match = (a["val_predictions"] == b["val_predictions"])
        if t_diff > 1e-6 or v_diff > 1e-6 or not preds_match:
            all_matched = False
        max_loss_diff = max(max_loss_diff, t_diff, v_diff)

        records.append({
            "epoch": ep,
            "train_loss_a": a["train_loss"],
            "train_loss_b": b["train_loss"],
            "train_loss_diff": t_diff,
            "val_loss_a": a["val_loss"],
            "val_loss_b": b["val_loss"],
            "val_loss_diff": v_diff,
            "preds_match": preds_match
        })

        print(f"{ep:<8} {a['train_loss']:<14.6f} {b['train_loss']:<14.6f} {t_diff:<12.2e} {a['val_loss']:<12.6f} {b['val_loss']:<12.6f} {v_diff:<12.2e} {str(preds_match):<10}")

    print("------------------------------------------------------------")
    print(f"Max Float Difference: {max_loss_diff:.2e}")
    print(f"Deterministic Exact Match: {'PASSED' if all_matched else 'FAILED'}")
    print("============================================================\n")

    out_file = "experiments/baseline/metrics/reproducibility_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_status": "PASSED" if all_matched else "FAILED",
            "seed": 42,
            "epochs_tested": len(hist_a),
            "max_numerical_discrepancy": max_loss_diff,
            "runs": records
        }, f, indent=2)
    print(f"[SUCCESS] Report saved to: {out_file}")


if __name__ == "__main__":
    main()
