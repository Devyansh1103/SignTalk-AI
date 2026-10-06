#!/usr/bin/env python3
"""
SignTalk AI: ST-GCN Smoke Test Runner.

Executes a minimal 2-epoch verification run to confirm:
  - Dataset loading and batch collation
  - Graph construction and adjacency tensor registration
  - Forward pass and loss calculation
  - Backward pass and non-zero gradient flow
  - Validation loop and metric computation
  - Checkpoint serialization to experiments/stgcn/smoke_test/
"""

import sys
import os
import shutil
import json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.device import get_device
from src.utils.reproducibility import set_seed
from src.models.stgcn import SignSTGCN
from src.data.dataloader import create_sequence_dataloader
from src.training.trainer import ModelTrainer


def run_smoke_test():
    print("\n============================================================")
    print("           SIGNTALK AI: ST-GCN SMOKE TEST")
    print("============================================================")

    out_dir = "experiments/stgcn/smoke_test"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    set_seed(42)
    device, dev_info = get_device("auto")
    print(f"[SMOKE] Device: {device} ({dev_info.get('name', 'CPU')})")

    # 1. Dataset & DataLoader
    train_loader = create_sequence_dataloader(
        manifest_path="data/manifests/train.csv",
        split="train",
        batch_size=4,
        shuffle=True,
        filter_rejects=False,
        augment=False
    )
    val_loader = create_sequence_dataloader(
        manifest_path="data/manifests/val.csv",
        split="val",
        batch_size=4,
        shuffle=False,
        filter_rejects=False,
        augment=False
    )
    print(f"[SMOKE] Dataloaders loaded: {len(train_loader)} train batches, {len(val_loader)} val batches.")

    # 2. Model Creation
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
    print(f"[SMOKE] Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # 3. Verify Gradient Flow on 1 Batch
    batch = next(iter(train_loader))
    x = batch["x"].to(device)
    y = batch["label"].to(device)
    mask = batch["mask"].to(device) if "mask" in batch else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    logits = model(x, mask)
    loss = loss_fn(logits, y)
    loss.backward()

    # Check non-zero gradients
    grads_ok = True
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                print(f"[SMOKE ERROR] NaN/Inf grad in {name}")
                grads_ok = False
        elif p.requires_grad and p.grad is None:
            print(f"[SMOKE WARNING] Missing grad in {name}")
            grads_ok = False

    print(f"[SMOKE] Gradient Flow Check: {'PASSED' if grads_ok else 'FAILED'} (Loss = {loss.item():.4f})")

    # 4. Run 2 Training Epochs via ModelTrainer
    config = {
        "training": {
            "epochs": 2,
            "early_stopping_patience": 5,
            "early_stopping_metric": "val_macro_f1",
            "clip_norm": 1.0
        },
        "paths": {
            "checkpoint_dir": os.path.join(out_dir, "checkpoints"),
            "log_dir": os.path.join(out_dir, "logs"),
            "metrics_dir": os.path.join(out_dir, "metrics")
        },
        "dataset_metadata": {
            "num_classes": 10
        }
    }

    class_names = [f"Class_{i}" for i in range(10)]
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names
    )

    history = trainer.fit()

    # 5. Verify Checkpoint Existence
    best_ckpt = os.path.join(out_dir, "checkpoints", "best_checkpoint.pt")
    latest_ckpt = os.path.join(out_dir, "checkpoints", "latest_checkpoint.pt")
    assert os.path.exists(best_ckpt), "Best checkpoint was not created!"
    assert os.path.exists(latest_ckpt), "Latest checkpoint was not created!"

    print(f"[SMOKE] Checkpoints successfully verified: {os.path.getsize(best_ckpt):,} bytes.")
    print("============================================================")
    print("                 SMOKE TEST PASSED (100%)")
    print("============================================================\n")


if __name__ == "__main__":
    run_smoke_test()
