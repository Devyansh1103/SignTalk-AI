#!/usr/bin/env python3
"""
SignTalk AI: Baseline Model Training CLI.

Executes reproducible training of the BiGRU sequence baseline model using
parameters defined in configs/baseline.yaml. Saves checkpoints and metrics
to experiments/baseline/.
"""

import sys
import os
import argparse
import json
import yaml
import torch
import torch.nn as nn

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.device import get_device
from src.utils.reproducibility import set_seed
from src.models.baseline import SignBaselineModel
from src.data.dataloader import create_sequence_dataloader
from src.training.trainer import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description="Train baseline BiGRU sequence classifier.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to baseline configuration YAML."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Reproducibility & Device Initialization
    train_cfg = config["training"]
    seed = train_cfg.get("seed", 42)
    seed_info = set_seed(seed)

    device_pref = train_cfg.get("device", "auto")
    device, device_info = get_device(device_pref)

    print("\n============================================================")
    print("      SIGNTALK AI: BASELINE MODEL TRAINING (BiGRU)")
    print("============================================================")
    print(f"Experiment Name:    {train_cfg.get('experiment_name')}")
    print(f"Allocated Device:   {device_info['allocated_device']} ({device_info['gpu_name'] or 'CPU'})")
    print(f"Random Seed:        {seed}")
    print(f"Dataset Version:    {config['dataset_metadata']['dataset_version']}")
    print("============================================================\n")

    # 2. DataLoaders
    paths_cfg = config["paths"]
    batch_size = train_cfg.get("batch_size", 8)
    filter_rej = train_cfg.get("filter_rejects", True)
    augment = train_cfg.get("augment_train", True)

    train_loader = create_sequence_dataloader(
        manifest_path=paths_cfg["train_manifest"],
        split="train",
        batch_size=batch_size,
        shuffle=True,
        filter_rejects=filter_rej,
        augment=augment,
        seed=seed
    )

    val_loader = create_sequence_dataloader(
        manifest_path=paths_cfg["val_manifest"],
        split="val",
        batch_size=batch_size,
        shuffle=False,
        filter_rejects=False,
        augment=False,
        seed=seed
    )

    print(f"[INFO] Train samples: {len(train_loader.dataset)} (Filter Rejects={filter_rej})")
    print(f"[INFO] Val samples:   {len(val_loader.dataset)}")

    # 3. Model Instantiation
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

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model initialized: {total_params:,} parameters ({trainable_params:,} trainable)")

    # 4. Optimizer & Scheduler
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    opt_name = train_cfg.get("optimizer", "adamw").lower()

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    epochs = train_cfg.get("epochs", 40)
    sched_name = train_cfg.get("scheduler", "cosine").lower()
    min_lr = train_cfg.get("min_lr", 1e-5)

    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    else:
        scheduler = None

    loss_fn = nn.CrossEntropyLoss()

    # Load class names from vocabulary if available
    class_names = None
    vocab_path = "assets/vocabularies/mvp_10.json"
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as vf:
            vocab = json.load(vf)
            c2g = vocab.get("class_id_to_gloss", {})
            class_names = [c2g.get(str(i), f"Class_{i}") for i in range(10)]

    # 5. Training Coordinator
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=config,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names
    )

    fit_summary = trainer.fit()

    # Save summary artifact
    summary_path = os.path.join(paths_cfg["experiment_dir"], "training_summary.json")
    fit_summary["hardware"] = device_info
    fit_summary["total_parameters"] = total_params
    fit_summary["trainable_parameters"] = trainable_params

    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(fit_summary, sf, indent=2)
    print(f"[SUCCESS] Saved training summary to: {summary_path}")


if __name__ == "__main__":
    main()
