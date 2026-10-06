#!/usr/bin/env python3
"""
SignTalk AI: ST-GCN Training CLI.

Trains the Spatial-Temporal Graph Convolutional Network using parameters
from configs/stgcn.yaml. Saves checkpoints and metrics to experiments/stgcn/.
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
from src.models.stgcn import SignSTGCN
from src.data.dataloader import create_sequence_dataloader
from src.training.trainer import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Spatial-Temporal Graph Convolutional Network (ST-GCN).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stgcn.yaml",
        help="Path to ST-GCN configuration YAML."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Reproducibility & Device Detection
    train_cfg = config["training"]
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    device_pref = config.get("device", "auto")
    device, device_info = get_device(device_pref)

    print("\n============================================================")
    print("      SIGNTALK AI: ST-GCN MODEL TRAINING")
    print("============================================================")
    print(f"Experiment Name:    {config.get('experiment_name')}")
    print(f"Allocated Device:   {device_info['allocated_device']} ({device_info['gpu_name'] or 'CPU'})")
    print(f"Random Seed:        {seed}")
    print(f"Dataset Version:    {config['dataset']['version']}")
    print("============================================================\n")

    # 2. DataLoaders
    paths_cfg = config["paths"]
    batch_size = train_cfg.get("batch_size", 8)
    dataset_cfg = config["dataset"]

    train_loader = create_sequence_dataloader(
        manifest_path=paths_cfg["train_manifest"],
        split="train",
        batch_size=batch_size,
        shuffle=True,
        filter_rejects=dataset_cfg.get("filter_rejects_train", False),
        augment=dataset_cfg.get("augment_train", False),
        include_velocity=dataset_cfg.get("include_velocity", False),
        seed=seed
    )

    val_loader = create_sequence_dataloader(
        manifest_path=paths_cfg["val_manifest"],
        split="val",
        batch_size=batch_size,
        shuffle=False,
        filter_rejects=dataset_cfg.get("filter_rejects_val", False),
        augment=False,
        include_velocity=dataset_cfg.get("include_velocity", False)
    )

    print(f"[DATA] Train Batches: {len(train_loader)} ({len(train_loader.dataset)} samples)")
    print(f"[DATA] Val Batches:   {len(val_loader)} ({len(val_loader.dataset)} samples)")

    # 3. Model Instantiation
    model_cfg = config["model"]
    graph_cfg = config["graph"]
    model = SignSTGCN(
        in_channels=dataset_cfg.get("in_channels", 3),
        num_classes=dataset_cfg.get("num_classes", 10),
        num_nodes=dataset_cfg.get("num_nodes", 93),
        sequence_length=dataset_cfg.get("sequence_length", 45),
        graph_strategy=graph_cfg.get("strategy", "spatial"),
        block_channels=model_cfg.get("block_channels", [64, 64, 128, 128, 256, 256]),
        block_strides=model_cfg.get("block_strides", [1, 1, 2, 1, 2, 1]),
        temporal_kernel_size=model_cfg.get("temporal_kernel_size", 9),
        dropout=model_cfg.get("dropout", 0.3),
        residual=model_cfg.get("residual", True),
        use_learnable_edge_weights=model_cfg.get("use_learnable_edge_weights", True)
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total Parameters:     {total_params:,}")
    print(f"[MODEL] Trainable Parameters: {trainable_params:,}")

    # 4. Optimizer & Scheduler
    lr = float(train_cfg.get("learning_rate", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer_name = train_cfg.get("optimizer", "AdamW")

    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    epochs = train_cfg.get("epochs", 80)
    scheduler_name = train_cfg.get("scheduler", "CosineAnnealingLR")
    if scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    else:
        scheduler = None

    # 5. Loss Function
    loss_fn = nn.CrossEntropyLoss()

    # 6. Class Names
    class_names = [f"Class_{i}" for i in range(dataset_cfg.get("num_classes", 10))]
    vocab_path = paths_cfg.get("vocabulary_file")
    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as vf:
            vocab = json.load(vf)
            c2g = vocab.get("class_id_to_gloss", {})
            class_names = [c2g.get(str(i), f"Class_{i}") for i in range(dataset_cfg.get("num_classes", 10))]

    # 7. Trainer Coordination
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        class_names=class_names
    )

    print("\n[TRAINER] Launching ST-GCN training loop...")
    history = trainer.fit()

    # Save training_history.json and metrics.csv for plotting compatibility
    metrics_dir = paths_cfg.get("metrics_dir", "experiments/stgcn/metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "training_history.json"), "w", encoding="utf-8") as hf:
        json.dump(trainer.history, hf, indent=2)

    import pandas as pd
    df_metrics = pd.DataFrame(trainer.history)
    df_metrics.to_csv(os.path.join(metrics_dir, "metrics.csv"), index=False)

    print("\n============================================================")
    print("             TRAINING COMPLETED SUCCESSFULLY")
    print("============================================================")
    print(f"Best Validation Epoch:     {history.get('best_epoch')}")
    print(f"Best Validation Macro F1:  {history.get('best_metric_val', 0.0):.4f}")
    print(f"Checkpoints directory:     {paths_cfg['checkpoint_dir']}")
    print(f"Metrics directory:         {metrics_dir}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
