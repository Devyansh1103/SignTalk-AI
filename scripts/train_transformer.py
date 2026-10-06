"""
SignTalk AI: Transformer Translation Layer Training Script.

Trains the SignLanguageTransformer and FeatureProjection layers using the
pre-trained and frozen SignSTGCN visual encoder:
  - Input: Standardized spatiotemporal landmark sequence [B, 3, 45, 93]
  - Target: Shifted token sequences [B, S]
  - Optimizes CrossEntropyLoss over token vocabulary
  - Evaluates token accuracy, sequence exact match, and validation macro F1
  - Saves latest and best validation checkpoints
"""

import os
import sys
import argparse
import time

# Ensure workspace root is in python path
sys.path.insert(0, os.getcwd())

import yaml
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.sign_sequence_dataset import SignSequenceDataset
from src.data.collate import sign_sequence_collate_fn
from src.models.sign_translation_model import SignTranslationModel
from src.evaluation.sequence_metrics import (
    compute_token_accuracy,
    compute_sequence_accuracy,
    compute_token_classification_metrics
)
from src.utils.reproducibility import set_seed
from src.utils.device import resolve_device


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_norm: float = 1.0,
    pad_idx: int = 0
) -> dict:
    """Trains model for a single epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        dec_in = batch["decoder_input"].to(device)
        dec_tgt = batch["decoder_target"].to(device)

        optimizer.zero_grad()

        # Forward pass through translation model
        # logits shape: [B, S, vocab_size]
        logits = model(x=x, target_tokens=dec_in, mask=mask)

        B, S, V = logits.shape
        loss = criterion(logits.view(-1, V), dec_tgt.view(-1))

        loss.backward()
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=clip_norm
            )
        optimizer.step()

        total_loss += loss.item() * B

        # Predictions for token and sequence metrics
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(dec_tgt.detach().cpu())

    epoch_preds = torch.cat(all_preds, dim=0)
    epoch_targets = torch.cat(all_targets, dim=0)

    token_acc = compute_token_accuracy(epoch_preds, epoch_targets, pad_idx=pad_idx)
    seq_acc = compute_sequence_accuracy(epoch_preds, epoch_targets, pad_idx=pad_idx)

    return {
        "loss": total_loss / len(dataloader.dataset),
        "token_acc": token_acc,
        "seq_acc": seq_acc
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_idx: int = 0
) -> dict:
    """Evaluates model on validation/test split."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_gen_seqs = []
    all_gt_seqs = []

    for batch in dataloader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        dec_in = batch["decoder_input"].to(device)
        dec_tgt = batch["decoder_target"].to(device)

        logits = model(x=x, target_tokens=dec_in, mask=mask)
        B, S, V = logits.shape
        loss = criterion(logits.view(-1, V), dec_tgt.view(-1))
        total_loss += loss.item() * B

        preds = logits.argmax(dim=-1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(dec_tgt.detach().cpu())

        # Also evaluate autoregressive generation
        gen_tokens, _ = model.generate(x=x, mask=mask, max_length=5)
        # Drop <BOS> for sequence exact match comparison against decoder_target
        gen_no_bos = gen_tokens[:, 1:].detach().cpu()
        all_gen_seqs.append(gen_no_bos)
        all_gt_seqs.append(dec_tgt.detach().cpu())

    epoch_preds = torch.cat(all_preds, dim=0)
    epoch_targets = torch.cat(all_targets, dim=0)
    gen_preds = torch.cat(all_gen_seqs, dim=0)
    gt_targets = torch.cat(all_gt_seqs, dim=0)

    token_acc = compute_token_accuracy(epoch_preds, epoch_targets, pad_idx=pad_idx)
    teacher_forced_seq_acc = compute_sequence_accuracy(epoch_preds, epoch_targets, pad_idx=pad_idx)
    gen_seq_acc = compute_sequence_accuracy(gen_preds, gt_targets, pad_idx=pad_idx)
    class_metrics = compute_token_classification_metrics(epoch_preds, epoch_targets, pad_idx=pad_idx)

    return {
        "loss": total_loss / len(dataloader.dataset),
        "token_acc": token_acc,
        "teacher_seq_acc": teacher_forced_seq_acc,
        "gen_seq_acc": gen_seq_acc,
        "macro_precision": class_metrics["macro_precision"],
        "macro_recall": class_metrics["macro_recall"],
        "macro_f1": class_metrics["macro_f1"],
        "weighted_f1": class_metrics["weighted_f1"]
    }


def main():
    parser = argparse.ArgumentParser(description="SignTalk AI — Train Sign Language Transformer")
    parser.add_argument("--config", type=str, default="configs/transformer.yaml", help="Path to YAML config")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["learning_rate"]

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = resolve_device(cfg.get("device", "auto"))
    print(f"[Train Transformer] Device: {device}, Seed: {seed}, Batch Size: {batch_size}, LR: {lr}")

    # Prepare directories
    paths = cfg["paths"]
    checkpoint_dir = paths["checkpoint_dir"]
    metrics_dir = paths["metrics_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Datasets and Loaders
    train_dataset = SignSequenceDataset(manifest_path=paths["train_manifest"], augment=cfg["dataset"].get("augment_train", False))
    val_dataset = SignSequenceDataset(manifest_path=paths["val_manifest"], augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sign_sequence_collate_fn,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sign_sequence_collate_fn,
        drop_last=False
    )

    print(f"[Train Transformer] Loaded {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    # Initialize Model
    stgcn_cfg = cfg.get("stgcn", {})
    transformer_cfg = cfg.get("transformer", {})
    stgcn_checkpoint = stgcn_cfg.get("checkpoint", None)

    model = SignTranslationModel(
        stgcn_config=stgcn_cfg,
        transformer_config=transformer_cfg,
        freeze_stgcn=stgcn_cfg.get("freeze", True),
        stgcn_checkpoint=stgcn_checkpoint
    ).to(device)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train Transformer] Parameters: Total={total_params:,}, Trainable={trainable_params:,}, Frozen={total_params - trainable_params:,}")

    # Setup Optimizer & Scheduler
    weight_decay = cfg["training"].get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5
    )

    pad_idx = transformer_cfg.get("pad_idx", 0)
    label_smoothing = cfg["training"].get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=label_smoothing)

    # Resume handling
    start_epoch = 1
    best_metric = -1.0
    history = []
    latest_ckpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    best_ckpt_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")

    if args.resume and os.path.exists(latest_ckpt_path):
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", -1.0)
        print(f"[Train Transformer] Resumed from epoch {ckpt['epoch']}")

    patience = cfg["training"].get("early_stopping_patience", 20)
    patience_counter = 0

    print(f"\n{'='*75}\nStarting Training: {epochs} Epochs\n{'='*75}")
    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        t_epoch_start = time.time()
        train_res = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            clip_norm=cfg["training"].get("clip_norm", 1.0),
            pad_idx=pad_idx
        )
        val_res = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            pad_idx=pad_idx
        )
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]
        epoch_sec = time.time() - t_epoch_start

        # Monitor generated sequence accuracy for early stopping
        primary_metric = val_res["gen_seq_acc"]
        is_best = primary_metric > best_metric
        if is_best:
            best_metric = primary_metric
            patience_counter = 0
        else:
            patience_counter += 1

        # Checkpoint saving
        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_res": train_res,
            "val_res": val_res,
            "best_metric": best_metric,
            "config": cfg,
            "seed": seed
        }
        torch.save(ckpt_state, latest_ckpt_path)
        if is_best:
            torch.save(ckpt_state, best_ckpt_path)

        # Log history
        record = {
            "epoch": epoch,
            "train_loss": round(train_res["loss"], 4),
            "train_token_acc": round(train_res["token_acc"], 4),
            "train_seq_acc": round(train_res["seq_acc"], 4),
            "val_loss": round(val_res["loss"], 4),
            "val_token_acc": round(val_res["token_acc"], 4),
            "val_teacher_seq_acc": round(val_res["teacher_seq_acc"], 4),
            "val_gen_seq_acc": round(val_res["gen_seq_acc"], 4),
            "val_macro_f1": round(val_res["macro_f1"], 4),
            "lr": f"{cur_lr:.6f}",
            "epoch_sec": round(epoch_sec, 2),
            "is_best": is_best
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_sec:.1f}s] | "
            f"Train Loss: {train_res['loss']:.4f} (TokAcc: {train_res['token_acc']*100:.1f}%) | "
            f"Val Loss: {val_res['loss']:.4f} (TokAcc: {val_res['token_acc']*100:.1f}%, GenSeqAcc: {val_res['gen_seq_acc']*100:.1f}%, F1: {val_res['macro_f1']*100:.1f}%) "
            f"{'[*] BEST' if is_best else ''}"
        )

        # Early stopping check
        if patience_counter >= patience:
            print(f"\n[Early Stopping] No improvement for {patience} consecutive epochs. Terminating training at Epoch {epoch}.")
            break

    total_training_sec = time.time() - start_time
    print(f"\nTraining completed in {total_training_sec:.2f}s. Best Val GenSeqAcc: {best_metric*100:.2f}%")

    # Save metrics CSV
    metrics_df = pd.DataFrame(history)
    csv_path = os.path.join(metrics_dir, "metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"[Train Transformer] Saved training history to {csv_path}")


if __name__ == "__main__":
    main()
