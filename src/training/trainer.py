"""
SignTalk AI: Unified Model Training and Evaluation Engine.

Provides an extensible, reusable training pipeline supporting:
  - Gradient clipping and mixed precision readiness
  - Multi-metric tracking (Accuracy, Macro/Weighted F1, Loss)
  - Model checkpointing (best and latest) with full optimizer/scheduler state
  - Configurable early stopping and learning rate scheduling
  - Logging of epoch histories to CSV and JSON formats
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.classification_metrics import compute_classification_metrics, format_metrics_table


class ModelTrainer:
    """
    Unified training and evaluation coordinator for SignTalk AI neural architectures.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        scheduler: Optional[Any] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        class_names: Optional[List[str]] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names

        training_cfg = config.get("training", {})
        self.epochs = training_cfg.get("epochs", 40)
        self.clip_norm = training_cfg.get("gradient_clip_norm", 1.0)
        self.patience = training_cfg.get("early_stopping_patience", 15)
        self.es_metric = training_cfg.get("early_stopping_metric", "val_macro_f1")
        self.es_mode = training_cfg.get("early_stopping_mode", "max")

        paths_cfg = config.get("paths", {})
        self.ckpt_dir = paths_cfg.get("checkpoint_dir", "experiments/baseline/checkpoints")
        self.log_dir = paths_cfg.get("log_dir", "experiments/baseline/logs")
        self.metrics_dir = paths_cfg.get("metrics_dir", "experiments/baseline/metrics")
        self.exp_name = training_cfg.get("experiment_name", "experiment_v1")

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.history: List[Dict[str, Any]] = []
        self.best_metric_val = -float("inf") if self.es_mode == "max" else float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, Any]]:
        """Executes one training epoch."""
        self.model.train()
        total_loss = 0.0
        all_true = []
        all_pred = []
        all_probs = []

        for batch in self.train_loader:
            x = batch["x"].to(self.device)
            mask = batch["mask"].to(self.device) if "mask" in batch else None
            targets = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x, mask)
            loss = self.loss_fn(logits, targets)
            loss.backward()

            if self.clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

            self.optimizer.step()

            total_loss += loss.item() * len(targets)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_true.extend(targets.cpu().numpy().tolist())
            all_pred.extend(preds.tolist())
            all_probs.extend(probs.tolist())

        avg_loss = total_loss / len(self.train_loader.dataset)
        num_classes = self.config.get("dataset_metadata", {}).get("num_classes", 10)
        metrics = compute_classification_metrics(
            all_true, all_pred, np.array(all_probs), num_classes=num_classes, class_names=self.class_names
        )
        return avg_loss, metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Evaluates model performance over a DataLoader."""
        self.model.eval()
        total_loss = 0.0
        all_true = []
        all_pred = []
        all_probs = []
        all_meta = []

        for batch in loader:
            x = batch["x"].to(self.device)
            mask = batch["mask"].to(self.device) if "mask" in batch else None
            targets = batch["label"].to(self.device)

            logits = self.model(x, mask)
            loss = self.loss_fn(logits, targets)
            total_loss += loss.item() * len(targets)

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_true.extend(targets.cpu().numpy().tolist())
            all_pred.extend(preds.tolist())
            all_probs.extend(probs.tolist())

            if "metadata" in batch:
                all_meta.extend(batch["metadata"])

        avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0
        num_classes = self.config.get("dataset_metadata", {}).get("num_classes", 10)
        metrics = compute_classification_metrics(
            all_true, all_pred, np.array(all_probs), num_classes=num_classes, class_names=self.class_names
        )

        return (
            avg_loss,
            metrics,
            np.array(all_true),
            np.array(all_pred),
            np.array(all_probs),
            all_meta
        )

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        val_metrics: Dict[str, Any],
        is_best: bool = False
    ) -> None:
        """Serializes model weights, optimizer, and training metadata to disk."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "val_metrics": val_metrics,
            "config": self.config,
            "is_best": is_best
        }
        torch.save(ckpt, filepath)

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Loads model weights and returns checkpoint metadata."""
        ckpt = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and self.optimizer:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and self.scheduler and ckpt["scheduler_state_dict"]:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt

    def fit(self) -> Dict[str, Any]:
        """Executes full training across configured epochs with validation and early stopping."""
        print(f"\n[INFO] Starting training for experiment '{self.exp_name}' on device: {self.device}")
        print(f"       Epochs: {self.epochs}, Patience: {self.patience}, Target Metric: {self.es_metric}\n")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            ep_start = time.time()
            train_loss, train_metrics = self.train_epoch(epoch)

            # Validation step
            val_loss, val_metrics, _, _, _, _ = self.evaluate(self.val_loader)

            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            ep_time = time.time() - ep_start

            # Record epoch statistics
            record = {
                "epoch": epoch,
                "lr": round(current_lr, 6),
                "train_loss": round(train_loss, 4),
                "train_acc": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": round(val_loss, 4),
                "val_acc": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "epoch_time_sec": round(ep_time, 2)
            }
            self.history.append(record)

            # Check improvement on early stopping metric
            current_metric = record.get(self.es_metric, val_metrics["macro_f1"])
            is_better = (current_metric > self.best_metric_val) if self.es_mode == "max" else (current_metric < self.best_metric_val)

            if is_better:
                self.best_metric_val = current_metric
                self.best_epoch = epoch
                self.patience_counter = 0
                best_ckpt_path = os.path.join(self.ckpt_dir, "best_checkpoint.pt")
                self.save_checkpoint(best_ckpt_path, epoch, val_metrics, is_best=True)
                marker = " [*BEST]"
            else:
                self.patience_counter += 1
                marker = ""

            # Save latest checkpoint
            latest_ckpt_path = os.path.join(self.ckpt_dir, "latest_checkpoint.pt")
            self.save_checkpoint(latest_ckpt_path, epoch, val_metrics, is_best=False)

            print(
                f"Epoch {epoch:02d}/{self.epochs:02d} [{ep_time:.1f}s] | "
                f"Train Loss: {train_loss:.4f} Acc: {train_metrics['accuracy']:.4f} F1: {train_metrics['macro_f1']:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['macro_f1']:.4f} "
                f"(LR: {current_lr:.6f}){marker}"
            )

            # Early stopping check
            if self.patience_counter >= self.patience:
                print(f"\n[INFO] Early stopping triggered at epoch {epoch} (no improvement for {self.patience} epochs).")
                break

        total_time = time.time() - start_time
        print(f"\n[SUCCESS] Training completed in {total_time:.2f}s. Best Epoch: {self.best_epoch} ({self.es_metric}: {self.best_metric_val:.4f})")

        # Save training metrics CSV
        history_df = pd.DataFrame(self.history)
        csv_path = os.path.join(self.metrics_dir, "training_history.csv")
        history_df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved training metrics to: {csv_path}")

        return {
            "best_epoch": self.best_epoch,
            "best_metric_val": self.best_metric_val,
            "total_epochs_trained": len(self.history),
            "total_time_seconds": round(total_time, 2),
            "history_csv": csv_path
        }
