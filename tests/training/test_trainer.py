"""
Unit tests for ModelTrainer training coordinator.
"""

import os
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.baseline import SignBaselineModel
from src.training.trainer import ModelTrainer


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=8):
        self.size = size
        self.x = torch.randn(size, 3, 45, 93)
        self.labels = torch.randint(0, 10, (size,))
        self.mask = torch.ones(size, 1, 45, 93)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "mask": self.mask[idx],
            "label": self.labels[idx],
            "metadata": {"sequence_id": f"dummy_{idx}"}
        }


def test_trainer_single_epoch(tmp_path):
    model = SignBaselineModel(proj_dim=32, hidden_dim=32, num_layers=1, num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    config = {
        "training": {
            "epochs": 1,
            "gradient_clip_norm": 1.0,
            "early_stopping_patience": 5,
            "early_stopping_metric": "val_acc",
            "early_stopping_mode": "max"
        },
        "paths": {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
            "metrics_dir": str(tmp_path / "metrics"),
            "experiment_dir": str(tmp_path)
        },
        "dataset_metadata": {"num_classes": 10}
    }

    train_loader = DataLoader(DummyDataset(8), batch_size=4, shuffle=True)
    val_loader = DataLoader(DummyDataset(4), batch_size=4, shuffle=False)

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )

    loss, metrics = trainer.train_epoch(epoch=1)
    assert loss > 0.0
    assert "accuracy" in metrics
    assert "macro_f1" in metrics


def test_checkpoint_save_and_load(tmp_path):
    model = SignBaselineModel(proj_dim=32, hidden_dim=32, num_layers=1, num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    config = {
        "training": {"epochs": 1},
        "paths": {
            "checkpoint_dir": str(tmp_path),
            "log_dir": str(tmp_path),
            "metrics_dir": str(tmp_path)
        },
        "dataset_metadata": {"num_classes": 10}
    }

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=config
    )

    ckpt_path = str(tmp_path / "test_ckpt.pt")
    val_metrics = {"accuracy": 0.85, "macro_f1": 0.82}
    trainer.save_checkpoint(ckpt_path, epoch=5, val_metrics=val_metrics, is_best=True)

    assert os.path.exists(ckpt_path)

    # Load checkpoint
    loaded_ckpt = trainer.load_checkpoint(ckpt_path)
    assert loaded_ckpt["epoch"] == 5
    assert loaded_ckpt["val_metrics"]["accuracy"] == 0.85
    assert loaded_ckpt["is_best"] is True
