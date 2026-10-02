"""
Unit tests for Sequence Utilities and PyTorch LandmarkDataset.
Tests missing landmark handling, temporal standardization, and tensor shapes.
"""

import os
import pytest
import numpy as np
import torch

from src.preprocessing.sequence_utils import (
    handle_missing_landmarks,
    pad_or_truncate_sequence,
    temporal_resample_sequence
)
from src.data.landmark_dataset import SignLandmarkDataset, create_landmark_dataloader


def test_missing_landmark_interpolation():
    T, V, C = 10, 93, 3
    coords = np.zeros((T, V, C), dtype=np.float32)
    mask = np.zeros((T, V), dtype=bool)

    # Node 21 (right wrist): observed at t=0 and t=4, missing in between (t=1, 2, 3)
    coords[0, 21] = [1.0, 1.0, 0.0]
    mask[0, 21] = True
    coords[4, 21] = [5.0, 5.0, 0.0]
    mask[4, 21] = True

    handled, updated_mask = handle_missing_landmarks(coords, mask, strategy="interpolate_and_mask")

    # t=2 is midpoint: should interpolate to [3.0, 3.0, 0.0]
    assert np.allclose(handled[2, 21], [3.0, 3.0, 0.0], atol=1e-5)
    assert updated_mask[2, 21] == True


def test_temporal_resampling():
    T_in, V, C = 30, 93, 3
    coords = np.random.randn(T_in, V, C).astype(np.float32)
    mask = np.ones((T_in, V), dtype=bool)

    target_T = 45
    resampled_coords, resampled_mask = temporal_resample_sequence(coords, mask, target_length=target_T)

    assert resampled_coords.shape == (45, 93, 3)
    assert resampled_mask.shape == (45, 93)


def test_pytorch_dataset_and_dataloader(tmp_path):
    # Create temporary mock processed NPZ files
    split_dir = tmp_path / "landmarks" / "train"
    split_dir.mkdir(parents=True)

    for i in range(5):
        sample_path = str(split_dir / f"sample_{i:02d}.npz")
        np.savez_compressed(
            sample_path,
            data=np.random.randn(3, 45, 93).astype(np.float32),
            mask=np.ones((1, 45, 93), dtype=bool),
            class_id=i % 3,
            class_label=f"class_{i%3}",
            signer_id="signer_01",
            sample_id=f"sample_{i:02d}",
            split="train",
            quality_score=0.85
        )

    dataset = SignLandmarkDataset(data_dir=str(tmp_path / "landmarks"), split="train")
    assert len(dataset) == 5

    data_t, mask_t, label_t, meta = dataset[0]
    assert data_t.shape == (3, 45, 93)
    assert mask_t.shape == (1, 45, 93)
    assert isinstance(label_t, int)
    assert meta["signer_id"] == "signer_01"

    # Test DataLoader collation
    loader = create_landmark_dataloader(
        data_dir=str(tmp_path / "landmarks"),
        split="train",
        batch_size=2,
        shuffle=False
    )

    batch = next(iter(loader))
    batch_data, batch_mask, batch_labels, batch_meta = batch

    # Verify expected ST-GCN input: [B, C, T, V]
    assert batch_data.shape == (2, 3, 45, 93)
    assert batch_mask.shape == (2, 1, 45, 93)
    assert batch_labels.shape == (2,)
    assert len(batch_meta) == 2
