"""
Unit tests for SignSequenceDataset.
"""

import os
import pytest
import torch

from src.data.sign_sequence_dataset import SignSequenceDataset


@pytest.fixture
def manifest_path():
    path = "data/manifests/sequence_manifest.csv"
    if not os.path.exists(path):
        pytest.skip("sequence_manifest.csv not generated yet")
    return path


def test_dataset_initialization(manifest_path):
    ds = SignSequenceDataset(manifest_path=manifest_path)
    assert len(ds) == 60


def test_dataset_train_split(manifest_path):
    ds_train = SignSequenceDataset(manifest_path=manifest_path, split="train", filter_rejects=False)
    assert len(ds_train) == 36


def test_dataset_filter_rejects(manifest_path):
    ds_clean = SignSequenceDataset(manifest_path=manifest_path, split="train", filter_rejects=True)
    # Filtered train samples must be <= 36 and contain no REJECT
    assert len(ds_clean) <= 36
    assert all(ds_clean.df["quality_status"] != "REJECT")


def test_dataset_getitem(manifest_path):
    ds = SignSequenceDataset(manifest_path=manifest_path)
    item = ds[0]
    assert "x" in item
    assert "mask" in item
    assert "label" in item
    assert "gloss_id" in item
    assert "decoder_input" in item
    assert "decoder_target" in item
    assert "attention_mask" in item

    assert item["x"].shape == (3, 45, 93)
    assert item["mask"].shape == (1, 45, 93)
    assert item["label"].dtype == torch.long
    assert item["decoder_input"].shape == (2,)


def test_dataset_velocity_and_augment(manifest_path):
    ds = SignSequenceDataset(manifest_path=manifest_path, include_velocity=True, augment=True)
    item = ds[0]
    assert item["x"].shape == (6, 45, 93)
