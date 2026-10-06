"""
Unit tests for Data Leakage and Split Integrity Auditing.
"""

import pytest
import os
import pandas as pd


def test_manifest_existence():
    train_csv = "data/manifests/train.csv"
    val_csv = "data/manifests/val.csv"
    test_csv = "data/manifests/test.csv"

    assert os.path.exists(train_csv), "Train manifest missing"
    assert os.path.exists(val_csv), "Val manifest missing"
    assert os.path.exists(test_csv), "Test manifest missing"


def test_no_sequence_id_leakage():
    train_df = pd.read_csv("data/manifests/train.csv")
    val_df = pd.read_csv("data/manifests/val.csv")
    test_df = pd.read_csv("data/manifests/test.csv")

    train_ids = set(train_df["sequence_id"])
    val_ids = set(val_df["sequence_id"])
    test_ids = set(test_df["sequence_id"])

    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val share sequence IDs!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test share sequence IDs!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test share sequence IDs!"


def test_no_exact_video_leakage():
    train_df = pd.read_csv("data/manifests/train.csv")
    val_df = pd.read_csv("data/manifests/val.csv")
    test_df = pd.read_csv("data/manifests/test.csv")

    train_vids = set(train_df["source_video_id"])
    val_vids = set(val_df["source_video_id"])
    test_vids = set(test_df["source_video_id"])

    assert len(train_vids.intersection(val_vids)) == 0, "Train and Val share source videos!"
    assert len(train_vids.intersection(test_vids)) == 0, "Train and Test share source videos!"
    assert len(val_vids.intersection(test_vids)) == 0, "Val and Test share source videos!"


def test_signer_split_audit():
    # Documents that current dataset has signer overlap across splits
    train_df = pd.read_csv("data/manifests/train.csv")
    test_df = pd.read_csv("data/manifests/test.csv")

    train_signers = set(train_df["signer_id"])
    test_signers = set(test_df["signer_id"])

    # This test asserts that the known signer overlap is explicitly tracked
    overlap = train_signers.intersection(test_signers)
    assert len(overlap) > 0, "Expected documented signer overlap across splits in current v1 dataset."
