"""
Unit tests for Collate functions.
"""

import pytest
import torch

from src.data.collate import sign_sequence_collate_fn, pad_variable_sequence_collate_fn


def test_sign_sequence_collate_fn():
    batch = [
        {
            "x": torch.zeros((3, 45, 93), dtype=torch.float32),
            "mask": torch.ones((1, 45, 93), dtype=torch.float32),
            "label": torch.tensor(0, dtype=torch.long),
            "gloss_id": torch.tensor(4, dtype=torch.long),
            "decoder_input": torch.tensor([2, 4], dtype=torch.long),
            "decoder_target": torch.tensor([4, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
            "metadata": {"seq_id": "seq_01"}
        },
        {
            "x": torch.ones((3, 45, 93), dtype=torch.float32),
            "mask": torch.ones((1, 45, 93), dtype=torch.float32),
            "label": torch.tensor(1, dtype=torch.long),
            "gloss_id": torch.tensor(5, dtype=torch.long),
            "decoder_input": torch.tensor([2, 5], dtype=torch.long),
            "decoder_target": torch.tensor([5, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
            "metadata": {"seq_id": "seq_02"}
        }
    ]

    out = sign_sequence_collate_fn(batch)
    assert out["x"].shape == (2, 3, 45, 93)
    assert out["mask"].shape == (2, 1, 45, 93)
    assert out["label"].shape == (2,)
    assert out["gloss_id"].shape == (2,)
    assert out["decoder_input"].shape == (2, 2)
    assert len(out["metadata"]) == 2


def test_pad_variable_sequence_collate_fn():
    batch = [
        {
            "x": torch.zeros((3, 40, 93), dtype=torch.float32),
            "mask": torch.ones((1, 40, 93), dtype=torch.float32),
            "label": torch.tensor(0, dtype=torch.long),
            "gloss_id": torch.tensor(4, dtype=torch.long),
            "decoder_input": torch.tensor([2, 4], dtype=torch.long),
            "decoder_target": torch.tensor([4, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
            "metadata": {"seq_id": "seq_01"}
        },
        {
            "x": torch.ones((3, 50, 93), dtype=torch.float32),
            "mask": torch.ones((1, 50, 93), dtype=torch.float32),
            "label": torch.tensor(1, dtype=torch.long),
            "gloss_id": torch.tensor(5, dtype=torch.long),
            "decoder_input": torch.tensor([2, 5], dtype=torch.long),
            "decoder_target": torch.tensor([5, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
            "metadata": {"seq_id": "seq_02"}
        }
    ]

    out = pad_variable_sequence_collate_fn(batch)
    assert out["x"].shape == (2, 3, 50, 93)
    assert out["mask"].shape == (2, 1, 50, 93)
    # Check that sample 0 has zeros in mask for t in [40..50)
    assert torch.all(out["mask"][0, :, 40:, :] == 0.0)
