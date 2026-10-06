"""
Unit tests for Transformer masks.
"""

import pytest
import torch
from src.models.transformer_masks import (
    generate_causal_mask,
    generate_causal_bool_mask,
    create_padding_mask,
    downsample_temporal_mask
)


def test_causal_masks():
    float_mask = generate_causal_mask(4)
    assert float_mask.shape == (4, 4)
    assert float_mask[0, 1] == float("-inf")
    assert float_mask[1, 0] == 0.0

    bool_mask = generate_causal_bool_mask(4)
    assert bool_mask.shape == (4, 4)
    assert bool_mask[0, 1].item() is True
    assert bool_mask[1, 0].item() is False


def test_padding_mask():
    tokens = torch.tensor([[2, 4, 3, 0], [2, 5, 0, 0]])
    pad_mask = create_padding_mask(tokens, pad_idx=0)
    assert pad_mask.shape == (2, 4)
    assert pad_mask[0, 3].item() is True
    assert pad_mask[0, 1].item() is False
    assert pad_mask[1, 2].item() is True


def test_downsample_temporal_mask():
    # Input mask [B, 1, T=45, V=93]
    mask_4d = torch.ones(2, 1, 45, 93)
    # Zero out last 15 frames for sample 1
    mask_4d[1, :, 30:, :] = 0.0

    downsampled = downsample_temporal_mask(mask_4d, target_len=12)
    assert downsampled.shape == (2, 12)
    # Sample 0 should be all False (all valid)
    assert not downsampled[0].any()
    # Sample 1 should have padded frames near the end (True)
    assert downsampled[1, -1].item() is True
