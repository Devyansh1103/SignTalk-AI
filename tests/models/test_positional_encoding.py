"""
Unit tests for positional encoding modules.
"""

import pytest
import torch
from src.models.layers.positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding
)


def test_sinusoidal_positional_encoding():
    pe = SinusoidalPositionalEncoding(embed_dim=128, max_len=64)
    x = torch.zeros(2, 12, 128)
    out = pe(x)
    assert out.shape == (2, 12, 128)
    # Ensure non-trivial positional values added
    assert not torch.allclose(out, x)


def test_learned_positional_encoding():
    lpe = LearnedPositionalEncoding(embed_dim=128, max_len=64)
    x = torch.zeros(2, 12, 128)
    out = lpe(x)
    assert out.shape == (2, 12, 128)
    assert not torch.allclose(out, x)
