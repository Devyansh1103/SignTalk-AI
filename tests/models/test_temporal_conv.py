"""
SignTalk AI: Unit tests for TemporalConv layer.
"""

import pytest
import torch

from src.models.layers.temporal_conv import TemporalConv


def test_temporal_conv_stride_1():
    layer = TemporalConv(in_channels=64, out_channels=64, kernel_size=9, stride=1)
    x = torch.randn(2, 64, 45, 93)
    y = layer(x)
    assert y.shape == (2, 64, 45, 93)


def test_temporal_conv_stride_2():
    # Stride 2 reduces temporal length: ceil(45 / 2) = 23
    layer = TemporalConv(in_channels=64, out_channels=128, kernel_size=9, stride=2)
    x = torch.randn(2, 64, 45, 93)
    y = layer(x)
    assert y.shape == (2, 128, 23, 93)
