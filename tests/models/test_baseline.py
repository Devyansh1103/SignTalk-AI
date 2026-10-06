"""
Unit tests for SignBaselineModel architecture.
"""

import pytest
import torch

from src.models.baseline import SignBaselineModel


def test_baseline_forward_4d():
    model = SignBaselineModel(
        in_channels=3,
        num_nodes=93,
        sequence_length=45,
        proj_dim=64,
        hidden_dim=64,
        num_layers=1,
        dropout=0.1,
        bidirectional=True,
        num_classes=10,
        pooling_type="mean_max"
    )
    # Batch size 2, Channels 3, Time 45, Nodes 93
    x = torch.randn(2, 3, 45, 93)
    out = model(x)
    assert out.shape == (2, 10)
    assert out.dtype == torch.float32


def test_baseline_forward_3d():
    model = SignBaselineModel(
        in_channels=3,
        num_nodes=93,
        sequence_length=45,
        proj_dim=64,
        hidden_dim=64,
        num_layers=1,
        num_classes=10
    )
    # Batch size 3, Time 45, Feature Dim 279
    x = torch.randn(3, 45, 279)
    out = model(x)
    assert out.shape == (3, 10)


def test_pooling_types():
    for p_type in ["mean", "max", "mean_max", "last"]:
        model = SignBaselineModel(
            in_channels=3,
            num_nodes=93,
            sequence_length=45,
            proj_dim=32,
            hidden_dim=32,
            num_layers=1,
            num_classes=5,
            pooling_type=p_type
        )
        x = torch.randn(2, 3, 45, 93)
        out = model(x)
        assert out.shape == (2, 5)


def test_invalid_input_dimension():
    model = SignBaselineModel()
    x = torch.randn(2, 45)  # 2D invalid
    with pytest.raises(ValueError):
        model(x)
