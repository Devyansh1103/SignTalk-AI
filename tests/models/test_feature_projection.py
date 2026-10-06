"""
Unit tests for FeatureProjection layer.
"""

import pytest
import torch
from src.models.layers.feature_projection import FeatureProjection


def test_feature_projection_forward():
    layer = FeatureProjection(in_dim=256, embed_dim=128, dropout=0.1, use_layer_norm=True)
    x = torch.randn(4, 12, 256)
    out = layer(x)
    assert out.shape == (4, 12, 128)
    assert not torch.isnan(out).any()


def test_feature_projection_shape_mismatch():
    layer = FeatureProjection(in_dim=256, embed_dim=128)
    with pytest.raises(ValueError):
        # Dim mismatch: in_dim is 100 instead of 256
        layer(torch.randn(4, 12, 100))

    with pytest.raises(ValueError):
        # 2D instead of 3D
        layer(torch.randn(4, 256))
