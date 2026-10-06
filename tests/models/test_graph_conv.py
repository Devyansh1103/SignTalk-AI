"""
SignTalk AI: Unit tests for SpatialGraphConv layer.
"""

import pytest
import torch

from src.models.graph import SignGraph
from src.models.layers.graph_conv import SpatialGraphConv


def test_spatial_graph_conv_forward():
    graph = SignGraph(strategy="spatial")
    A = graph.get_adjacency_tensor()
    layer = SpatialGraphConv(
        in_channels=3,
        out_channels=64,
        num_subsets=3,
        num_nodes=93,
        use_learnable_edge_weights=True
    )
    x = torch.randn(2, 3, 45, 93)
    y = layer(x, A)
    assert y.shape == (2, 64, 45, 93)


def test_spatial_graph_conv_backward():
    graph = SignGraph(strategy="spatial")
    A = graph.get_adjacency_tensor()
    layer = SpatialGraphConv(
        in_channels=3,
        out_channels=32,
        num_subsets=3,
        num_nodes=93,
        use_learnable_edge_weights=True
    )
    x = torch.randn(2, 3, 10, 93, requires_grad=True)
    y = layer(x, A)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert layer.conv.weight.grad is not None
    assert layer.edge_importance.grad is not None
