"""
SignTalk AI: Unit tests for STGCNBlock.
"""

import pytest
import torch

from src.models.graph import SignGraph
from src.models.layers.stgcn_block import STGCNBlock


def test_stgcn_block_identity_residual():
    graph = SignGraph(strategy="spatial")
    A = graph.get_adjacency_tensor()
    block = STGCNBlock(
        in_channels=64,
        out_channels=64,
        num_subsets=3,
        num_nodes=93,
        stride=1,
        residual=True
    )
    x = torch.randn(2, 64, 45, 93)
    y = block(x, A)
    assert y.shape == (2, 64, 45, 93)


def test_stgcn_block_downsampling_residual():
    graph = SignGraph(strategy="spatial")
    A = graph.get_adjacency_tensor()
    block = STGCNBlock(
        in_channels=64,
        out_channels=128,
        num_subsets=3,
        num_nodes=93,
        stride=2,
        residual=True
    )
    x = torch.randn(2, 64, 45, 93)
    y = block(x, A)
    assert y.shape == (2, 128, 23, 93)
