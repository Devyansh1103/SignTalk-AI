"""
SignTalk AI: Unit tests for 93-node skeletal graph topology and adjacency matrices.
"""

import pytest
import numpy as np
import torch

from src.models.graph import SignGraph


def test_graph_initialization():
    graph = SignGraph(strategy="spatial")
    assert graph.num_nodes == 93
    assert len(graph.edges) == 210  # 105 undirected edges * 2 directions = 210 directed pairs


def test_spatial_partitioning_shape():
    graph = SignGraph(strategy="spatial")
    tensor_A = graph.get_adjacency_tensor()
    assert tensor_A.shape == (3, 93, 93)
    assert isinstance(tensor_A, torch.Tensor)
    # Subset 0 should be identity (self-loops)
    np_A0 = tensor_A[0].numpy()
    assert np.allclose(np.diag(np_A0), 1.0)


def test_uniform_and_distance_strategies():
    graph_u = SignGraph(strategy="uniform")
    tensor_u = graph_u.get_adjacency_tensor()
    assert tensor_u.shape == (1, 93, 93)

    graph_d = SignGraph(strategy="distance")
    tensor_d = graph_d.get_adjacency_tensor()
    assert tensor_d.shape == (2, 93, 93)


def test_node_validation():
    graph = SignGraph()
    graph.validate_input_nodes(93)
    with pytest.raises(ValueError):
        graph.validate_input_nodes(54)
