"""
Unit tests for ST-GCN Tensor utilities.
"""

import pytest
import numpy as np
import torch

from src.data.stgcn_tensor import (
    get_kinematic_edges,
    build_spatial_adjacency_matrix,
    landmarks_to_stgcn_tensor,
    compute_temporal_velocity,
    validate_stgcn_shape,
    NUM_NODES,
    SEQUENCE_LENGTH,
    CHANNELS
)


def test_kinematic_edges():
    edges = get_kinematic_edges()
    assert len(edges) > 50
    for u, v in edges:
        assert 0 <= u < NUM_NODES
        assert 0 <= v < NUM_NODES
        assert u != v


def test_spatial_adjacency_matrix():
    adj = build_spatial_adjacency_matrix()
    assert adj.shape == (3, NUM_NODES, NUM_NODES)
    assert adj.dtype == np.float32
    # Check self loops in partition 0 (Root)
    assert np.allclose(np.diag(adj[0]), 1.0)
    # Check partition 1 and 2 have no self loops
    assert np.allclose(np.diag(adj[1]), 0.0)
    assert np.allclose(np.diag(adj[2]), 0.0)


def test_landmarks_to_stgcn_tensor():
    data = np.zeros((3, SEQUENCE_LENGTH, NUM_NODES), dtype=np.float32)
    mask = np.ones((1, SEQUENCE_LENGTH, NUM_NODES), dtype=np.float32)
    x, m = landmarks_to_stgcn_tensor(data, mask)
    assert x.shape == (3, 45, 93)
    assert m.shape == (1, 45, 93)
    assert x.dtype == torch.float32
    assert validate_stgcn_shape(x) is True


def test_compute_temporal_velocity():
    x = torch.zeros((4, 3, 45, 93), dtype=torch.float32)
    # Inject linear displacement in channel 0
    t = torch.linspace(0, 1, 45).view(1, 1, 45, 1)
    x[:, :1, :, :] = t
    x_aug = compute_temporal_velocity(x)
    assert x_aug.shape == (4, 6, 45, 93)
    # Check first velocity frame is zero
    assert torch.all(x_aug[:, 3:, 0, :] == 0.0)
    # Check subsequent velocity frames are non-zero positive
    assert torch.all(x_aug[:, 3:4, 1:, :] > 0.0)
