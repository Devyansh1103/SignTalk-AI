"""
Unit tests for Real-Time Torso-Scale Normalization and Temporal Anchor Smoothing.
"""

import pytest
import numpy as np

from src.realtime.landmark_stream import RealTimeLandmarkStream
from src.realtime.realtime_config import RealTimeConfig, NormalizationConfig
from src.data.node_schema import TOTAL_NODES


def test_realtime_torso_scale_centering():
    """Verify torso centering places mid-shoulder at origin (0, 0, 0)."""
    cfg = RealTimeConfig()
    cfg.normalization.use_temporal_anchor_smoothing = False
    stream = RealTimeLandmarkStream(cfg)

    raw_coords = np.zeros((TOTAL_NODES, 3), dtype=np.float32)
    mask = np.zeros((TOTAL_NODES,), dtype=bool)

    # Set shoulders: Left shoulder (node 47) at (0.4, 0.5, 0.0), Right shoulder (node 48) at (0.6, 0.5, 0.0)
    raw_coords[47] = [0.4, 0.5, 0.0]
    raw_coords[48] = [0.6, 0.5, 0.0]
    mask[47] = True
    mask[48] = True

    # Nose (node 42) at (0.5, 0.3, 0.0)
    raw_coords[42] = [0.5, 0.3, 0.0]
    mask[42] = True

    norm_coords, meta = stream._normalize_realtime_frame(raw_coords, mask)

    # Mid-shoulder should be at origin: (norm_47 + norm_48) / 2 == (0, 0, 0)
    mid_shoulder = (norm_coords[47] + norm_coords[48]) / 2.0
    assert np.allclose(mid_shoulder, 0.0, atol=1e-5)

    # Distance between shoulders should be 1.0 in normalized space
    norm_dist = np.linalg.norm(norm_coords[47] - norm_coords[48])
    assert np.isclose(norm_dist, 1.0, atol=1e-5)

    # Nose should be centered horizontally at x=0
    assert np.isclose(norm_coords[42, 0], 0.0, atol=1e-5)
    # Nose y should be (0.3 - 0.5) / 0.2 = -1.0
    assert np.isclose(norm_coords[42, 1], -1.0, atol=1e-5)

    stream.close()


def test_temporal_anchor_smoothing():
    """Verify EMA smoothing prevents sudden jumps in normalization center/scale."""
    cfg = RealTimeConfig()
    cfg.normalization.use_temporal_anchor_smoothing = True
    cfg.normalization.anchor_smoothing_alpha = 0.5
    stream = RealTimeLandmarkStream(cfg)

    raw_coords1 = np.zeros((TOTAL_NODES, 3), dtype=np.float32)
    mask = np.zeros((TOTAL_NODES,), dtype=bool)
    mask[47] = mask[48] = True

    # Frame 1: shoulders at (0.4, 0.5) and (0.6, 0.5) -> center = 0.5, scale = 0.2
    raw_coords1[47] = [0.4, 0.5, 0.0]
    raw_coords1[48] = [0.6, 0.5, 0.0]
    _, meta1 = stream._normalize_realtime_frame(raw_coords1, mask)
    assert np.isclose(meta1["scale"], 0.2, atol=1e-5)

    # Frame 2: Sudden detection noise in right shoulder: moves to (0.8, 0.5) -> instant center = 0.6, instant scale = 0.4
    raw_coords2 = np.zeros((TOTAL_NODES, 3), dtype=np.float32)
    raw_coords2[47] = [0.4, 0.5, 0.0]
    raw_coords2[48] = [0.8, 0.5, 0.0]

    _, meta2 = stream._normalize_realtime_frame(raw_coords2, mask)
    # With alpha=0.5: smoothed_scale = 0.5 * 0.2 + 0.5 * 0.4 = 0.3
    assert np.isclose(meta2["scale"], 0.3, atol=1e-4)

    stream.close()
