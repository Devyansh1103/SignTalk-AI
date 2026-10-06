"""
Unit tests for Real-Time Landmark Quality Validator.
"""

import pytest
import numpy as np

from src.realtime.types import ModalityDetection
from src.realtime.landmark_stream import RealTimeLandmarkStream
from src.realtime.realtime_config import RealTimeConfig
from src.data.node_schema import TOTAL_NODES


def test_quality_checker_nan_detection():
    """Verify NaN detection triggers REJECT classification."""
    stream = RealTimeLandmarkStream(RealTimeConfig())

    norm_coords = np.zeros((TOTAL_NODES, 3), dtype=np.float32)
    norm_coords[0, 0] = np.nan
    mask = np.ones((TOTAL_NODES,), dtype=bool)
    modality = ModalityDetection(pose_detected=True, left_hand_detected=True)

    report = stream._validate_frame_quality(
        norm_coords=norm_coords,
        mask=mask,
        modality_detection=modality,
        raw_coords=np.zeros((TOTAL_NODES, 3))
    )

    assert report.has_nan_or_inf is True
    assert report.classification == "REJECT"
    assert report.is_valid is False
    assert any("NaN" in w for w in report.warnings)

    stream.close()


def test_quality_checker_no_pose():
    """Verify absence of pose triggers REJECT."""
    stream = RealTimeLandmarkStream(RealTimeConfig())

    norm_coords = np.zeros((TOTAL_NODES, 3), dtype=np.float32)
    mask = np.zeros((TOTAL_NODES,), dtype=bool)
    modality = ModalityDetection(pose_detected=False, left_hand_detected=False)

    report = stream._validate_frame_quality(
        norm_coords=norm_coords,
        mask=mask,
        modality_detection=modality,
        raw_coords=np.zeros((TOTAL_NODES, 3))
    )

    assert report.classification == "REJECT"
    assert report.is_valid is False

    stream.close()


def test_quality_checker_optimal_frame():
    """Verify high-confidence pose and hands trigger GOOD classification."""
    stream = RealTimeLandmarkStream(RealTimeConfig())

    norm_coords = np.zeros((TOTAL_NODES, 3), dtype=np.float32)
    mask = np.ones((TOTAL_NODES,), dtype=bool)
    modality = ModalityDetection(
        pose_detected=True,
        left_hand_detected=True,
        right_hand_detected=True,
        pose_confidence=0.9,
        left_hand_confidence=0.9,
        right_hand_confidence=0.9
    )

    raw_coords = np.full((TOTAL_NODES, 3), 0.5, dtype=np.float32)

    report = stream._validate_frame_quality(
        norm_coords=norm_coords,
        mask=mask,
        modality_detection=modality,
        raw_coords=raw_coords
    )

    assert report.classification == "GOOD"
    assert report.is_valid is True
    assert report.quality_score >= 0.75
    assert report.missing_ratio == 0.0

    stream.close()


def test_quality_checker_sudden_jump_detection():
    """Verify joint displacement exceeding jump_distance_threshold triggers warning."""
    cfg = RealTimeConfig()
    cfg.quality.jump_distance_threshold = 0.35
    stream = RealTimeLandmarkStream(cfg)

    # Set previous frame
    stream._previous_coords = np.zeros((TOTAL_NODES, 3), dtype=np.float32)
    stream._previous_mask = np.ones((TOTAL_NODES,), dtype=bool)

    # Current frame: joint 0 jumps by 1.0 (exceeds 0.35)
    current_coords = np.zeros((TOTAL_NODES, 3), dtype=np.float32)
    current_coords[0] = [1.0, 0.0, 0.0]
    mask = np.ones((TOTAL_NODES,), dtype=bool)

    modality = ModalityDetection(pose_detected=True, left_hand_detected=True, pose_confidence=0.8, left_hand_confidence=0.8)

    report = stream._validate_frame_quality(
        norm_coords=current_coords,
        mask=mask,
        modality_detection=modality,
        raw_coords=np.full((TOTAL_NODES, 3), 0.5, dtype=np.float32)
    )

    assert report.sudden_jumps_detected is True
    assert report.max_jump_distance >= 1.0
    assert any("jump" in w.lower() for w in report.warnings)

    stream.close()
