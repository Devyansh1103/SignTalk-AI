"""
Unit tests for QualityChecker module.
Tests frame quality scoring and sequence classification (GOOD, ACCEPTABLE, REVIEW, REJECT).
"""

import pytest
import numpy as np
from src.preprocessing.quality_checker import QualityChecker


def test_frame_quality_score():
    qc = QualityChecker()

    # Case 1: High quality full detection
    f_stat_good = {
        "pose_detected": True,
        "pose_confidence": 0.95,
        "left_hand_detected": True,
        "left_hand_confidence": 0.90,
        "right_hand_detected": True,
        "right_hand_confidence": 0.85,
        "face_detected": True
    }
    score_good = qc.compute_frame_quality(f_stat_good)
    assert 0.85 <= score_good <= 1.0

    # Case 2: Only pose detected, hands occluded
    f_stat_pose_only = {
        "pose_detected": True,
        "pose_confidence": 0.80,
        "left_hand_detected": False,
        "right_hand_detected": False,
        "face_detected": False
    }
    score_pose_only = qc.compute_frame_quality(f_stat_pose_only)
    assert np.isclose(score_pose_only, 0.40 * 0.80, atol=1e-3)


def test_sequence_evaluation():
    qc = QualityChecker(min_valid_frames=10)

    # Synthetic sequence of 30 frames: high quality
    stats_good = [{
        "pose_detected": True,
        "pose_confidence": 0.9,
        "left_hand_detected": False,
        "right_hand_detected": True,
        "right_hand_confidence": 0.85,
        "face_detected": False
    } for _ in range(30)]
    mask = np.ones((30, 93), dtype=bool)

    res_good = qc.evaluate_sequence(stats_good, mask)
    assert res_good["classification"] in ["GOOD", "ACCEPTABLE"]
    assert res_good["valid_frames"] == 30

    # Synthetic sequence: insufficient valid frames (< 10)
    stats_bad = [{
        "pose_detected": False,
        "left_hand_detected": False,
        "right_hand_detected": False,
        "face_detected": False
    } for _ in range(30)]

    res_bad = qc.evaluate_sequence(stats_bad, np.zeros((30, 93), dtype=bool))
    assert res_bad["classification"] == "REJECT"
