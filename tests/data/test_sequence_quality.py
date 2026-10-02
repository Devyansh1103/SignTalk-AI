"""
Unit tests for SequenceQualityScorer.
"""

import pytest
import numpy as np

from src.data.sequence_quality import SequenceQualityScorer


@pytest.fixture
def scorer():
    return SequenceQualityScorer()


def test_perfect_quality(scorer):
    # Mask of ones: all joints detected across all 45 frames
    mask = np.ones((1, 45, 93), dtype=np.float32)
    res = scorer.compute_quality(mask)

    assert res["quality_score"] == 1.0
    assert res["quality_status"] == "GOOD"
    assert res["pose_detection_rate"] == 100.0
    assert res["left_hand_detection_rate"] == 100.0
    assert res["right_hand_detection_rate"] == 100.0
    assert res["valid_frames"] == 45


def test_reject_quality_insufficient_frames(scorer):
    # Mask where only 2 frames have detected hands
    mask = np.zeros((1, 45, 93), dtype=np.float32)
    mask[:, :, 42:53] = 1.0  # pose detected across all 45 frames
    mask[:, :2, 21:42] = 1.0  # right hand detected for only 2 frames

    res = scorer.compute_quality(mask)
    assert res["quality_status"] == "REJECT"
    assert res["valid_frames"] == 2


def test_acceptable_quality(scorer):
    # Mask where pose is 100% and dominant hand is detected for 25 frames
    mask = np.zeros((1, 45, 93), dtype=np.float32)
    mask[:, :, 42:53] = 1.0  # pose 100%
    mask[:, :25, 21:42] = 1.0  # dominant hand detected 25 frames (55.5%)

    res = scorer.compute_quality(mask)
    assert res["quality_status"] in ["ACCEPTABLE", "GOOD"]
    assert res["valid_frames"] == 25
