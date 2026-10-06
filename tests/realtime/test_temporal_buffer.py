"""
Unit tests for TemporalBuffer (Phase 4 Part 2).
Validates rolling temporal buffer behavior, length tracking, window extraction,
linear interpolation over dropped frames, and quality gating.
"""

import pytest
import numpy as np
import time

from src.realtime.temporal_buffer import TemporalBuffer
from src.realtime.types import LandmarkFrame, ModalityDetection, QualityReport
from src.data.node_schema import TOTAL_NODES, CHANNELS, TARGET_SEQUENCE_LENGTH


def make_dummy_landmark_frame(frame_id: int, timestamp: float, has_hands: bool = True, quality_score: float = 0.90) -> LandmarkFrame:
    """Helper to generate mock LandmarkFrame instances."""
    coords = np.zeros((TOTAL_NODES, CHANNELS), dtype=np.float32)
    norm = np.zeros((TOTAL_NODES, CHANNELS), dtype=np.float32)
    mask = np.ones((TOTAL_NODES,), dtype=bool)
    vis = np.ones((TOTAL_NODES,), dtype=np.float32)

    mod = ModalityDetection(
        left_hand_detected=has_hands,
        right_hand_detected=has_hands,
        pose_detected=True,
        face_detected=False
    )
    qual = QualityReport(
        is_valid=(quality_score >= 0.50),
        quality_score=quality_score,
        classification="GOOD" if quality_score >= 0.75 else "ACCEPTABLE",
        missing_ratio=0.10,
        has_nan_or_inf=False,
        coordinate_in_range=True,
        sudden_jumps_detected=False
    )

    return LandmarkFrame(
        frame_id=frame_id,
        timestamp=timestamp,
        raw_coords=coords,
        normalized_coords=norm,
        mask=mask,
        visibility=vis,
        modality_stats=mod,
        quality=qual
    )


class TestTemporalBuffer:
    def test_buffer_initialization(self):
        buf = TemporalBuffer(max_length=45)
        assert buf.length() == 0
        assert len(buf) == 0
        assert not buf.is_ready()
        assert not buf.is_window_valid()

    def test_empty_buffer_get_window_raises(self):
        buf = TemporalBuffer(max_length=45)
        with pytest.raises(ValueError, match="Insufficient frames"):
            buf.get_window()

    def test_partially_filled_buffer(self):
        buf = TemporalBuffer(max_length=45)
        for i in range(20):
            buf.append(make_dummy_landmark_frame(i, i * 0.04))

        assert buf.length() == 20
        assert not buf.is_ready()
        meta = buf.get_window_metadata()
        assert meta["count"] == 20
        assert not meta["is_ready"]

    def test_exactly_full_buffer(self):
        buf = TemporalBuffer(max_length=45)
        for i in range(45):
            buf.append(make_dummy_landmark_frame(i, i * 0.04))

        assert buf.length() == 45
        assert buf.is_ready()
        assert buf.is_window_valid()

        window = buf.get_window()
        assert len(window) == 45
        assert window[0].frame_id == 0
        assert window[-1].frame_id == 44

    def test_overflow_and_sliding_window(self):
        buf = TemporalBuffer(max_length=45)
        for i in range(55):
            buf.append(make_dummy_landmark_frame(i, i * 0.04))

        # Size is capped at 45
        assert buf.length() == 45
        assert buf.is_ready()

        window = buf.get_window()
        assert len(window) == 45
        # Oldest 10 frames (0..9) were pushed out
        assert window[0].frame_id == 10
        assert window[-1].frame_id == 54

    def test_clear_buffer(self):
        buf = TemporalBuffer(max_length=45)
        for i in range(30):
            buf.append(make_dummy_landmark_frame(i, i * 0.04))
        assert buf.length() == 30
        buf.clear()
        assert buf.length() == 0
        assert not buf.is_ready()

    def test_linear_interpolation_on_dropped_frames(self):
        buf = TemporalBuffer(max_length=45, interpolate_missing_frames=True, max_consecutive_interpolated=5)
        f0 = make_dummy_landmark_frame(0, 0.0)
        # Next frame is frame_id 3 (frames 1 and 2 dropped)
        f3 = make_dummy_landmark_frame(3, 0.12)

        buf.append(f0)
        buf.append(f3)

        # Buffer should have 4 frames: 0, 1 (interp), 2 (interp), 3
        assert buf.length() == 4
        assert buf._buffer[0].frame_id == 0
        assert buf._buffer[1].frame_id == 1
        assert buf._buffer[2].frame_id == 2
        assert buf._buffer[3].frame_id == 3

    def test_quality_gate_rejects_no_hands(self):
        buf = TemporalBuffer(max_length=45, min_valid_frames=15)
        for i in range(45):
            # No hands present in any frame
            buf.append(make_dummy_landmark_frame(i, i * 0.04, has_hands=False))

        assert buf.is_ready()
        # Should fail quality gate due to lack of hand activity
        assert not buf.is_window_valid()
