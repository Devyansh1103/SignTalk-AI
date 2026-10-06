"""
Functional edge case and known-limitation tests (Phase 4 Part 2).
Evaluates sliding-window behavior under:
  - No person detected
  - No hands detected
  - Single active hand
  - Dual active hands
  - Fast signing / rapid displacements
  - Temporary occlusions / bridged dropped frames
"""

import pytest
import os
import numpy as np
import torch

from src.realtime.temporal_buffer import TemporalBuffer
from src.realtime.model_runner import STGCNRunner
from src.realtime.types import LandmarkFrame, ModalityDetection, QualityReport
from src.data.node_schema import TOTAL_NODES, CHANNELS, TARGET_SEQUENCE_LENGTH


CHECKPOINT_PATH = "experiments/stgcn/checkpoints/best_checkpoint.pt"


def make_frame_with_modalities(frame_id: int, pose: bool = True, lh: bool = False, rh: bool = False, score: float = 0.9) -> LandmarkFrame:
    coords = np.zeros((TOTAL_NODES, CHANNELS), dtype=np.float32)
    norm = np.zeros((TOTAL_NODES, CHANNELS), dtype=np.float32)
    mask = np.zeros((TOTAL_NODES,), dtype=bool)
    vis = np.zeros((TOTAL_NODES,), dtype=np.float32)

    if pose:
        mask[42:53] = True
        vis[42:53] = 0.9
    if lh:
        mask[0:21] = True
        vis[0:21] = 0.85
    if rh:
        mask[21:42] = True
        vis[21:42] = 0.85

    classification = "GOOD" if score >= 0.75 else ("ACCEPTABLE" if score >= 0.50 else "REJECT")

    return LandmarkFrame(
        frame_id=frame_id,
        timestamp=frame_id * 0.04,
        raw_coords=coords,
        normalized_coords=norm,
        mask=mask,
        visibility=vis,
        modality_stats=ModalityDetection(
            pose_detected=pose,
            left_hand_detected=lh,
            right_hand_detected=rh,
            face_detected=False
        ),
        quality=QualityReport(
            is_valid=(score >= 0.50),
            quality_score=score,
            classification=classification,
            missing_ratio=float(1.0 - np.mean(mask)),
            has_nan_or_inf=False,
            coordinate_in_range=True,
            sudden_jumps_detected=False
        )
    )


class TestKnownLimitations:
    @pytest.fixture
    def runner(self):
        if not os.path.exists(CHECKPOINT_PATH):
            pytest.skip("Checkpoint missing")
        return STGCNRunner(checkpoint_path=CHECKPOINT_PATH, device="cpu", warmup_iterations=0)

    def test_no_person_behavior(self, runner):
        """When no person is in front of camera, buffer flags window as invalid."""
        buf = TemporalBuffer(max_length=TARGET_SEQUENCE_LENGTH, min_valid_frames=15)
        for i in range(TARGET_SEQUENCE_LENGTH):
            buf.append(make_frame_with_modalities(i, pose=False, lh=False, rh=False, score=0.0))

        assert buf.is_ready()
        assert not buf.is_window_valid()

        # Model runner still executes safely without crashing, returning valid prediction with low input_quality
        window = buf.get_window()
        pred = runner.predict_window(window)
        assert pred.input_quality == 0.0
        assert not pred.is_valid_quality

    def test_no_hands_behavior(self, runner):
        """When pose is visible but no hands are in frame, window is flagged."""
        buf = TemporalBuffer(max_length=TARGET_SEQUENCE_LENGTH, min_valid_frames=15)
        for i in range(TARGET_SEQUENCE_LENGTH):
            buf.append(make_frame_with_modalities(i, pose=True, lh=False, rh=False, score=0.6))

        assert buf.is_ready()
        assert not buf.is_window_valid()  # Hand ratio < 0.15 threshold

    def test_single_hand_behavior(self, runner):
        """One hand visible (dominant hand signing). Window satisfies quality gate."""
        buf = TemporalBuffer(max_length=TARGET_SEQUENCE_LENGTH, min_valid_frames=15)
        for i in range(TARGET_SEQUENCE_LENGTH):
            buf.append(make_frame_with_modalities(i, pose=True, lh=False, rh=True, score=0.85))

        assert buf.is_ready()
        assert buf.is_window_valid()
        pred = runner.predict_window(buf.get_window())
        assert pred.is_valid_quality
        assert 0 <= pred.class_id < 10

    def test_dual_hand_behavior(self, runner):
        """Two hands visible (bimanual signing). Window satisfies quality gate."""
        buf = TemporalBuffer(max_length=TARGET_SEQUENCE_LENGTH, min_valid_frames=15)
        for i in range(TARGET_SEQUENCE_LENGTH):
            buf.append(make_frame_with_modalities(i, pose=True, lh=True, rh=True, score=0.95))

        assert buf.is_ready()
        assert buf.is_window_valid()
        pred = runner.predict_window(buf.get_window())
        assert pred.is_valid_quality

    def test_temporary_occlusion_recovery(self, runner):
        """Temporary occlusion of 3 dropped frames is bridged via linear interpolation."""
        buf = TemporalBuffer(max_length=TARGET_SEQUENCE_LENGTH, interpolate_missing_frames=True, max_consecutive_interpolated=5)
        f_start = make_frame_with_modalities(0, pose=True, lh=True, rh=True, score=0.9)
        f_end = make_frame_with_modalities(4, pose=True, lh=True, rh=True, score=0.9)  # Frames 1, 2, 3 dropped

        buf.append(f_start)
        buf.append(f_end)

        assert buf.length() == 5
        # Interp frames exist
        assert [f.frame_id for f in buf._buffer] == [0, 1, 2, 3, 4]
