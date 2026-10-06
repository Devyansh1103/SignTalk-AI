"""
Unit tests for sliding window generation (Phase 4 Part 2).
Validates window extraction, sliding stride progression, temporal interval tracking,
and metadata consistency across rolling windows.
"""

import pytest
import numpy as np

from src.realtime.temporal_buffer import TemporalBuffer
from src.realtime.inference_scheduler import InferenceScheduler
from src.realtime.types import LandmarkFrame, ModalityDetection, QualityReport
from src.data.node_schema import TOTAL_NODES, CHANNELS, TARGET_SEQUENCE_LENGTH


def make_frame(frame_id: int, timestamp: float) -> LandmarkFrame:
    return LandmarkFrame(
        frame_id=frame_id,
        timestamp=timestamp,
        raw_coords=np.zeros((TOTAL_NODES, CHANNELS), dtype=np.float32),
        normalized_coords=np.zeros((TOTAL_NODES, CHANNELS), dtype=np.float32),
        mask=np.ones((TOTAL_NODES,), dtype=bool),
        visibility=np.ones((TOTAL_NODES,), dtype=np.float32),
        modality_stats=ModalityDetection(True, True, True, False),
        quality=QualityReport(
            is_valid=True,
            quality_score=0.95,
            classification="GOOD",
            missing_ratio=0.05,
            has_nan_or_inf=False,
            coordinate_in_range=True,
            sudden_jumps_detected=False
        )
    )


class TestWindowGeneration:
    def test_window_generation_exact_t(self):
        buf = TemporalBuffer(max_length=45)
        for i in range(45):
            buf.append(make_frame(i, i * 0.04))

        window = buf.get_window()
        assert len(window) == 45
        assert window[0].frame_id == 0
        assert window[-1].frame_id == 44

        meta = buf.get_window_metadata()
        assert meta["start_frame_id"] == 0
        assert meta["end_frame_id"] == 44
        assert meta["count"] == 45
        assert np.isclose(meta["start_time"], 0.0)
        assert np.isclose(meta["end_time"], 44 * 0.04)

    def test_sliding_window_progression(self):
        buf = TemporalBuffer(max_length=45)
        stride = 5
        sched = InferenceScheduler(stride=stride)

        # Ingest 44 frames: not ready yet
        for i in range(44):
            buf.append(make_frame(i, i * 0.04))
            sched.on_frame_appended()
            assert not sched.should_infer(buf)

        # Ingest frame 45: buffer full, should infer
        buf.append(make_frame(44, 44 * 0.04))
        sched.on_frame_appended()
        assert sched.should_infer(buf)

        sched.record_inference_start()
        window1 = buf.get_window()
        sched.record_inference_finish()
        assert window1[0].frame_id == 0
        assert window1[-1].frame_id == 44

        # Next 4 frames (frames 45..48): should not infer because stride=5
        for i in range(45, 49):
            buf.append(make_frame(i, i * 0.04))
            sched.on_frame_appended()
            assert not sched.should_infer(buf)

        # Frame 49 (5th frame since last inference): should infer!
        buf.append(make_frame(49, 49 * 0.04))
        sched.on_frame_appended()
        assert sched.should_infer(buf)

        sched.record_inference_start()
        window2 = buf.get_window()
        sched.record_inference_finish()
        assert window2[0].frame_id == 5
        assert window2[-1].frame_id == 49

        # Verify overlap: window2 contains 40 frames from window1
        assert window1[5].frame_id == window2[0].frame_id
