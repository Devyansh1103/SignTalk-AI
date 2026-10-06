"""
Unit tests for InferenceScheduler (Phase 4 Part 2).
Validates sliding stride coordination, stale window prevention, and frame pacing.
"""

import pytest

from src.realtime.inference_scheduler import InferenceScheduler
from src.realtime.temporal_buffer import TemporalBuffer
from src.realtime.types import LandmarkFrame, ModalityDetection, QualityReport
import numpy as np


def make_frame(i: int) -> LandmarkFrame:
    return LandmarkFrame(
        frame_id=i,
        timestamp=i * 0.04,
        raw_coords=np.zeros((93, 3), dtype=np.float32),
        normalized_coords=np.zeros((93, 3), dtype=np.float32),
        mask=np.ones((93,), dtype=bool),
        visibility=np.ones((93,), dtype=np.float32),
        modality_stats=ModalityDetection(True, True, True, False),
        quality=QualityReport(True, 0.9, "GOOD", 0.0, False, True, False)
    )


class TestInferenceScheduler:
    def test_stride_cadence(self):
        buf = TemporalBuffer(max_length=45)
        sched = InferenceScheduler(stride=4, drop_stale_windows=True)

        for i in range(45):
            buf.append(make_frame(i))
            sched.on_frame_appended()

        # At frame 45 (0..44), buffer is full, frames_since_last_inference is 45 >= 4
        assert sched.should_infer(buf)
        sched.record_inference_start()
        sched.record_inference_finish()
        assert sched.inferences_triggered == 1

        # Next 3 frames: stride=4, so should not infer
        for i in range(45, 48):
            buf.append(make_frame(i))
            sched.on_frame_appended()
            assert not sched.should_infer(buf)

        # 4th frame: should infer
        buf.append(make_frame(48))
        sched.on_frame_appended()
        assert sched.should_infer(buf)
        sched.record_inference_start()
        sched.record_inference_finish()
        assert sched.inferences_triggered == 2

    def test_stale_window_dropping(self):
        buf = TemporalBuffer(max_length=45)
        sched = InferenceScheduler(stride=1, drop_stale_windows=True)

        for i in range(45):
            buf.append(make_frame(i))
            sched.on_frame_appended()

        # Mark inference in-flight
        sched.record_inference_start()

        # New frame arrives while inference is in flight
        buf.append(make_frame(45))
        sched.on_frame_appended()

        # Should drop stale window because drop_stale_windows=True and inference in-flight
        assert not sched.should_infer(buf)
        assert sched.windows_skipped == 1

        # Once finish is called, next eligible window can proceed
        sched.record_inference_finish()
        buf.append(make_frame(46))
        sched.on_frame_appended()
        assert sched.should_infer(buf)
