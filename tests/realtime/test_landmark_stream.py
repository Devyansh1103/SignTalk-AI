"""
Unit and integration tests for RealTimeLandmarkStream.
"""

import pytest
import os
import cv2
import numpy as np

from src.realtime.landmark_stream import RealTimeLandmarkStream
from src.realtime.realtime_config import RealTimeConfig
from src.realtime.types import FramePacket, LandmarkFrame


@pytest.fixture(scope="module")
def stream():
    cfg = RealTimeConfig()
    cfg.runtime.display_preview = False
    s = RealTimeLandmarkStream(cfg)
    yield s
    s.close()


def test_stream_synthetic_frame(stream):
    """Verify processing of a synthetic blank frame."""
    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
    packet = FramePacket(
        frame_id=1,
        timestamp=1.0,
        capture_time=1.0,
        image=blank_img,
        width=640,
        height=480,
        is_rgb=False
    )

    lf = stream.process_frame(packet)

    assert isinstance(lf, LandmarkFrame)
    assert lf.frame_id == 1
    assert lf.raw_coords.shape == (93, 3)
    assert lf.normalized_coords.shape == (93, 3)
    assert lf.mask.shape == (93,)
    # Blank frame should not detect pose
    assert lf.modality_stats.pose_detected is False
    assert lf.quality.classification == "REJECT"
    assert lf.total_latency_ms >= 0.0


def test_stream_metrics_tracking(stream):
    """Verify rolling metrics accumulation and latency tracking."""
    metrics = stream.get_metrics()
    assert metrics.total_processed_frames >= 1
    assert metrics.total_dropped_frames == 0


def test_render_debug_overlay(stream):
    """Verify HUD overlay renders without errors on RGB/BGR frame."""
    bgr_img = np.zeros((480, 640, 3), dtype=np.uint8)
    packet = FramePacket(
        frame_id=2,
        timestamp=1.04,
        capture_time=1.04,
        image=bgr_img,
        width=640,
        height=480
    )

    lf = stream.process_frame(packet)
    hud_canvas = stream.render_debug_overlay(bgr_img, lf, fps_display=25.0)

    assert hud_canvas.shape == (480, 640, 3)
    assert hud_canvas.dtype == np.uint8
    # Canvas should have drawn pixels (not pure zero)
    assert np.any(hud_canvas > 0)
