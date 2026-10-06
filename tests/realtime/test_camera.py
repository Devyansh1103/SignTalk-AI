"""
Unit tests for Camera capture abstraction.
"""

import pytest
import os
import cv2
import numpy as np

from src.realtime.camera import (
    Camera,
    CameraInitializationError,
    CameraError
)
from src.realtime.types import FramePacket


def test_camera_invalid_device():
    """Verify Camera raises CameraInitializationError on non-existent hardware or file."""
    with pytest.raises(CameraInitializationError):
        cam = Camera(device_index="non_existent_file_path_12345.mp4")
        cam.start()


def test_camera_from_video_file_synchronous():
    """Verify synchronous frame reading from a known video sample."""
    sample_path = "data/interim/landmark_pilot/pilot_sample_01.mp4"
    if not os.path.exists(sample_path):
        pytest.skip("Pilot video sample not found")

    cam = Camera(
        device_index=sample_path,
        width=426,
        height=240,
        fps=25.0,
        threaded=False
    )
    cam.start()

    assert cam.is_running() is True

    packet = cam.read()
    assert packet is not None
    assert isinstance(packet, FramePacket)
    assert packet.frame_id == 1
    assert packet.timestamp > 0.0
    assert packet.image.shape == (240, 426, 3)
    assert packet.width == 426
    assert packet.height == 240

    # Read second frame
    packet2 = cam.read()
    assert packet2 is not None
    assert packet2.frame_id == 2
    assert packet2.timestamp >= packet.timestamp

    cam.stop()
    assert cam.is_running() is False


def test_camera_context_manager():
    """Verify Camera works as a clean context manager."""
    sample_path = "data/interim/landmark_pilot/pilot_sample_01.mp4"
    if not os.path.exists(sample_path):
        pytest.skip("Pilot video sample not found")

    with Camera(device_index=sample_path, threaded=False) as cam:
        assert cam.is_running() is True
        packet = cam.read()
        assert packet is not None

    assert cam.is_running() is False
