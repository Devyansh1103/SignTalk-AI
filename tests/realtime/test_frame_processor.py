"""
Unit tests for FrameProcessor module.
"""

import pytest
import numpy as np
import cv2

from src.realtime.frame_processor import FrameProcessor, FrameProcessingError
from src.realtime.types import FramePacket
from src.realtime.realtime_config import ProcessingConfig


def test_frame_processor_bgr_to_rgb():
    """Verify BGR to RGB color conversion."""
    processor = FrameProcessor(flip_horizontal=False, color_format="RGB")

    # Create synthetic frame with distinct BGR pixel: Blue=255, Green=100, Red=50
    bgr_img = np.zeros((100, 100, 3), dtype=np.uint8)
    bgr_img[:, :] = [255, 100, 50]

    packet = FramePacket(
        frame_id=1,
        timestamp=1.0,
        capture_time=1.0,
        image=bgr_img,
        width=100,
        height=100,
        is_rgb=False,
        is_flipped=False
    )

    out_packet = processor.process(packet)

    assert out_packet.is_rgb is True
    # In RGB, Red=50, Green=100, Blue=255
    assert out_packet.image[0, 0, 0] == 50
    assert out_packet.image[0, 0, 1] == 100
    assert out_packet.image[0, 0, 2] == 255


def test_frame_processor_horizontal_flip():
    """Verify horizontal flipping (mirroring)."""
    processor = FrameProcessor(flip_horizontal=True, color_format="BGR")

    # Create image with marker on left half
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :50] = 255

    packet = FramePacket(
        frame_id=1,
        timestamp=1.0,
        capture_time=1.0,
        image=img,
        width=100,
        height=100,
        is_rgb=False,
        is_flipped=False
    )

    out_packet = processor.process(packet)

    assert out_packet.is_flipped is True
    # Marker should now be on the right half (> 50)
    assert np.all(out_packet.image[:, 50:] == 255)
    assert np.all(out_packet.image[:, :50] == 0)


def test_frame_processor_resizing():
    """Verify resizing to target dimensions."""
    processor = FrameProcessor(
        flip_horizontal=False,
        color_format="BGR",
        target_width=320,
        target_height=240
    )

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    packet = FramePacket(
        frame_id=1,
        timestamp=1.0,
        capture_time=1.0,
        image=img,
        width=640,
        height=480
    )

    out_packet = processor.process(packet)
    assert out_packet.width == 320
    assert out_packet.height == 240
    assert out_packet.image.shape == (240, 320, 3)


def test_frame_processor_invalid_inputs():
    """Verify error handling on invalid image buffers."""
    processor = FrameProcessor()

    # Empty image
    empty_packet = FramePacket(1, 1.0, 1.0, np.zeros((0, 0, 3), dtype=np.uint8), 0, 0)
    with pytest.raises(FrameProcessingError):
        processor.process(empty_packet)

    # Wrong dtype
    float_packet = FramePacket(1, 1.0, 1.0, np.zeros((10, 10, 3), dtype=np.float32), 10, 10)
    with pytest.raises(FrameProcessingError):
        processor.process(float_packet)

    # Grayscale image (2 channels)
    gray_packet = FramePacket(1, 1.0, 1.0, np.zeros((10, 10), dtype=np.uint8), 10, 10)
    with pytest.raises(FrameProcessingError):
        processor.process(gray_packet)
