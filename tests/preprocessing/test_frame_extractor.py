"""
Unit tests for FrameExtractor module.
Tests video/image extraction, sampling rate, timestamp tracking, and error handling.
"""

import os
import pytest
import numpy as np
import cv2

from src.data.frame_extractor import FrameExtractor, FrameExtractionError


@pytest.fixture
def sample_video(tmp_path):
    """Creates a temporary synthetic 30-frame video for testing."""
    v_path = str(tmp_path / "test_synth.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(v_path, fourcc, 25.0, (160, 120))
    for i in range(30):
        # Create patterned frame
        frame = np.full((120, 160, 3), i * 8, dtype=np.uint8)
        out.write(frame)
    out.release()
    return v_path


@pytest.fixture
def sample_image(tmp_path):
    """Creates a temporary synthetic image for testing."""
    img_path = str(tmp_path / "test_synth.jpg")
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.imwrite(img_path, img)
    return img_path


def test_frame_extractor_basic(sample_video):
    extractor = FrameExtractor(target_fps=25.0, frame_sampling_rate=1, image_size=(160, 120))
    result = extractor.extract_from_video(sample_video)

    frames = result["frames"]
    meta = result["metadata"]

    assert len(frames) == 30
    assert frames[0].shape == (120, 160, 3)
    assert meta["original_frame_count"] == 30
    assert meta["target_fps"] == 25.0
    assert len(meta["frame_records"]) == 30
    assert meta["frame_records"][0]["timestamp_sec"] == 0.0


def test_frame_extractor_resampling(sample_video):
    # Test sampling with frame_sampling_rate=2
    extractor = FrameExtractor(target_fps=25.0, frame_sampling_rate=2, uniform_resampling=False)
    result = extractor.extract_from_video(sample_video)
    assert len(result["frames"]) == 15


def test_frame_extractor_image(sample_image):
    extractor = FrameExtractor(image_size=(80, 60))
    result = extractor.extract_from_image(sample_image)
    frames = result["frames"]
    assert len(frames) == 1
    assert frames[0].shape == (60, 80, 3)
    assert result["metadata"]["extracted_frame_count"] == 1


def test_frame_extractor_missing_file():
    extractor = FrameExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract_from_video("non_existent_file.mp4")
