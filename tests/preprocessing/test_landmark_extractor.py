"""
Integration tests for LandmarkExtractor module.
Tests 93-node architecture, coordinate shapes, and MediaPipe Tasks integration.
"""

import os
import pytest
import numpy as np
import cv2

from src.preprocessing.landmark_extractor import LandmarkExtractor


@pytest.fixture(scope="module")
def extractor():
    """Initializes landmark extractor once for the test module."""
    return LandmarkExtractor(
        pose_model_path="models/mediapipe/pose_landmarker_full.task",
        hand_model_path="models/mediapipe/hand_landmarker.task",
        face_model_path="models/mediapipe/face_landmarker.task",
        enable_face_mesh=False
    )


def test_extractor_single_frame(extractor):
    # Test on a blank synthetic image
    synthetic_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    res = extractor.extract_frame_landmarks(synthetic_frame)

    coords = res["coords"]
    visibility = res["visibility"]
    mask = res["mask"]

    # Verify fixed 93-node structure
    assert coords.shape == (93, 3)
    assert visibility.shape == (93,)
    assert mask.shape == (93,)
    assert coords.dtype == np.float32

    # On a blank image, all masks should be False and coords 0.0
    assert not np.any(mask)
    assert np.all(coords == 0.0)


def test_extractor_real_pilot_frame(extractor):
    # Test on real pilot sample frame
    cap = cv2.VideoCapture("data/interim/landmark_pilot/pilot_sample_01.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
    ret, frame = cap.read()
    cap.release()

    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = extractor.extract_frame_landmarks(rgb)

        # On frame 25 of pilot sample, pose should be detected
        assert res["stats"]["pose_detected"] == True
        assert res["coords"].shape == (93, 3)
        # Pose nodes 42-52 should have valid mask
        assert np.any(res["mask"][42:53])
