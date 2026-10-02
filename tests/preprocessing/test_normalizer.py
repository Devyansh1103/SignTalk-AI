"""
Unit tests for CoordinateNormalizer module.
Tests torso centering, scale invariance, and handedness mirroring.
"""

import pytest
import numpy as np
from src.preprocessing.normalizer import CoordinateNormalizer


def test_torso_centering_and_scale():
    normalizer = CoordinateNormalizer(method="torso_scale")

    coords = np.zeros((93, 3), dtype=np.float32)
    mask = np.zeros((93,), dtype=bool)

    # Left shoulder = node 47, Right shoulder = node 48
    # Set shoulders at (0.4, 0.5, 0.0) and (0.6, 0.5, 0.0)
    coords[47] = [0.4, 0.5, 0.0]
    coords[48] = [0.6, 0.5, 0.0]
    mask[47] = True
    mask[48] = True

    # Center is (0.5, 0.5, 0.0), Distance is 0.2
    # Set nose = node 42 at (0.5, 0.3, 0.0)
    coords[42] = [0.5, 0.3, 0.0]
    mask[42] = True

    norm_coords, meta = normalizer.normalize_frame(coords, mask)

    # Mid-shoulder after normalization:
    # ls_norm_x = (0.4 - 0.5) / 0.2 = -0.5
    # rs_norm_x = (0.6 - 0.5) / 0.2 = +0.5
    assert np.isclose(norm_coords[47, 0], -0.5, atol=1e-5)
    assert np.isclose(norm_coords[48, 0], 0.5, atol=1e-5)
    assert np.isclose((norm_coords[47, 0] + norm_coords[48, 0]) / 2.0, 0.0, atol=1e-5)

    # Nose x should be exactly 0.0 (centered horizontally)
    assert np.isclose(norm_coords[42, 0], 0.0, atol=1e-5)
    # Nose y should be (0.3 - 0.5) / 0.2 = -1.0
    assert np.isclose(norm_coords[42, 1], -1.0, atol=1e-5)


def test_handedness_mirroring():
    coords = np.zeros((93, 3), dtype=np.float32)
    mask = np.zeros((93,), dtype=bool)

    # Left hand wrist (node 0) at x = -0.3, y = 0.2, z = 0.1
    coords[0] = [-0.3, 0.2, 0.1]
    mask[0] = True

    # Right hand wrist (node 21) is initially empty
    mask[21] = False

    # Left eye (node 43) at (-0.1, -0.4, 0.0)
    coords[43] = [-0.1, -0.4, 0.0]
    mask[43] = True

    mirrored_coords, mirrored_mask = CoordinateNormalizer.mirror_horizontal(coords, mask)

    # After horizontal reflection:
    # 1. Old left hand (node 0) swapped into right hand (node 21)
    # 2. X coordinate negated: -(-0.3) = +0.3
    assert mirrored_mask[21] == True
    assert np.isclose(mirrored_coords[21, 0], 0.3, atol=1e-5)
    assert np.isclose(mirrored_coords[21, 1], 0.2, atol=1e-5)

    # Node 0 (left hand) should now be False
    assert mirrored_mask[0] == False

    # Left eye (43) should swap with right eye (44)
    assert mirrored_mask[44] == True
    assert np.isclose(mirrored_coords[44, 0], 0.1, atol=1e-5)
