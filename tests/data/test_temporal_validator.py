"""
Unit tests for TemporalValidator.
"""

import os
import pytest
import numpy as np

from src.data.temporal_validator import TemporalValidator


@pytest.fixture
def validator():
    return TemporalValidator(target_fps=25.0, max_joint_jump=2.5)


def test_valid_recording(validator, tmp_path):
    fpath = tmp_path / "valid.npz"
    # Create valid continuous motion
    t = np.linspace(0, 1, 45)[:, np.newaxis]  # (45, 1)
    data = np.zeros((45, 93, 3), dtype=np.float32)
    data[:, :, 0] = t * 0.5  # smooth displacement (45, 93)
    np.savez_compressed(fpath, data=data)

    rep = validator.validate_recording(str(fpath), "valid")
    assert rep["valid"] is True
    assert rep["has_gaps"] is False
    assert rep["has_duplicates"] is False
    assert rep["has_nan_inf"] is False


def test_nan_inf_detection(validator, tmp_path):
    fpath = tmp_path / "nan.npz"
    data = np.zeros((45, 93, 3), dtype=np.float32)
    data[10, 5, 0] = np.nan
    np.savez_compressed(fpath, data=data)

    rep = validator.validate_recording(str(fpath), "nan")
    assert rep["valid"] is False
    assert rep["has_nan_inf"] is True


def test_catastrophic_jump_detection(validator, tmp_path):
    fpath = tmp_path / "jump.npz"
    data = np.zeros((45, 93, 3), dtype=np.float32)
    data[20, 0, 0] = 50.0  # huge jump > 10.0 units
    np.savez_compressed(fpath, data=data)

    rep = validator.validate_recording(str(fpath), "jump")
    assert rep["valid"] is False
    assert rep["has_coordinate_jumps"] is True
