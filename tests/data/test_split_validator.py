"""
Unit tests for SplitValidator.
"""

import pytest
import pandas as pd

from src.data.split_validator import SplitValidator, SplitLeakageError


def test_split_validator_clean():
    df = pd.DataFrame([
        {"sequence_id": "seq_01", "source_recording_id": "rec_01", "source_video_id": "vid_01.mp4", "split": "train", "signer_id": "s1"},
        {"sequence_id": "seq_02", "source_recording_id": "rec_02", "source_video_id": "vid_02.mp4", "split": "train", "signer_id": "s2"},
        {"sequence_id": "seq_03", "source_recording_id": "rec_03", "source_video_id": "vid_03.mp4", "split": "val", "signer_id": "s1"},
        {"sequence_id": "seq_04", "source_recording_id": "rec_04", "source_video_id": "vid_04.mp4", "split": "test", "signer_id": "s2"},
    ])
    validator = SplitValidator(df)
    rep = validator.check_leakage()
    assert rep["passed"] is True
    assert rep["train_unique_videos"] == 2
    assert rep["val_unique_videos"] == 1
    assert rep["test_unique_videos"] == 1


def test_split_validator_leakage_detection():
    # Artificially inject vid_01 into both train and test
    df = pd.DataFrame([
        {"sequence_id": "seq_01", "source_recording_id": "rec_01", "source_video_id": "vid_01.mp4", "split": "train"},
        {"sequence_id": "seq_02", "source_recording_id": "rec_01", "source_video_id": "vid_01.mp4", "split": "test"},
    ])
    validator = SplitValidator(df)
    with pytest.raises(SplitLeakageError):
        validator.check_leakage()
