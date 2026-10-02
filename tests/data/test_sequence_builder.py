"""
Unit tests for SequenceBuilder.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data.sequence_builder import SequenceBuilder
from src.data.stgcn_tensor import SEQUENCE_LENGTH, NUM_NODES, CHANNELS


@pytest.fixture
def builder():
    return SequenceBuilder(config_path="configs/sequence_generation.yaml")


def test_builder_initialization(builder):
    assert builder is not None
    assert builder.seq_params["target_sequence_length"] == 45
    assert len(builder.label_mapping) == 10
    assert "HELLO" in builder.vocabulary["token_to_id"]


def test_process_sample(builder, tmp_path):
    # Create mock landmark npz
    mock_data = np.zeros((3, 45, 93), dtype=np.float32)
    mock_mask = np.ones((1, 45, 93), dtype=np.float32)
    mock_file = tmp_path / "raw_test.npz"
    np.savez_compressed(
        mock_file,
        data=mock_data,
        mask=mock_mask,
        sample_id="raw_test",
        class_id=0,
        signer_id="signer_01",
        split="train",
        fps=25.0
    )

    meta, arrays = builder.process_sample(str(mock_file), sequence_index=1)

    assert meta["sequence_id"] == "seq_0001"
    assert meta["label"] == "hello"
    assert meta["gloss"] == "HELLO"
    assert meta["split"] == "train"
    assert meta["target_frames"] == 45
    assert arrays["data"].shape == (3, 45, 93)
    assert arrays["mask"].shape == (1, 45, 93)
    assert arrays["label"] == 0
    assert arrays["gloss_id"] == 4
