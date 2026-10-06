"""
Unit tests for Real-Time Landmark Shapes and Model Input Schema Validation.
"""

import pytest
import numpy as np
import torch

from src.realtime.types import LandmarkFrame, ModalityDetection, QualityReport
from src.data.node_schema import TOTAL_NODES, CHANNELS, validate_node_tensor
from src.data.stgcn_tensor import validate_stgcn_shape, NUM_NODES


def test_landmark_frame_shape_integrity():
    """Verify LandmarkFrame enforces exact (93, 3) coordinate arrays."""
    modality = ModalityDetection()
    quality = QualityReport(
        is_valid=True,
        quality_score=0.8,
        classification="GOOD",
        missing_ratio=0.2,
        has_nan_or_inf=False,
        coordinate_in_range=True,
        sudden_jumps_detected=False
    )

    # Valid shapes
    lf = LandmarkFrame(
        frame_id=1,
        timestamp=10.0,
        raw_coords=np.zeros((93, 3), dtype=np.float32),
        normalized_coords=np.zeros((93, 3), dtype=np.float32),
        mask=np.zeros((93,), dtype=bool),
        visibility=np.zeros((93,), dtype=np.float32),
        modality_stats=modality,
        quality=quality
    )

    assert lf.raw_coords.shape == (TOTAL_NODES, CHANNELS)
    assert lf.normalized_coords.shape == (TOTAL_NODES, CHANNELS)
    assert lf.mask.shape == (TOTAL_NODES,)
    assert lf.visibility.shape == (TOTAL_NODES,)

    # Incompatible shape must raise ValueError
    with pytest.raises(ValueError):
        LandmarkFrame(
            frame_id=1,
            timestamp=10.0,
            raw_coords=np.zeros((92, 3), dtype=np.float32),  # Wrong node count
            normalized_coords=np.zeros((93, 3), dtype=np.float32),
            mask=np.zeros((93,), dtype=bool),
            visibility=np.zeros((93,), dtype=np.float32),
            modality_stats=modality,
            quality=quality
        )


def test_stgcn_tensor_conversion():
    """Verify conversion to ST-GCN ready PyTorch tensors."""
    modality = ModalityDetection()
    quality = QualityReport(
        is_valid=True, quality_score=0.8, classification="GOOD",
        missing_ratio=0.2, has_nan_or_inf=False,
        coordinate_in_range=True, sudden_jumps_detected=False
    )

    lf = LandmarkFrame(
        frame_id=1,
        timestamp=10.0,
        raw_coords=np.zeros((93, 3), dtype=np.float32),
        normalized_coords=np.zeros((93, 3), dtype=np.float32),
        mask=np.ones((93,), dtype=bool),
        visibility=np.ones((93,), dtype=np.float32),
        modality_stats=modality,
        quality=quality
    )

    # Single frame: [3, 1, 93]
    tx, tm = lf.to_stgcn_frame_tensor(include_batch=False)
    assert tx.shape == (3, 1, 93)
    assert tm.shape == (1, 1, 93)
    assert tx.dtype == torch.float32
    assert tm.dtype == torch.float32

    # Batched single frame: [1, 3, 1, 93]
    tx_b, tm_b = lf.to_stgcn_frame_tensor(include_batch=True)
    assert tx_b.shape == (1, 3, 1, 93)
    assert tm_b.shape == (1, 1, 1, 93)
    assert tx_b.dtype == torch.float32


def test_45_frame_sequence_shape():
    """Verify that stacking 45 real-time landmark frames yields exactly [1, 3, 45, 93]."""
    modality = ModalityDetection()
    quality = QualityReport(
        is_valid=True, quality_score=0.8, classification="GOOD",
        missing_ratio=0.2, has_nan_or_inf=False,
        coordinate_in_range=True, sudden_jumps_detected=False
    )

    frames = [
        LandmarkFrame(
            frame_id=i,
            timestamp=float(i) / 25.0,
            raw_coords=np.zeros((93, 3), dtype=np.float32),
            normalized_coords=np.zeros((93, 3), dtype=np.float32),
            mask=np.ones((93,), dtype=bool),
            visibility=np.ones((93,), dtype=np.float32),
            modality_stats=modality,
            quality=quality
        )
        for i in range(45)
    ]

    # Stack into sequence array (45, 93, 3)
    seq_coords = np.stack([f.normalized_coords for f in frames], axis=0)  # (45, 93, 3)
    assert seq_coords.shape == (45, 93, 3)

    # Transpose to (3, 45, 93) [C, T, V]
    tensor_c_t_v = np.transpose(seq_coords, (2, 0, 1))
    tensor_b_c_t_v = torch.from_numpy(tensor_c_t_v).unsqueeze(0)  # [1, 3, 45, 93]

    assert tensor_b_c_t_v.shape == (1, 3, 45, 93)
    assert validate_stgcn_shape(tensor_b_c_t_v) is True
