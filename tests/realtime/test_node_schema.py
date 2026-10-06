"""
Unit tests for Central Node Schema (93-node multimodal specification).
"""

import pytest
import numpy as np

from src.data.node_schema import (
    TOTAL_NODES,
    CHANNELS,
    NODE_DEFINITIONS,
    NODE_ID_TO_NAME,
    NAME_TO_NODE_ID,
    LEFT_HAND_RANGE,
    RIGHT_HAND_RANGE,
    UPPER_POSE_RANGE,
    FACE_CONTOURS_RANGE,
    get_modality_slice,
    is_node_in_modality,
    validate_node_tensor
)


def test_node_schema_cardinality():
    """Verify exactly 93 canonical nodes are registered."""
    assert TOTAL_NODES == 93
    assert len(NODE_DEFINITIONS) == 93
    assert len(NODE_ID_TO_NAME) == 93
    assert len(NAME_TO_NODE_ID) == 93


def test_modality_ranges():
    """Verify modality boundaries are contiguous and non-overlapping."""
    assert LEFT_HAND_RANGE == (0, 21)
    assert RIGHT_HAND_RANGE == (21, 42)
    assert UPPER_POSE_RANGE == (42, 53)
    assert FACE_CONTOURS_RANGE == (53, 93)

    assert get_modality_slice("left_hand") == slice(0, 21)
    assert get_modality_slice("right_hand") == slice(21, 42)
    assert get_modality_slice("upper_pose") == slice(42, 53)
    assert get_modality_slice("face_contours") == slice(53, 93)


def test_node_modality_membership():
    """Verify individual node lookup by modality."""
    # Left hand
    assert is_node_in_modality(0, "left_hand")
    assert is_node_in_modality(20, "left_hand")
    assert not is_node_in_modality(21, "left_hand")

    # Right hand
    assert is_node_in_modality(21, "right_hand")
    assert is_node_in_modality(41, "right_hand")
    assert not is_node_in_modality(42, "right_hand")

    # Pose
    assert is_node_in_modality(42, "upper_pose")
    assert is_node_in_modality(47, "upper_pose")  # Left shoulder
    assert is_node_in_modality(48, "upper_pose")  # Right shoulder
    assert is_node_in_modality(52, "upper_pose")

    # Face
    assert is_node_in_modality(53, "face_contours")
    assert is_node_in_modality(92, "face_contours")
    assert not is_node_in_modality(93, "face_contours")


def test_anatomical_key_anchors():
    """Verify specific anatomical anchors match expected nodes."""
    assert NODE_DEFINITIONS[0].name == "LH_WRIST"
    assert NODE_DEFINITIONS[21].name == "RH_WRIST"
    assert NODE_DEFINITIONS[42].name == "POSE_NOSE"
    assert NODE_DEFINITIONS[47].name == "POSE_LEFT_SHOULDER"
    assert NODE_DEFINITIONS[48].name == "POSE_RIGHT_SHOULDER"
    assert NODE_DEFINITIONS[51].name == "POSE_LEFT_WRIST_POSE"
    assert NODE_DEFINITIONS[52].name == "POSE_RIGHT_WRIST_POSE"


def test_validate_node_tensor():
    """Verify shape validation helper across 2D, 3D, and 4D tensors."""
    assert validate_node_tensor(np.zeros((93, 3))) is True
    assert validate_node_tensor(np.zeros((45, 93, 3))) is True
    assert validate_node_tensor(np.zeros((3, 45, 93))) is True
    assert validate_node_tensor(np.zeros((1, 3, 45, 93))) is True

    # Incompatible shapes
    assert validate_node_tensor(np.zeros((92, 3))) is False
    assert validate_node_tensor(np.zeros((93, 2))) is False
    assert validate_node_tensor(np.zeros((10, 10))) is False
