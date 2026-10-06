"""
SignTalk AI - Canonical 93-Node Multimodal Skeletal Schema
Defines the authoritative node index ordering, anatomical annotations, and modality
partitions for the 93-node spatiotemporal graph representation used across
SignTalk AI training, offline preprocessing, and real-time inference.

Schema: 93-node-v1
  - Nodes 00 - 20: Left Hand (21 phalanx keypoints)
  - Nodes 21 - 41: Right Hand (21 phalanx keypoints)
  - Nodes 42 - 52: Upper Pose (11 torso & cranial anatomical anchors)
  - Nodes 53 - 92: Facial Non-Manual Contours (40 eyebrow, lip, and jawline markers)
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NodeDefinition:
    node_id: int
    name: str
    modality: str
    anatomical_meaning: str
    parent_node_id: Optional[int] = None
    mediapipe_source: str = ""
    mediapipe_raw_index: Optional[int] = None


# MediaPipe hand landmark names (0..20)
HAND_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# MediaPipe Pose indices used for 11 upper pose anchors
# [Nose(0), LeftEye(2), RightEye(5), LeftEar(7), RightEar(8),
#  LeftShoulder(11), RightShoulder(12), LeftElbow(13), RightElbow(14),
#  LeftWrist(15), RightWrist(16)]
POSE_INDICES = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]

POSE_LANDMARK_NAMES = [
    ("NOSE", "Cranial central reference anchor", 0),
    ("LEFT_EYE", "Left ocular gaze / head orientation anchor", 2),
    ("RIGHT_EYE", "Right ocular gaze / head orientation anchor", 5),
    ("LEFT_EAR", "Left cranial lateral anchor", 7),
    ("RIGHT_EAR", "Right cranial lateral anchor", 8),
    ("LEFT_SHOULDER", "Left torso origin / shoulder width baseline", 11),
    ("RIGHT_SHOULDER", "Right torso origin / shoulder width baseline", 12),
    ("LEFT_ELBOW", "Left upper arm articulatory pivot", 13),
    ("RIGHT_ELBOW", "Right upper arm articulatory pivot", 14),
    ("LEFT_WRIST_POSE", "Pose left arm terminal bridge to hand", 15),
    ("RIGHT_WRIST_POSE", "Pose right arm terminal bridge to hand", 16),
]

# 40 Facial keypoints selected from 468 MediaPipe face mesh:
# 8 Eyebrows: left [70, 63, 105, 66], right [300, 293, 334, 296]
# 16 Lips: outer [61, 146, 91, 181, 84, 17, 314, 405], inner [78, 95, 88, 178, 87, 14, 317, 402]
# 16 Jawline: [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
FACE_KEYPOINT_MAPPINGS = [
    # Eyebrows (8 nodes: 53-60)
    (70, "LEFT_EYEBROW_OUTER", "Left eyebrow outer contour"),
    (63, "LEFT_EYEBROW_MID_OUTER", "Left eyebrow upper mid contour"),
    (105, "LEFT_EYEBROW_MID_INNER", "Left eyebrow inner mid contour"),
    (66, "LEFT_EYEBROW_INNER", "Left eyebrow inner terminal"),
    (300, "RIGHT_EYEBROW_INNER", "Right eyebrow inner terminal"),
    (293, "RIGHT_EYEBROW_MID_INNER", "Right eyebrow inner mid contour"),
    (334, "RIGHT_EYEBROW_MID_OUTER", "Right eyebrow upper mid contour"),
    (296, "RIGHT_EYEBROW_OUTER", "Right eyebrow outer contour"),

    # Outer Lips (8 nodes: 61-68)
    (61, "MOUTH_CORNER_LEFT", "Mouth left oral corner"),
    (146, "MOUTH_OUTER_UPPER_LEFT", "Upper outer lip left segment"),
    (91, "MOUTH_OUTER_LOWER_LEFT", "Lower outer lip left segment"),
    (181, "MOUTH_OUTER_LOWER_MID_LEFT", "Lower outer lip mid-left segment"),
    (84, "MOUTH_OUTER_LOWER_MID_RIGHT", "Lower outer lip mid-right segment"),
    (17, "MOUTH_OUTER_LOWER_CENTER", "Lower outer lip center terminal"),
    (314, "MOUTH_OUTER_LOWER_RIGHT", "Lower outer lip right segment"),
    (405, "MOUTH_CORNER_RIGHT", "Mouth right oral corner"),

    # Inner Lips (8 nodes: 69-76)
    (78, "MOUTH_INNER_CORNER_LEFT", "Inner mouth left corner"),
    (95, "MOUTH_INNER_UPPER_LEFT", "Inner upper lip left segment"),
    (88, "MOUTH_INNER_LOWER_LEFT", "Inner lower lip left segment"),
    (178, "MOUTH_INNER_LOWER_MID_LEFT", "Inner lower lip mid-left segment"),
    (87, "MOUTH_INNER_LOWER_MID_RIGHT", "Inner lower lip mid-right segment"),
    (14, "MOUTH_INNER_LOWER_CENTER", "Inner lower lip center terminal"),
    (317, "MOUTH_INNER_LOWER_RIGHT", "Inner lower lip right segment"),
    (402, "MOUTH_INNER_CORNER_RIGHT", "Inner mouth right corner"),

    # Mandibular & Jawline contour (16 nodes: 77-92)
    (172, "JAWLINE_01", "Left jaw upper mandibular angle"),
    (136, "JAWLINE_02", "Left jaw ramus contour"),
    (150, "JAWLINE_03", "Left jaw lower ramus"),
    (149, "JAWLINE_04", "Left jaw posterior body"),
    (176, "JAWLINE_05", "Left jaw anterior body"),
    (148, "JAWLINE_06", "Left jaw mental tubercle lateral"),
    (152, "JAWLINE_CHIN_CENTER", "Mentum / central gnathion apex"),
    (377, "JAWLINE_08", "Right jaw mental tubercle lateral"),
    (400, "JAWLINE_09", "Right jaw anterior body"),
    (378, "JAWLINE_10", "Right jaw posterior body"),
    (379, "JAWLINE_11", "Right jaw lower ramus"),
    (365, "JAWLINE_12", "Right jaw ramus contour"),
    (397, "JAWLINE_13", "Right jaw upper mandibular angle"),
    (288, "JAWLINE_14", "Right jaw post-auricular boundary"),
    (361, "JAWLINE_15", "Right jaw zygomatic-mandibular junction"),
    (323, "JAWLINE_16", "Right jaw superior margin anchor")
]


def _build_canonical_node_registry() -> Dict[int, NodeDefinition]:
    registry: Dict[int, NodeDefinition] = {}

    # 1. Left Hand (0..20)
    for idx, name in enumerate(HAND_LANDMARK_NAMES):
        node_id = idx
        parent = 0 if idx in [1, 5, 9, 13, 17] else (idx - 1 if idx > 0 else None)
        registry[node_id] = NodeDefinition(
            node_id=node_id,
            name=f"LH_{name}",
            modality="left_hand",
            anatomical_meaning=f"Left Hand {name.replace('_', ' ').title()}",
            parent_node_id=parent,
            mediapipe_source="HandLandmarker",
            mediapipe_raw_index=idx
        )

    # 2. Right Hand (21..41)
    for idx, name in enumerate(HAND_LANDMARK_NAMES):
        node_id = 21 + idx
        parent = 21 if idx in [1, 5, 9, 13, 17] else (21 + idx - 1 if idx > 0 else None)
        registry[node_id] = NodeDefinition(
            node_id=node_id,
            name=f"RH_{name}",
            modality="right_hand",
            anatomical_meaning=f"Right Hand {name.replace('_', ' ').title()}",
            parent_node_id=parent,
            mediapipe_source="HandLandmarker",
            mediapipe_raw_index=idx
        )

    # 3. Upper Pose (42..52)
    for idx, (p_name, desc, raw_idx) in enumerate(POSE_LANDMARK_NAMES):
        node_id = 42 + idx
        registry[node_id] = NodeDefinition(
            node_id=node_id,
            name=f"POSE_{p_name}",
            modality="upper_pose",
            anatomical_meaning=desc,
            parent_node_id=47 if "LEFT" in p_name else (48 if "RIGHT" in p_name else None),
            mediapipe_source="PoseLandmarker",
            mediapipe_raw_index=raw_idx
        )

    # 4. Face Contours (53..92)
    for idx, (raw_idx, f_name, desc) in enumerate(FACE_KEYPOINT_MAPPINGS):
        node_id = 53 + idx
        registry[node_id] = NodeDefinition(
            node_id=node_id,
            name=f"FACE_{f_name}",
            modality="face_contours",
            anatomical_meaning=desc,
            parent_node_id=42,  # Rooted to cranial nose anchor
            mediapipe_source="FaceLandmarker",
            mediapipe_raw_index=raw_idx
        )

    return registry


NODE_DEFINITIONS: Dict[int, NodeDefinition] = _build_canonical_node_registry()
NODE_ID_TO_NAME: Dict[int, str] = {nid: nd.name for nid, nd in NODE_DEFINITIONS.items()}
NAME_TO_NODE_ID: Dict[str, int] = {nd.name: nid for nid, nd in NODE_DEFINITIONS.items()}

# Partition Slices & Ranges
LEFT_HAND_RANGE: Tuple[int, int] = (0, 21)
RIGHT_HAND_RANGE: Tuple[int, int] = (21, 42)
UPPER_POSE_RANGE: Tuple[int, int] = (42, 53)
FACE_CONTOURS_RANGE: Tuple[int, int] = (53, 93)

MODALITY_SLICES: Dict[str, slice] = {
    "left_hand": slice(0, 21),
    "right_hand": slice(21, 42),
    "upper_pose": slice(42, 53),
    "face_contours": slice(53, 93)
}

TOTAL_NODES: int = 93
CHANNELS: int = 3
TARGET_SEQUENCE_LENGTH: int = 45


def get_modality_slice(modality: str) -> slice:
    """Returns slice corresponding to given modality."""
    if modality not in MODALITY_SLICES:
        raise KeyError(f"Unknown modality '{modality}'. Expected one of {list(MODALITY_SLICES.keys())}")
    return MODALITY_SLICES[modality]


def is_node_in_modality(node_id: int, modality: str) -> bool:
    """Returns True if node_id belongs to the specified modality."""
    sl = get_modality_slice(modality)
    return sl.start <= node_id < sl.stop


def validate_node_tensor(tensor: np.ndarray) -> bool:
    """
    Validates that a landmark tensor has the canonical 93 nodes and 3 coordinates.
    Accepts (93, 3), (T, 93, 3), or (3, T, 93).
    """
    if tensor.ndim == 2:
        return tensor.shape == (TOTAL_NODES, CHANNELS)
    elif tensor.ndim == 3:
        if tensor.shape[1] == TOTAL_NODES and tensor.shape[2] == CHANNELS:
            return True  # (T, 93, 3)
        if tensor.shape[0] == CHANNELS and tensor.shape[2] == TOTAL_NODES:
            return True  # (3, T, 93)
    elif tensor.ndim == 4:
        if tensor.shape[1] == CHANNELS and tensor.shape[3] == TOTAL_NODES:
            return True  # (B, 3, T, 93)
    return False
