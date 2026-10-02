"""
SignTalk AI - MediaPipe Landmark Extractor
Implements the 93-node multimodal landmark architecture:
  - 21 Left Hand Articulators (Indices 0 - 20)
  - 21 Right Hand Articulators (Indices 21 - 41)
  - 11 Upper-Body Pose Anchors (Indices 42 - 52)
  - 40 Salient Facial Non-Manual Markers (Indices 53 - 92)
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions


# Pose mapping: MediaPipe Pose landmark indices for our 11 upper pose anchors
POSE_KEYPOINT_MAP = [
    0,   # 42: Nose
    2,   # 43: Left Eye
    5,   # 44: Right Eye
    7,   # 45: Left Ear
    8,   # 46: Right Ear
    11,  # 47: Left Shoulder (Key anatomical anchor)
    12,  # 48: Right Shoulder (Key anatomical anchor)
    13,  # 49: Left Elbow
    14,  # 50: Right Elbow
    15,  # 51: Left Wrist (Pose bridge)
    16   # 52: Right Wrist (Pose bridge)
]

# Selected 40 facial landmarks from 468 MediaPipe face mesh
# 8 Eyebrow contours (4 left, 4 right)
# 16 Lip & mouth contours (8 outer, 8 inner)
# 16 Lower jawline contours
FACE_KEYPOINT_MAP = [
    # Eyebrows (8): Left brow [70, 63, 105, 66], Right brow [300, 293, 334, 296]
    70, 63, 105, 66, 300, 293, 334, 296,
    # Lip contours (16): Outer [61, 146, 91, 181, 84, 17, 314, 405], Inner [78, 95, 88, 178, 87, 14, 317, 402]
    61, 146, 91, 181, 84, 17, 314, 405, 78, 95, 88, 178, 87, 14, 317, 402,
    # Jawline (16): Lower contour tracing chin and mandibular curve
    172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323
]


class LandmarkExtractionError(Exception):
    """Raised when landmark extraction encounters unrecoverable errors."""
    pass


class LandmarkExtractor:
    """
    Extracts multimodal skeletal landmarks from video frames according to the
    SignTalk AI 93-node specification using MediaPipe Tasks Vision API.
    """

    def __init__(
        self,
        pose_model_path: str = "models/mediapipe/pose_landmarker_full.task",
        hand_model_path: str = "models/mediapipe/hand_landmarker.task",
        face_model_path: str = "models/mediapipe/face_landmarker.task",
        pose_confidence: float = 0.5,
        hand_confidence: float = 0.35,
        face_confidence: float = 0.4,
        enable_face_mesh: bool = False
    ):
        self.pose_model_path = pose_model_path
        self.hand_model_path = hand_model_path
        self.face_model_path = face_model_path
        self.enable_face_mesh = enable_face_mesh

        # Verify model files exist
        for m_path in [self.pose_model_path, self.hand_model_path]:
            if not os.path.exists(m_path):
                raise FileNotFoundError(f"MediaPipe model asset not found: {m_path}")

        # Initialize Pose Landmarker
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.pose_model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=pose_confidence,
            min_pose_presence_confidence=pose_confidence
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)

        # Initialize Hand Landmarker
        hand_opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.hand_model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=hand_confidence,
            min_hand_presence_confidence=hand_confidence
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)

        # Initialize Face Landmarker if enabled
        self.face_landmarker = None
        if self.enable_face_mesh:
            if os.path.exists(self.face_model_path):
                face_opts = vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=self.face_model_path),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=face_confidence,
                    min_face_presence_confidence=face_confidence
                )
                self.face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)

    def extract_frame_landmarks(self, rgb_frame: np.ndarray) -> Dict[str, Any]:
        """
        Extracts 93 landmarks from a single RGB frame.

        Args:
            rgb_frame: Numpy array (H, W, 3) in RGB format.

        Returns:
            Dictionary containing:
                - 'coords': np.ndarray of shape (93, 3) [x, y, z]
                - 'visibility': np.ndarray of shape (93,) [confidence/visibility]
                - 'mask': np.ndarray of shape (93,) boolean detection mask
                - 'stats': Dictionary of modality detection flags
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 1. Run Pose Detection
        pose_res = self.pose_landmarker.detect(mp_image)

        # 2. Run Hand Detection
        hand_res = self.hand_landmarker.detect(mp_image)

        # 3. Run Face Detection if enabled
        face_res = None
        if self.face_landmarker is not None:
            face_res = self.face_landmarker.detect(mp_image)

        # Initialize buffers for 93 nodes
        coords = np.zeros((93, 3), dtype=np.float32)
        visibility = np.zeros((93,), dtype=np.float32)
        mask = np.zeros((93,), dtype=bool)

        stats = {
            "pose_detected": False,
            "left_hand_detected": False,
            "right_hand_detected": False,
            "face_detected": False,
            "left_hand_confidence": 0.0,
            "right_hand_confidence": 0.0,
            "pose_confidence": 0.0
        }

        # --- A. Hands Processing ---
        # Nodes 0-20: Left Hand, Nodes 21-41: Right Hand
        if hand_res.hand_landmarks and hand_res.handedness:
            for handedness_info, landmarks_list in zip(hand_res.handedness, hand_res.hand_landmarks):
                hand_label = handedness_info[0].category_name  # 'Left' or 'Right'
                hand_score = float(handedness_info[0].score)

                if hand_label == "Left":
                    node_offset = 0
                    stats["left_hand_detected"] = True
                    stats["left_hand_confidence"] = hand_score
                else:
                    node_offset = 21
                    stats["right_hand_detected"] = True
                    stats["right_hand_confidence"] = hand_score

                for j_idx, lm in enumerate(landmarks_list[:21]):
                    node_idx = node_offset + j_idx
                    coords[node_idx] = [lm.x, lm.y, lm.z]
                    # In MediaPipe Tasks, visibility or presence may be None or float
                    raw_vis = getattr(lm, 'visibility', None)
                    vis = float(raw_vis) if raw_vis is not None else hand_score
                    visibility[node_idx] = vis
                    mask[node_idx] = True

        # --- B. Pose Processing ---
        # Nodes 42-52: 11 Upper-Body Pose Anchors
        if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
            pose_lms = pose_res.pose_landmarks[0]
            stats["pose_detected"] = True
            pose_confs = []

            for target_node_offset, pose_idx in enumerate(POSE_KEYPOINT_MAP):
                node_idx = 42 + target_node_offset
                if pose_idx < len(pose_lms):
                    lm = pose_lms[pose_idx]
                    coords[node_idx] = [lm.x, lm.y, lm.z]
                    raw_vis = getattr(lm, 'visibility', None)
                    vis = float(raw_vis) if raw_vis is not None else 1.0
                    visibility[node_idx] = vis
                    mask[node_idx] = True
                    pose_confs.append(vis)

            if pose_confs:
                stats["pose_confidence"] = float(np.mean(pose_confs))

        # --- C. Face Processing ---
        # Nodes 53-92: 40 Salient Facial Non-Manual Markers
        if face_res is not None and face_res.face_landmarks and len(face_res.face_landmarks) > 0:
            face_lms = face_res.face_landmarks[0]
            stats["face_detected"] = True

            for target_node_offset, face_idx in enumerate(FACE_KEYPOINT_MAP):
                node_idx = 53 + target_node_offset
                if face_idx < len(face_lms):
                    lm = face_lms[face_idx]
                    coords[node_idx] = [lm.x, lm.y, lm.z]
                    raw_vis = getattr(lm, 'visibility', None)
                    vis = float(raw_vis) if raw_vis is not None else 1.0
                    visibility[node_idx] = vis
                    mask[node_idx] = True
        else:
            # Face mesh not detected or disabled:
            # Upper facial pose anchors (nose 42, eyes 43-44, ears 45-46) provide stable reference
            stats["face_detected"] = False

        return {
            "coords": coords,
            "visibility": visibility,
            "mask": mask,
            "stats": stats
        }

    def extract_sequence_landmarks(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Extracts landmarks across an entire sequence of RGB frames.

        Args:
            frames: List of RGB numpy arrays (H, W, 3).

        Returns:
            Dictionary containing:
                - 'coords': np.ndarray of shape (T, 93, 3)
                - 'visibility': np.ndarray of shape (T, 93)
                - 'mask': np.ndarray of shape (T, 93) boolean mask
                - 'frame_stats': List of stats per frame
                - 'sequence_stats': Aggregate detection statistics
        """
        T = len(frames)
        all_coords = np.zeros((T, 93, 3), dtype=np.float32)
        all_visibility = np.zeros((T, 93), dtype=np.float32)
        all_masks = np.zeros((T, 93), dtype=bool)
        frame_stats_list = []

        pose_count = 0
        lh_count = 0
        rh_count = 0
        face_count = 0

        for t_idx, frame in enumerate(frames):
            res = self.extract_frame_landmarks(frame)
            all_coords[t_idx] = res["coords"]
            all_visibility[t_idx] = res["visibility"]
            all_masks[t_idx] = res["mask"]
            frame_stats_list.append(res["stats"])

            if res["stats"]["pose_detected"]:
                pose_count += 1
            if res["stats"]["left_hand_detected"]:
                lh_count += 1
            if res["stats"]["right_hand_detected"]:
                rh_count += 1
            if res["stats"]["face_detected"]:
                face_count += 1

        sequence_stats = {
            "total_frames": T,
            "pose_detection_rate": round(pose_count / T * 100.0, 2) if T > 0 else 0.0,
            "left_hand_detection_rate": round(lh_count / T * 100.0, 2) if T > 0 else 0.0,
            "right_hand_detection_rate": round(rh_count / T * 100.0, 2) if T > 0 else 0.0,
            "face_detection_rate": round(face_count / T * 100.0, 2) if T > 0 else 0.0,
            "either_hand_detection_rate": round(sum(1 for s in frame_stats_list if s["left_hand_detected"] or s["right_hand_detected"]) / T * 100.0, 2) if T > 0 else 0.0
        }

        return {
            "coords": all_coords,
            "visibility": all_visibility,
            "mask": all_masks,
            "frame_stats": frame_stats_list,
            "sequence_stats": sequence_stats
        }

    def close(self):
        """Release underlying MediaPipe task landmarker resources."""
        if hasattr(self, 'pose_landmarker') and self.pose_landmarker:
            self.pose_landmarker.close()
        if hasattr(self, 'hand_landmarker') and self.hand_landmarker:
            self.hand_landmarker.close()
        if hasattr(self, 'face_landmarker') and self.face_landmarker:
            self.face_landmarker.close()
