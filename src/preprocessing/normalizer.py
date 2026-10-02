"""
SignTalk AI - Landmark Normalizer
Implements anatomical translation and scale transformations:
  - Torso-Centering & Scale Invariance (Primary Blueprint method)
  - Wrist-Centering Hand Normalization
  - Handedness Invariant Mirroring Transformation
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


class CoordinateNormalizer:
    """
    Transforms raw pixel/image coordinates into scale- and translation-invariant
    body-centric coordinate systems suitable for spatial-temporal graph modeling.
    """

    def __init__(
        self,
        method: str = "torso_scale",
        min_scale_epsilon: float = 1.0e-4,
        z_scale_factor: float = 1.0
    ):
        """
        Args:
            method: Normalization strategy ('torso_scale', 'wrist_centered', 'min_max', 'none').
            min_scale_epsilon: Guard to prevent division by zero during occlusion.
            z_scale_factor: Multiplier for depth coordinates.
        """
        self.method = method
        self.min_scale_epsilon = float(min_scale_epsilon)
        self.z_scale_factor = float(z_scale_factor)

    def normalize_frame(
        self,
        coords: np.ndarray,      # (93, 3)
        mask: np.ndarray         # (93,)
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Normalizes 3D coordinates for a single frame.

        Args:
            coords: Raw joint coordinates (93, 3) [x, y, z].
            mask: Boolean array (93,) indicating joint validity.

        Returns:
            Tuple of:
                - normalized_coords: (93, 3) normalized array
                - norm_meta: Dictionary of center and scale factors used
        """
        if self.method == "none":
            return coords.copy(), {"method": "none", "scale": 1.0}

        norm_coords = coords.copy()

        # Pose shoulder anchors: Left Shoulder = 47, Right Shoulder = 48
        left_shoulder_valid = mask[47]
        right_shoulder_valid = mask[48]

        # Calculate torso center and scale
        if left_shoulder_valid and right_shoulder_valid:
            p_ls = coords[47]
            p_rs = coords[48]
            c_torso = (p_ls + p_rs) / 2.0
            shoulder_dist = np.linalg.norm(p_ls - p_rs)
            scale = max(float(shoulder_dist), self.min_scale_epsilon)
        elif left_shoulder_valid:
            c_torso = coords[47].copy()
            scale = 0.25  # Anatomical default scale estimation
        elif right_shoulder_valid:
            c_torso = coords[48].copy()
            scale = 0.25
        else:
            # Fallback if both shoulders occluded: use centroid of valid pose joints
            valid_pose = np.where(mask[42:53])[0]
            if len(valid_pose) > 0:
                c_torso = np.mean(coords[42 + valid_pose], axis=0)
                scale = 0.25
            else:
                # Default origin
                c_torso = np.array([0.5, 0.5, 0.0], dtype=np.float32)
                scale = 1.0

        if self.method == "torso_scale":
            # Torso centering across all 93 nodes
            for i in range(93):
                if mask[i]:
                    norm_coords[i, 0] = (coords[i, 0] - c_torso[0]) / scale
                    norm_coords[i, 1] = (coords[i, 1] - c_torso[1]) / scale
                    norm_coords[i, 2] = (coords[i, 2] - c_torso[2]) / scale * self.z_scale_factor
                else:
                    norm_coords[i] = 0.0

        elif self.method == "wrist_centered":
            # Pose normalized relative to torso
            for i in range(42, 93):
                if mask[i]:
                    norm_coords[i, 0] = (coords[i, 0] - c_torso[0]) / scale
                    norm_coords[i, 1] = (coords[i, 1] - c_torso[1]) / scale
                    norm_coords[i, 2] = (coords[i, 2] - c_torso[2]) / scale * self.z_scale_factor
                else:
                    norm_coords[i] = 0.0

            # Left Hand (0-20) normalized relative to Left Wrist (0)
            if mask[0]:
                lw = coords[0].copy()
                hand_scale = np.linalg.norm(coords[9] - lw) if mask[9] else scale * 0.4
                hand_scale = max(float(hand_scale), self.min_scale_epsilon)
                for i in range(21):
                    if mask[i]:
                        norm_coords[i] = (coords[i] - lw) / hand_scale
                    else:
                        norm_coords[i] = 0.0
            else:
                norm_coords[0:21] = 0.0

            # Right Hand (21-41) normalized relative to Right Wrist (21)
            if mask[21]:
                rw = coords[21].copy()
                hand_scale = np.linalg.norm(coords[30] - rw) if mask[30] else scale * 0.4
                hand_scale = max(float(hand_scale), self.min_scale_epsilon)
                for i in range(21, 42):
                    if mask[i]:
                        norm_coords[i] = (coords[i] - rw) / hand_scale
                    else:
                        norm_coords[i] = 0.0
            else:
                norm_coords[21:42] = 0.0

        elif self.method == "min_max":
            valid_pts = coords[mask]
            if len(valid_pts) > 0:
                min_val = np.min(valid_pts, axis=0)
                max_val = np.max(valid_pts, axis=0)
                diff = np.maximum(max_val - min_val, self.min_scale_epsilon)
                for i in range(93):
                    if mask[i]:
                        norm_coords[i] = (coords[i] - min_val) / diff
                    else:
                        norm_coords[i] = 0.0
            scale = 1.0

        norm_meta = {
            "method": self.method,
            "center": c_torso.tolist(),
            "scale": float(scale),
            "left_shoulder_valid": bool(left_shoulder_valid),
            "right_shoulder_valid": bool(right_shoulder_valid)
        }

        return norm_coords, norm_meta

    def normalize_sequence(
        self,
        coords: np.ndarray,      # (T, 93, 3)
        mask: np.ndarray         # (T, 93)
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Normalizes a temporal sequence of landmarks.
        Uses sequence-smoothed torso center and scale to prevent high-frequency
        normalization jitter when shoulders move slightly.

        Args:
            coords: (T, 93, 3) raw coordinate array.
            mask: (T, 93) boolean array.

        Returns:
            Tuple of (normalized_coords, sequence_normalization_metadata).
        """
        T = coords.shape[0]
        norm_seq = np.zeros_like(coords, dtype=np.float32)

        # Precompute stable sequence-wide anchor scale if available
        both_shoulders_mask = mask[:, 47] & mask[:, 48]
        if np.any(both_shoulders_mask):
            seq_ls = coords[both_shoulders_mask, 47]
            seq_rs = coords[both_shoulders_mask, 48]
            global_center = np.mean((seq_ls + seq_rs) / 2.0, axis=0)
            global_scale = float(np.mean(np.linalg.norm(seq_ls - seq_rs, axis=1)))
            global_scale = max(global_scale, self.min_scale_epsilon)
            stable_reference = True
        else:
            global_center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
            global_scale = 1.0
            stable_reference = False

        frame_scales = []
        for t in range(T):
            norm_frame, f_meta = self.normalize_frame(coords[t], mask[t])
            # If stable sequence reference is available, use global scale for temporal consistency
            if stable_reference and self.method == "torso_scale":
                # Re-scale with global sequence scale to prevent breathing/jitter artifacts
                for i in range(93):
                    if mask[t, i]:
                        norm_frame[i, 0] = (coords[t, i, 0] - global_center[0]) / global_scale
                        norm_frame[i, 1] = (coords[t, i, 1] - global_center[1]) / global_scale
                        norm_frame[i, 2] = (coords[t, i, 2] - global_center[2]) / global_scale * self.z_scale_factor
                    else:
                        norm_frame[i] = 0.0
            norm_seq[t] = norm_frame
            frame_scales.append(f_meta.get("scale", 1.0))

        seq_meta = {
            "method": self.method,
            "stable_reference": stable_reference,
            "mean_scale": float(np.mean(frame_scales)) if frame_scales else 1.0,
            "global_center": global_center.tolist(),
            "global_scale": float(global_scale)
        }

        return norm_seq, seq_meta

    @staticmethod
    def mirror_horizontal(
        coords: np.ndarray,      # (T, 93, 3) or (93, 3)
        mask: np.ndarray         # (T, 93) or (93,)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs handedness-inverting horizontal reflection:
        1. Negates x coordinate: x' = -x
        2. Swaps Left Hand (0-20) and Right Hand (21-41)
        3. Swaps Left/Right Pose Anchors:
           - Left Eye 43 <-> Right Eye 44
           - Left Ear 45 <-> Right Ear 46
           - Left Shoulder 47 <-> Right Shoulder 48
           - Left Elbow 49 <-> Right Elbow 50
           - Left Wrist 51 <-> Right Wrist 52
        """
        is_seq = (coords.ndim == 3)
        if not is_seq:
            coords = coords[np.newaxis, ...]
            mask = mask[np.newaxis, ...]

        T, V, C = coords.shape
        mirrored_coords = coords.copy()
        mirrored_mask = mask.copy()

        # 1. Flip X axis
        mirrored_coords[:, :, 0] = -mirrored_coords[:, :, 0]

        # 2. Swap Hands (0-20 <-> 21-41)
        lh_c = mirrored_coords[:, 0:21].copy()
        rh_c = mirrored_coords[:, 21:42].copy()
        mirrored_coords[:, 0:21] = rh_c
        mirrored_coords[:, 21:42] = lh_c

        lh_m = mirrored_mask[:, 0:21].copy()
        rh_m = mirrored_mask[:, 21:42].copy()
        mirrored_mask[:, 0:21] = rh_m
        mirrored_mask[:, 21:42] = lh_m

        # 3. Swap Bilateral Pose Anchors
        swap_pairs = [
            (43, 44),  # Eyes
            (45, 46),  # Ears
            (47, 48),  # Shoulders
            (49, 50),  # Elbows
            (51, 52)   # Wrist pose anchors
        ]
        for p1, p2 in swap_pairs:
            tmp_c = mirrored_coords[:, p1].copy()
            mirrored_coords[:, p1] = mirrored_coords[:, p2]
            mirrored_coords[:, p2] = tmp_c

            tmp_m = mirrored_mask[:, p1].copy()
            mirrored_mask[:, p1] = mirrored_mask[:, p2]
            mirrored_mask[:, p2] = tmp_m

        if not is_seq:
            return mirrored_coords[0], mirrored_mask[0]
        return mirrored_coords, mirrored_mask
