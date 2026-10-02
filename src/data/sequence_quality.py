"""
SignTalk AI: Sequence Quality Scoring and Filtering Module.

Evaluates multi-factor sequence quality scores based on:
  - Pose tracking stability
  - Dominant and non-dominant hand presence
  - Coordinate continuity and physical plausibility
  - Total active articulation frames
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class SequenceQualityScorer:
    """
    Computes deterministic sequence-level quality metrics and classifies
    samples into GOOD, ACCEPTABLE, REVIEW, or REJECT categories.
    """

    def __init__(
        self,
        weight_pose: float = 0.40,
        weight_dominant_hand: float = 0.40,
        weight_nondominant_hand: float = 0.15,
        weight_face: float = 0.05,
        good_threshold: float = 0.75,
        acceptable_threshold: float = 0.50,
        review_threshold: float = 0.35,
        min_valid_frames: int = 8
    ):
        self.w_pose = weight_pose
        self.w_dom = weight_dominant_hand
        self.w_nondom = weight_nondominant_hand
        self.w_face = weight_face
        self.good_thresh = good_threshold
        self.acc_thresh = acceptable_threshold
        self.rev_thresh = review_threshold
        self.min_valid_frames = min_valid_frames

    def compute_quality(
        self,
        mask: np.ndarray,
        data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Computes the sequence quality score from the binary validity mask.
        
        mask: shape (1, T, V) or (T, V) where V = 93
              0..20: Left Hand
              21..41: Right Hand
              42..52: Upper Pose
              53..92: Facial Anchors
        """
        if mask.ndim == 3:
            m = mask[0]  # (T, V)
        else:
            m = mask
            
        T, V = m.shape
        
        # Modality detection rates (percentage of frames with active detection)
        # Left Hand (nodes 0..20)
        lh_active_per_frame = np.mean(m[:, 0:21], axis=-1) > 0.3  # at least 30% of hand nodes detected
        lh_rate = float(np.mean(lh_active_per_frame) * 100.0)
        
        # Right Hand (nodes 21..41)
        rh_active_per_frame = np.mean(m[:, 21:42], axis=-1) > 0.3
        rh_rate = float(np.mean(rh_active_per_frame) * 100.0)
        
        # Upper Pose (nodes 42..52)
        pose_active_per_frame = np.mean(m[:, 42:53], axis=-1) > 0.5
        pose_rate = float(np.mean(pose_active_per_frame) * 100.0)
        
        # Face (nodes 53..92)
        face_active_per_frame = np.mean(m[:, 53:93], axis=-1) > 0.3
        face_rate = float(np.mean(face_active_per_frame) * 100.0)
        
        # Determine dominant vs non-dominant hand for this sequence
        if rh_rate >= lh_rate:
            dom_rate = rh_rate
            nondom_rate = lh_rate
        else:
            dom_rate = lh_rate
            nondom_rate = rh_rate
            
        # Composite Quality Score [0.0, 1.0]
        score = (
            self.w_pose * (pose_rate / 100.0) +
            self.w_dom * (dom_rate / 100.0) +
            self.w_nondom * (nondom_rate / 100.0) +
            self.w_face * (face_rate / 100.0)
        )
        score = float(np.clip(score, 0.0, 1.0))
        
        # Count total active frames (frames where at least one hand is detected)
        any_hand_active = np.logical_or(lh_active_per_frame, rh_active_per_frame)
        valid_frames = int(np.sum(any_hand_active))
        
        # Classification Logic
        if valid_frames < self.min_valid_frames or score < self.rev_thresh:
            status = "REJECT"
            reason = f"Insufficient active hand frames ({valid_frames} < {self.min_valid_frames}) or low quality ({score:.2f} < {self.rev_thresh})"
        elif score >= self.good_thresh and pose_rate >= 90.0 and valid_frames >= 15:
            status = "GOOD"
            reason = "High landmark quality and continuous pose tracking"
        elif score >= self.acc_thresh and valid_frames >= 10:
            status = "ACCEPTABLE"
            reason = "Acceptable detection coverage for sequence modeling"
        else:
            status = "REVIEW"
            reason = f"Marginal landmark detection rates ({valid_frames} valid frames, quality {score:.2f})"
            
        return {
            "quality_score": round(score, 4),
            "quality_status": status,
            "classification_reason": reason,
            "valid_frames": valid_frames,
            "pose_detection_rate": round(pose_rate, 2),
            "left_hand_detection_rate": round(lh_rate, 2),
            "right_hand_detection_rate": round(rh_rate, 2),
            "face_detection_rate": round(face_rate, 2),
            "missing_landmark_ratio": round(1.0 - float(np.mean(m)), 4)
        }
