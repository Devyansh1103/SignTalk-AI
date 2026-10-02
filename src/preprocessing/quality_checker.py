"""
SignTalk AI - Landmark Quality Checker
Evaluates frame-level detection confidence and sequence-level completeness.
Classifies sequences into: GOOD, ACCEPTABLE, REVIEW, REJECT.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class QualityChecker:
    """
    Computes frame-level and sequence-level quality metrics for landmark sequences.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold_good: float = 0.75,
        threshold_acceptable: float = 0.50,
        threshold_review: float = 0.35,
        min_valid_frames: int = 15
    ):
        self.weights = weights or {
            "pose": 0.40,
            "dominant_hand": 0.40,
            "non_dominant_hand": 0.15,
            "face": 0.05
        }
        self.threshold_good = threshold_good
        self.threshold_acceptable = threshold_acceptable
        self.threshold_review = threshold_review
        self.min_valid_frames = min_valid_frames

    def compute_frame_quality(self, frame_stat: Dict[str, Any]) -> float:
        """
        Computes a scalar quality score [0.0, 1.0] for an individual frame.

        Args:
            frame_stat: Dictionary of detection flags and confidence scores.

        Returns:
            Float quality score between 0.0 and 1.0.
        """
        pose_score = frame_stat.get("pose_confidence", 1.0) if frame_stat.get("pose_detected", False) else 0.0
        lh_score = frame_stat.get("left_hand_confidence", 1.0) if frame_stat.get("left_hand_detected", False) else 0.0
        rh_score = frame_stat.get("right_hand_confidence", 1.0) if frame_stat.get("right_hand_detected", False) else 0.0
        face_score = 1.0 if frame_stat.get("face_detected", False) else 0.0

        # Designate dominant vs non-dominant for this frame
        dom_hand_score = max(lh_score, rh_score)
        non_dom_hand_score = min(lh_score, rh_score)

        score = (
            self.weights["pose"] * pose_score +
            self.weights["dominant_hand"] * dom_hand_score +
            self.weights["non_dominant_hand"] * non_dom_hand_score +
            self.weights["face"] * face_score
        )

        return float(np.clip(score, 0.0, 1.0))

    def evaluate_sequence(
        self,
        frame_stats: List[Dict[str, Any]],
        mask: np.ndarray  # (T, 93)
    ) -> Dict[str, Any]:
        """
        Evaluates sequence-level quality and assigns classification status.

        Args:
            frame_stats: List of per-frame detection dictionaries.
            mask: Boolean array (T, 93) indicating valid node detections.

        Returns:
            Dictionary containing sequence metrics and quality classification.
        """
        T = len(frame_stats)
        if T == 0:
            return {
                "total_frames": 0,
                "valid_frames": 0,
                "average_quality": 0.0,
                "classification": "REJECT",
                "rejection_reason": "Zero frames in sequence"
            }

        frame_scores = [self.compute_frame_quality(s) for s in frame_stats]
        avg_quality = float(np.mean(frame_scores))

        pose_detected_count = sum(1 for s in frame_stats if s.get("pose_detected", False))
        lh_detected_count = sum(1 for s in frame_stats if s.get("left_hand_detected", False))
        rh_detected_count = sum(1 for s in frame_stats if s.get("right_hand_detected", False))
        either_hand_count = sum(1 for s in frame_stats if s.get("left_hand_detected", False) or s.get("right_hand_detected", False))
        face_count = sum(1 for s in frame_stats if s.get("face_detected", False))

        # A frame is valid if pose is present and at least one hand or face was detected
        valid_frames_count = sum(1 for s in frame_stats if s.get("pose_detected", False) and (s.get("left_hand_detected", False) or s.get("right_hand_detected", False)))

        pct_missing_pose = round((1.0 - (pose_detected_count / T)) * 100.0, 2)
        pct_missing_both_hands = round((1.0 - (either_hand_count / T)) * 100.0, 2)
        pct_missing_lh = round((1.0 - (lh_count / T if (lh_count := lh_detected_count) else 0.0)) * 100.0, 2)
        pct_missing_rh = round((1.0 - (rh_count / T if (rh_count := rh_detected_count) else 0.0)) * 100.0, 2)
        pct_missing_face = round((1.0 - (face_count / T)) * 100.0, 2)

        # Classification decision logic
        if valid_frames_count < 8:
            classification = "REJECT"
            reason = f"Insufficient valid frames ({valid_frames_count} < 8)"
        elif avg_quality >= self.threshold_good and pct_missing_pose <= 10.0 and valid_frames_count >= self.min_valid_frames:
            classification = "GOOD"
            reason = "High landmark quality and continuous pose tracking"
        elif avg_quality >= self.threshold_acceptable and valid_frames_count >= 10:
            classification = "ACCEPTABLE"
            reason = "Acceptable detection coverage for sequence modeling"
        elif avg_quality >= self.threshold_review or valid_frames_count >= 8:
            classification = "REVIEW"
            reason = f"Marginal landmark detection rates ({valid_frames_count} valid frames, quality {avg_quality:.2f}); recommended for manual inspection"
        else:
            classification = "REJECT"
            reason = f"Average quality ({avg_quality:.2f}) below minimum review threshold ({self.threshold_review})"

        return {
            "total_frames": T,
            "valid_frames": valid_frames_count,
            "average_quality": round(avg_quality, 4),
            "min_frame_quality": round(float(np.min(frame_scores)), 4) if frame_scores else 0.0,
            "max_frame_quality": round(float(np.max(frame_scores)), 4) if frame_scores else 0.0,
            "pose_detection_rate": round(pose_detected_count / T * 100.0, 2),
            "either_hand_detection_rate": round(either_hand_count / T * 100.0, 2),
            "left_hand_detection_rate": round(lh_detected_count / T * 100.0, 2),
            "right_hand_detection_rate": round(rh_detected_count / T * 100.0, 2),
            "face_detection_rate": round(face_count / T * 100.0, 2),
            "pct_missing_pose": pct_missing_pose,
            "pct_missing_both_hands": pct_missing_both_hands,
            "pct_missing_left_hand": pct_missing_lh,
            "pct_missing_right_hand": pct_missing_rh,
            "pct_missing_face": pct_missing_face,
            "classification": classification,
            "classification_reason": reason,
            "frame_qualities": [round(q, 3) for q in frame_scores]
        }
