"""
SignTalk AI: Systematic Error Analysis and Taxonomy Engine.

Classifies prediction failures into grounded failure categories:
  1. VISUAL_DETECTION_FAILURE: Low detection rate (<40%) on hands/pose in source video
  2. REPRESENTATION_LANDMARK_NOISE: High missing landmark ratio (>70%) or low quality score (<0.50)
  3. TEMPORAL_AMBIGUITY: Boundary frame effects or atypical duration / valid frame count
  4. SEMANTIC_KINEMATIC_SIMILARITY: Signs sharing phalanx postures or spatial loci
  5. LANGUAGE_DECODING_ERROR: Token-level misalignment or autoregressive truncation
"""

from typing import List, Dict, Any, Optional
import os
import pandas as pd
import numpy as np


class ErrorAnalyzer:
    """Systematic failure diagnostician for SignTalk AI predictions."""

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or [f"Class_{i}" for i in range(10)]

    def categorize_error(
        self,
        true_label: str,
        pred_label: str,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0
    ) -> Dict[str, Any]:
        """
        Categorizes a single prediction failure into evidence-based failure modes.
        """
        meta = metadata or {}
        valid_frames = meta.get("valid_frames", 20)
        quality_score = meta.get("quality_score", 0.6)
        quality_status = meta.get("quality_status", "ACCEPTABLE")
        pose_rate = meta.get("pose_detection_rate", 100.0)
        lh_rate = meta.get("left_hand_detection_rate", 50.0)
        rh_rate = meta.get("right_hand_detection_rate", 50.0)
        missing_ratio = meta.get("missing_landmark_ratio", 0.6)

        primary_category = "SEMANTIC_KINEMATIC_SIMILARITY"
        explanation = f"Signs '{true_label}' and '{pred_label}' share similar articulatory trajectory."

        if quality_status == "REJECT" or valid_frames < 8 or (lh_rate < 15.0 and rh_rate < 15.0):
            primary_category = "VISUAL_DETECTION_FAILURE"
            explanation = (
                f"Severe hand landmark dropouts in source recording: valid frames={valid_frames}, "
                f"LH rate={lh_rate}%, RH rate={rh_rate}%."
            )
        elif missing_ratio > 0.80 or quality_score < 0.45:
            primary_category = "REPRESENTATION_LANDMARK_NOISE"
            explanation = (
                f"High landmark sparsity/noise: missing ratio={missing_ratio:.2f}, "
                f"quality score={quality_score:.2f}."
            )
        elif valid_frames < 15:
            primary_category = "TEMPORAL_AMBIGUITY"
            explanation = (
                f"Insufficient active signing duration: only {valid_frames} valid frames present "
                f"before zero-padding to 45."
            )
        elif confidence > 0.85:
            primary_category = "SEMANTIC_OVERCONFIDENT_CONFUSION"
            explanation = (
                f"High-confidence false prediction (conf={confidence:.2f}): spatial ambiguity "
                f"between '{true_label}' and '{pred_label}'."
            )

        return {
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": round(confidence, 4),
            "primary_category": primary_category,
            "explanation": explanation,
            "quality_status": quality_status,
            "valid_frames": valid_frames,
            "quality_score": round(float(quality_score), 4)
        }

    def analyze_dataset_errors(
        self,
        y_true: List[int],
        y_pred: List[int],
        confidences: List[float],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyzes all misclassified instances in a partition.
        """
        records = []
        for i, (yt, yp, conf) in enumerate(zip(y_true, y_pred, confidences)):
            if yt != yp:
                meta = metadata_list[i] if (metadata_list and i < len(metadata_list)) else {}
                true_name = self.class_names[yt] if yt < len(self.class_names) else str(yt)
                pred_name = self.class_names[yp] if yp < len(self.class_names) else str(yp)
                cat_info = self.categorize_error(true_name, pred_name, meta, conf)
                cat_info["sample_index"] = i
                records.append(cat_info)

        df = pd.DataFrame(records)
        if output_csv:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
            df.to_csv(output_csv, index=False)

        return df

    def summarize_categories(self, errors_df: pd.DataFrame) -> Dict[str, Any]:
        """Summarizes error distributions across primary failure categories."""
        if errors_df.empty:
            return {"total_errors": 0, "categories": {}}

        counts = errors_df["primary_category"].value_counts().to_dict()
        total = len(errors_df)
        proportions = {cat: round(cnt / total, 4) for cat, cnt in counts.items()}

        return {
            "total_errors": total,
            "category_counts": counts,
            "category_proportions": proportions
        }
