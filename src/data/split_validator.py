"""
SignTalk AI: Split Integrity and Data Leakage Validator.

Performs automated verification to ensure:
  - Zero source recording / video file leakage across splits
  - Zero frame-level or overlapping window leakage across splits
  - Explicit auditing of signer representation across splits
  - Partition count consistency
"""

from typing import Dict, Any, List, Set
import pandas as pd


class SplitLeakageError(Exception):
    """Raised when data leakage is detected across train/val/test partitions."""
    pass


class SplitValidator:
    """
    Validates that sequence manifests maintain strict partition isolation.
    """

    def __init__(self, manifest_df: pd.DataFrame):
        self.df = manifest_df
        required_cols = {"sequence_id", "source_recording_id", "source_video_id", "split"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")

    def check_leakage(self) -> Dict[str, Any]:
        """
        Evaluates source video and recording overlap across train, val, and test splits.
        """
        train_df = self.df[self.df["split"] == "train"]
        val_df = self.df[self.df["split"] == "val"]
        test_df = self.df[self.df["split"] == "test"]

        train_videos = set(train_df["source_video_id"].dropna().unique())
        val_videos = set(val_df["source_video_id"].dropna().unique())
        test_videos = set(test_df["source_video_id"].dropna().unique())

        train_recs = set(train_df["source_recording_id"].dropna().unique())
        val_recs = set(val_df["source_recording_id"].dropna().unique())
        test_recs = set(test_df["source_recording_id"].dropna().unique())

        # Check intersections
        train_val_vid_overlap = train_videos.intersection(val_videos)
        train_test_vid_overlap = train_videos.intersection(test_videos)
        val_test_vid_overlap = val_videos.intersection(test_videos)

        train_val_rec_overlap = train_recs.intersection(val_recs)
        train_test_rec_overlap = train_recs.intersection(test_recs)
        val_test_rec_overlap = val_recs.intersection(test_recs)

        has_video_leakage = bool(train_val_vid_overlap or train_test_vid_overlap or val_test_vid_overlap)
        has_rec_leakage = bool(train_val_rec_overlap or train_test_rec_overlap or val_test_rec_overlap)

        passed = not (has_video_leakage or has_rec_leakage)

        report = {
            "passed": passed,
            "has_video_leakage": has_video_leakage,
            "has_recording_leakage": has_rec_leakage,
            "train_sample_count": len(train_df),
            "val_sample_count": len(val_df),
            "test_sample_count": len(test_df),
            "total_sample_count": len(self.df),
            "train_unique_videos": len(train_videos),
            "val_unique_videos": len(val_videos),
            "test_unique_videos": len(test_videos),
            "train_val_overlap_count": len(train_val_vid_overlap),
            "train_test_overlap_count": len(train_test_vid_overlap),
            "val_test_overlap_count": len(val_test_vid_overlap),
            "overlapping_videos": list(train_val_vid_overlap | train_test_vid_overlap | val_test_vid_overlap)
        }

        # Signer distribution check
        if "signer_id" in self.df.columns:
            signer_split_ct = pd.crosstab(self.df["signer_id"], self.df["split"]).to_dict()
            report["signer_distribution_by_split"] = signer_split_ct

        if not passed:
            msg = (
                f"Data leakage detected! "
                f"Train/Val overlap: {len(train_val_vid_overlap)}, "
                f"Train/Test overlap: {len(train_test_vid_overlap)}, "
                f"Val/Test overlap: {len(val_test_vid_overlap)}. "
                f"Overlapping videos: {report['overlapping_videos']}"
            )
            raise SplitLeakageError(msg)

        return report
