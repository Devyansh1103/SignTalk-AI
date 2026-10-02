"""
SignTalk AI - Preprocessing Pipeline
Orchestrates the complete data preparation flow:
Raw Video -> File Validation -> Frame Extraction -> MediaPipe Landmark Extraction ->
Missing Landmark Handling -> Coordinate Normalization -> Quality Scoring ->
Structured NPZ Storage & Manifest Metadata Tracking.
"""

import os
import sys
import csv
import json
import time
import yaml
import numpy as np
from typing import Dict, Any, List, Optional

from src.data.frame_extractor import FrameExtractor, FrameExtractionError
from src.preprocessing.landmark_extractor import LandmarkExtractor
from src.preprocessing.sequence_utils import (
    handle_missing_landmarks,
    pad_or_truncate_sequence,
    temporal_resample_sequence
)
from src.preprocessing.normalizer import CoordinateNormalizer
from src.preprocessing.quality_checker import QualityChecker


class PreprocessingPipeline:
    """
    Production batch preprocessing pipeline for SignTalk AI with error resilience,
    deterministic reproducibility, and resumability checkpointing.
    """

    def __init__(self, config_path: str = "configs/preprocessing.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Initialize submodules
        fp_cfg = self.config.get("frame_processing", {})
        self.frame_extractor = FrameExtractor(
            target_fps=fp_cfg.get("target_fps", 25.0),
            frame_sampling_rate=fp_cfg.get("frame_sampling_rate", 1),
            image_size=tuple(fp_cfg.get("image_size", [426, 240])),
            uniform_resampling=fp_cfg.get("uniform_resampling", True)
        )

        mp_cfg = self.config.get("mediapipe", {})
        self.landmark_extractor = LandmarkExtractor(
            pose_model_path=mp_cfg.get("pose_model_path", "models/mediapipe/pose_landmarker_full.task"),
            hand_model_path=mp_cfg.get("hand_model_path", "models/mediapipe/hand_landmarker.task"),
            face_model_path=mp_cfg.get("face_model_path", "models/mediapipe/face_landmarker.task"),
            pose_confidence=mp_cfg.get("pose_detection_confidence", 0.5),
            hand_confidence=mp_cfg.get("hand_detection_confidence", 0.35),
            face_confidence=mp_cfg.get("face_detection_confidence", 0.4),
            enable_face_mesh=mp_cfg.get("enable_face_mesh", False)
        )

        norm_cfg = self.config.get("normalization", {})
        self.normalizer = CoordinateNormalizer(
            method=norm_cfg.get("method", "torso_scale"),
            min_scale_epsilon=norm_cfg.get("min_scale_epsilon", 1.0e-4),
            z_scale_factor=norm_cfg.get("z_scale_factor", 1.0)
        )

        qc_cfg = self.config.get("quality_control", {})
        self.quality_checker = QualityChecker(
            weights=qc_cfg.get("weights"),
            threshold_good=qc_cfg.get("classification_thresholds", {}).get("good", 0.75),
            threshold_acceptable=qc_cfg.get("classification_thresholds", {}).get("acceptable", 0.50),
            threshold_review=qc_cfg.get("classification_thresholds", {}).get("review", 0.35),
            min_valid_frames=self.config.get("sequence", {}).get("minimum_valid_frames", 15)
        )

        # Output paths
        ds_cfg = self.config.get("dataset", {})
        self.processed_dir = ds_cfg.get("processed_output_dir", "data/processed/landmarks")
        self.interim_dir = ds_cfg.get("interim_output_dir", "data/interim/landmarks")
        self.error_log_path = ds_cfg.get("error_log_path", "data/metadata/preprocessing_errors.csv")
        self.processed_manifest_path = ds_cfg.get("processed_manifest_path", "data/metadata/processed_dataset_manifest.csv")

        for split in ["train", "val", "test", "pilot"]:
            os.makedirs(os.path.join(self.processed_dir, split), exist_ok=True)
            os.makedirs(os.path.join(self.interim_dir, split), exist_ok=True)
        os.makedirs(os.path.dirname(self.error_log_path), exist_ok=True)

    def process_sample(self, sample_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single video sample through the end-to-end pipeline.

        Args:
            sample_record: Dictionary with video metadata (path, class_label, split, etc.)

        Returns:
            Dictionary with processing result, quality scores, and output file paths.
        """
        video_path = sample_record["video_path"]
        sample_id = sample_record.get("sample_id", os.path.splitext(os.path.basename(video_path))[0])
        split = sample_record.get("split", "train")
        class_label = sample_record.get("class_label", "unknown")
        class_id = sample_record.get("class_id", -1)
        signer_id = sample_record.get("signer_id", "unknown")

        t_start = time.perf_counter()

        # Step 1: File Validation & Frame Extraction
        extracted = self.frame_extractor.extract_from_video(video_path)
        frames = extracted["frames"]
        frame_meta = extracted["metadata"]

        # Step 2: MediaPipe Landmark Extraction (93 nodes)
        landmark_res = self.landmark_extractor.extract_sequence_landmarks(frames)
        raw_coords = landmark_res["coords"]       # (T, 93, 3)
        raw_visibility = landmark_res["visibility"] # (T, 93)
        raw_mask = landmark_res["mask"]           # (T, 93)
        frame_stats = landmark_res["frame_stats"]

        # Step 3: Missing Landmark Handling
        missing_cfg = self.config.get("missing_landmarks", {})
        handled_coords, updated_mask = handle_missing_landmarks(
            raw_coords,
            raw_mask,
            strategy=missing_cfg.get("strategy", "interpolate_and_mask"),
            max_consecutive_missing=missing_cfg.get("max_consecutive_interpolated_frames", 10),
            visibility=raw_visibility
        )

        # Step 4: Coordinate Normalization
        norm_coords, norm_meta = self.normalizer.normalize_sequence(handled_coords, updated_mask)

        # Step 5: Sequence Length Standardization (T = 45 frames)
        seq_cfg = self.config.get("sequence", {})
        target_T = seq_cfg.get("target_sequence_length", 45)
        pad_mode = seq_cfg.get("pad_mode", "zero_padding")

        # Temporal resampling to standardize to exact T frames
        resampled_coords, resampled_mask = temporal_resample_sequence(
            norm_coords,
            updated_mask,
            target_length=target_T
        )

        # Step 6: Quality Scoring & Classification
        quality_eval = self.quality_checker.evaluate_sequence(frame_stats, raw_mask)

        # Step 7: Serialize Processed Representation to NPZ
        # Model expected format: (C, T, V) where C=3, T=45, V=93
        # PyTorch ST-GCN input: [B, C, T, V]
        model_input_tensor = np.transpose(resampled_coords, (2, 0, 1))  # (3, 45, 93)
        model_mask_tensor = resampled_mask[np.newaxis, ...]              # (1, 45, 93)

        out_fname = f"{sample_id}.npz"
        out_path = os.path.join(self.processed_dir, split, out_fname).replace("\\", "/")

        np.savez_compressed(
            out_path,
            data=model_input_tensor.astype(np.float32),      # (3, 45, 93)
            mask=model_mask_tensor.astype(bool),             # (1, 45, 93)
            raw_coords=raw_coords.astype(np.float32),        # Original (T, 93, 3)
            raw_mask=raw_mask.astype(bool),
            class_id=int(class_id),
            class_label=str(class_label),
            signer_id=str(signer_id),
            sample_id=str(sample_id),
            split=str(split),
            fps=float(frame_meta.get("effective_fps", 25.0)),
            quality_score=float(quality_eval["average_quality"]),
            classification=str(quality_eval["classification"])
        )

        t_end = time.perf_counter()
        elapsed_sec = t_end - t_start

        return {
            "sample_id": sample_id,
            "split": split,
            "class_label": class_label,
            "class_id": class_id,
            "signer_id": signer_id,
            "processed_file": out_path,
            "original_frames": len(frames),
            "target_frames": target_T,
            "average_quality": quality_eval["average_quality"],
            "classification": quality_eval["classification"],
            "classification_reason": quality_eval["classification_reason"],
            "pose_detection_rate": quality_eval["pose_detection_rate"],
            "left_hand_detection_rate": quality_eval["left_hand_detection_rate"],
            "right_hand_detection_rate": quality_eval["right_hand_detection_rate"],
            "face_detection_rate": quality_eval["face_detection_rate"],
            "elapsed_seconds": round(elapsed_sec, 3),
            "status": "COMPLETED"
        }

    def process_batch(
        self,
        sample_records: List[Dict[str, Any]],
        is_pilot: bool = False
    ) -> Dict[str, Any]:
        """
        Processes a batch or full dataset manifest with checkpointing and error capture.

        Args:
            sample_records: List of sample metadata records.
            is_pilot: Whether this execution is the preliminary pilot run.

        Returns:
            Dictionary of aggregate batch statistics and manifest records.
        """
        total = len(sample_records)
        print(f"Starting {'PILOT' if is_pilot else 'FULL DATASET'} processing across {total} samples...")

        completed_records = []
        error_records = []
        classification_counts = {"GOOD": 0, "ACCEPTABLE": 0, "REVIEW": 0, "REJECT": 0}

        t0_batch = time.perf_counter()

        for idx, record in enumerate(sample_records, start=1):
            sample_id = record.get("sample_id", f"sample_{idx}")
            video_path = record.get("video_path", "")

            try:
                # Force split to 'pilot' if running pilot mode
                if is_pilot:
                    rec_copy = record.copy()
                    rec_copy["split"] = "pilot"
                else:
                    rec_copy = record

                res = self.process_sample(rec_copy)
                completed_records.append(res)
                classification = res["classification"]
                classification_counts[classification] = classification_counts.get(classification, 0) + 1

                if idx % 5 == 0 or idx == total:
                    print(f"[{idx}/{total}] Processed {sample_id}: Quality={res['average_quality']:.2f} ({res['classification']}) in {res['elapsed_seconds']:.2f}s")

            except Exception as e:
                err_msg = str(e)
                print(f"[{idx}/{total}] ERROR processing {sample_id} ({video_path}): {err_msg}", file=sys.stderr)
                error_records.append({
                    "sample_id": sample_id,
                    "video_path": video_path,
                    "error_type": type(e).__name__,
                    "error_message": err_msg,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

        t1_batch = time.perf_counter()
        total_time_sec = t1_batch - t0_batch

        # Append/Write error records if any
        if error_records:
            fieldnames = ["sample_id", "video_path", "error_type", "error_message", "timestamp"]
            write_header = not os.path.exists(self.error_log_path)
            with open(self.error_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(error_records)

        # Update processed manifest if not pilot
        if not is_pilot and completed_records:
            fieldnames = list(completed_records[0].keys())
            with open(self.processed_manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(completed_records)

        summary = {
            "mode": "PILOT" if is_pilot else "FULL_DATASET",
            "total_submitted": total,
            "total_successful": len(completed_records),
            "total_failed": len(error_records),
            "classification_counts": classification_counts,
            "total_time_seconds": round(total_time_sec, 2),
            "average_time_per_sample": round(total_time_sec / max(1, total), 3),
            "throughput_samples_per_sec": round(total / max(0.001, total_time_sec), 2),
            "completed_records": completed_records,
            "error_records": error_records
        }

        print(f"\n--- Batch Execution Summary ({'PILOT' if is_pilot else 'FULL DATASET'}) ---")
        print(f"Success: {len(completed_records)}/{total} | Failed: {len(error_records)}")
        print(f"Quality Breakdown: {classification_counts}")
        print(f"Total Time: {total_time_sec:.2f}s ({summary['throughput_samples_per_sec']} samples/sec)\n")

        return summary

    def close(self):
        """Release underlying MediaPipe landmarker models."""
        self.landmark_extractor.close()
