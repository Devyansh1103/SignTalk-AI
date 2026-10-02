"""
SignTalk AI: Canonical Sequence Builder Pipeline.

Builds standardized sequence representations from preprocessed landmark recordings:
  - Preserves temporal order, boundary timestamps, and frame continuity
  - Standardizes temporal duration to T=45 frames (1.80s @ 25 FPS)
  - Maps classes to canonical labels, glosses, and translation targets
  - Computes multi-factor sequence quality scores
  - Generates sequence manifests and rejected sequence logs
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import json
import glob
import numpy as np
import pandas as pd
import yaml

from src.data.temporal_validator import TemporalValidator
from src.data.sequence_quality import SequenceQualityScorer
from src.data.stgcn_tensor import NUM_NODES, SEQUENCE_LENGTH, CHANNELS


class SequenceBuilder:
    """
    Constructs, validates, and serializes canonical sequence archives and manifests.
    """

    def __init__(self, config_path: str = "configs/sequence_generation.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.paths = self.config["paths"]
        self.seq_params = self.config["sequence_parameters"]
        self.quality_thresh = self.config["quality_thresholds"]
        self.dataset_meta = self.config["dataset_metadata"]

        self.temporal_validator = TemporalValidator(target_fps=self.seq_params["target_fps"])
        self.quality_scorer = SequenceQualityScorer(
            good_threshold=self.quality_thresh["good_min_quality"],
            acceptable_threshold=self.quality_thresh["acceptable_min_quality"],
            review_threshold=self.quality_thresh["review_min_quality"],
            min_valid_frames=self.seq_params["min_valid_frames"]
        )

        # Load label mapping and vocabulary
        self.label_mapping = self._load_label_mapping(self.paths["label_mapping_path"])
        self.vocabulary = self._load_vocabulary(self.paths["vocabulary_path"])

    def _load_label_mapping(self, path: str) -> Dict[int, Dict[str, Any]]:
        df = pd.read_csv(path)
        mapping = {}
        for _, row in df.iterrows():
            cid = int(row["canonical_class_id"])
            mapping[cid] = {
                "canonical_label": str(row["canonical_label"]),
                "gloss": str(row["gloss"]),
                "translation": str(row["translation"]),
                "source_dataset": str(row["source_dataset"]),
                "source_label": str(row["source_label"])
            }
        return mapping

    def _load_vocabulary(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def process_sample(
        self,
        landmark_filepath: str,
        sequence_index: int,
        raw_manifest_row: Optional[pd.Series] = None
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """
        Processes a single preprocessed landmark file into a canonical sequence record.
        """
        seq_id = f"seq_{sequence_index:04d}"
        
        with np.load(landmark_filepath) as arc:
            data = arc["data"]  # (3, 45, 93)
            mask = arc["mask"]  # (1, 45, 93)
            raw_coords = arc["raw_coords"] if "raw_coords" in arc else np.zeros((45, 93, 3), dtype=np.float32)
            
            sample_id = str(arc["sample_id"]) if "sample_id" in arc else os.path.splitext(os.path.basename(landmark_filepath))[0]
            class_id = int(arc["class_id"]) if "class_id" in arc else 0
            signer_id = str(arc["signer_id"]) if "signer_id" in arc else "unknown"
            split = str(arc["split"]) if "split" in arc else "train"
            fps = float(arc["fps"]) if "fps" in arc else 25.0

        # Run temporal validation
        temp_val = self.temporal_validator.validate_recording(landmark_filepath, sample_id)
        
        # Run quality scoring
        quality_info = self.quality_scorer.compute_quality(mask, data)
        
        # Label & Vocabulary lookup
        lbl_info = self.label_mapping.get(class_id, {
            "canonical_label": f"class_{class_id}",
            "gloss": f"GLOSS_{class_id}",
            "translation": f"Class {class_id}",
            "source_dataset": "INCLUDE-50",
            "source_label": f"class_{class_id}"
        })
        
        token_id = class_id + 4  # special tokens offset: 0:PAD, 1:UNK, 2:BOS, 3:EOS

        # Raw video metadata lookup if available
        if raw_manifest_row is not None:
            source_video_id = str(raw_manifest_row["video_file"])
            orig_frames = int(raw_manifest_row["frame_count"])
            orig_duration = float(raw_manifest_row["duration_sec"])
        else:
            source_video_id = f"{sample_id}.mp4"
            orig_frames = SEQUENCE_LENGTH
            orig_duration = round(SEQUENCE_LENGTH / fps, 3)

        metadata = {
            "sequence_id": seq_id,
            "source_dataset": lbl_info["source_dataset"],
            "source_recording_id": sample_id,
            "source_video_id": source_video_id,
            "signer_id": signer_id,
            "split": split,
            "class_id": class_id,
            "label": lbl_info["canonical_label"],
            "gloss": lbl_info["gloss"],
            "translation": lbl_info["translation"],
            "sequence_start_frame": 0,
            "sequence_end_frame": orig_frames - 1,
            "sequence_start_time": 0.0,
            "sequence_end_time": orig_duration,
            "frame_count": orig_frames,
            "target_frames": SEQUENCE_LENGTH,
            "fps": fps,
            "duration_seconds": orig_duration,
            "landmark_schema_version": self.dataset_meta["landmark_schema"],
            "preprocessing_version": self.dataset_meta["preprocessing_version"],
            "dataset_version": self.dataset_meta["dataset_version"],
            "quality_score": quality_info["quality_score"],
            "quality_status": quality_info["quality_status"],
            "classification_reason": quality_info["classification_reason"],
            "valid_frames": quality_info["valid_frames"],
            "pose_detection_rate": quality_info["pose_detection_rate"],
            "left_hand_detection_rate": quality_info["left_hand_detection_rate"],
            "right_hand_detection_rate": quality_info["right_hand_detection_rate"],
            "face_detection_rate": quality_info["face_detection_rate"],
            "missing_landmark_ratio": quality_info["missing_landmark_ratio"],
            "sequence_generation_method": self.seq_params["generation_mode"],
            "window_size": SEQUENCE_LENGTH,
            "stride": SEQUENCE_LENGTH,
            "padding_length": 0,
            "mask_available": True,
            "augmentation_status": "none",
            "file_path": f"data/processed/sequences/{split}/{seq_id}.npz"
        }

        arrays = {
            "data": data.astype(np.float32),
            "mask": mask.astype(np.float32),
            "raw_coords": raw_coords.astype(np.float32),
            "label": np.int64(class_id),
            "gloss_id": np.int64(token_id),
            "sample_id": sample_id,
            "sequence_id": seq_id,
            "signer_id": signer_id,
            "split": split
        }

        return metadata, arrays

    def build_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Builds the entire sequence dataset across train, val, and test splits.
        """
        raw_manifest_path = self.paths["raw_manifest"]
        raw_manifest_df = pd.read_csv(raw_manifest_path) if os.path.exists(raw_manifest_path) else None

        input_dir = self.paths["input_landmarks_dir"]
        output_dir = self.paths["output_sequences_dir"]
        manifest_dir = self.paths["output_manifest_dir"]

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(manifest_dir, exist_ok=True)

        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(output_dir, split), exist_ok=True)

        # Collect all preprocessed landmark files across official partitions
        files = []
        for s in ["train", "val", "test"]:
            s_dir = os.path.join(input_dir, s)
            if os.path.exists(s_dir):
                files.extend(sorted(glob.glob(os.path.join(s_dir, "raw_*.npz"))))

        if not files:
            raise FileNotFoundError(f"No preprocessed landmark files found in {input_dir}")

        all_metadata: List[Dict[str, Any]] = []
        rejected_records: List[Dict[str, Any]] = []

        print(f"[INFO] Building canonical sequence dataset from {len(files)} landmark files...")

        for idx, fpath in enumerate(files, start=1):
            sample_id = os.path.splitext(os.path.basename(fpath))[0]
            raw_row = None
            if raw_manifest_df is not None:
                matches = raw_manifest_df[raw_manifest_df["sample_id"] == sample_id]
                if not matches.empty:
                    raw_row = matches.iloc[0]

            meta, arrays = self.process_sample(fpath, idx, raw_row)
            all_metadata.append(meta)

            if meta["quality_status"] == "REJECT":
                rejected_records.append({
                    "sequence_id": meta["sequence_id"],
                    "source_recording_id": meta["source_recording_id"],
                    "source_video_id": meta["source_video_id"],
                    "signer_id": meta["signer_id"],
                    "split": meta["split"],
                    "class_id": meta["class_id"],
                    "quality_score": meta["quality_score"],
                    "reason": meta["classification_reason"],
                    "valid_frames": meta["valid_frames"],
                    "pipeline_version": meta["dataset_version"]
                })

            # Save serialized sequence archive
            out_file = os.path.join(output_dir, meta["split"], f"{meta['sequence_id']}.npz")
            np.savez_compressed(out_file, **arrays)

        # Compile Master Manifests
        master_df = pd.DataFrame(all_metadata)
        master_manifest_path = os.path.join(manifest_dir, "sequence_manifest.csv")
        master_df.to_csv(master_manifest_path, index=False)
        print(f"[SUCCESS] Saved master sequence manifest: {master_manifest_path} ({len(master_df)} rows)")

        # Compile Partition-Specific Manifests
        for split in ["train", "val", "test"]:
            split_df = master_df[master_df["split"] == split]
            split_path = os.path.join(manifest_dir, f"{split}.csv")
            split_df.to_csv(split_path, index=False)
            print(f"[SUCCESS] Saved {split} manifest: {split_path} ({len(split_df)} rows)")

        # Compile Rejected Manifest
        rej_df = pd.DataFrame(rejected_records)
        rej_path = os.path.join(manifest_dir, "rejected_sequences.csv")
        rej_df.to_csv(rej_path, index=False)
        print(f"[SUCCESS] Saved rejected sequences manifest: {rej_path} ({len(rej_df)} rows)")

        return master_df, rej_df
