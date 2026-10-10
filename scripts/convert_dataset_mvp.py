#!/usr/bin/env python3
"""
SignTalk AI: Convert Dataset MVP Videos to Trainable Format.

Converts videos from 'dataset/INCLUDE' corresponding to MVP classes
(happy, teacher - 35 videos total) into the canonical trainable format:
1. MediaPipe 93-node multimodal landmark extraction (normalized coordinates)
   -> data/processed/landmarks/{split}/{sample_id}.npz
2. Canonical standardized sequence dataset generation (T=45 frames)
   -> data/processed/sequences/{split}/{seq_id}.npz
3. Manifest registration and synchronization:
   -> data/manifests/sequence_manifest.csv, train.csv, val.csv, test.csv
   -> data/manifests/include_mvp_manifest.csv
   -> data/metadata/include_mvp_processed_manifest.csv
"""

import os
import sys
import time
import shutil
import cv2
import json
import numpy as np
import pandas as pd

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.data.temporal_validator import TemporalValidator
from src.data.sequence_quality import SequenceQualityScorer
from src.data.stgcn_tensor import NUM_NODES, SEQUENCE_LENGTH, CHANNELS


def get_mvp_video_records():
    include_root = os.path.join("dataset", "INCLUDE")

    def strip_ext(p):
        return os.path.splitext(p.replace("/", os.sep))[0].lower()

    with open("data/metadata/include/include_train.txt", "r", encoding="utf-8") as f:
        train_lines = set(strip_ext(line.strip()) for line in f if line.strip())
    with open("data/metadata/include/include_val.txt", "r", encoding="utf-8") as f:
        val_lines = set(strip_ext(line.strip()) for line in f if line.strip())
    with open("data/metadata/include/include_test.txt", "r", encoding="utf-8") as f:
        test_lines = set(strip_ext(line.strip()) for line in f if line.strip())

    targets = [
        ("happy", 3, "HAPPY", "Happy", os.path.join(include_root, "Adjectives", "3. happy")),
        ("teacher", 9, "TEACHER", "Teacher", os.path.join(include_root, "Jobs", "84. Teacher"))
    ]

    records = []
    idx = 1
    for cls_label, cls_id, gloss, trans, folder_path in targets:
        if not os.path.exists(folder_path):
            print(f"[WARN] Target folder not found: {folder_path}")
            continue

        files = sorted(os.listdir(folder_path))
        for f in files:
            if not f.lower().endswith((".mp4", ".mov")):
                continue

            full_path = os.path.join(folder_path, f)
            rel_path = os.path.relpath(full_path, include_root)
            key = strip_ext(rel_path)

            if key in val_lines:
                split = "val"
            elif key in test_lines:
                split = "test"
            else:
                split = "train"

            # Parse signer from filename, e.g., MVI_5183.MOV -> signer_51
            base_no_ext = os.path.splitext(f)[0]
            parts = base_no_ext.split("_")
            signer_id = "signer_unknown"
            if len(parts) >= 2 and parts[-1].isdigit():
                num_part = parts[-1]
                signer_id = f"signer_{num_part[:2]}"

            sample_id = f"raw_{60 + idx:04d}"
            seq_id = f"seq_{60 + idx:04d}"

            records.append({
                "sample_id": sample_id,
                "sequence_id": seq_id,
                "video_file": f,
                "video_path": full_path,
                "class_label": cls_label,
                "class_id": cls_id,
                "gloss": gloss,
                "translation": trans,
                "signer_id": signer_id,
                "split": split,
                "source_dataset": "INCLUDE"
            })
            idx += 1

    return records


def main():
    print("=" * 65)
    print("SignTalk AI: Dataset Video Conversion to Trainable Format")
    print("Target Classes: 'happy' (id 3) and 'teacher' (id 9)")
    print("=" * 65)

    records = get_mvp_video_records()
    total = len(records)
    print(f"Total candidate videos found: {total}")
    if total == 0:
        print("[ERROR] No candidate videos found. Exiting.")
        return

    # Initialize components
    pipeline = PreprocessingPipeline(config_path="configs/preprocessing.yaml")
    temporal_validator = TemporalValidator(target_fps=25.0)
    quality_scorer = SequenceQualityScorer(
        good_threshold=0.75,
        acceptable_threshold=0.50,
        review_threshold=0.35,
        min_valid_frames=8
    )

    landmarks_base = "data/processed/landmarks"
    sequences_base = "data/processed/sequences"
    for s in ["train", "val", "test"]:
        os.makedirs(os.path.join(landmarks_base, s), exist_ok=True)
        os.makedirs(os.path.join(sequences_base, s), exist_ok=True)

    completed_preprocessing = []
    completed_sequences = []

    t_start_all = time.time()

    for i, rec in enumerate(records, start=1):
        sample_id = rec["sample_id"]
        seq_id = rec["sequence_id"]
        vpath = rec["video_path"]
        split = rec["split"]
        cls_label = rec["class_label"]
        cls_id = rec["class_id"]
        signer_id = rec["signer_id"]

        print(f"\n[{i}/{total}] Processing {sample_id} ({cls_label}, {split}): {os.path.basename(vpath)} ...")
        t0 = time.time()

        try:
            # 1. Landmark Preprocessing
            prep_res = pipeline.process_sample(rec)
            completed_preprocessing.append(prep_res)
            lnd_file = prep_res["processed_file"]

            # 2. Sequence Construction
            with np.load(lnd_file) as arc:
                data = arc["data"]          # (3, 45, 93) float32
                mask = arc["mask"]          # (1, 45, 93) bool
                raw_coords = arc["raw_coords"] if "raw_coords" in arc else np.zeros((45, 93, 3), dtype=np.float32)
                fps = float(arc["fps"]) if "fps" in arc else 25.0

            q_info = quality_scorer.compute_quality(mask, data)
            token_id = cls_id + 4

            # Inspect original video duration and frame count
            cap = cv2.VideoCapture(vpath)
            orig_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else SEQUENCE_LENGTH
            orig_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 25.0
            cap.release()
            orig_duration = round(orig_frames / (orig_fps if orig_fps > 0 else 25.0), 3)

            seq_out_dir = os.path.join(sequences_base, split)
            seq_out_path = os.path.join(seq_out_dir, f"{seq_id}.npz").replace("\\", "/")

            # Save canonical sequence archive
            np.savez_compressed(
                seq_out_path,
                data=data.astype(np.float32),
                mask=mask.astype(np.float32),
                raw_coords=raw_coords.astype(np.float32),
                label=np.int64(cls_id),
                gloss_id=np.int64(token_id),
                sample_id=str(sample_id),
                sequence_id=str(seq_id),
                signer_id=str(signer_id),
                split=str(split)
            )

            seq_meta = {
                "sequence_id": seq_id,
                "source_dataset": "INCLUDE",
                "source_recording_id": sample_id,
                "source_video_id": os.path.basename(vpath),
                "signer_id": signer_id,
                "split": split,
                "class_id": cls_id,
                "label": cls_label,
                "gloss": rec["gloss"],
                "translation": rec["translation"],
                "sequence_start_frame": 0,
                "sequence_end_frame": max(0, orig_frames - 1),
                "sequence_start_time": 0.0,
                "sequence_end_time": orig_duration,
                "frame_count": orig_frames,
                "target_frames": SEQUENCE_LENGTH,
                "fps": 25.0,
                "duration_seconds": orig_duration,
                "landmark_schema_version": "93-node-v1",
                "preprocessing_version": "landmarks-v1.0",
                "dataset_version": "signTalk-seq-v1.0.0",
                "quality_score": q_info["quality_score"],
                "quality_status": q_info["quality_status"],
                "classification_reason": q_info["classification_reason"],
                "valid_frames": q_info["valid_frames"],
                "pose_detection_rate": q_info["pose_detection_rate"],
                "left_hand_detection_rate": q_info["left_hand_detection_rate"],
                "right_hand_detection_rate": q_info["right_hand_detection_rate"],
                "face_detection_rate": q_info["face_detection_rate"],
                "missing_landmark_ratio": q_info["missing_landmark_ratio"],
                "sequence_generation_method": "full_recording",
                "window_size": SEQUENCE_LENGTH,
                "stride": SEQUENCE_LENGTH,
                "padding_length": 0,
                "mask_available": True,
                "augmentation_status": "none",
                "file_path": seq_out_path
            }
            completed_sequences.append(seq_meta)

            t_elapsed = time.time() - t0
            print(f"  -> Finished {sample_id} -> {seq_id}.npz in {t_elapsed:.2f}s | Quality={q_info['quality_score']:.3f} ({q_info['quality_status']})")

        except Exception as e:
            print(f"  [ERROR] Failed to process {sample_id}: {e}", file=sys.stderr)

    pipeline.close()
    t_total = time.time() - t_start_all

    print("\n" + "=" * 65)
    print("CONVERSION COMPLETE")
    print(f"Successfully processed: {len(completed_sequences)} / {total} videos")
    print(f"Total time elapsed: {t_total:.2f} seconds ({t_total/60:.2f} mins)")
    print("=" * 65)

    # 3. Save Manifests
    new_seq_df = pd.DataFrame(completed_sequences)
    mvp_manifest_path = "data/manifests/include_mvp_manifest.csv"
    new_seq_df.to_csv(mvp_manifest_path, index=False)
    print(f"\nSaved new batch manifest: {mvp_manifest_path}")

    # Processed log
    prep_df = pd.DataFrame(completed_preprocessing)
    prep_log_path = "data/metadata/include_mvp_processed_manifest.csv"
    prep_df.to_csv(prep_log_path, index=False)
    print(f"Saved preprocessing log: {prep_log_path}")

    # Synchronize with master manifests
    master_manifest_path = "data/manifests/sequence_manifest.csv"
    if os.path.exists(master_manifest_path):
        backup_path = "data/manifests/sequence_manifest_backup.csv"
        if not os.path.exists(backup_path):
            shutil.copy2(master_manifest_path, backup_path)
        existing_df = pd.read_csv(master_manifest_path)
        combined_df = pd.concat([existing_df, new_seq_df], ignore_index=True)
        # Drop duplicates if re-run
        combined_df = combined_df.drop_duplicates(subset=["sequence_id"], keep="last")
        combined_df.to_csv(master_manifest_path, index=False)
        print(f"Updated master manifest: {master_manifest_path} (Total rows: {len(combined_df)})")

        # Update train/val/test split manifests
        for split_name in ["train", "val", "test"]:
            split_manifest_path = f"data/manifests/{split_name}.csv"
            split_df = combined_df[combined_df["split"] == split_name].reset_index(drop=True)
            split_df.to_csv(split_manifest_path, index=False)
            print(f"Updated {split_name} split manifest: {split_manifest_path} ({len(split_df)} sequences)")


if __name__ == "__main__":
    main()
