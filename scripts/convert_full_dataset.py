#!/usr/bin/env python3
"""
SignTalk AI: Full Dataset Conversion Pipeline.

Converts all 700 videos from 'dataset/INCLUDE' into the canonical trainable format:
1. MediaPipe 93-node multimodal landmark extraction (normalized coordinates)
   -> data/processed/landmarks/{split}/raw_inc_{idx:04d}.npz
2. Canonical standardized sequence dataset generation (T=45 frames)
   -> data/processed/sequences/{split}/seq_inc_{idx:04d}.npz
3. Manifest registration and synchronization:
   -> data/manifests/sequence_manifest.csv, train.csv, val.csv, test.csv
   -> data/manifests/include_full_sequence_manifest.csv
   -> data/metadata/include_full_processed_manifest.csv

Fully resumable: automatically skips already processed files.
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


def get_all_dataset_records():
    include_root = os.path.join("dataset", "INCLUDE")

    with open("data/metadata/include/label_map_include.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)

    def strip_ext(p):
        return os.path.splitext(p.replace("/", os.sep))[0].lower()

    with open("data/metadata/include/include_train.txt", "r", encoding="utf-8") as f:
        train_lines = set(strip_ext(line.strip()) for line in f if line.strip())
    with open("data/metadata/include/include_val.txt", "r", encoding="utf-8") as f:
        val_lines = set(strip_ext(line.strip()) for line in f if line.strip())
    with open("data/metadata/include/include_test.txt", "r", encoding="utf-8") as f:
        test_lines = set(strip_ext(line.strip()) for line in f if line.strip())

    records = []
    idx = 1

    # Deterministic directory traversal
    for root, dirs, files in os.walk(include_root):
        dirs.sort()
        for f in sorted(files):
            if not f.lower().endswith((".mp4", ".mov")):
                continue

            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, include_root)
            key = strip_ext(rel_path)

            parent = os.path.basename(root)
            if parent.lower() == "extra":
                parent = os.path.basename(os.path.dirname(root))

            clean_label = parent.split(".", 1)[-1].strip().lower().replace(" ", "")
            cls_id = label_map.get(clean_label, -1)
            gloss = clean_label.upper()
            trans = clean_label.capitalize()

            if key in val_lines:
                split = "val"
            elif key in test_lines:
                split = "test"
            else:
                split = "train"

            base_no_ext = os.path.splitext(f)[0]
            parts = base_no_ext.split("_")
            signer_id = "signer_unknown"
            if len(parts) >= 2 and parts[-1].isdigit():
                num_part = parts[-1]
                signer_id = f"signer_{num_part[:2]}"

            sample_id = f"raw_inc_{idx:04d}"
            seq_id = f"seq_inc_{idx:04d}"

            records.append({
                "sample_id": sample_id,
                "sequence_id": seq_id,
                "video_file": f,
                "video_path": full_path,
                "class_label": clean_label,
                "class_id": cls_id,
                "gloss": gloss,
                "translation": trans,
                "signer_id": signer_id,
                "split": split,
                "source_dataset": "INCLUDE"
            })
            idx += 1

    return records


def save_manifests(completed_sequences, completed_preprocessing):
    # Save standalone manifests
    if completed_sequences:
        new_seq_df = pd.DataFrame(completed_sequences)
        full_manifest_path = "data/manifests/include_full_sequence_manifest.csv"
        new_seq_df.to_csv(full_manifest_path, index=False)

        # Synchronize with master sequence_manifest.csv
        master_manifest_path = "data/manifests/sequence_manifest.csv"
        if os.path.exists(master_manifest_path):
            existing_df = pd.read_csv(master_manifest_path)
            combined_df = pd.concat([existing_df, new_seq_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["sequence_id"], keep="last")
            combined_df.to_csv(master_manifest_path, index=False)

            # Update split manifests
            for split_name in ["train", "val", "test"]:
                split_manifest_path = f"data/manifests/{split_name}.csv"
                split_df = combined_df[combined_df["split"] == split_name].reset_index(drop=True)
                split_df.to_csv(split_manifest_path, index=False)

    if completed_preprocessing:
        prep_df = pd.DataFrame(completed_preprocessing)
        prep_log_path = "data/metadata/include_full_processed_manifest.csv"
        prep_df.to_csv(prep_log_path, index=False)


def main():
    print("=" * 70, flush=True)
    print("SignTalk AI: Full Dataset Conversion Pipeline (700 Videos)", flush=True)
    print("=" * 70, flush=True)

    records = get_all_dataset_records()
    total = len(records)
    print(f"Total videos indexed from dataset/INCLUDE: {total}", flush=True)
    if total == 0:
        print("[ERROR] No candidate videos found. Exiting.", flush=True)
        return

    pipeline = PreprocessingPipeline(config_path="configs/preprocessing.yaml")
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

    # Check for existing records in full manifest to resume
    full_manifest_path = "data/manifests/include_full_sequence_manifest.csv"
    existing_seq_ids = set()
    if os.path.exists(full_manifest_path):
        try:
            prev_df = pd.read_csv(full_manifest_path)
            completed_sequences = prev_df.to_dict("records")
            existing_seq_ids = set(prev_df["sequence_id"])
            print(f"Resuming from existing manifest: {len(existing_seq_ids)} sequences already indexed.", flush=True)
        except Exception:
            pass

    t_start_all = time.time()
    processed_count = 0
    skipped_count = 0

    for i, rec in enumerate(records, start=1):
        sample_id = rec["sample_id"]
        seq_id = rec["sequence_id"]
        vpath = rec["video_path"]
        split = rec["split"]
        cls_label = rec["class_label"]
        cls_id = rec["class_id"]
        signer_id = rec["signer_id"]

        seq_out_dir = os.path.join(sequences_base, split)
        seq_out_path = os.path.join(seq_out_dir, f"{seq_id}.npz").replace("\\", "/")
        lnd_out_dir = os.path.join(landmarks_base, split)
        lnd_out_path = os.path.join(lnd_out_dir, f"{sample_id}.npz").replace("\\", "/")

        # Check if already processed and valid
        if seq_id in existing_seq_ids and os.path.exists(seq_out_path) and os.path.exists(lnd_out_path):
            skipped_count += 1
            if i % 50 == 0:
                print(f"[{i}/{total}] Already processed {seq_id} ({cls_label}), skipping...", flush=True)
            continue

        t0 = time.time()
        try:
            # 1. Landmark Preprocessing
            prep_res = pipeline.process_sample(rec)
            completed_preprocessing.append(prep_res)

            # 2. Sequence Construction
            with np.load(lnd_out_path) as arc:
                data = arc["data"]
                mask = arc["mask"]
                raw_coords = arc["raw_coords"] if "raw_coords" in arc else np.zeros((45, 93, 3), dtype=np.float32)

            q_info = quality_scorer.compute_quality(mask, data)
            token_id = cls_id + 4

            cap = cv2.VideoCapture(vpath)
            orig_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else SEQUENCE_LENGTH
            orig_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 25.0
            cap.release()
            orig_duration = round(orig_frames / (orig_fps if orig_fps > 0 else 25.0), 3)

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
            existing_seq_ids.add(seq_id)
            processed_count += 1

            t_elapsed = time.time() - t0
            avg_per_item = (time.time() - t_start_all) / max(1, processed_count)
            remaining_sec = (total - i) * avg_per_item

            if processed_count % 5 == 0 or i == total:
                print(f"[{i}/{total}] {sample_id} ({cls_label}, {split}) in {t_elapsed:.1f}s | "
                      f"Q={q_info['quality_score']:.2f} ({q_info['quality_status']}) | "
                      f"ETA: {remaining_sec/60:.1f}m", flush=True)

            # Checkpoint manifests every 25 samples
            if processed_count % 25 == 0:
                save_manifests(completed_sequences, completed_preprocessing)

        except Exception as e:
            print(f"[{i}/{total}] ERROR on {sample_id} ({vpath}): {e}", file=sys.stderr, flush=True)

    pipeline.close()
    t_total = time.time() - t_start_all

    # Final save of manifests
    save_manifests(completed_sequences, completed_preprocessing)

    print("\n" + "=" * 70, flush=True)
    print("FULL DATASET CONVERSION FINISHED", flush=True)
    print(f"Total videos processed this run: {processed_count}", flush=True)
    print(f"Total sequences indexed: {len(completed_sequences)} / {total}", flush=True)
    print(f"Total elapsed time: {t_total:.2f}s ({t_total/60:.1f} minutes)", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
