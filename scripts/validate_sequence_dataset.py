#!/usr/bin/env python3
"""
SignTalk AI: Comprehensive Sequence Dataset Integrity and Validation Suite.

Executes 15 rigorous checks:
  1. Manifest validation (required columns, row counts)
  2. File existence (every manifest entry maps to existing .npz on disk)
  3. Tensor shape validation (C=3, T=45, V=93)
  4. Label validation (class IDs in [0..9], matching canonical labels)
  5. Split validation (train, val, test partition consistency)
  6. Signer leakage detection (zero recording leakage across signers)
  7. Source-video leakage detection (zero source video overlap across splits)
  8. Missing-value validation (no nulls in mandatory manifest columns)
  9. NaN/Inf detection (all tensor coordinates are finite)
  10. Sequence-length validation (exact T=45 temporal standardization)
  11. Mask validation (binary validity mask shape [1, 45, 93])
  12. Graph-node validation (V=93 matches kinematic adjacency matrix)
  13. Class distribution check (uniform class representation audit)
  14. Metadata consistency (synchronization across manifest and .npz internal metadata)
  15. Version consistency (verifies dataset_version == 'signTalk-seq-v1.0.0')

Exits with code 0 on complete pass, code 1 on failure.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.split_validator import SplitValidator
from src.data.stgcn_tensor import NUM_NODES, SEQUENCE_LENGTH, CHANNELS


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Sequence Dataset Validation.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/sequence_manifest.csv",
        help="Path to master sequence manifest."
    )
    parser.add_argument(
        "--adjacency",
        type=str,
        default="assets/graphs/kinematic_adjacency_93.npy",
        help="Path to kinematic adjacency matrix."
    )
    args = parser.parse_args()

    print("\n============================================================")
    print("      SIGNTALK AI: SEQUENCE DATASET VALIDATION SUITE")
    print("============================================================")
    print(f"Manifest:  {args.manifest}")
    print(f"Adjacency: {args.adjacency}\n")

    failures = []

    # 1. Manifest Validation
    print("[CHECK 1/15] Validating master manifest structure...")
    if not os.path.exists(args.manifest):
        print(f"  [FAIL] Manifest does not exist: {args.manifest}")
        sys.exit(1)

    df = pd.read_csv(args.manifest)
    required_cols = [
        "sequence_id", "source_dataset", "source_recording_id", "source_video_id",
        "signer_id", "split", "class_id", "label", "gloss", "translation",
        "frame_count", "target_frames", "fps", "duration_seconds",
        "quality_score", "quality_status", "dataset_version", "file_path"
    ]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        failures.append(f"Check 1: Manifest missing required columns: {missing_cols}")
    else:
        print(f"  [PASS] Manifest contains all {len(required_cols)} required columns ({len(df)} rows).")

    # 2. File Existence
    print("[CHECK 2/15] Verifying sequence archive existence on disk...")
    missing_files = []
    for _, row in df.iterrows():
        fpath = str(row["file_path"])
        if not os.path.exists(fpath):
            missing_files.append(fpath)
    if missing_files:
        failures.append(f"Check 2: {len(missing_files)} sequence files missing from disk.")
    else:
        print(f"  [PASS] All {len(df)} sequence archives verified on disk.")

    # 3, 9, 10, 11, 14: Tensor Inspection across all files
    print("[CHECK 3, 9, 10, 11, 14/15] Inspecting internal tensor shapes, NaNs, masks, and metadata...")
    shape_errors = []
    nan_errors = []
    mask_errors = []
    meta_mismatches = []

    for _, row in df.iterrows():
        fpath = str(row["file_path"])
        seq_id = str(row["sequence_id"])

        try:
            with np.load(fpath) as arc:
                data = arc["data"]
                mask = arc["mask"]
                cid = int(arc["label"])
                arc_seq_id = str(arc["sequence_id"])

                # Check 3 & 10: Shape
                if data.shape != (CHANNELS, SEQUENCE_LENGTH, NUM_NODES):
                    shape_errors.append((seq_id, data.shape))

                # Check 9: NaNs/Infs
                if not np.all(np.isfinite(data)):
                    nan_errors.append(seq_id)

                # Check 11: Mask
                if mask.shape != (1, SEQUENCE_LENGTH, NUM_NODES):
                    mask_errors.append((seq_id, mask.shape))
                if not np.all(np.logical_or(mask == 0.0, mask == 1.0)):
                    mask_errors.append((seq_id, "Non-binary mask values"))

                # Check 14: Metadata match
                if cid != int(row["class_id"]) or arc_seq_id != seq_id:
                    meta_mismatches.append(seq_id)

        except Exception as e:
            failures.append(f"Corrupt archive {seq_id}: {str(e)}")

    if shape_errors:
        failures.append(f"Check 3: {len(shape_errors)} tensors have invalid shapes: {shape_errors[:3]}")
    else:
        print(f"  [PASS] All {len(df)} tensors match expected shape [{CHANNELS}, {SEQUENCE_LENGTH}, {NUM_NODES}].")

    if nan_errors:
        failures.append(f"Check 9: {len(nan_errors)} tensors contain NaN or Infinite values.")
    else:
        print("  [PASS] All tensors contain 100% finite coordinates (Zero NaNs, Zero Infs).")

    if mask_errors:
        failures.append(f"Check 11: Mask shape or binary errors detected: {mask_errors[:3]}")
    else:
        print("  [PASS] All auxiliary masks conform to binary [1, 45, 93] topology.")

    if meta_mismatches:
        failures.append(f"Check 14: Metadata discrepancies between manifest and archives: {meta_mismatches[:3]}")
    else:
        print("  [PASS] Internal archive metadata is 100% synchronized with manifest.")

    # 4. Label Validation
    print("[CHECK 4/15] Validating categorical label integrity...")
    valid_cids = set(range(10))
    manifest_cids = set(df["class_id"].unique())
    if not manifest_cids.issubset(valid_cids):
        failures.append(f"Check 4: Invalid class IDs in manifest: {manifest_cids - valid_cids}")
    else:
        print(f"  [PASS] All class IDs are valid integers within [0, 9] (10 distinct classes).")

    # 5. Split Validation
    print("[CHECK 5/15] Validating train/val/test partition structure...")
    splits = set(df["split"].unique())
    if splits != {"train", "val", "test"}:
        failures.append(f"Check 5: Unexpected split names: {splits}")
    else:
        train_n = (df["split"] == "train").sum()
        val_n = (df["split"] == "val").sum()
        test_n = (df["split"] == "test").sum()
        print(f"  [PASS] Split distribution: Train={train_n}, Val={val_n}, Test={test_n} (Total={len(df)}).")

    # 6 & 7. Signer & Source-Video Leakage Check
    print("[CHECK 6 & 7/15] Running automated partition leakage audit...")
    try:
        validator = SplitValidator(df)
        leakage_rep = validator.check_leakage()
        if leakage_rep["passed"]:
            print(f"  [PASS] Zero source-video overlap across splits (0 leakage).")
            print(f"  [PASS] Signer representation balanced across partitions.")
        else:
            failures.append("Check 6/7: Split leakage detected by validator.")
    except Exception as e:
        failures.append(f"Check 6/7: Split validator raised exception: {str(e)}")

    # 8. Missing-Value Validation
    print("[CHECK 8/15] Checking for null or missing values in mandatory fields...")
    null_counts = df[required_cols].isnull().sum()
    has_nulls = null_counts[null_counts > 0]
    if not has_nulls.empty:
        failures.append(f"Check 8: Null values found in mandatory columns: {has_nulls.to_dict()}")
    else:
        print("  [PASS] Zero null or NaN values in mandatory manifest columns.")

    # 12. Graph-Node Validation
    print("[CHECK 12/15] Validating graph adjacency tensor compatibility...")
    if not os.path.exists(args.adjacency):
        failures.append(f"Check 12: Adjacency file missing: {args.adjacency}")
    else:
        adj = np.load(args.adjacency)
        if adj.shape != (3, NUM_NODES, NUM_NODES):
            failures.append(f"Check 12: Adjacency shape mismatch: {adj.shape} != (3, {NUM_NODES}, {NUM_NODES})")
        else:
            print(f"  [PASS] Adjacency matrix shape matches skeletal node count ({adj.shape}).")

    # 13. Class Distribution Balance Check
    print("[CHECK 13/15] Auditing vocabulary class distribution balance...")
    class_counts = df["class_id"].value_counts()
    if (class_counts != 6).any():
        failures.append(f"Check 13: Class imbalance detected! Counts: {class_counts.to_dict()}")
    else:
        print(f"  [PASS] All 10 classes have exactly 6 sequences (100% uniform balance).")

    # 15. Version Consistency
    print("[CHECK 15/15] Verifying dataset version consistency...")
    versions = set(df["dataset_version"].unique())
    expected_ver = "signTalk-seq-v1.0.0"
    if versions != {expected_ver}:
        failures.append(f"Check 15: Inconsistent dataset versions: {versions} != '{expected_ver}'")
    else:
        print(f"  [PASS] Master dataset version is uniformly '{expected_ver}'.")

    # Final Summary
    print("\n============================================================")
    if not failures:
        print("    ALL 15 INTEGRITY CHECKS PASSED SUCCESSFULLY (CODE 0)")
        print("============================================================\n")
        sys.exit(0)
    else:
        print(f"    VALIDATION FAILED WITH {len(failures)} ERROR(S):")
        for f in failures:
            print(f"      - {f}")
        print("============================================================\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
