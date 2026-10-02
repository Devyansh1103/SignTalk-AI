"""
SignTalk AI - Validate Processed Data
Performs comprehensive quality, shape, distribution, and PyTorch compatibility validation
on preprocessed landmark datasets.
Usage:
  python scripts/validate_processed_data.py
"""

import os
import sys
import glob
import json
import csv
import numpy as np

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.landmark_dataset import (
    SignLandmarkDataset,
    create_landmark_dataloader,
    verify_dataloader_tensor_shapes
)


def validate_processed_directory(
    root_dir: str = "data/processed/landmarks",
    expected_c: int = 3,
    expected_t: int = 45,
    expected_v: int = 93
) -> dict:
    """
    Scans processed directory and validates each NPZ file.
    """
    print(f"Scanning processed directory: {root_dir}")
    splits = ["train", "val", "test"]
    stats = {
        "total_files": 0,
        "valid_files": 0,
        "corrupt_files": 0,
        "nan_or_inf_files": 0,
        "shape_mismatches": 0,
        "splits": {},
        "classes": {},
        "signers": {},
        "classifications": {"GOOD": 0, "ACCEPTABLE": 0, "REVIEW": 0, "REJECT": 0},
        "quality_scores": []
    }

    for s in splits:
        s_dir = os.path.join(root_dir, s)
        if not os.path.exists(s_dir):
            stats["splits"][s] = {"count": 0, "status": "MISSING"}
            continue

        files = sorted(glob.glob(os.path.join(s_dir, "*.npz")))
        stats["splits"][s] = {"count": len(files), "status": "OK"}
        stats["total_files"] += len(files)

        for f in files:
            try:
                npz = np.load(f, allow_pickle=True)
                data = npz["data"]
                mask = npz["mask"]
                class_label = str(npz["class_label"])
                signer_id = str(npz["signer_id"])
                quality = float(npz["quality_score"])
                classification = str(npz.get("classification", "UNKNOWN"))

                # 1. Shape validation
                C, T, V = data.shape
                if C != expected_c or T != expected_t or V != expected_v:
                    stats["shape_mismatches"] += 1
                    continue

                # 2. Numerical validation
                if not np.all(np.isfinite(data)):
                    stats["nan_or_inf_files"] += 1
                    continue

                # 3. Tally distributions
                stats["classes"][class_label] = stats["classes"].get(class_label, 0) + 1
                stats["signers"][signer_id] = stats["signers"].get(signer_id, 0) + 1
                stats["classifications"][classification] = stats["classifications"].get(classification, 0) + 1
                stats["quality_scores"].append(quality)
                stats["valid_files"] += 1

            except Exception as e:
                stats["corrupt_files"] += 1

    return stats


def main():
    print("============================================================")
    print("SignTalk AI: Processed Data Validation Suite")
    print("============================================================\n")

    stats = validate_processed_directory()

    print(f"Total Processed Files: {stats['total_files']}")
    print(f"Valid Files: {stats['valid_files']}")
    print(f"Corrupt Files: {stats['corrupt_files']}")
    print(f"Shape Mismatches: {stats['shape_mismatches']}")
    print(f"NaN/Inf Occurrences: {stats['nan_or_inf_files']}\n")

    print("--- Split Distribution ---")
    for s, info in stats["splits"].items():
        print(f"  Split '{s}': {info['count']} samples ({info['status']})")

    print("\n--- Quality Classification Breakdown ---")
    for c_type, count in stats["classifications"].items():
        pct = (count / max(1, stats["valid_files"])) * 100.0
        print(f"  {c_type:12s}: {count:4d} ({pct:5.1f}%)")

    if stats["quality_scores"]:
        mean_q = float(np.mean(stats["quality_scores"]))
        p95_q = float(np.percentile(stats["quality_scores"], 95))
        p05_q = float(np.percentile(stats["quality_scores"], 5))
        print(f"\nMean Sequence Quality: {mean_q:.4f} (5th pct: {p05_q:.3f}, 95th pct: {p95_q:.3f})")

    print("\n--- Class Coverage ---")
    print(f"Total Unique Classes Present: {len(stats['classes'])}")
    for cls, cnt in sorted(stats["classes"].items()):
        print(f"  {cls:15s}: {cnt:3d} samples")

    # PyTorch DataLoader Sanity Check
    print("\n--- PyTorch DataLoader Sanity Check ---")
    train_check = verify_dataloader_tensor_shapes(split="train")
    print(f"Train DataLoader Check: {train_check['status']}")
    if train_check["status"] == "PASSED":
        print(f"  Input Tensor Shape  : {train_check['data_shape']} [B, C, T, V]")
        print(f"  Mask Tensor Shape   : {train_check['mask_shape']} [B, 1, T, V]")
        print(f"  Labels Shape        : {train_check['labels_shape']} [B]")
        print(f"  Data Type           : {train_check['dtype']}")
        print(f"  All Values Finite   : {train_check['is_finite']}")
    else:
        print(f"  Reason: {train_check.get('reason', 'Unknown error')}")

    # Save validation report JSON
    os.makedirs("data/metadata", exist_ok=True)
    report_path = "data/metadata/processed_validation_report.json"
    clean_stats = stats.copy()
    clean_stats["dataloader_check"] = train_check
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(clean_stats, f, indent=2)

    print(f"\nValidation complete. Report written to {report_path}")


if __name__ == "__main__":
    main()
