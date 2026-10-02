#!/usr/bin/env python3
"""
SignTalk AI: Sequence Split Leakage Verification Script.

Executes automated checks to verify zero source video or recording overlap
between train, val, and test splits. Exits with code 0 on success, code 1 on failure.
"""

import sys
import os
import argparse
import pandas as pd

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.split_validator import SplitValidator, SplitLeakageError


def main():
    parser = argparse.ArgumentParser(description="Validate sequence dataset split integrity.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/sequence_manifest.csv",
        help="Path to sequence manifest CSV."
    )
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"[ERROR] Manifest not found at: {args.manifest}")
        sys.exit(1)

    print(f"[INFO] Reading sequence manifest from: {args.manifest}")
    df = pd.read_csv(args.manifest)

    try:
        validator = SplitValidator(df)
        report = validator.check_leakage()
        print("\n============================================================")
        print("          SEQUENCE SPLIT LEAKAGE AUDIT REPORT")
        print("============================================================")
        print(f"Total Sequences:       {report['total_sample_count']}")
        print(f"Train Sequences:       {report['train_sample_count']}")
        print(f"Val Sequences:         {report['val_sample_count']}")
        print(f"Test Sequences:        {report['test_sample_count']}")
        print(f"Train Unique Videos:   {report['train_unique_videos']}")
        print(f"Val Unique Videos:     {report['val_unique_videos']}")
        print(f"Test Unique Videos:    {report['test_unique_videos']}")
        print("------------------------------------------------------------")
        print(f"Train/Val Overlap:     {report['train_val_overlap_count']}")
        print(f"Train/Test Overlap:    {report['train_test_overlap_count']}")
        print(f"Val/Test Overlap:      {report['val_test_overlap_count']}")
        print("------------------------------------------------------------")
        print(f"STATUS:                {'PASSED (ZERO LEAKAGE)' if report['passed'] else 'FAILED'}")
        print("============================================================\n")

        if "signer_distribution_by_split" in report:
            print("[INFO] Signer Distribution by Split:")
            for split, signers in report["signer_distribution_by_split"].items():
                print(f"  Split '{split}': {signers}")

        print("\n[SUCCESS] Split validation passed with zero data leakage.")
        sys.exit(0)

    except SplitLeakageError as e:
        print(f"\n[CRITICAL FAILURE] {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during split validation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
