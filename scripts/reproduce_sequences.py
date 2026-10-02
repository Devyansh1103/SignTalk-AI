#!/usr/bin/env python3
"""
SignTalk AI: End-to-End Sequence Dataset Reproduction Script.

Rebuilds the entire sequence dataset from preprocessed landmarks,
validates split integrity, and tests DataLoader functionality.
"""

import sys
import os
import subprocess
import time


def run_command(cmd, desc):
    print(f"\n>>> [STEP] {desc}")
    print(f"    Command: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[FAILED] Error executing step ({elapsed:.2f}s):")
        print(res.stderr)
        print(res.stdout)
        sys.exit(res.returncode)
    else:
        print(f"[SUCCESS] Completed in {elapsed:.2f}s")
        if res.stdout:
            lines = res.stdout.strip().split("\n")
            summary = "\n    ".join(lines[-5:])
            print(f"    Output tail:\n    {summary}")


def main():
    print("============================================================")
    print("      SIGNTALK AI: SEQUENCE DATASET REPRODUCTION")
    print("============================================================")
    py = sys.executable

    # 1. Build Sequences
    run_command([py, "scripts/build_sequences.py"], "1. Building canonical sequence dataset")

    # 2. Validate Splits
    run_command([py, "scripts/validate_sequence_splits.py"], "2. Auditing split isolation & leakage")

    # 3. Test DataLoader
    run_command([py, "scripts/test_dataloader.py"], "3. Validating PyTorch DataLoaders & batch shapes")

    print("\n============================================================")
    print("      REPRODUCTION PIPELINE COMPLETED SUCCESSFULLY")
    print("============================================================\n")


if __name__ == "__main__":
    main()
