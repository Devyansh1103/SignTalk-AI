#!/usr/bin/env python3
"""
SignTalk AI: Sequence Dataset Generation Script.

Executes the deterministic sequence generation pipeline using
configs/sequence_generation.yaml. Produces standardized sequence archives
in data/processed/sequences/ and manifests in data/manifests/.
"""

import sys
import os
import argparse
import time

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.sequence_builder import SequenceBuilder


def main():
    parser = argparse.ArgumentParser(description="Build canonical sequence dataset for SignTalk AI.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sequence_generation.yaml",
        help="Path to sequence generation configuration YAML."
    )
    args = parser.parse_args()

    print("\n============================================================")
    print("      SIGNTALK AI: CANONICAL SEQUENCE DATASET BUILDER")
    print("============================================================")
    print(f"Configuration: {args.config}")
    start_time = time.time()

    builder = SequenceBuilder(config_path=args.config)
    master_df, rej_df = builder.build_dataset()

    elapsed = time.time() - start_time
    print("\n------------------------------------------------------------")
    print("                   BUILD SUMMARY")
    print("------------------------------------------------------------")
    print(f"Total Sequences Generated:  {len(master_df)}")
    print(f"Train Sequences:            {(master_df['split'] == 'train').sum()}")
    print(f"Validation Sequences:       {(master_df['split'] == 'val').sum()}")
    print(f"Test Sequences:             {(master_df['split'] == 'test').sum()}")
    print(f"Rejected Sequences:         {len(rej_df)}")
    print(f"Total Processing Time:      {elapsed:.2f} seconds")
    print(f"Throughput:                 {len(master_df)/elapsed:.2f} seq/sec")
    print("============================================================\n")


if __name__ == "__main__":
    main()
