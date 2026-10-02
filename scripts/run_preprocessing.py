"""
SignTalk AI - Run Preprocessing
CLI runner for pilot and full-dataset landmark extraction and preprocessing.
Usage:
  python scripts/run_preprocessing.py --mode pilot
  python scripts/run_preprocessing.py --mode full
"""

import os
import sys
import argparse
import csv
import json
import yaml

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.pipeline import PreprocessingPipeline


def main():
    parser = argparse.ArgumentParser(description="SignTalk AI Landmark Preprocessing Pipeline")
    parser.add_argument("--mode", type=str, choices=["pilot", "full"], default="pilot",
                        help="Execution mode: 'pilot' for preliminary validation or 'full' for complete dataset.")
    parser.add_argument("--config", type=str, default="configs/preprocessing.yaml",
                        help="Path to YAML configuration file.")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Optional override path to raw video manifest CSV.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional limit on number of samples to process.")
    args = parser.parse_args()

    print("============================================================")
    print("SignTalk AI: Preprocessing Pipeline Runner")
    print(f"Mode: {args.mode.upper()}")
    print(f"Config: {args.config}")
    print("============================================================\n")

    pipeline = PreprocessingPipeline(config_path=args.config)

    # Load raw manifest
    manifest_path = args.manifest or pipeline.config["dataset"]["manifest_path"]
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_records = list(reader)

    print(f"Total records in raw manifest: {len(all_records)}")

    if args.mode == "pilot":
        # Select representative pilot: multiple classes, all signers, difficult conditions
        # Pick 1 sample per class across signers (10 samples)
        pilot_records = []
        classes_seen = set()
        for rec in all_records:
            cls = rec["class_label"]
            if cls not in classes_seen:
                pilot_records.append(rec)
                classes_seen.add(cls)
            if len(pilot_records) >= 10:
                break

        print(f"Selected {len(pilot_records)} representative samples for PILOT across {len(classes_seen)} classes.")
        summary = pipeline.process_batch(pilot_records, is_pilot=True)

        # Save pilot summary to interim
        os.makedirs("data/interim/landmarks", exist_ok=True)
        with open("data/interim/landmarks/pilot_summary.json", "w", encoding="utf-8") as out_f:
            json.dump(summary, out_f, indent=2)
        print("Pilot processing completed. Results saved to data/interim/landmarks/pilot_summary.json")

    else:
        # Full dataset mode
        records_to_process = all_records[:args.max_samples] if args.max_samples else all_records
        summary = pipeline.process_batch(records_to_process, is_pilot=False)

        # Save full summary
        with open("data/processed/landmarks/full_dataset_summary.json", "w", encoding="utf-8") as out_f:
            json.dump(summary, out_f, indent=2)
        print("Full dataset processing completed. Summary saved to data/processed/landmarks/full_dataset_summary.json")

    pipeline.close()


if __name__ == "__main__":
    main()
