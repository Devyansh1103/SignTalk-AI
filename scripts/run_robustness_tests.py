#!/usr/bin/env python3
"""
SignTalk AI: Robustness Testing CLI.

Executes perturbation stress tests on the final ST-GCN candidate model:
  - Additive Gaussian noise
  - Temporal frame dropout
  - Temporal speed variation
  - Anatomical subsystem occlusion
  - Horizontal sign mirroring
Exports:
  - results/robustness/robustness_matrix.csv
"""

import sys
import os
import argparse
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.device import get_device
from src.evaluation.evaluator import ModelEvaluator
from src.data.dataloader import create_sequence_dataloader
from src.evaluation.robustness import evaluate_robustness_suite


def main():
    parser = argparse.ArgumentParser(description="Run robustness perturbation suite.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/stgcn/checkpoints/best_checkpoint.pt",
        help="Path to trained ST-GCN checkpoint."
    )
    parser.add_argument(
        "--test-manifest",
        type=str,
        default="data/manifests/test.csv",
        help="Path to test partition manifest."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/robustness",
        help="Output directory for robustness results."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run experiments on."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device, _ = get_device(args.device)

    print("=" * 70)
    print("SignTalk AI — Executing Robustness Stress Testing Suite")
    print(f"Candidate Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Test Manifest: {args.test_manifest}")
    print("=" * 70)

    loader = create_sequence_dataloader(
        manifest_path=args.test_manifest,
        split="test",
        batch_size=8,
        shuffle=False,
        filter_rejects=False,
        augment=False
    )

    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        model_type="stgcn",
        device=args.device
    )

    results = evaluate_robustness_suite(
        model=evaluator.model,
        dataloader=loader,
        device=device,
        class_names=evaluator.class_names,
        num_classes=len(evaluator.class_names)
    )

    df = pd.DataFrame(results)
    out_csv = os.path.join(args.output_dir, "robustness_matrix.csv")
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("                     ROBUSTNESS STRESS MATRIX")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"\n[SUCCESS] Robustness matrix saved to: {out_csv}\n")


if __name__ == "__main__":
    main()
