#!/usr/bin/env python3
"""
SignTalk AI: ST-GCN Model Evaluation CLI.

Evaluates the Spatial-Temporal Graph Convolutional Network (ST-GCN) model
on test or validation splits using the standardized ModelEvaluator engine.
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.report import generate_model_summary_markdown
from src.data.dataloader import create_sequence_dataloader


def main():
    parser = argparse.ArgumentParser(description="Evaluate ST-GCN on sign sequence partitions.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/stgcn/checkpoints/best_checkpoint.pt",
        help="Path to ST-GCN checkpoint."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Split to evaluate."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Manifest CSV path (defaults to data/manifests/<split>.csv)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Evaluation batch size."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/stgcn",
        help="Directory to save evaluation artifacts."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('auto', 'cpu', 'cuda')."
    )
    args = parser.parse_args()

    manifest_path = args.manifest or f"data/manifests/{args.split}.csv"
    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"\n[SignTalk AI] Evaluating ST-GCN Model")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split:      {args.split} ({manifest_path})")
    print(f"Output Dir: {args.output_dir}\n")

    loader = create_sequence_dataloader(
        manifest_path=manifest_path,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        filter_rejects=False,
        augment=False
    )

    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        model_type="stgcn",
        device=args.device
    )

    results = evaluator.evaluate(
        dataloader=loader,
        split_name=args.split,
        output_dir=args.output_dir
    )

    # Generate Markdown Summary
    report_md_path = os.path.join(args.output_dir, "report.md")
    generate_model_summary_markdown("SignSTGCN", results, output_path=report_md_path)

    metrics = results["metrics"]
    bench = results["benchmark"]
    calib = results["calibration"]
    print("=" * 60)
    print("              ST-GCN EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Top-1 Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Top-3 Accuracy:    {metrics['top3_accuracy']:.4f}")
    print(f"Macro F1-Score:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    print(f"Mean Latency:      {bench['mean_latency_ms']} ms")
    print(f"Throughput:        {bench['throughput_fps']} FPS")
    print(f"Total Parameters:  {bench['total_parameters']:,}")
    print(f"ECE Calibration:   {calib['ece']:.4f}")
    print(f"Checkpoint Size:   {results['model_size_mb']} MB")
    print("=" * 60)
    print(f"[SUCCESS] All artifacts written to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
