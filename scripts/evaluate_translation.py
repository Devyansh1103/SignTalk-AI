#!/usr/bin/env python3
"""
SignTalk AI: Translation & NLP Layer Evaluation CLI.

Evaluates the SignTranslationModel (ST-GCN + Transformer) on sequence/gloss
prediction tasks, computing token accuracy, exact match, BLEU, and ROUGE.
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.evaluator import ModelEvaluator
from src.data.dataloader import create_sequence_dataloader


def main():
    parser = argparse.ArgumentParser(description="Evaluate SignTranslationModel (ST-GCN + Transformer).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/transformer/checkpoints/best_checkpoint.pt",
        help="Path to transformer checkpoint."
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
        default="results/translation",
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

    print(f"\n[SignTalk AI] Evaluating ST-GCN + Transformer Translation Model")
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
        model_type="transformer",
        device=args.device
    )

    results = evaluator.evaluate(
        dataloader=loader,
        split_name=args.split,
        output_dir=args.output_dir
    )

    metrics = results["metrics"]
    bench = results["benchmark"]
    bleu = metrics["bleu"]
    rouge = metrics["rouge"]

    print("=" * 60)
    print("          TRANSFORMER TRANSLATION EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Token Accuracy:       {metrics['token_accuracy']:.4f}")
    print(f"Sequence Exact Match: {metrics['sequence_exact_match']:.4f}")
    print(f"BLEU-1:               {bleu.get('bleu_1', 0.0):.4f}")
    print(f"BLEU-2:               {bleu.get('bleu_2', 0.0):.4f}")
    print(f"BLEU-4:               {bleu.get('bleu_4', 0.0):.4f}")
    print(f"ROUGE-L F1:           {rouge.get('rouge_l_f1', 0.0):.4f}")
    print(f"Mean Latency:         {bench['mean_latency_ms']} ms")
    print(f"Throughput:           {bench['throughput_fps']} FPS")
    print(f"Total Parameters:     {bench['total_parameters']:,}")
    print(f"Checkpoint Size:      {results['model_size_mb']} MB")
    print("=" * 60)
    print(f"[SUCCESS] All artifacts written to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
