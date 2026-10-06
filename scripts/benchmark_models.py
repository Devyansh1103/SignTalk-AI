#!/usr/bin/env python3
"""
SignTalk AI: Hardware and Multi-Model Benchmarking CLI.

Measures:
  - Total inference latency distribution (mean, median, p95, p99, min, max, std)
  - Throughput (FPS)
  - Parameter counts (Total, Trainable)
  - Checkpoint disk size
  - Host execution hardware platform
Exports:
  - results/benchmarks/benchmark_summary.csv
  - results/benchmarks/benchmark_summary.json
"""

import sys
import os
import argparse
import json
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.device import get_device
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.benchmark import get_hardware_environment


def main():
    parser = argparse.ArgumentParser(description="Benchmark candidate models.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmarks",
        help="Directory to save benchmark results."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of timed benchmark iterations."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('auto', 'cpu', 'cuda')."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device, _ = get_device(args.device)

    print("=" * 70)
    print("SignTalk AI — Multi-Model Latency and Resource Benchmark")
    print(f"Device: {device}")
    print(f"Iterations: {args.iterations}")
    print("=" * 70)

    hardware = get_hardware_environment()

    candidates = [
        ("Baseline (BiLSTM)", "baseline", "experiments/baseline/checkpoints/best_checkpoint.pt"),
        ("ST-GCN (Graph Conv)", "stgcn", "experiments/stgcn/checkpoints/best_checkpoint.pt"),
        ("ST-GCN + Transformer", "transformer", "experiments/transformer/checkpoints/best_checkpoint.pt"),
    ]

    records = []
    full_json_data = {
        "hardware": hardware,
        "models": {}
    }

    dummy_x = torch.zeros(1, 3, 45, 93)

    for name, mtype, ckpt in candidates:
        if not os.path.exists(ckpt):
            print(f"[SKIP] Checkpoint not found for {name}: {ckpt}")
            continue

        print(f"\nBenchmarking: {name}...")
        evaluator = ModelEvaluator(checkpoint_path=ckpt, model_type=mtype, device=args.device)
        from src.evaluation.benchmark import benchmark_model_inference
        bench = benchmark_model_inference(
            model=evaluator.model,
            sample_input=dummy_x,
            device=device,
            num_warmup=10,
            num_iterations=args.iterations
        )
        ckpt_sz = os.path.getsize(ckpt) / (1024 * 1024)

        full_json_data["models"][name] = {
            "checkpoint": ckpt,
            "model_size_mb": round(ckpt_sz, 2),
            "benchmark": bench
        }

        records.append({
            "model": name,
            "checkpoint": ckpt,
            "mean_latency_ms": bench["mean_latency_ms"],
            "median_latency_ms": bench["median_latency_ms"],
            "p95_latency_ms": bench["p95_latency_ms"],
            "min_latency_ms": bench["min_latency_ms"],
            "max_latency_ms": bench["max_latency_ms"],
            "throughput_fps": bench["throughput_fps"],
            "total_parameters": bench["total_parameters"],
            "trainable_parameters": bench["trainable_parameters"],
            "model_size_mb": round(ckpt_sz, 2),
            "meets_realtime_25fps": bench["meets_realtime_target"]
        })

    df = pd.DataFrame(records)
    out_csv = os.path.join(args.output_dir, "benchmark_summary.csv")
    out_json = os.path.join(args.output_dir, "benchmark_summary.json")

    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_json_data, f, indent=2)

    print("\n" + "=" * 80)
    print("                     BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"\n[SUCCESS] Benchmark artifacts saved to {args.output_dir}\n")


if __name__ == "__main__":
    main()
