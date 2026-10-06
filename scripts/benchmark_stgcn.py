#!/usr/bin/env python3
"""
SignTalk AI: ST-GCN Inference Latency and Throughput Benchmark.

Measures inference latency on the ST-GCN model:
  - Preprocessing time (data normalization, device transfer)
  - Pure model inference time (PyTorch forward pass over graph)
  - Post-processing time (Softmax + Top-1/Top-3 argmax extraction)

Evaluates:
  - Batch size 1 (real-time stream simulation)
  - Batch size 8 (batched evaluation)
Calculates:
  - Mean latency
  - Median latency
  - P95 latency
  - Throughput (sequences/sec)
"""

import sys
import os
import time
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.device import get_device
from src.models.stgcn import SignSTGCN


def benchmark_latency(model, device, batch_size=1, num_runs=200, warmup_runs=20):
    model.eval()

    # Input tensor shape: [B, C, T, V] = [B, 3, 45, 93]
    dummy_input = torch.randn(batch_size, 3, 45, 93, dtype=torch.float32)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            x = dummy_input.to(device)
            out = model(x)
            _ = torch.softmax(out, dim=-1)

    prep_times = []
    infer_times = []
    post_times = []
    total_times = []

    with torch.no_grad():
        for _ in range(num_runs):
            # Preprocessing: device transfer
            t0 = time.perf_counter()
            x = dummy_input.to(device)
            t1 = time.perf_counter()

            # Pure model inference
            logits = model(x)
            t2 = time.perf_counter()

            # Post-processing: softmax & top-1 prediction
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            t3 = time.perf_counter()

            prep_times.append((t1 - t0) * 1000.0)      # ms
            infer_times.append((t2 - t1) * 1000.0)     # ms
            post_times.append((t3 - t2) * 1000.0)      # ms
            total_times.append((t3 - t0) * 1000.0)     # ms

    results = {
        "batch_size": batch_size,
        "num_runs": num_runs,
        "prep_ms": {
            "mean": float(np.mean(prep_times)),
            "median": float(np.median(prep_times)),
            "p95": float(np.percentile(prep_times, 95))
        },
        "infer_ms": {
            "mean": float(np.mean(infer_times)),
            "median": float(np.median(infer_times)),
            "p95": float(np.percentile(infer_times, 95))
        },
        "post_ms": {
            "mean": float(np.mean(post_times)),
            "median": float(np.median(post_times)),
            "p95": float(np.percentile(post_times, 95))
        },
        "total_ms": {
            "mean": float(np.mean(total_times)),
            "median": float(np.median(total_times)),
            "p95": float(np.percentile(total_times, 95))
        },
        "throughput_fps": float((batch_size * num_runs) / (np.sum(total_times) / 1000.0))
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Inference Benchmark for SignSTGCN.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/stgcn/checkpoints/best_checkpoint.pt"
    )
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    device, dev_info = get_device("auto")
    print(f"[BENCHMARK] Device: {device} ({dev_info.get('name', 'CPU')})")

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model_cfg = ckpt["config"]["model"]
        dataset_cfg = ckpt["config"]["dataset"]
        graph_cfg = ckpt["config"]["graph"]
        model = SignSTGCN(
            in_channels=dataset_cfg.get("in_channels", 3),
            num_classes=dataset_cfg.get("num_classes", 10),
            num_nodes=dataset_cfg.get("num_nodes", 93),
            sequence_length=dataset_cfg.get("sequence_length", 45),
            graph_strategy=graph_cfg.get("strategy", "spatial"),
            block_channels=model_cfg.get("block_channels", [64, 64, 128, 128, 256, 256]),
            block_strides=model_cfg.get("block_strides", [1, 1, 2, 1, 2, 1]),
            temporal_kernel_size=model_cfg.get("temporal_kernel_size", 9),
            dropout=model_cfg.get("dropout", 0.3),
            residual=model_cfg.get("residual", True),
            use_learnable_edge_weights=model_cfg.get("use_learnable_edge_weights", True)
        )
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = SignSTGCN()

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n--- Benchmarking Batch Size 1 (Real-time Stream) ---")
    res_b1 = benchmark_latency(model, device, batch_size=1, num_runs=args.runs)
    print(f"Preprocessing Latency: Mean={res_b1['prep_ms']['mean']:.3f} ms, Median={res_b1['prep_ms']['median']:.3f} ms, P95={res_b1['prep_ms']['p95']:.3f} ms")
    print(f"Model Inference Latency: Mean={res_b1['infer_ms']['mean']:.3f} ms, Median={res_b1['infer_ms']['median']:.3f} ms, P95={res_b1['infer_ms']['p95']:.3f} ms")
    print(f"Postprocessing Latency: Mean={res_b1['post_ms']['mean']:.3f} ms, Median={res_b1['post_ms']['median']:.3f} ms, P95={res_b1['post_ms']['p95']:.3f} ms")
    print(f"Total Pipeline Latency: Mean={res_b1['total_ms']['mean']:.3f} ms, Median={res_b1['total_ms']['median']:.3f} ms, P95={res_b1['total_ms']['p95']:.3f} ms")
    print(f"Throughput: {res_b1['throughput_fps']:.1f} sequences/sec")

    print("\n--- Benchmarking Batch Size 8 (Batched Evaluation) ---")
    res_b8 = benchmark_latency(model, device, batch_size=8, num_runs=args.runs)
    print(f"Total Pipeline Latency: Mean={res_b8['total_ms']['mean']:.3f} ms, Median={res_b8['total_ms']['median']:.3f} ms, P95={res_b8['total_ms']['p95']:.3f} ms")
    print(f"Throughput: {res_b8['throughput_fps']:.1f} sequences/sec")

    # Save benchmark report
    out_dir = "experiments/stgcn/metrics"
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "hardware": dev_info,
        "model_parameters": {
            "total": total_params,
            "trainable": trainable_params
        },
        "batch_1": res_b1,
        "batch_8": res_b8
    }
    with open(os.path.join(out_dir, "inference_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SUCCESS] ST-GCN benchmark report saved to: {os.path.join(out_dir, 'inference_benchmark.json')}")


if __name__ == "__main__":
    main()
