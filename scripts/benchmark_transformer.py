"""
SignTalk AI: Transformer Translation Layer Inference Benchmark.

Benchmarks computational latency and throughput:
  - ST-GCN visual feature extraction latency
  - Feature projection latency
  - Autoregressive decoding latency
  - Total end-to-end translation latency (mean, median, P95)
  - Parameter footprint and model checkpoint size
"""

import os
import sys
import argparse
import json
import time

sys.path.insert(0, os.getcwd())

import yaml
import numpy as np
import torch

from src.models.sign_translation_model import SignTranslationModel
from src.nlp.tokenizer import SignLanguageTokenizer
from src.utils.device import get_device


def benchmark_transformer(
    config_path: str = "configs/transformer.yaml",
    num_runs: int = 100
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device, hw_info = get_device(cfg.get("device", "auto"))
    paths = cfg["paths"]
    checkpoint_file = os.path.join(paths["checkpoint_dir"], "best_checkpoint.pt")
    metrics_dir = paths["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)

    print("=" * 70)
    print("SignTalk AI — Transformer Translation Inference Benchmark")
    print(f"Device: {device} | Iterations: {num_runs}")
    print("=" * 70)

    # 1. Instantiate Model & Tokenizer
    tokenizer = SignLanguageTokenizer(vocab_path=cfg["transformer"]["vocabulary_file"])
    model = SignTranslationModel(
        stgcn_config=cfg.get("stgcn", {}),
        transformer_config=cfg.get("transformer", {}),
        freeze_stgcn=True
    ).to(device)

    if os.path.exists(checkpoint_file):
        ckpt = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from: {checkpoint_file}")
    model.eval()

    # Parameter footprint
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stgcn_params = sum(p.numel() for p in model.stgcn.parameters())
    transformer_params = sum(p.numel() for p in model.transformer.parameters())
    projection_params = sum(p.numel() for p in model.feature_projection.parameters())

    ckpt_size_mb = os.path.getsize(checkpoint_file) / (1024 * 1024) if os.path.exists(checkpoint_file) else 0.0

    print(f"Model Parameters:")
    print(f"  Total:        {total_params:,}")
    print(f"  ST-GCN:       {stgcn_params:,} (Frozen)")
    print(f"  Projection:   {projection_params:,}")
    print(f"  Transformer:  {transformer_params:,}")
    print(f"  Trainable:    {trainable_params:,}")
    print(f"  Size on disk: {ckpt_size_mb:.2f} MB")

    # 2. Warmup
    dummy_x = torch.randn(1, 3, 45, 93, device=device)
    print("\nWarming up model...")
    for _ in range(10):
        with torch.no_grad():
            _ = model.generate(dummy_x, max_length=5)

    # 3. Timed Iterations
    vis_latencies = []
    dec_latencies = []
    total_latencies = []

    print(f"Running {num_runs} timed inference passes...")
    for _ in range(num_runs):
        t0 = time.perf_counter()

        with torch.no_grad():
            # Visual feature extraction
            t_vis0 = time.perf_counter()
            proj_features, visual_mask = model.extract_visual_embeddings(dummy_x)
            t_vis1 = time.perf_counter()

            # Autoregressive decoding
            t_dec0 = time.perf_counter()
            tokens, confs = model.transformer.generate(
                visual_features=proj_features,
                visual_padding_mask=visual_mask,
                max_length=5
            )
            t_dec1 = time.perf_counter()

        t_end = time.perf_counter()

        vis_latencies.append((t_vis1 - t_vis0) * 1000.0)
        dec_latencies.append((t_dec1 - t_dec0) * 1000.0)
        total_latencies.append((t_end - t0) * 1000.0)

    # Compute statistics
    vis_arr = np.array(vis_latencies)
    dec_arr = np.array(dec_latencies)
    tot_arr = np.array(total_latencies)

    results = {
        "hardware": hw_info,
        "parameters": {
            "total": total_params,
            "stgcn_frozen": stgcn_params,
            "feature_projection": projection_params,
            "transformer": transformer_params,
            "trainable": trainable_params,
            "checkpoint_size_mb": round(ckpt_size_mb, 2)
        },
        "benchmark": {
            "num_runs": num_runs,
            "batch_size": 1,
            "visual_encoder_latency_ms": {
                "mean": round(float(np.mean(vis_arr)), 3),
                "median": round(float(np.median(vis_arr)), 3),
                "p95": round(float(np.percentile(vis_arr, 95)), 3)
            },
            "decoder_latency_ms": {
                "mean": round(float(np.mean(dec_arr)), 3),
                "median": round(float(np.median(dec_arr)), 3),
                "p95": round(float(np.percentile(dec_arr, 95)), 3)
            },
            "total_latency_ms": {
                "mean": round(float(np.mean(tot_arr)), 3),
                "median": round(float(np.median(tot_arr)), 3),
                "p95": round(float(np.percentile(tot_arr, 95)), 3)
            },
            "throughput_seq_per_sec": round(float(1000.0 / np.mean(tot_arr)), 2)
        }
    }

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY (Batch Size = 1):")
    print("=" * 70)
    print(f"Visual Encoder Latency:  Mean={results['benchmark']['visual_encoder_latency_ms']['mean']:.2f} ms | Median={results['benchmark']['visual_encoder_latency_ms']['median']:.2f} ms | P95={results['benchmark']['visual_encoder_latency_ms']['p95']:.2f} ms")
    print(f"Transformer Decoder:     Mean={results['benchmark']['decoder_latency_ms']['mean']:.2f} ms | Median={results['benchmark']['decoder_latency_ms']['median']:.2f} ms | P95={results['benchmark']['decoder_latency_ms']['p95']:.2f} ms")
    print(f"Total Pipeline Latency:  Mean={results['benchmark']['total_latency_ms']['mean']:.2f} ms | Median={results['benchmark']['total_latency_ms']['median']:.2f} ms | P95={results['benchmark']['total_latency_ms']['p95']:.2f} ms")
    print(f"Throughput:              {results['benchmark']['throughput_seq_per_sec']:.2f} sequences/sec")
    print("=" * 70)

    out_file = os.path.join(metrics_dir, "inference_benchmark.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark results to: {out_file}")
    return results


if __name__ == "__main__":
    benchmark_transformer()
