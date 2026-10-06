"""
SignTalk AI - Real-Time Inference Micro-Benchmark
Phase 4 Part 2: Sliding-Window Inference & Temporal Prediction.

Measures latency across pipeline stages over N windows:
  1. Temporal buffer append / update
  2. Tensor conversion (LandmarkFrame window -> [1, 3, 45, 93])
  3. ST-GCN model inference forward pass
  4. Softmax probability computation
  5. Prediction formatting & Top-K calculation
  6. Total end-to-end temporal prediction cycle
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.realtime.temporal_buffer import TemporalBuffer
from src.realtime.model_runner import STGCNRunner
from src.realtime.types import LandmarkFrame, ModalityDetection, QualityReport
from src.data.node_schema import TOTAL_NODES, CHANNELS, TARGET_SEQUENCE_LENGTH
from src.data.label_map import NUM_CLASSES


def generate_synthetic_window(t_length: int = 45):
    window = []
    for i in range(t_length):
        coords = np.random.randn(TOTAL_NODES, CHANNELS).astype(np.float32)
        norm = np.random.randn(TOTAL_NODES, CHANNELS).astype(np.float32)
        mask = np.ones((TOTAL_NODES,), dtype=bool)
        vis = np.ones((TOTAL_NODES,), dtype=np.float32)
        mod = ModalityDetection(True, True, True, False)
        qual = QualityReport(True, 0.92, "GOOD", 0.05, False, True, False)

        lf = LandmarkFrame(
            frame_id=i,
            timestamp=i * 0.04,
            raw_coords=coords,
            normalized_coords=norm,
            mask=mask,
            visibility=vis,
            modality_stats=mod,
            quality=qual
        )
        window.append(lf)
    return window


def run_benchmark(num_iterations: int = 100, device: str = "cpu"):
    print("=" * 65)
    print("  SignTalk AI — ST-GCN Sliding Window Inference Benchmark")
    print("=" * 65)
    print(f"Device               : {device.upper()}")
    print(f"Sequence Length (T)  : {TARGET_SEQUENCE_LENGTH} frames")
    print(f"Total Nodes (V)      : {TOTAL_NODES} nodes")
    print(f"Feature Channels (C) : {CHANNELS} (X, Y, Z)")
    print(f"Benchmark Windows    : {num_iterations}")
    print("=" * 65)

    runner = STGCNRunner(
        checkpoint_path="experiments/stgcn/checkpoints/best_checkpoint.pt",
        device=device,
        top_k=3,
        warmup_iterations=10
    )

    buffer = TemporalBuffer(max_length=TARGET_SEQUENCE_LENGTH)

    t_buffer_update = []
    t_tensor_conv = []
    t_inference = []
    t_softmax = []
    t_formatting = []
    t_total = []

    # Prime buffer
    initial_window = generate_synthetic_window(TARGET_SEQUENCE_LENGTH)
    for f in initial_window:
        buffer.append(f)

    for it in range(num_iterations):
        t_start_total = time.perf_counter()

        # 1. Temporal buffer append (simulating 1 frame slide)
        new_frame = generate_synthetic_window(1)[0]
        t0 = time.perf_counter()
        buffer.append(new_frame)
        window = buffer.get_window()
        t1 = time.perf_counter()
        t_buffer_update.append((t1 - t0) * 1000.0)

        # 2. Tensor conversion
        t0 = time.perf_counter()
        x, mask = runner.window_to_tensor(window)
        t1 = time.perf_counter()
        t_tensor_conv.append((t1 - t0) * 1000.0)

        # 3. Model forward pass
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = runner.model(x, mask)
        t1 = time.perf_counter()
        t_inference.append((t1 - t0) * 1000.0)

        # 4. Softmax
        t0 = time.perf_counter()
        probs = torch.softmax(logits, dim=-1)
        t1 = time.perf_counter()
        t_softmax.append((t1 - t0) * 1000.0)

        # 5. Prediction formatting & Top-K
        t0 = time.perf_counter()
        probs_np = probs[0].cpu().numpy()
        pred_cid = int(np.argmax(probs_np))
        top_indices = np.argsort(probs_np)[::-1][:3]
        _ = [(int(idx), float(probs_np[idx])) for idx in top_indices]
        t1 = time.perf_counter()
        t_formatting.append((t1 - t0) * 1000.0)

        t_end_total = time.perf_counter()
        t_total.append((t_end_total - t_start_total) * 1000.0)

    stages = [
        ("Temporal Buffer Update", t_buffer_update),
        ("Tensor Conversion", t_tensor_conv),
        ("ST-GCN Forward Pass", t_inference),
        ("Softmax Computation", t_softmax),
        ("Prediction Formatting (Top-3)", t_formatting),
        ("Total Prediction Processing", t_total)
    ]

    print(f"\n{'Stage':<32} | {'Mean (ms)':<9} | {'Median':<9} | {'P95 (ms)':<9} | {'Max (ms)':<9}")
    print("-" * 75)
    for name, data in stages:
        arr = np.array(data)
        print(
            f"{name:<32} | {np.mean(arr):>8.2f}  | {np.median(arr):>8.2f}  | {np.percentile(arr, 95):>8.2f}  | {np.max(arr):>8.2f}"
        )

    print("-" * 75)
    inference_fps = 1000.0 / np.mean(t_inference)
    throughput_fps = 1000.0 / np.mean(t_total)
    print(f"Pure ST-GCN Inference Throughput : {inference_fps:.1f} windows/second")
    print(f"End-to-End Prediction Throughput : {throughput_fps:.1f} windows/second")
    print(f"Temporal Observation Window      : {TARGET_SEQUENCE_LENGTH / 25.0:.2f} s ({TARGET_SEQUENCE_LENGTH} frames @ 25 FPS)")
    print(f"Cadence at Stride=5              : {5.0 / 25.0 * 1000.0:.0f} ms update period (~5.0 predictions/sec)")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmark(num_iterations=100, device="cpu")
