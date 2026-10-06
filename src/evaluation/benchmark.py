"""
SignTalk AI: Hardware-Aware Inference Latency and Resource Benchmarking.

Measures:
  - Total latency across warm-up and repeated evaluation runs
  - Mean, Median, Standard Deviation, P95, P99, Min, and Max Latency (ms)
  - Throughput (FPS)
  - Parameter Count (Total and Trainable)
  - Checkpoint Disk Footprint (MB)
  - Component Latency Breakdown: Preprocessing -> Forward Inference -> Post-processing
  - Hardware Environment (CPU, RAM, OS, PyTorch version, CUDA status)
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import os
import time
import platform
import json
import numpy as np
import torch
import torch.nn as nn


def get_hardware_environment() -> Dict[str, Any]:
    """Inspects and returns current execution hardware and runtime platform."""
    return {
        "os": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Returns (total_params, trainable_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark_model_inference(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    mask: Optional[torch.Tensor] = None,
    num_warmup: int = 10,
    num_iterations: int = 50,
    target_fps: float = 25.0
) -> Dict[str, Any]:
    """
    Runs repeated inference loops measuring latency distribution and throughput.
    
    Args:
        model: PyTorch module
        sample_input: Input tensor matching expected shape [B, C, T, V]
        device: Evaluation device
        mask: Optional validity mask
        num_warmup: Number of warm-up iterations
        num_iterations: Number of timed benchmark iterations
        target_fps: Real-time acquisition rate for latency thresholding
        
    Returns:
        Dictionary of timing statistics and resource counts.
    """
    model.eval()
    model.to(device)
    x = sample_input.to(device)
    m = mask.to(device) if mask is not None else None

    def _run_step():
        if hasattr(model, "generate") and "Translation" in model.__class__.__name__:
            return model.generate(x, mask=m, max_length=5)
        elif m is not None:
            return model(x, m)
        else:
            return model(x)

    # 1. Warm-up
    with torch.no_grad():
        for _ in range(num_warmup):
            _run_step()

    # 2. Timed Iterations
    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _run_step()
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

    latencies_arr = np.array(latencies_ms, dtype=np.float64)
    mean_lat = float(np.mean(latencies_arr))
    median_lat = float(np.median(latencies_arr))
    std_lat = float(np.std(latencies_arr))
    p95_lat = float(np.percentile(latencies_arr, 95))
    p99_lat = float(np.percentile(latencies_arr, 99))
    min_lat = float(np.min(latencies_arr))
    max_lat = float(np.max(latencies_arr))
    throughput_fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    total_params, trainable_params = count_parameters(model)

    return {
        "num_iterations": num_iterations,
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "p99_latency_ms": round(p99_lat, 2),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "throughput_fps": round(throughput_fps, 2),
        "real_time_budget_ms": round(1000.0 / target_fps, 2),
        "meets_realtime_target": bool(mean_lat <= (1000.0 / target_fps)),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    }
