"""
SignTalk AI - High-Resolution Real-Time Performance Profiler
Phase 4 Part 4: End-to-End Pipeline Timing, Resource Monitoring & Latency Statistics.

Tracks microsecond-resolution stage timings, calculates empirical latency
percentiles (mean, median, p50, p90, p95, p99, min, max, std), and monitors
system RAM and GPU memory utilization.
"""

import time
import os
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class StageTimer:
    """Context manager for microsecond-resolution timing of an individual pipeline stage."""

    def __init__(self, profiler: "PerformanceProfiler", stage_name: str):
        self.profiler = profiler
        self.stage_name = stage_name
        self.t_start = 0.0

    def __enter__(self):
        self.t_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.t_start) * 1000.0
        self.profiler.record_stage(self.stage_name, duration_ms)


class PerformanceProfiler:
    """
    Thread-safe performance profiler recording granular stage latencies,
    evaluating empirical statistics over rolling or accumulated samples,
    and monitoring host and GPU memory footprints.
    """

    KNOWN_STAGES = [
        "capture",
        "frame_preprocess",
        "mediapipe",
        "normalization",
        "temporal_buffer",
        "tensor_prep",
        "stgcn_inference",
        "confidence_filter",
        "smoothing",
        "state_machine",
        "deduplicator",
        "sequence",
        "translation",
        "display_render",
        "end_to_end"
    ]

    def __init__(self, history_size: int = 500, enabled: bool = True):
        """
        Args:
            history_size: Maximum rolling measurement entries retained per stage.
            enabled: If False, measurements are bypassed for minimal overhead.
        """
        self.history_size = max(10, int(history_size))
        self.enabled = enabled
        self._stage_samples: Dict[str, deque] = {
            stage: deque(maxlen=self.history_size) for stage in self.KNOWN_STAGES
        }
        self._process = psutil.Process(os.getpid()) if HAS_PSUTIL else None
        self._sample_count = 0
        self._start_time = time.time()

    def time_stage(self, stage_name: str) -> StageTimer:
        """Returns a StageTimer context manager for timing a block."""
        return StageTimer(self, stage_name)

    def record_stage(self, stage_name: str, duration_ms: float) -> None:
        """Records a duration in milliseconds for a designated stage."""
        if not self.enabled:
            return

        if stage_name not in self._stage_samples:
            self._stage_samples[stage_name] = deque(maxlen=self.history_size)

        self._stage_samples[stage_name].append(max(0.0, float(duration_ms)))
        if stage_name == "end_to_end":
            self._sample_count += 1

    def get_stage_stats(self, stage_name: str) -> Dict[str, float]:
        """Calculates comprehensive latency statistics for a given stage."""
        samples = list(self._stage_samples.get(stage_name, []))
        if not samples:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
            }

        arr = np.array(samples, dtype=np.float64)
        return {
            "count": int(len(arr)),
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p90": round(float(np.percentile(arr, 90)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "p99": round(float(np.percentile(arr, 99)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "std": round(float(np.std(arr)), 3),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Returns statistical summaries for all tracked stages."""
        result = {}
        for stage in self._stage_samples:
            if len(self._stage_samples[stage]) > 0:
                result[stage] = self.get_stage_stats(stage)
        return result

    def get_memory_metrics(self) -> Dict[str, Any]:
        """Returns current host RAM and GPU memory metrics."""
        metrics: Dict[str, Any] = {
            "ram_rss_mb": 0.0,
            "ram_vms_mb": 0.0,
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "gpu_available": False,
        }

        if self._process is not None:
            try:
                mem_info = self._process.memory_info()
                metrics["ram_rss_mb"] = round(mem_info.rss / (1024 * 1024), 2)
                metrics["ram_vms_mb"] = round(mem_info.vms / (1024 * 1024), 2)
            except Exception:
                pass

        if HAS_TORCH and torch.cuda.is_available():
            metrics["gpu_available"] = True
            metrics["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
            metrics["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)

        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Returns an integrated performance summary containing latencies and memory."""
        all_stats = self.get_all_stats()
        mem = self.get_memory_metrics()
        e2e = all_stats.get("end_to_end", {})
        fps = (1000.0 / e2e["mean"]) if e2e.get("mean", 0.0) > 0 else 0.0

        return {
            "sample_count": self._sample_count,
            "elapsed_sec": round(time.time() - self._start_time, 2),
            "estimated_fps": round(fps, 2),
            "stages": all_stats,
            "memory": mem,
        }

    def reset(self) -> None:
        """Clears all accumulated measurement queues."""
        for q in self._stage_samples.values():
            q.clear()
        self._sample_count = 0
        self._start_time = time.time()
