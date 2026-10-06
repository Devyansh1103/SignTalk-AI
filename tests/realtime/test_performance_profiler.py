"""
Tests for Real-Time Performance Profiler (Phase 4 Part 4).
Validates microsecond stage timing, statistical calculations, memory tracking, and reset behavior.
"""

import time
import pytest
import numpy as np

from src.realtime.performance import PerformanceProfiler, StageTimer


class TestPerformanceProfiler:
    def test_profiler_initialization(self):
        profiler = PerformanceProfiler(history_size=100, enabled=True)
        assert profiler.history_size == 100
        assert profiler.enabled is True
        summary = profiler.get_summary()
        assert summary["sample_count"] == 0

    def test_record_stage_and_statistics(self):
        profiler = PerformanceProfiler(history_size=50, enabled=True)
        # Record 10 deterministic latencies: 10, 20, 30, ... 100
        for i in range(1, 11):
            profiler.record_stage("stgcn_inference", float(i * 10.0))

        stats = profiler.get_stage_stats("stgcn_inference")
        assert stats["count"] == 10
        assert stats["min"] == 10.0
        assert stats["max"] == 100.0
        assert stats["mean"] == 55.0
        assert stats["median"] == 55.0
        assert stats["p90"] == 91.0 or stats["p90"] == 90.0 or stats["p90"] >= 85.0

    def test_stage_timer_context_manager(self):
        profiler = PerformanceProfiler(history_size=50, enabled=True)
        with profiler.time_stage("capture"):
            time.sleep(0.01)  # 10ms

        stats = profiler.get_stage_stats("capture")
        assert stats["count"] == 1
        assert stats["mean"] >= 5.0  # At least 5ms

    def test_memory_metrics(self):
        profiler = PerformanceProfiler()
        mem = profiler.get_memory_metrics()
        assert isinstance(mem, dict)
        assert "ram_rss_mb" in mem
        assert mem["ram_rss_mb"] >= 0.0

    def test_reset_behavior(self):
        profiler = PerformanceProfiler()
        profiler.record_stage("capture", 15.0)
        profiler.record_stage("end_to_end", 30.0)
        assert profiler.get_stage_stats("capture")["count"] == 1

        profiler.reset()
        assert profiler.get_stage_stats("capture")["count"] == 0
        assert profiler.get_summary()["sample_count"] == 0
