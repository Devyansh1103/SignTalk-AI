"""Performance Benchmark & Latency Breakdown (Phase 4 Part 3).

Measures detailed latency profiles (mean, median, p95, max) for:
- ST-GCN inference
- Confidence filtering
- Prediction history management
- Temporal smoothing (Majority vote, Confidence weighted, Temporal stability)
- Stability state machine
- Event deduplication
- Total pipeline latency breakdown
"""

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.realtime.confidence_filter import ConfidenceFilter
from src.realtime.event_deduplicator import EventDeduplicator
from src.realtime.model_runner import STGCNRunner
from src.realtime.prediction_history import PredictionHistory, PredictionRecord
from src.realtime.sign_event import SignEvent
from src.realtime.sign_state_machine import SignStateMachine
from src.realtime.smoothing import (
    ConfidenceWeightedSmoother,
    MajorityVoteSmoother,
    TemporalStabilitySmoother,
)
from src.realtime.temporal_buffer import TemporalBuffer
from src.realtime.types import LandmarkFrame, ModalityDetection, QualityReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BenchmarkTemporalPipeline")

CHECKPOINT_PATH = "experiments/stgcn/checkpoints/best_checkpoint.pt"


def compute_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute mean, median, p95, max in milliseconds."""
    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(arr)), 4),
        "median_ms": round(float(np.median(arr)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "max_ms": round(float(np.max(arr)), 4),
        "min_ms": round(float(np.min(arr)), 4),
    }


def run_benchmark(num_iterations: int = 100, output_json: str = "results/temporal_stability/benchmark_latency.json") -> Dict[str, Any]:
    logger.info("Initializing components for benchmark (%d iterations)...", num_iterations)
    runner = STGCNRunner(checkpoint_path=CHECKPOINT_PATH, device="cpu", warmup_iterations=5)

    cfilter = ConfidenceFilter(confidence_threshold=0.65, min_input_quality=0.40)
    history = PredictionHistory(max_history=20)
    mv_smoother = MajorityVoteSmoother(history_size=5, min_votes=3)
    cw_smoother = ConfidenceWeightedSmoother(history_size=5, recency_decay=0.85)
    ts_smoother = TemporalStabilitySmoother(min_consecutive=2)
    fsm = SignStateMachine(min_consecutive_predictions=2, min_confidence=0.65)
    dedup = EventDeduplicator(minimum_gap_ms=800.0)

    # Synthetic window
    buf = TemporalBuffer(45)
    for i in range(45):
        coords = np.random.randn(93, 3).astype(np.float32)
        buf.append(LandmarkFrame(
            frame_id=i,
            timestamp=i * 0.04,
            raw_coords=coords,
            normalized_coords=coords,
            mask=np.ones(93, bool),
            visibility=np.ones(93, np.float32),
            modality_stats=ModalityDetection(pose_detected=True, left_hand_detected=True, right_hand_detected=True),
            quality=QualityReport(is_valid=True, quality_score=0.85, classification="GOOD", missing_ratio=0.1, has_nan_or_inf=False, coordinate_in_range=True, sudden_jumps_detected=False),
        ))
    window = buf.get_window()

    stgcn_lats = []
    filter_lats = []
    hist_lats = []
    mv_lats = []
    cw_lats = []
    ts_lats = []
    fsm_lats = []
    dedup_lats = []
    pure_overhead_lats = []

    # Warm-up
    for _ in range(10):
        _ = runner.predict_window(window)

    logger.info("Running timed benchmark passes...")
    for step in range(num_iterations):
        # 1. ST-GCN
        t0 = time.perf_counter()
        pred = runner.predict_window(window)
        stgcn_lats.append((time.perf_counter() - t0) * 1000.0)

        # 2. Confidence Filter
        t0 = time.perf_counter()
        filt = cfilter.filter(pred)
        filter_lats.append((time.perf_counter() - t0) * 1000.0)

        # 3. History Append
        rec = PredictionRecord(
            timestamp=pred.timestamp,
            window_start=pred.window_start_time,
            window_end=pred.window_end_time,
            class_id=filt.class_id,
            label=filt.label,
            confidence=filt.confidence,
            input_quality=filt.input_quality,
        )
        t0 = time.perf_counter()
        history.append(rec)
        hist_lats.append((time.perf_counter() - t0) * 1000.0)

        # 4. Smoothing methods
        t0 = time.perf_counter()
        sm_mv = mv_smoother.smooth(history)
        mv_lats.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        sm_cw = cw_smoother.smooth(history)
        cw_lats.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        sm_ts = ts_smoother.smooth(history)
        ts_lats.append((time.perf_counter() - t0) * 1000.0)

        # 5. State Machine
        t0 = time.perf_counter()
        state, completed = fsm.process(sm_mv)
        fsm_lats.append((time.perf_counter() - t0) * 1000.0)

        # 6. Deduplication
        fake_evt = SignEvent.create(class_id=pred.class_id, label=pred.label, start_time=1.0, end_time=2.0, confidence=0.85, input_quality=0.85, stability_score=0.9)
        t0 = time.perf_counter()
        _ = dedup.process(fake_evt)
        dedup_lats.append((time.perf_counter() - t0) * 1000.0)

        # Pure post-STGCN processing overhead (Filter + History + MV + FSM + Dedup)
        pure_overhead = (
            filter_lats[-1] + hist_lats[-1] + mv_lats[-1] + fsm_lats[-1] + dedup_lats[-1]
        )
        pure_overhead_lats.append(pure_overhead)

    benchmark_results = {
        "num_iterations": num_iterations,
        "device": "cpu",
        "stgcn_inference": compute_stats(stgcn_lats),
        "confidence_filter": compute_stats(filter_lats),
        "prediction_history": compute_stats(hist_lats),
        "smoothing_majority_vote": compute_stats(mv_lats),
        "smoothing_confidence_weighted": compute_stats(cw_lats),
        "smoothing_temporal_stability": compute_stats(ts_lats),
        "sign_state_machine": compute_stats(fsm_lats),
        "event_deduplication": compute_stats(dedup_lats),
        "total_added_overhead_post_stgcn": compute_stats(pure_overhead_lats),
        "total_end_to_end_pipeline_ms": {
            "mean_ms": round(compute_stats(stgcn_lats)["mean_ms"] + compute_stats(pure_overhead_lats)["mean_ms"], 4),
            "median_ms": round(compute_stats(stgcn_lats)["median_ms"] + compute_stats(pure_overhead_lats)["median_ms"], 4),
            "p95_ms": round(compute_stats(stgcn_lats)["p95_ms"] + compute_stats(pure_overhead_lats)["p95_ms"], 4),
            "max_ms": round(compute_stats(stgcn_lats)["max_ms"] + compute_stats(pure_overhead_lats)["max_ms"], 4),
        },
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    logger.info("Saved benchmark results to %s:\n%s", output_json, json.dumps(benchmark_results, indent=2))
    return benchmark_results


if __name__ == "__main__":
    run_benchmark(num_iterations=100)
