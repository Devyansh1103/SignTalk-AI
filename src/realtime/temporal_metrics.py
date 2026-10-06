"""Temporal stability and event evaluation metrics.

Calculates temporal stability metrics including:
- Prediction flip rate (rapid A -> B -> A alternations)
- Stability duration (run lengths of contiguous predictions)
- Duplicate rate (proportions of duplicate vs unique events)
- Event detection delay (observation to emission latency)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.realtime.sign_event import SignEvent


def compute_prediction_flip_rate(predictions: Sequence[Union[str, int]]) -> float:
    """Calculate the flip rate: frequency of rapid A -> B -> A reversals.

    Args:
        predictions: Sequence of prediction labels or class IDs.

    Returns:
        Fraction of 3-step triplets exhibiting an A -> B -> A flip pattern.
        Returns 0.0 if sequence length < 3.
    """
    if len(predictions) < 3:
        return 0.0

    flips = 0
    total_triplets = len(predictions) - 2

    for i in range(total_triplets):
        a = predictions[i]
        b = predictions[i + 1]
        c = predictions[i + 2]
        if a != b and a == c:
            flips += 1

    return float(flips / total_triplets)


def compute_stability_durations(
    predictions: Sequence[Union[str, int]],
    step_duration_s: float = 0.20,
) -> Dict[str, float]:
    """Compute run lengths (stability durations) of identical contiguous predictions.

    Args:
        predictions: Sequence of prediction labels or class IDs.
        step_duration_s: Duration in seconds of each prediction step (e.g. 5 frames @ 25fps = 0.20s).

    Returns:
        Dictionary with mean_s, median_s, max_s, min_s of run durations.
    """
    if not predictions:
        return {"mean_s": 0.0, "median_s": 0.0, "max_s": 0.0, "min_s": 0.0}

    runs: List[int] = []
    current_run = 1

    for i in range(1, len(predictions)):
        if predictions[i] == predictions[i - 1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)

    run_durations_s = [r * step_duration_s for r in runs]
    return {
        "mean_s": float(np.mean(run_durations_s)),
        "median_s": float(np.median(run_durations_s)),
        "max_s": float(np.max(run_durations_s)),
        "min_s": float(np.min(run_durations_s)),
    }


def compute_duplicate_rate(
    total_candidate_events: int,
    emitted_events_count: int,
) -> float:
    """Compute event duplicate suppression rate.

    Args:
        total_candidate_events: Total events received before deduplication.
        emitted_events_count: Events permitted after deduplication.

    Returns:
        Proportion of candidates suppressed as duplicates.
    """
    if total_candidate_events == 0:
        return 0.0
    suppressed = max(0, total_candidate_events - emitted_events_count)
    return float(suppressed / total_candidate_events)


def compute_event_detection_latency(
    events: Sequence[SignEvent],
) -> Dict[str, float]:
    """Compute summary statistics for event durations / observation delays.

    Args:
        events: Sequence of detected SignEvents.

    Returns:
        Dictionary with mean_ms, median_ms, p95_ms, max_ms duration.
    """
    if not events:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }

    durations = [e.duration_ms for e in events]
    return {
        "mean_ms": float(np.mean(durations)),
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "max_ms": float(np.max(durations)),
    }


def evaluate_temporal_pipeline(
    raw_predictions: Sequence[Union[str, int]],
    smoothed_predictions: Sequence[Union[str, int]],
    events: Sequence[SignEvent],
    raw_candidate_count: int,
    step_duration_s: float = 0.20,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Calculate and export complete temporal stability metrics report.

    Args:
        raw_predictions: Sequence of raw class IDs or labels.
        smoothed_predictions: Sequence of smoothed class IDs or labels.
        events: Emitted SignEvents.
        raw_candidate_count: Candidate events evaluated by deduplicator.
        step_duration_s: Stride duration in seconds.
        output_path: Optional path to save JSON metrics.

    Returns:
        Structured dictionary of metrics.
    """
    raw_flip_rate = compute_prediction_flip_rate(raw_predictions)
    smoothed_flip_rate = compute_prediction_flip_rate(smoothed_predictions)

    stability_durations = compute_stability_durations(
        smoothed_predictions, step_duration_s=step_duration_s
    )
    duplicate_rate = compute_duplicate_rate(
        raw_candidate_count, len(events)
    )
    event_latencies = compute_event_detection_latency(events)

    metrics = {
        "flip_rate": {
            "raw": round(raw_flip_rate, 4),
            "smoothed": round(smoothed_flip_rate, 4),
            "flip_reduction_percent": round(
                ((raw_flip_rate - smoothed_flip_rate) / raw_flip_rate * 100.0)
                if raw_flip_rate > 0
                else 0.0,
                2,
            ),
        },
        "duplicate_rate": round(duplicate_rate, 4),
        "total_candidates": raw_candidate_count,
        "emitted_events_count": len(events),
        "stability_duration_s": {
            k: round(v, 4) for k, v in stability_durations.items()
        },
        "event_latency_ms": {
            k: round(v, 2) for k, v in event_latencies.items()
        },
        "formal_evaluation_note": (
            "Formal event-level evaluation (precision, recall, F1 against continuous ground truth) "
            "is not currently supported by the available annotations because the dataset contains "
            "isolated sign sequences."
        ),
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return metrics
