"""Unit tests for Temporal Metrics module (Phase 4 Part 3).

Validates:
- compute_prediction_flip_rate on rapid A -> B -> A alternations
- compute_stability_durations on contiguous run lengths
- compute_duplicate_rate
- compute_event_detection_latency
- evaluate_temporal_pipeline JSON report creation
"""

import tempfile
from pathlib import Path
import pytest

from src.realtime.sign_event import SignEvent
from src.realtime.temporal_metrics import (
    compute_prediction_flip_rate,
    compute_stability_durations,
    compute_duplicate_rate,
    compute_event_detection_latency,
    evaluate_temporal_pipeline,
)


class TestTemporalMetrics:
    def test_flip_rate_calculation(self):
        # A -> B -> A pattern:
        # Sequence: [A, B, A] -> 1 flip in 1 triplet = 1.0
        seq = ["A", "B", "A"]
        assert compute_prediction_flip_rate(seq) == 1.0

        # Sequence: [A, A, A, A] -> 0 flips
        assert compute_prediction_flip_rate(["A", "A", "A", "A"]) == 0.0

        # Sequence: [A, B, A, B, A] -> 3 flips in 3 triplets = 1.0
        assert compute_prediction_flip_rate(["A", "B", "A", "B", "A"]) == 1.0

        # Sequence length < 3 -> 0.0
        assert compute_prediction_flip_rate(["A", "B"]) == 0.0

    def test_stability_durations(self):
        # 3 steps of A, 2 steps of B, step_duration = 0.2s
        seq = ["A", "A", "A", "B", "B"]
        dur = compute_stability_durations(seq, step_duration_s=0.20)

        # Runs are [3, 2] -> durations [0.6s, 0.4s]
        assert dur["mean_s"] == pytest.approx(0.50, abs=0.01)
        assert dur["max_s"] == pytest.approx(0.60, abs=0.01)
        assert dur["min_s"] == pytest.approx(0.40, abs=0.01)

    def test_duplicate_rate(self):
        # 10 candidate events, 4 emitted -> 6 duplicates (60%)
        rate = compute_duplicate_rate(10, 4)
        assert rate == 0.60

        # 0 candidate events -> 0.0
        assert compute_duplicate_rate(0, 0) == 0.0

    def test_event_latency(self):
        e1 = SignEvent.create(0, "hello", 1.0, 2.0, 0.9, 0.8, 0.9)  # duration 1000ms
        e2 = SignEvent.create(1, "thankyou", 3.0, 4.5, 0.9, 0.8, 0.9)  # duration 1500ms

        lat = compute_event_detection_latency([e1, e2])
        assert lat["mean_ms"] == pytest.approx(1250.0, abs=1.0)
        assert lat["max_ms"] == pytest.approx(1500.0, abs=1.0)

    def test_evaluate_temporal_pipeline_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "metrics.json"
            raw = ["A", "B", "A", "A"]
            smoothed = ["A", "A", "A", "A"]
            events = [SignEvent.create(0, "A", 1.0, 2.0, 0.9, 0.8, 0.9)]

            rep = evaluate_temporal_pipeline(
                raw_predictions=raw,
                smoothed_predictions=smoothed,
                events=events,
                raw_candidate_count=2,
                step_duration_s=0.20,
                output_path=json_path,
            )

            assert json_path.exists()
            assert rep["flip_rate"]["raw"] > 0
            assert rep["flip_rate"]["smoothed"] == 0
            assert rep["emitted_events_count"] == 1
            assert "formal_evaluation_note" in rep
