"""Unit tests for ConfidenceWeightedSmoother (Phase 4 Part 3).

Validates:
- Weighted evidence favors sustained sign over high-confidence noise outlier (Section 11)
- Recency decay gives greater weight to recent observations
- Tie / low evidence produces UNSTABLE
"""

import pytest

from src.realtime.prediction_history import PredictionHistory, PredictionRecord
from src.realtime.smoothing.confidence_weighted import ConfidenceWeightedSmoother
from src.realtime.smoothing.base import UNSTABLE_LABEL, UNSTABLE_CLASS_ID


def make_record(class_id: int, label: str, confidence: float, timestamp: float) -> PredictionRecord:
    return PredictionRecord(
        timestamp=timestamp,
        window_start=timestamp - 1.8,
        window_end=timestamp,
        class_id=class_id,
        label=label,
        gloss=label.upper(),
        confidence=confidence,
        input_quality=0.85,
    )


class TestConfidenceWeightedSmoother:
    def test_weighted_evidence_favors_sustained_sign(self):
        """Prompt Section 11 scenario:
        HELP 0.90, HELP 0.80, HELLO 0.95, HELP 0.85
        Weighted evidence favors HELP despite HELLO having single peak confidence.
        """
        smoother = ConfidenceWeightedSmoother(history_size=4, recency_decay=0.90, min_confidence=0.50)
        history = PredictionHistory(max_history=10)

        history.append(make_record(3, "help", 0.90, 1.0))
        history.append(make_record(3, "help", 0.80, 1.2))
        history.append(make_record(0, "hello", 0.95, 1.4))  # High-confidence noise outlier
        history.append(make_record(3, "help", 0.85, 1.6))

        smoothed = smoother.smooth(history)
        assert smoothed.is_stable is True
        assert smoothed.class_id == 3
        assert smoothed.label == "happy"

    def test_recency_decay_favors_recent_evidence(self):
        # 1 old 'hello' (id=0) vs 2 newer 'happy' (id=3)
        smoother = ConfidenceWeightedSmoother(history_size=3, recency_decay=0.70)
        history = PredictionHistory(max_history=10)

        history.append(make_record(0, "hello", 0.99, 1.0))  # Oldest
        history.append(make_record(3, "happy", 0.80, 1.2))
        history.append(make_record(3, "happy", 0.85, 1.4))   # Newest

        smoothed = smoother.smooth(history)
        assert smoothed.class_id == 3
        assert smoothed.label == "happy"


    def test_empty_or_subthreshold_records(self):
        smoother = ConfidenceWeightedSmoother(history_size=5, min_confidence=0.70)
        history = PredictionHistory(max_history=10)

        history.append(make_record(1, "thankyou", 0.40, 1.0))
        history.append(make_record(1, "thankyou", 0.50, 1.2))

        smoothed = smoother.smooth(history)
        assert smoothed.is_stable is False
        assert smoothed.class_id == UNSTABLE_CLASS_ID
