"""Unit tests for ConfidenceFilter (Phase 4 Part 3).

Validates:
- Predictions above threshold are accepted
- Predictions below threshold are marked UNCERTAIN
- Predictions exactly at threshold are accepted
- Predictions with low input quality are rejected
- Model confidence and landmark quality remain decoupled
"""

import pytest
import numpy as np

from src.realtime.confidence_filter import (
    ConfidenceFilter,
    FilteredPrediction,
    UNCERTAIN_LABEL,
    UNCERTAIN_CLASS_ID,
)
from src.realtime.prediction_history import PredictionRecord


def make_record(
    class_id: int = 0,
    label: str = "hello",
    confidence: float = 0.85,
    input_quality: float = 0.90,
    timestamp: float = 1.0,
) -> PredictionRecord:
    return PredictionRecord(
        timestamp=timestamp,
        window_start=timestamp - 1.8,
        window_end=timestamp,
        class_id=class_id,
        label=label,
        gloss=label.upper(),
        confidence=confidence,
        input_quality=input_quality,
        inference_latency_ms=15.0,
        is_valid_quality=True,
    )


class TestConfidenceFilter:
    def test_above_threshold_accepted(self):
        cfilter = ConfidenceFilter(confidence_threshold=0.65, min_input_quality=0.40)
        rec = make_record(class_id=2, label="good", confidence=0.88, input_quality=0.90)
        filtered = cfilter.filter(rec)

        assert filtered.is_accepted is True
        assert filtered.class_id == 2
        assert filtered.label == "good"
        assert filtered.confidence == 0.88
        assert filtered.input_quality == 0.90
        assert filtered.rejection_reason is None

    def test_below_threshold_marked_uncertain(self):
        cfilter = ConfidenceFilter(confidence_threshold=0.65, min_input_quality=0.40)
        rec = make_record(class_id=2, label="good", confidence=0.55, input_quality=0.90)
        filtered = cfilter.filter(rec)

        assert filtered.is_accepted is False
        assert filtered.class_id == UNCERTAIN_CLASS_ID
        assert filtered.label == UNCERTAIN_LABEL
        assert filtered.confidence == 0.55
        assert filtered.input_quality == 0.90
        assert filtered.rejection_reason == "low_confidence"

    def test_exact_threshold_accepted(self):
        cfilter = ConfidenceFilter(confidence_threshold=0.65, min_input_quality=0.40)
        rec = make_record(class_id=1, label="thankyou", confidence=0.65, input_quality=0.50)
        filtered = cfilter.filter(rec)

        assert filtered.is_accepted is True
        assert filtered.class_id == 1
        assert filtered.label == "thankyou"

    def test_low_quality_rejected_even_with_high_confidence(self):
        cfilter = ConfidenceFilter(confidence_threshold=0.65, min_input_quality=0.40)
        rec = make_record(class_id=0, label="hello", confidence=0.98, input_quality=0.25)
        filtered = cfilter.filter(rec)

        assert filtered.is_accepted is False
        assert filtered.class_id == UNCERTAIN_CLASS_ID
        assert filtered.rejection_reason == "low_input_quality"
        # Confidence score must be preserved without mathematical distortion
        assert filtered.confidence == 0.98
        assert filtered.input_quality == 0.25

    def test_filter_prediction_alias(self):
        cfilter = ConfidenceFilter(confidence_threshold=0.70)
        rec = make_record(confidence=0.80)
        res1 = cfilter.filter(rec)
        res2 = cfilter.filter_prediction(rec)
        assert res1.is_accepted == res2.is_accepted
