"""Unit tests for MajorityVoteSmoother (Phase 4 Part 3).

Validates:
- Majority consensus over stable sequence
- Rejection of ties as UNSTABLE
- Filtering of low-confidence predictions from vote count
- Behavior under noisy alternations
- Enforcement of min_votes quorum
"""

import pytest

from src.realtime.prediction_history import PredictionHistory, PredictionRecord
from src.realtime.smoothing.majority_vote import MajorityVoteSmoother
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


class TestMajorityVoteSmoother:
    def test_stable_sequence_majority(self):
        smoother = MajorityVoteSmoother(history_size=5, min_votes=3)
        history = PredictionHistory(max_history=10)

        # 4 out of 5 votes for 'help' (class_id=3)
        history.append(make_record(3, "help", 0.88, 1.0))
        history.append(make_record(3, "help", 0.90, 1.2))
        history.append(make_record(0, "hello", 0.70, 1.4))  # 1 transient noise
        history.append(make_record(3, "help", 0.85, 1.6))
        history.append(make_record(3, "help", 0.92, 1.8))

        smoothed = smoother.smooth(history)
        assert smoothed.is_stable is True
        assert smoothed.class_id == 3
        assert smoothed.label == "happy"
        assert smoothed.stability_score == 4 / 5  # 0.80


    def test_tie_produces_unstable(self):
        smoother = MajorityVoteSmoother(history_size=4, min_votes=2)
        history = PredictionHistory(max_history=10)

        # 2 votes for 'help' (id=3), 2 votes for 'hello' (id=0)
        history.append(make_record(3, "help", 0.85, 1.0))
        history.append(make_record(0, "hello", 0.85, 1.2))
        history.append(make_record(3, "help", 0.85, 1.4))
        history.append(make_record(0, "hello", 0.85, 1.6))

        smoothed = smoother.smooth(history)
        assert smoothed.is_stable is False
        assert smoothed.class_id == UNSTABLE_CLASS_ID
        assert smoothed.label == UNSTABLE_LABEL
        assert smoothed.details["reason"] == "tie_detected"

    def test_low_confidence_votes_ignored(self):
        smoother = MajorityVoteSmoother(history_size=5, min_votes=3, min_confidence=0.65)
        history = PredictionHistory(max_history=10)

        # Only 2 votes have confidence >= 0.65
        history.append(make_record(3, "help", 0.80, 1.0))
        history.append(make_record(3, "help", 0.50, 1.2))  # Below threshold
        history.append(make_record(3, "help", 0.55, 1.4))  # Below threshold
        history.append(make_record(3, "help", 0.85, 1.6))
        history.append(make_record(0, "hello", 0.40, 1.8)) # Below threshold

        smoothed = smoother.smooth(history)
        # Winner has only 2 eligible votes, but min_votes is 3 -> unstable
        assert smoothed.is_stable is False

    def test_empty_history_returns_unstable(self):
        smoother = MajorityVoteSmoother(history_size=5, min_votes=3)
        history = PredictionHistory(max_history=10)

        smoothed = smoother.smooth(history)
        assert smoothed.is_stable is False
        assert smoothed.class_id == UNSTABLE_CLASS_ID
