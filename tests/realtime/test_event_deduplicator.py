"""Unit tests for EventDeduplicator (Phase 4 Part 3).

Validates:
- Repeated same sign within minimum_gap_ms is suppressed
- Different sign class is immediately emitted
- Repeated sign after valid gap is emitted as distinct occurrence
- Ongoing continuation extends boundary of initial event
"""

import pytest

from src.realtime.event_deduplicator import EventDeduplicator
from src.realtime.sign_event import SignEvent


def make_event(
    class_id: int,
    label: str,
    start_time: float,
    end_time: float,
    confidence: float = 0.85,
) -> SignEvent:
    return SignEvent.create(
        class_id=class_id,
        label=label,
        start_time=start_time,
        end_time=end_time,
        confidence=confidence,
        input_quality=0.85,
        stability_score=0.90,
    )


class TestEventDeduplicator:
    def test_first_event_always_emits(self):
        dedup = EventDeduplicator(minimum_gap_ms=800.0)
        e1 = make_event(0, "hello", 1.0, 1.8)
        out = dedup.process(e1)

        assert out is not None
        assert out.label == "hello"
        assert dedup.total_emitted == 1
        assert dedup.total_suppressed == 0

    def test_repeated_same_sign_within_gap_suppressed(self):
        dedup = EventDeduplicator(minimum_gap_ms=800.0)

        # Event 1: [1.0s - 1.8s]
        e1 = make_event(0, "hello", 1.0, 1.8)
        dedup.process(e1)

        # Event 2: same sign, starts at 2.0s (gap is 0.2s = 200ms < 800ms)
        e2 = make_event(0, "hello", 2.0, 2.5)
        out2 = dedup.process(e2)

        assert out2 is None
        assert dedup.total_suppressed == 1
        # Boundary of last emitted event extended
        assert dedup.last_emitted_event.end_time == 2.5

    def test_different_sign_always_emits(self):
        dedup = EventDeduplicator(minimum_gap_ms=800.0)

        e1 = make_event(0, "hello", 1.0, 1.8)
        out1 = dedup.process(e1)
        assert out1 is not None

        # Different sign (class_id=1) shortly after
        e2 = make_event(1, "thankyou", 1.9, 2.7)
        out2 = dedup.process(e2)
        assert out2 is not None
        assert out2.label == "thankyou"
        assert dedup.total_emitted == 2

    def test_repeated_same_sign_after_valid_gap_emits(self):
        dedup = EventDeduplicator(minimum_gap_ms=800.0)

        # First instance: [1.0s - 1.8s]
        e1 = make_event(3, "help", 1.0, 1.8)
        dedup.process(e1)

        # Second instance: starts at 3.0s (gap is 1.2s = 1200ms >= 800ms)
        e2 = make_event(3, "help", 3.0, 3.8)
        out2 = dedup.process(e2)

        assert out2 is not None
        assert out2.label == "help"
        assert dedup.total_emitted == 2
        assert dedup.duplicate_rate == 0.0
