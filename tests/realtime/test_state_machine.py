"""Unit tests for SignStateMachine (Phase 4 Part 3).

Validates:
- Lifecycle transitions: IDLE -> CANDIDATE -> ACTIVE -> ENDING -> IDLE
- Prompt Section 38 requirements:
  * HELP, HELP, HELP produces one stable HELP event
  * HELP, HELLO, HELP, UNKNOWN does not generate four sign events
  * HELP, HELP, WATER, WATER produces HELP, WATER
- Force end flush behavior
"""

import pytest

from src.realtime.sign_state_machine import SignStateMachine, SignState
from src.realtime.smoothing.base import SmoothedPrediction


def make_pred(
    class_id: int,
    label: str,
    confidence: float = 0.85,
    is_stable: bool = True,
    timestamp: float = 1.0,
    quality: float = 0.80,
) -> SmoothedPrediction:
    return SmoothedPrediction(
        class_id=class_id,
        label=label,
        gloss=label.upper(),
        translation=label.title(),
        confidence=confidence,
        stability_score=0.85 if is_stable else 0.20,
        is_stable=is_stable,
        timestamp=timestamp,
        method="majority_vote",
        input_quality=quality,
        window_start=timestamp - 1.8,
        window_end=timestamp,
    )


class TestSignStateMachine:
    def test_state_transitions_lifecycle(self):
        fsm = SignStateMachine(min_consecutive_predictions=2, min_confidence=0.65)
        assert fsm.state == SignState.IDLE

        # Step 1: Valid prediction enters CANDIDATE
        state1, evt1 = fsm.process(make_pred(0, "hello", timestamp=1.0))
        assert state1 == SignState.CANDIDATE
        assert evt1 is None
        assert fsm.consecutive_count == 1

        # Step 2: Matching prediction enters ACTIVE
        state2, evt2 = fsm.process(make_pred(0, "hello", timestamp=1.2))
        assert state2 == SignState.ACTIVE
        assert evt2 is None
        assert fsm.consecutive_count == 2

        # Step 3: Different class triggers ENDING & candidate for new sign
        state3, evt3 = fsm.process(make_pred(1, "thankyou", timestamp=1.4))
        # Finished the "hello" event!
        assert evt3 is not None
        assert evt3.label == "hello"
        assert evt3.class_id == 0
        assert evt3.start_time == pytest.approx(1.0 - 1.8, abs=0.01)
        assert state3 == SignState.CANDIDATE

    def test_section_38_stable_triple_prediction(self):
        """Prompt Section 38: HELP, HELP, HELP should produce one stable HELP event."""
        fsm = SignStateMachine(min_consecutive_predictions=2, min_confidence=0.65)

        fsm.process(make_pred(3, "help", timestamp=1.0))
        fsm.process(make_pred(3, "help", timestamp=1.2))
        fsm.process(make_pred(3, "help", timestamp=1.4))

        # Stream completes: force_end() finalizes the sustained gesture
        event = fsm.force_end()
        assert event is not None
        assert event.label == "help"
        assert event.class_id == 3
        assert event.duration_ms > 0

    def test_section_38_noisy_sequence_no_spurious_events(self):
        """Prompt Section 38: HELP, HELLO, HELP, UNKNOWN should not generate four sign events."""
        fsm = SignStateMachine(min_consecutive_predictions=2, min_confidence=0.65)

        events = []
        for cid, lbl, ts in [
            (3, "help", 1.0),
            (0, "hello", 1.2),
            (3, "help", 1.4),
            (-1, "UNKNOWN", 1.6),
        ]:
            _, evt = fsm.process(make_pred(cid, lbl, is_stable=(cid >= 0), timestamp=ts))
            if evt:
                events.append(evt)

        flush_evt = fsm.force_end()
        if flush_evt:
            events.append(flush_evt)

        # Zero events must be generated from this unstable noise
        assert len(events) == 0

    def test_section_38_two_distinct_signs_sequential(self):
        """Prompt Section 38: HELP, HELP, WATER, WATER should produce: HELP, WATER."""
        fsm = SignStateMachine(min_consecutive_predictions=2, min_confidence=0.65)

        events = []
        # HELP 1
        _, e1 = fsm.process(make_pred(3, "help", timestamp=1.0))
        if e1: events.append(e1)
        # HELP 2 (ACTIVE)
        _, e2 = fsm.process(make_pred(3, "help", timestamp=1.2))
        if e2: events.append(e2)

        # WATER 1 (Transition ends HELP, starts WATER CANDIDATE)
        _, e3 = fsm.process(make_pred(5, "water", timestamp=1.4))
        if e3: events.append(e3)

        # WATER 2 (WATER ACTIVE)
        _, e4 = fsm.process(make_pred(5, "water", timestamp=1.6))
        if e4: events.append(e4)

        # Stream concludes
        flush_evt = fsm.force_end()
        if flush_evt:
            events.append(flush_evt)

        labels = [e.label for e in events]
        assert labels == ["help", "water"]


class TestEmitOnConfirm:
    def _run(self, fsm, labels):
        events = []
        for i, (cid, lab) in enumerate(labels):
            p = make_pred(cid, lab, timestamp=1.0 + 0.2 * i) if cid >= 0 else make_pred(-1, "UNCERTAIN", is_stable=False, timestamp=1.0 + 0.2 * i)
            _, ev = fsm.process(p)
            events.append(ev.label if ev is not None else None)
        return events

    def test_emits_when_confirmed_not_at_end(self):
        fsm = SignStateMachine(min_consecutive_predictions=3, min_confidence=0.30, emit_on_confirm=True)
        seq = [(5, "hello")] * 5 + [(-1, "")] * 2
        events = self._run(fsm, seq)
        assert events[2] == "hello"           # third consecutive prediction confirms it
        assert [e for e in events if e] == ["hello"]  # no second emission when the sign ends
        assert fsm.force_end() is None

    def test_two_signs_each_emitted_once(self):
        fsm = SignStateMachine(min_consecutive_predictions=2, min_confidence=0.30, emit_on_confirm=True)
        seq = [(5, "hello")] * 3 + [(7, "thankyou")] * 3
        assert [e for e in self._run(fsm, seq) if e] == ["hello", "thankyou"]

    def test_default_still_emits_at_end(self):
        fsm = SignStateMachine(min_consecutive_predictions=2, min_confidence=0.30)
        events = self._run(fsm, [(5, "hello")] * 3 + [(-1, "")])
        assert events[:3] == [None, None, None] and events[3] == "hello"
