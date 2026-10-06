"""Unit tests for SignSequenceBuffer (Phase 4 Part 3).

Validates:
- Appending SignEvents
- Clearing buffer
- Bounded maximum length enforcement
- Formatting human-readable timeline
- Formatting concatenated gloss string
- Saving events to CSV
"""

import tempfile
from pathlib import Path
import pytest

from src.realtime.sign_event import SignEvent
from src.realtime.sign_sequence import SignSequenceBuffer


def make_event(class_id: int, label: str, start: float, end: float) -> SignEvent:
    return SignEvent.create(
        class_id=class_id,
        label=label,
        start_time=start,
        end_time=end,
        confidence=0.88,
        input_quality=0.80,
        stability_score=0.90,
    )


class TestSignSequenceBuffer:
    def test_append_and_retrieve(self):
        buf = SignSequenceBuffer(max_events=10)
        e1 = make_event(0, "hello", 1.0, 1.8)
        e2 = make_event(1, "thankyou", 2.2, 3.0)

        buf.append_event(e1)
        buf.append_event(e2)

        assert len(buf) == 2
        assert buf.get_labels() == ["hello", "thankyou"]
        assert buf.last_event().label == "thankyou"

    def test_bounded_max_length(self):
        buf = SignSequenceBuffer(max_events=3)
        for i in range(5):
            buf.append_event(make_event(i, f"sign_{i}", float(i), float(i) + 0.8))

        assert len(buf) == 3
        # Should retain only the 3 most recent
        assert buf.get_labels() == ["sign_2", "sign_3", "sign_4"]

    def test_clear_buffer(self):
        buf = SignSequenceBuffer(max_events=10)
        buf.append_event(make_event(0, "hello", 1.0, 1.8))
        assert len(buf) == 1

        buf.clear()
        assert len(buf) == 0
        assert buf.last_event() is None

    def test_timeline_formatting(self):
        buf = SignSequenceBuffer(max_events=10)
        buf.append_event(make_event(3, "help", 12.2, 13.1))
        buf.append_event(make_event(5, "water", 13.5, 14.3))

        timeline = buf.format_timeline()
        assert "HELP" in timeline
        assert "start: 12.2s" in timeline
        assert "end: 13.1s" in timeline
        assert "WATER" in timeline
        assert "start: 13.5s" in timeline
        assert "end: 14.3s" in timeline

    def test_gloss_string_formatting(self):
        buf = SignSequenceBuffer(max_events=10)
        buf.append_event(make_event(3, "help", 1.0, 1.8))
        buf.append_event(make_event(5, "water", 2.0, 2.8))
        buf.append_event(make_event(7, "hospital", 3.0, 3.8))

        gloss_str = buf.format_gloss_string()
        assert gloss_str == "HELP -> WATER -> HOSPITAL"

    def test_save_to_csv(self):
        buf = SignSequenceBuffer(max_events=10)
        buf.append_event(make_event(0, "hello", 1.0, 1.8))

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_events.csv"
            buf.save_to_csv(csv_path)

            assert csv_path.exists()
            content = csv_path.read_text(encoding="utf-8")
            assert "event_id,label,class_id" in content
            assert "hello" in content
