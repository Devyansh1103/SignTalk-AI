"""
Tests for Real-Time Translation & Linguistic Layer (Phase 4 Part 4).
Validates bilingual translation (English and Hindi), phrase template matching,
pause-based sentence finalization, and live transcript tracking.
"""

import time
import pytest

from src.realtime.translator import RealTimeTranslator, TranslationResult
from src.realtime.sign_event import SignEvent


class TestRealTimeTranslator:
    def test_translator_initialization(self):
        translator = RealTimeTranslator(pause_threshold_sec=1.5, min_confidence=0.50)
        assert translator.pause_threshold_sec == 1.5
        assert translator.min_confidence == 0.50
        assert len(translator.finalized_phrases) == 0

    def test_single_sign_translation(self):
        translator = RealTimeTranslator()
        res = translator.translate_signs(["hello"])
        assert res.status == "SUCCESS"
        assert "Hello" in res.text
        assert "नमस्ते" in res.hindi_text

    def test_phrase_template_matching(self):
        translator = RealTimeTranslator()
        res = translator.translate_signs(["hello", "teacher"])
        assert res.status == "SUCCESS"
        assert res.text == "Hello teacher!"
        assert res.hindi_text == "नमस्ते शिक्षक!"

    def test_multi_sign_concatenation(self):
        translator = RealTimeTranslator()
        res = translator.translate_signs(["car", "time"])
        assert res.status == "SUCCESS"
        assert "car" in res.tokens
        assert "time" in res.tokens
        assert len(res.text) > 0
        assert len(res.hindi_text) > 0

    def test_empty_sequence_handling(self):
        translator = RealTimeTranslator()
        res = translator.translate_signs([])
        assert res.status == "INSUFFICIENT_INPUT"
        assert res.text == ""

    def test_low_confidence_sign_rejection(self):
        translator = RealTimeTranslator(min_confidence=0.60)
        evt = SignEvent.create(
            class_id=0,
            label="hello",
            start_time=1.0,
            end_time=2.0,
            confidence=0.45,  # Sub-threshold
            input_quality=0.85,
            stability_score=0.80,
            event_id="evt_01"
        )
        res = translator.process_event(evt)
        assert res.status == "LOW_CONFIDENCE"
        assert "Uncertain" in res.text

    def test_pause_sentence_finalization(self):
        translator = RealTimeTranslator(pause_threshold_sec=1.0)
        evt1 = SignEvent.create(class_id=0, label="hello", start_time=1.0, end_time=1.8, confidence=0.90, input_quality=0.90, stability_score=0.85, event_id="evt_01")
        evt2 = SignEvent.create(class_id=9, label="teacher", start_time=2.2, end_time=3.0, confidence=0.92, input_quality=0.92, stability_score=0.90, event_id="evt_02")

        r1 = translator.process_event(evt1)
        assert r1.is_final is False
        assert len(translator.finalized_phrases) == 0

        r2 = translator.process_event(evt2)
        assert r2.is_final is False
        assert len(translator.finalized_phrases) == 0

        # Simulate elapsed time 3.5s (silence = 0.5s < 1.0s) -> no finalization yet
        p1 = translator.check_pause(3.5)
        assert p1 is None

        # Simulate elapsed time 4.5s (silence = 1.5s >= 1.0s) -> finalizes phrase!
        p2 = translator.check_pause(4.5)
        assert p2 is not None
        assert p2.is_final is True
        assert len(translator.finalized_phrases) == 1
        assert translator.finalized_phrases[0]["text"] == "Hello teacher!"

    def test_reset_behavior(self):
        translator = RealTimeTranslator()
        evt = SignEvent.create(class_id=0, label="hello", start_time=1.0, end_time=1.8, confidence=0.90, input_quality=0.90, stability_score=0.85, event_id="evt_01")
        translator.process_event(evt)
        translator.finalize_phrase()
        assert len(translator.finalized_phrases) == 1

        translator.reset()
        assert len(translator.finalized_phrases) == 0
        assert translator.current_translation is None
