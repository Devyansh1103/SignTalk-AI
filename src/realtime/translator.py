"""
SignTalk AI - Real-Time Translation & Linguistic Formatting Layer
Phase 4 Part 4: Sign-to-Text Translation, Grammar Harmonization, and Live Transcript Management.

Converts debounced SignEvent sequences and gloss tokens into fluent natural language
translations (English and Hindi) with pause-based sentence finalization and structured transcript tracking.
"""

from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import time
import logging

from src.realtime.sign_event import SignEvent
from src.data.label_map import (
    NUM_CLASSES,
    CANONICAL_CLASSES,
    get_label,
    get_gloss,
    get_translation,
    LABEL_TO_ID
)

logger = logging.getLogger("SignTalk.RealTime.Translator")


# Bilingual lexical dictionaries for MVP-10 vocabulary
ENGLISH_LEXICON: Dict[str, str] = {
    "hello": "Hello",
    "thankyou": "Thank you",
    "good": "Good",
    "happy": "Happy",
    "monday": "Monday",
    "car": "Car",
    "bird": "Bird",
    "house": "House",
    "time": "Time",
    "teacher": "Teacher",
}

HINDI_LEXICON: Dict[str, str] = {
    "hello": "नमस्ते",
    "thankyou": "धन्यवाद",
    "good": "अच्छा",
    "happy": "खुश",
    "monday": "सोमवार",
    "car": "गाड़ी",
    "bird": "पक्षी",
    "house": "घर",
    "time": "समय",
    "teacher": "शिक्षक",
}

# Linguistic phrase composition templates for natural multi-sign communication
PHRASE_TEMPLATES: Dict[Tuple[str, ...], Tuple[str, str]] = {
    ("hello", "teacher"): ("Hello teacher!", "नमस्ते शिक्षक!"),
    ("good", "monday"): ("Good Monday!", "शुभ सोमवार!"),
    ("happy", "monday"): ("Happy Monday!", "शुभ सोमवार!"),
    ("good", "teacher"): ("Good teacher.", "अच्छे शिक्षक।"),
    ("happy", "bird"): ("Happy bird.", "खुश पक्षी।"),
    ("hello", "happy"): ("Hello, I am happy.", "नमस्ते, मैं खुश हूँ।"),
    ("thankyou", "teacher"): ("Thank you, teacher.", "धन्यवाद, शिक्षक।"),
    ("time", "house"): ("Time to go home.", "घर जाने का समय।"),
    ("car", "house"): ("The car is at the house.", "गाड़ी घर पर है।"),
    ("bird", "house"): ("A bird is near the house.", "घर के पास पक्षी है।"),
    ("good", "time"): ("Have a good time.", "अच्छा समय बीते।"),
}


@dataclass(frozen=True)
class TranslationResult:
    """
    Structured outcome of sign-to-text linguistic processing.
    """
    text: str
    hindi_text: str
    confidence: float
    tokens: List[str]
    source_signs: List[str]
    timestamp: float
    status: str = "SUCCESS"  # SUCCESS, LOW_CONFIDENCE, INSUFFICIENT_INPUT, NO_TRANSLATION, ERROR
    is_final: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "hindi_text": self.hindi_text,
            "confidence": round(self.confidence, 4),
            "tokens": list(self.tokens),
            "source_signs": list(self.source_signs),
            "timestamp": round(self.timestamp, 3),
            "status": self.status,
            "is_final": self.is_final,
        }


class RealTimeTranslator:
    """
    Translates incoming SignEvents into grammatical English and Hindi captions,
    tracks streaming phrase accumulation, and finalizes sentences upon inter-sign pauses.
    """

    def __init__(
        self,
        pause_threshold_sec: float = 1.5,
        min_confidence: float = 0.50,
        max_phrase_signs: int = 10,
    ):
        """
        Args:
            pause_threshold_sec: Temporal silence required to finalize a phrase (default: 1.5s).
            min_confidence: Threshold required to accept individual signs into translation.
            max_phrase_signs: Maximum number of sign events per conversational phrase.
        """
        self.pause_threshold_sec = max(0.5, float(pause_threshold_sec))
        self.min_confidence = float(min_confidence)
        self.max_phrase_signs = max(1, int(max_phrase_signs))

        # Transcript state
        self._active_events: List[SignEvent] = []
        self._last_event_time: Optional[float] = None
        self._finalized_phrases: List[Dict[str, Any]] = []
        self._current_translation: Optional[TranslationResult] = None

    @property
    def current_translation(self) -> Optional[TranslationResult]:
        """Returns the most recent provisional or finalized translation."""
        return self._current_translation

    @property
    def finalized_phrases(self) -> List[Dict[str, Any]]:
        """Returns the chronological history of finalized phrases."""
        return list(self._finalized_phrases)

    def process_event(self, event: SignEvent) -> TranslationResult:
        """
        Ingests a newly finalized SignEvent, updates active phrase, and produces provisional translation.
        """
        if event.confidence < self.min_confidence:
            return TranslationResult(
                text="Uncertain sign",
                hindi_text="अनिश्चित संकेत",
                confidence=event.confidence,
                tokens=[event.label],
                source_signs=[event.label],
                timestamp=event.end_time,
                status="LOW_CONFIDENCE",
                is_final=False,
            )

        self._active_events.append(event)
        self._last_event_time = event.end_time

        # If phrase length exceeds limit, auto-finalize
        if len(self._active_events) >= self.max_phrase_signs:
            return self.finalize_phrase(reason="max_length")

        res = self._translate_events(self._active_events, is_final=False)
        self._current_translation = res
        return res

    def check_pause(self, current_time: float) -> Optional[TranslationResult]:
        """
        Evaluates elapsed silence since last sign event; finalizes phrase if pause threshold exceeded.
        """
        if not self._active_events or self._last_event_time is None:
            return None

        elapsed = current_time - self._last_event_time
        if elapsed >= self.pause_threshold_sec:
            return self.finalize_phrase(reason="pause_timeout")

        return None

    def finalize_phrase(self, reason: str = "manual") -> TranslationResult:
        """
        Commits active sign events into a finalized phrase entry in the transcript history.
        """
        if not self._active_events:
            return TranslationResult(
                text="",
                hindi_text="",
                confidence=1.0,
                tokens=[],
                source_signs=[],
                timestamp=time.time(),
                status="NO_TRANSLATION",
                is_final=True,
            )

        final_res = self._translate_events(self._active_events, is_final=True)
        self._finalized_phrases.append({
            "text": final_res.text,
            "hindi_text": final_res.hindi_text,
            "tokens": final_res.tokens,
            "source_signs": final_res.source_signs,
            "start_time": self._active_events[0].start_time,
            "end_time": self._active_events[-1].end_time,
            "confidence": final_res.confidence,
            "reason": reason,
        })

        # Clear active buffer for the next sentence
        self._active_events = []
        self._last_event_time = None
        self._current_translation = final_res
        return final_res

    def translate_signs(self, sign_labels: List[str]) -> TranslationResult:
        """
        Utility to translate an explicit list of sign label strings directly.
        """
        if not sign_labels:
            return TranslationResult(
                text="",
                hindi_text="",
                confidence=1.0,
                tokens=[],
                source_signs=[],
                timestamp=time.time(),
                status="INSUFFICIENT_INPUT",
                is_final=True,
            )

        normalized = [s.lower().replace("_", "").replace("-", "") for s in sign_labels]
        now = time.time()

        # Check multi-sign template matching
        tuple_key = tuple(normalized)
        if tuple_key in PHRASE_TEMPLATES:
            en, hi = PHRASE_TEMPLATES[tuple_key]
            return TranslationResult(
                text=en,
                hindi_text=hi,
                confidence=0.95,
                tokens=list(normalized),
                source_signs=list(sign_labels),
                timestamp=now,
                status="SUCCESS",
                is_final=True,
            )

        # Concatenate individual lexicon translations
        en_words = [ENGLISH_LEXICON.get(s, s.capitalize()) for s in normalized]
        hi_words = [HINDI_LEXICON.get(s, s) for s in normalized]

        en_sentence = " ".join(en_words) + "."
        hi_sentence = " ".join(hi_words) + "।"

        return TranslationResult(
            text=en_sentence,
            hindi_text=hi_sentence,
            confidence=0.90,
            tokens=list(normalized),
            source_signs=list(sign_labels),
            timestamp=now,
            status="SUCCESS",
            is_final=True,
        )

    def _translate_events(self, events: List[SignEvent], is_final: bool) -> TranslationResult:
        """Internal generator translating active SignEvent list."""
        labels = [e.label for e in events]
        confs = [e.confidence for e in events]
        mean_conf = float(sum(confs) / len(confs)) if confs else 0.0

        res = self.translate_signs(labels)
        return TranslationResult(
            text=res.text,
            hindi_text=res.hindi_text,
            confidence=round(mean_conf, 4),
            tokens=res.tokens,
            source_signs=labels,
            timestamp=events[-1].end_time if events else time.time(),
            status="SUCCESS",
            is_final=is_final,
        )

    def get_transcript_state(self) -> Dict[str, Any]:
        """Provides a structured state dictionary for overlay rendering or frontend streaming."""
        active_labels = [e.label.upper() for e in self._active_events]
        curr_text = self._current_translation.text if self._current_translation else ""
        curr_hindi = self._current_translation.hindi_text if self._current_translation else ""

        return {
            "active_signs": active_labels,
            "provisional_text": curr_text,
            "provisional_hindi": curr_hindi,
            "finalized_phrases": list(self._finalized_phrases),
            "finalized_count": len(self._finalized_phrases),
        }

    def reset(self) -> None:
        """Resets active phrase buffer and finalized transcript history."""
        self._active_events.clear()
        self._last_event_time = None
        self._finalized_phrases.clear()
        self._current_translation = None
