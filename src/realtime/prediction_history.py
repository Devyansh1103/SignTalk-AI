"""
SignTalk AI - Prediction History Buffer
Phase 4 Part 3: Confidence, Smoothing & Continuous Sign Detection.

Maintains a bounded chronological history of recent model predictions for
temporal smoothing, confidence filtering, and stability evaluation.
"""

from typing import List, Optional, Dict, Any, Tuple
from collections import deque
from dataclasses import dataclass
import time
import numpy as np

from src.realtime.types import PredictionResult
from src.data.label_map import get_label, get_gloss, get_translation


@dataclass
class PredictionRecord:
    """
    Lightweight bounded representation of a single window prediction.
    """
    timestamp: float
    window_start: float
    window_end: float
    class_id: int
    label: str
    confidence: float
    input_quality: float
    gloss: str = ""
    inference_latency_ms: float = 0.0
    is_valid_quality: bool = True
    probabilities: Optional[np.ndarray] = None


    @classmethod
    def from_prediction_result(cls, res: PredictionResult) -> "PredictionRecord":
        return cls(
            timestamp=res.timestamp,
            window_start=res.window_start_time,
            window_end=res.window_end_time,
            class_id=res.class_id,
            label=res.label,
            gloss=res.gloss,
            confidence=float(res.confidence),
            input_quality=float(res.input_quality),
            inference_latency_ms=float(res.inference_latency_ms),
            is_valid_quality=bool(res.is_valid_quality),
            probabilities=res.probabilities
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 4),
            "window_start": round(self.window_start, 4),
            "window_end": round(self.window_end, 4),
            "class_id": self.class_id,
            "label": self.label,
            "gloss": self.gloss,
            "confidence": round(self.confidence, 4),
            "input_quality": round(self.input_quality, 4),
            "inference_latency_ms": round(self.inference_latency_ms, 2),
            "is_valid_quality": self.is_valid_quality
        }


class PredictionHistory:
    """
    Bounded rolling FIFO queue storing the most recent prediction records.
    """

    def __init__(self, max_length: int = 30, max_history: Optional[int] = None):
        if max_history is not None:
            max_length = max_history
        self.max_length = max(1, int(max_length))
        self._history: deque = deque(maxlen=self.max_length)
        self._total_added = 0

    def append(self, record: Any) -> None:
        """Appends a new prediction record."""
        from src.realtime.confidence_filter import FilteredPrediction
        if isinstance(record, FilteredPrediction):
            record = PredictionRecord(
                timestamp=record.timestamp,
                window_start=record.window_start,
                window_end=record.window_end,
                class_id=record.class_id,
                label=record.label,
                gloss=record.gloss,
                confidence=record.confidence,
                input_quality=record.input_quality,
            )
        elif not isinstance(record, PredictionRecord):
            raise TypeError(f"Expected PredictionRecord or FilteredPrediction, got {type(record)}")
        self._history.append(record)
        self._total_added += 1


    def add(self, record: PredictionRecord) -> None:
        """Alias for append()."""
        self.append(record)


    def append_result(self, result: PredictionResult) -> None:
        """Helper to append directly from a PredictionResult."""
        self.append(PredictionRecord.from_prediction_result(result))

    def __len__(self) -> int:
        """Number of items currently in history."""
        return len(self._history)

    def __iter__(self):
        """Iterate over history items."""
        return iter(self._history)

    def __getitem__(self, item):
        """Support indexing and slicing over history items."""
        if isinstance(item, slice):
            return list(self._history)[item]
        return self._history[item]

    def to_list(self) -> List[PredictionRecord]:
        """Convert history buffer to a list."""
        return list(self._history)

    def get_recent(self, n: Optional[int] = None) -> List[PredictionRecord]:
        """
        Retrieves the N most recent prediction records (oldest to newest).
        If n is None or n >= len, returns the entire history.
        """
        if not self._history:
            return []
        if n is None or n >= len(self._history):
            return list(self._history)
        return list(self._history)[-n:]

    def latest(self) -> Optional[PredictionRecord]:
        """Returns the most recent prediction record, or None if empty."""
        return self._history[-1] if self._history else None


    def length(self) -> int:
        """Returns current number of stored records."""
        return len(self._history)

    def __len__(self) -> int:
        return self.length()

    def is_empty(self) -> bool:
        """Checks if history is empty."""
        return len(self._history) == 0

    def clear(self) -> None:
        """Clears all records."""
        self._history.clear()

    def mean_confidence(self, n: Optional[int] = None) -> float:
        """Computes average confidence over the last N predictions."""
        recent = self.get_recent(n)
        if not recent:
            return 0.0
        return float(np.mean([r.confidence for r in recent]))

    def mean_quality(self, n: Optional[int] = None) -> float:
        """Computes average input quality over the last N predictions."""
        recent = self.get_recent(n)
        if not recent:
            return 0.0
        return float(np.mean([r.input_quality for r in recent]))

    @property
    def total_added(self) -> int:
        return self._total_added
