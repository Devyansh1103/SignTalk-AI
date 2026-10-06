"""
SignTalk AI - Temporal Smoothing Abstract Base
Phase 4 Part 3: Confidence, Smoothing & Continuous Sign Detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from src.realtime.prediction_history import PredictionRecord

UNSTABLE_LABEL = "UNSTABLE"
UNSTABLE_GLOSS = "UNSTABLE"
UNSTABLE_CLASS_ID = -1


@dataclass
class SmoothedPrediction:

    """
    Result of temporal smoothing across recent prediction records.
    """
    class_id: int
    label: str
    gloss: str
    translation: str
    confidence: float
    stability_score: float     # Ratio of evidence supporting winning class [0.0..1.0]
    is_stable: bool           # True if passes stability / vote threshold
    timestamp: float
    method: str
    input_quality: float = 1.0
    window_start: float = 0.0
    window_end: float = 0.0
    details: Optional[Dict[str, Any]] = None

    @property
    def is_valid(self) -> bool:
        """True if prediction is stable, non-negative class, and not UNCERTAIN/UNSTABLE."""
        return (
            self.is_stable
            and self.class_id >= 0
            and self.label not in ("UNSTABLE", "UNCERTAIN", "UNKNOWN", "IDLE")
        )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "class_id": self.class_id,
            "label": self.label,
            "gloss": self.gloss,
            "translation": self.translation,
            "confidence": round(self.confidence, 4),
            "stability_score": round(self.stability_score, 4),
            "is_stable": self.is_stable,
            "timestamp": round(self.timestamp, 4),
            "method": self.method
        }


class BaseSmoother(ABC):
    """Abstract interface for temporal prediction smoothing algorithms."""

    @abstractmethod
    def smooth(self, records: List[PredictionRecord]) -> SmoothedPrediction:
        """
        Executes smoothing over a chronological list of recent PredictionRecords.
        """
        pass
