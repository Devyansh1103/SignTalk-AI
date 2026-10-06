"""
SignTalk AI - Confidence & Input Quality Filter
Phase 4 Part 3: Confidence, Smoothing & Continuous Sign Detection.

Filters incoming raw predictions by validating both model confidence score
and landmark input quality against empirically established validation thresholds.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
import numpy as np

from src.realtime.prediction_history import PredictionRecord
from src.realtime.types import PredictionResult


UNCERTAIN_LABEL = "UNCERTAIN"
UNCERTAIN_GLOSS = "UNCERTAIN"
UNCERTAIN_CLASS_ID = -1


@dataclass
class FilteredPrediction:
    """
    Representation of a prediction after confidence and quality filtering.
    """
    class_id: int
    label: str
    gloss: str
    translation: str
    confidence: float
    input_quality: float
    is_accepted: bool
    rejection_reason: Optional[str]
    timestamp: float
    window_start: float
    window_end: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "gloss": self.gloss,
            "translation": self.translation,
            "confidence": round(self.confidence, 4),
            "input_quality": round(self.input_quality, 4),
            "is_accepted": self.is_accepted,
            "rejection_reason": self.rejection_reason,
            "timestamp": round(self.timestamp, 4),
            "window_start": round(self.window_start, 4),
            "window_end": round(self.window_end, 4)
        }


class ConfidenceFilter:
    """
    Evaluates raw predictions against confidence and landmark quality thresholds.
    Ensures low-confidence inferences are flagged as UNCERTAIN without crashing downstream logic.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        min_input_quality: float = 0.40,
        threshold: Optional[float] = None,
        uncertain_label: str = UNCERTAIN_LABEL,
    ):
        """
        Args:
            confidence_threshold: Minimum model confidence required to accept prediction (default: 0.65).
            min_input_quality: Minimum landmark quality score required (default: 0.40).
            threshold: Alias for confidence_threshold.
            uncertain_label: Label to assign to uncertain predictions.
        """
        if threshold is not None:
            confidence_threshold = threshold
        self.confidence_threshold = float(confidence_threshold)
        self.min_input_quality = float(min_input_quality)
        self.uncertain_label = uncertain_label

    def filter(self, prediction: Union[PredictionResult, PredictionRecord]) -> FilteredPrediction:
        """
        Applies confidence and quality gates to a prediction instance.
        """

        if isinstance(prediction, PredictionResult):
            rec = PredictionRecord.from_prediction_result(prediction)
        elif isinstance(prediction, PredictionRecord):
            rec = prediction
        else:
            raise TypeError(f"Expected PredictionResult or PredictionRecord, got {type(prediction)}")

        conf = rec.confidence
        qual = rec.input_quality


        # Quality check first
        if qual < self.min_input_quality:
            return FilteredPrediction(
                class_id=UNCERTAIN_CLASS_ID,
                label=UNCERTAIN_LABEL,
                gloss=UNCERTAIN_GLOSS,
                translation="Uncertain",
                confidence=conf,
                input_quality=qual,
                is_accepted=False,
                rejection_reason="low_input_quality",
                timestamp=rec.timestamp,
                window_start=rec.window_start,
                window_end=rec.window_end
            )

        # Confidence threshold check
        if conf < self.confidence_threshold:
            return FilteredPrediction(
                class_id=UNCERTAIN_CLASS_ID,
                label=UNCERTAIN_LABEL,
                gloss=UNCERTAIN_GLOSS,
                translation="Uncertain",
                confidence=conf,
                input_quality=qual,
                is_accepted=False,
                rejection_reason="low_confidence",
                timestamp=rec.timestamp,
                window_start=rec.window_start,
                window_end=rec.window_end
            )

        # Accepted prediction
        return FilteredPrediction(
            class_id=rec.class_id,
            label=rec.label,
            gloss=rec.gloss,
            translation=get_translation_safe(rec.class_id),
            confidence=conf,
            input_quality=qual,
            is_accepted=True,
            rejection_reason=None,
            timestamp=rec.timestamp,
            window_start=rec.window_start,
            window_end=rec.window_end
        )

    def filter_prediction(self, prediction: Union[PredictionResult, PredictionRecord]) -> FilteredPrediction:
        """Alias for filter()."""
        return self.filter(prediction)



def get_translation_safe(class_id: int) -> str:
    from src.data.label_map import get_translation
    try:
        return get_translation(class_id)
    except Exception:
        return "Unknown"
