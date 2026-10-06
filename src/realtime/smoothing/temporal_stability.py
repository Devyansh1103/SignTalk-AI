"""
SignTalk AI - Consecutive Temporal Stability Smoother
Phase 4 Part 3: Confidence, Smoothing & Continuous Sign Detection.

Requires K consecutive identical predictions above confidence threshold
before promoting candidate prediction to stable status.
"""

from typing import List, Optional, Dict, Any
import numpy as np

from src.realtime.prediction_history import PredictionRecord
from src.realtime.smoothing.base import BaseSmoother, SmoothedPrediction
from src.data.label_map import get_label, get_gloss, get_translation


UNSTABLE_LABEL = "UNSTABLE"
UNSTABLE_GLOSS = "UNSTABLE"
UNSTABLE_CLASS_ID = -1


class TemporalStabilitySmoother(BaseSmoother):
    """
    Enforces that K consecutive sliding-window inferences predict the identical class
    above confidence threshold before declaring a stable prediction.
    """

    def __init__(
        self,
        min_consecutive: int = 3,
        min_confidence: float = 0.55
    ):
        """
        Args:
            min_consecutive: Required consecutive identical prediction count (default: 3).
            min_confidence: Minimum confidence threshold per consecutive prediction (default: 0.55).
        """
        self.min_consecutive = max(1, int(min_consecutive))
        self.min_confidence = float(min_confidence)

    def smooth(self, records: List[PredictionRecord]) -> SmoothedPrediction:
        if not records or len(records) < self.min_consecutive:
            latest_ts = records[-1].timestamp if records else 0.0
            return SmoothedPrediction(
                class_id=UNSTABLE_CLASS_ID,
                label=UNSTABLE_LABEL,
                gloss=UNSTABLE_GLOSS,
                translation="Unstable",
                confidence=0.0,
                stability_score=float(len(records) / self.min_consecutive) if records else 0.0,
                is_stable=False,
                timestamp=latest_ts,
                method="temporal_stability",
                details={"reason": "insufficient_history"}
            )

        window = records[-self.min_consecutive:]
        latest_ts = window[-1].timestamp

        target_cid = window[-1].class_id
        if target_cid < 0 or window[-1].confidence < self.min_confidence:
            return SmoothedPrediction(
                class_id=UNSTABLE_CLASS_ID,
                label=UNSTABLE_LABEL,
                gloss=UNSTABLE_GLOSS,
                translation="Unstable",
                confidence=window[-1].confidence,
                stability_score=0.0,
                is_stable=False,
                timestamp=latest_ts,
                method="temporal_stability",
                details={"reason": "latest_below_confidence"}
            )

        # Check that all K frames have the exact same class and meet min_confidence
        match_count = sum(1 for r in window if r.class_id == target_cid and r.confidence >= self.min_confidence)
        is_stable = (match_count == self.min_consecutive)
        stability_score = float(match_count / self.min_consecutive)
        mean_conf = float(np.mean([r.confidence for r in window if r.class_id == target_cid]))

        mean_quality = float(np.mean([r.input_quality for r in window])) if window else 1.0

        if is_stable:
            return SmoothedPrediction(
                class_id=target_cid,
                label=get_label(target_cid),
                gloss=get_gloss(target_cid),
                translation=get_translation(target_cid),
                confidence=mean_conf,
                stability_score=stability_score,
                is_stable=True,
                timestamp=latest_ts,
                method="temporal_stability",
                input_quality=mean_quality,
                window_start=window[0].window_start,
                window_end=window[-1].window_end,
                details={"consecutive_matches": match_count}
            )
        else:
            return SmoothedPrediction(
                class_id=UNSTABLE_CLASS_ID,
                label=UNSTABLE_LABEL,
                gloss=UNSTABLE_GLOSS,
                translation="Unstable",
                confidence=mean_conf,
                stability_score=stability_score,
                is_stable=False,
                timestamp=latest_ts,
                method="temporal_stability",
                input_quality=mean_quality,
                window_start=window[0].window_start,
                window_end=window[-1].window_end,
                details={"consecutive_matches": match_count}
            )

