"""
SignTalk AI - Confidence-Weighted Temporal Smoother
Phase 4 Part 3: Confidence, Smoothing & Continuous Sign Detection.

Aggregates predictions weighted by individual confidence scores and recency decay.
"""

from typing import List, Optional, Dict, Any
from collections import defaultdict
import numpy as np

from src.realtime.prediction_history import PredictionRecord
from src.realtime.smoothing.base import BaseSmoother, SmoothedPrediction
from src.data.label_map import get_label, get_gloss, get_translation


UNSTABLE_LABEL = "UNSTABLE"
UNSTABLE_GLOSS = "UNSTABLE"
UNSTABLE_CLASS_ID = -1


class ConfidenceWeightedSmoother(BaseSmoother):
    """
    Computes confidence-weighted consensus across recent prediction windows.
    Weights evidence by prediction confidence, optionally decaying older predictions.
    """

    def __init__(
        self,
        history_size: int = 5,
        min_confidence: float = 0.40,
        min_stability_ratio: float = 0.50,
        recency_decay: float = 0.90,
        decay_factor: Optional[float] = None,
    ):
        """
        Args:
            history_size: Number of trailing records to aggregate (default: 5).
            min_confidence: Ignore predictions below this threshold (default: 0.40).
            min_stability_ratio: Fraction of total weighted evidence needed for stability (default: 0.50).
            recency_decay: Exponential decay multiplier per time step back (default: 0.90).
            decay_factor: Alias for recency_decay.
        """
        if decay_factor is not None:
            recency_decay = decay_factor
        self.history_size = max(1, int(history_size))
        self.min_confidence = float(min_confidence)
        self.min_stability_ratio = float(min_stability_ratio)
        self.recency_decay = float(recency_decay)


    def smooth(self, records: List[PredictionRecord]) -> SmoothedPrediction:
        if not records:
            return SmoothedPrediction(
                class_id=UNSTABLE_CLASS_ID,
                label=UNSTABLE_LABEL,
                gloss=UNSTABLE_GLOSS,
                translation="Unstable",
                confidence=0.0,
                stability_score=0.0,
                is_stable=False,
                timestamp=0.0,
                method="confidence_weighted"
            )

        window = records[-self.history_size:]
        latest_ts = window[-1].timestamp

        class_weights = defaultdict(float)
        class_confidences = defaultdict(list)
        total_weight = 0.0

        n = len(window)
        for idx, rec in enumerate(window):
            if rec.class_id < 0 or rec.confidence < self.min_confidence:
                continue

            # Exponential decay based on distance from latest record
            steps_back = n - 1 - idx
            decay = self.recency_decay ** steps_back
            effective_weight = rec.confidence * decay

            class_weights[rec.class_id] += effective_weight
            class_confidences[rec.class_id].append(rec.confidence)
            total_weight += effective_weight

        if total_weight <= 0.0:
            return SmoothedPrediction(
                class_id=UNSTABLE_CLASS_ID,
                label=UNSTABLE_LABEL,
                gloss=UNSTABLE_GLOSS,
                translation="Unstable",
                confidence=0.0,
                stability_score=0.0,
                is_stable=False,
                timestamp=latest_ts,
                method="confidence_weighted",
                details={"reason": "zero_weight"}
            )

        # Select class with maximum accumulated weight
        winner_cid = max(class_weights.keys(), key=lambda cid: class_weights[cid])
        winner_weight = class_weights[winner_cid]
        stability_score = float(winner_weight / total_weight)
        is_stable = (stability_score >= self.min_stability_ratio)

        mean_conf = float(np.mean(class_confidences[winner_cid]))
        winning_records = [r for r in window if r.class_id == winner_cid]
        mean_quality = float(np.mean([r.input_quality for r in winning_records])) if winning_records else 1.0

        return SmoothedPrediction(
            class_id=winner_cid,
            label=get_label(winner_cid),
            gloss=get_gloss(winner_cid),
            translation=get_translation(winner_cid),
            confidence=mean_conf,
            stability_score=stability_score,
            is_stable=is_stable,
            timestamp=latest_ts,
            method="confidence_weighted",
            input_quality=mean_quality,
            window_start=window[0].window_start,
            window_end=window[-1].window_end,
            details={
                "winner_weight": round(winner_weight, 3),
                "total_weight": round(total_weight, 3),
                "stability_ratio": round(stability_score, 3)
            }
        )

