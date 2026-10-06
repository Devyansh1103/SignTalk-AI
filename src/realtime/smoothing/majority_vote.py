"""
SignTalk AI - Majority Vote Temporal Smoother
Phase 4 Part 3: Confidence, Smoothing & Continuous Sign Detection.

Performs robust majority voting over the most recent N sliding window predictions.
Handles confidence filtering, tie detection, and minimum vote consensus.
"""

from typing import List, Optional, Dict, Any
from collections import Counter
import numpy as np

from src.realtime.prediction_history import PredictionRecord
from src.realtime.smoothing.base import BaseSmoother, SmoothedPrediction
from src.data.label_map import get_label, get_gloss, get_translation


UNSTABLE_LABEL = "UNSTABLE"
UNSTABLE_GLOSS = "UNSTABLE"
UNSTABLE_CLASS_ID = -1


class MajorityVoteSmoother(BaseSmoother):
    """
    Stabilizes real-time predictions by taking a majority vote over a rolling history of N predictions.
    """

    def __init__(
        self,
        history_size: int = 5,
        min_votes: int = 3,
        min_confidence: float = 0.50
    ):
        """
        Args:
            history_size: Number of recent predictions to include in voting window (default: 5).
            min_votes: Minimum votes required for a majority consensus (default: 3).
            min_confidence: Ignore individual predictions below this confidence threshold (default: 0.50).
        """
        self.history_size = max(1, int(history_size))
        self.min_votes = max(1, int(min_votes))
        self.min_confidence = float(min_confidence)

    def smooth(self, records: List[PredictionRecord]) -> SmoothedPrediction:
        """
        Evaluates voting consensus over the trailing records.
        """
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
                method="majority_vote"
            )

        window = records[-self.history_size:]
        latest_ts = window[-1].timestamp

        # Filter records by minimum confidence
        eligible = [r for r in window if r.confidence >= self.min_confidence and r.class_id >= 0]

        if not eligible:
            return SmoothedPrediction(
                class_id=UNSTABLE_CLASS_ID,
                label=UNSTABLE_LABEL,
                gloss=UNSTABLE_GLOSS,
                translation="Unstable",
                confidence=float(np.mean([r.confidence for r in window])),
                stability_score=0.0,
                is_stable=False,
                timestamp=latest_ts,
                method="majority_vote",
                details={"reason": "no_eligible_predictions", "window_size": len(window)}
            )

        # Count votes by class_id
        counts = Counter(r.class_id for r in eligible)
        most_common = counts.most_common()

        winner_cid, max_votes = most_common[0]

        # Check for tie with second place
        if len(most_common) > 1 and most_common[1][1] == max_votes:
            return SmoothedPrediction(
                class_id=UNSTABLE_CLASS_ID,
                label=UNSTABLE_LABEL,
                gloss=UNSTABLE_GLOSS,
                translation="Tie",
                confidence=0.0,
                stability_score=float(max_votes / len(eligible)),
                is_stable=False,
                timestamp=latest_ts,
                method="majority_vote",
                details={"reason": "tie_detected", "tied_votes": max_votes}
            )

        stability_score = float(max_votes / len(window))
        is_stable = (max_votes >= self.min_votes)

        # Mean confidence and quality of winning votes
        winning_records = [r for r in eligible if r.class_id == winner_cid]
        mean_win_conf = float(np.mean([r.confidence for r in winning_records]))
        mean_quality = float(np.mean([r.input_quality for r in winning_records])) if winning_records else 1.0

        return SmoothedPrediction(
            class_id=winner_cid,
            label=get_label(winner_cid),
            gloss=get_gloss(winner_cid),
            translation=get_translation(winner_cid),
            confidence=mean_win_conf,
            stability_score=stability_score,
            is_stable=is_stable,
            timestamp=latest_ts,
            method="majority_vote",
            input_quality=mean_quality,
            window_start=window[0].window_start,
            window_end=window[-1].window_end,
            details={
                "votes": max_votes,
                "total_window": len(window),
                "total_eligible": len(eligible)
            }
        )

