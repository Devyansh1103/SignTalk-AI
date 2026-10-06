"""Temporal prediction smoothing module.

Provides smoothing algorithms to transform noisy real-time predictions
into stable sign candidates.
"""

from src.realtime.smoothing.base import (
    BaseSmoother,
    SmoothedPrediction,
    UNSTABLE_LABEL,
    UNSTABLE_GLOSS,
    UNSTABLE_CLASS_ID,
)
from src.realtime.smoothing.confidence_weighted import ConfidenceWeightedSmoother
from src.realtime.smoothing.majority_vote import MajorityVoteSmoother
from src.realtime.smoothing.temporal_stability import TemporalStabilitySmoother

__all__ = [
    "BaseSmoother",
    "SmoothedPrediction",
    "MajorityVoteSmoother",
    "ConfidenceWeightedSmoother",
    "TemporalStabilitySmoother",
    "UNSTABLE_LABEL",
    "UNSTABLE_GLOSS",
    "UNSTABLE_CLASS_ID",
]

