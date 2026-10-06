"""Sign state machine for temporal prediction stability.

Implements the 4-state lifecycle:
IDLE -> CANDIDATE -> ACTIVE -> ENDING -> IDLE
to distinguish transient noise from sustained sign gestures.
"""

from enum import Enum
import logging
from typing import Any, Dict, List, Optional

from src.realtime.sign_event import SignEvent
from src.realtime.smoothing.base import SmoothedPrediction

logger = logging.getLogger(__name__)


class SignState(str, Enum):
    """Lifecycle states for sign gesture detection."""

    IDLE = "IDLE"
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    ENDING = "ENDING"


class SignStateMachine:
    """Finite State Machine tracking sign stability and emission.

    Lifecycle:
        IDLE:
            Waiting for consistent sign evidence.
        CANDIDATE:
            A promising valid prediction has appeared. Tracking consecutive
            reinforcements.
        ACTIVE:
            The sign has met consecutive count and confidence requirements.
            Sign gesture is actively occurring.
        ENDING:
            Prediction has changed or confidence dropped below threshold.
            Gesture has concluded, yielding a candidate SignEvent.

    Attributes:
        min_consecutive_predictions: Consecutive predictions required for ACTIVE.
        min_confidence: Confidence threshold required to consider a prediction valid.
        min_input_quality: Minimum landmark quality required for candidate/active state.
        state: Current SignState.
    """

    def __init__(
        self,
        min_consecutive_predictions: int = 2,
        min_confidence: float = 0.65,
        min_input_quality: float = 0.40,
    ) -> None:
        """Initialize SignStateMachine.

        Args:
            min_consecutive_predictions: Count of consecutive matching predictions to enter ACTIVE.
            min_confidence: Minimum model confidence to treat prediction as valid candidate.
            min_input_quality: Minimum landmark tracking quality score.
        """
        self.min_consecutive_predictions = max(1, min_consecutive_predictions)
        self.min_confidence = min_confidence
        self.min_input_quality = min_input_quality

        self._state: SignState = SignState.IDLE

        # Tracking state variables
        self._current_class_id: Optional[int] = None
        self._current_label: Optional[str] = None
        self._consecutive_count: int = 0
        self._gesture_start_time: float = 0.0
        self._gesture_end_time: float = 0.0
        self._window_start_idx: int = 0
        self._window_end_idx: int = 0

        # Accumulated metrics during gesture
        self._confidences: List[float] = []
        self._qualities: List[float] = []
        self._stability_scores: List[float] = []

        # Last finalized event produced
        self._last_event: Optional[SignEvent] = None

    @property
    def state(self) -> SignState:
        """Current state of the machine."""
        return self._state

    @property
    def current_label(self) -> Optional[str]:
        """Label currently being tracked (in CANDIDATE or ACTIVE)."""
        return self._current_label

    @property
    def consecutive_count(self) -> int:
        """Count of consecutive matching predictions."""
        return self._consecutive_count

    def is_prediction_valid(self, pred: SmoothedPrediction) -> bool:
        """Check if smoothed prediction meets validity criteria."""
        if not pred.is_valid:
            return False
        if pred.confidence < self.min_confidence:
            return False
        if pred.input_quality < self.min_input_quality:
            return False
        if pred.label in ("UNKNOWN", "UNCERTAIN", "UNSTABLE", "NO_SIGN", "IDLE"):
            return False
        return True

    def process(
        self, pred: SmoothedPrediction
    ) -> tuple[SignState, Optional[SignEvent]]:
        """Process one smoothed prediction step and advance the state machine.

        Args:
            pred: The latest SmoothedPrediction from temporal smoothing.

        Returns:
            Tuple of (current_state, sign_event_if_completed).
        """
        valid = self.is_prediction_valid(pred)
        event_to_emit: Optional[SignEvent] = None

        if self._state == SignState.IDLE:
            if valid:
                self._transition_to_candidate(pred)
            # Else remain IDLE

        elif self._state == SignState.CANDIDATE:
            if valid and pred.class_id == self._current_class_id:
                self._consecutive_count += 1
                self._accumulate_metrics(pred)
                if self._consecutive_count >= self.min_consecutive_predictions:
                    self._transition_to_active(pred)
            elif valid and pred.class_id != self._current_class_id:
                # Class changed before achieving stability - switch candidate
                self._transition_to_candidate(pred)
            else:
                # Dropped to invalid/uncertain - reset to IDLE
                self._reset_to_idle()

        elif self._state == SignState.ACTIVE:
            if valid and pred.class_id == self._current_class_id:
                # Continuation of active sign
                self._consecutive_count += 1
                self._accumulate_metrics(pred)
                # Keep ACTIVE
            else:
                # Sign concluded: prediction changed or confidence dropped
                event_to_emit = self._finalize_active_event()
                self._state = SignState.ENDING

                # Decide next state from ENDING:
                if valid:
                    # Immediately transition into candidate for the new sign
                    self._transition_to_candidate(pred)
                else:
                    self._reset_to_idle()

        elif self._state == SignState.ENDING:
            # Transitional state safety handling
            if valid:
                self._transition_to_candidate(pred)
            else:
                self._reset_to_idle()

        return self._state, event_to_emit

    def force_end(self) -> Optional[SignEvent]:
        """Force finalize any active sign (e.g. at end of stream or timeout).

        Returns:
            Completed SignEvent if in ACTIVE state, else None.
        """
        if self._state == SignState.ACTIVE:
            event = self._finalize_active_event()
            self._reset_to_idle()
            return event
        self._reset_to_idle()
        return None

    def reset(self) -> None:
        """Reset state machine completely to IDLE without emitting."""
        self._reset_to_idle()

    # Internal transition helpers

    def _transition_to_candidate(self, pred: SmoothedPrediction) -> None:
        self._state = SignState.CANDIDATE
        self._current_class_id = pred.class_id
        self._current_label = pred.label
        self._consecutive_count = 1
        self._gesture_start_time = pred.window_start
        self._gesture_end_time = pred.window_end
        self._window_start_idx = 0
        self._window_end_idx = 0
        self._confidences = [pred.confidence]
        self._qualities = [pred.input_quality]
        self._stability_scores = [pred.stability_score]

    def _transition_to_active(self, pred: SmoothedPrediction) -> None:
        self._state = SignState.ACTIVE
        self._accumulate_metrics(pred)

    def _accumulate_metrics(self, pred: SmoothedPrediction) -> None:
        self._gesture_end_time = max(self._gesture_end_time, pred.window_end)
        self._confidences.append(pred.confidence)
        self._qualities.append(pred.input_quality)
        self._stability_scores.append(pred.stability_score)

    def _finalize_active_event(self) -> SignEvent:
        avg_conf = (
            sum(self._confidences) / len(self._confidences)
            if self._confidences
            else 0.0
        )
        avg_quality = (
            sum(self._qualities) / len(self._qualities)
            if self._qualities
            else 0.0
        )
        avg_stability = (
            sum(self._stability_scores) / len(self._stability_scores)
            if self._stability_scores
            else 1.0
        )

        event = SignEvent.create(
            class_id=self._current_class_id if self._current_class_id is not None else -1,
            label=self._current_label or "UNKNOWN",
            start_time=self._gesture_start_time,
            end_time=self._gesture_end_time,
            confidence=avg_conf,
            input_quality=avg_quality,
            stability_score=avg_stability,
            source_window_start=self._window_start_idx,
            source_window_end=self._window_end_idx,
            metadata={"num_samples": len(self._confidences)},
        )
        self._last_event = event
        return event

    def _reset_to_idle(self) -> None:
        self._state = SignState.IDLE
        self._current_class_id = None
        self._current_label = None
        self._consecutive_count = 0
        self._gesture_start_time = 0.0
        self._gesture_end_time = 0.0
        self._confidences.clear()
        self._qualities.clear()
        self._stability_scores.clear()
