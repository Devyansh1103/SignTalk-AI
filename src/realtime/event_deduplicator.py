"""Duplicate event suppression module.

Prevents repeated detections of the same continuous sign gesture from producing
spurious duplicate events, while preserving distinct repetitions separated by a
sufficient temporal gap.
"""

import logging
from typing import Optional

from src.realtime.sign_event import SignEvent

logger = logging.getLogger(__name__)


class EventDeduplicator:
    """Filters duplicate sign events based on class identity and temporal spacing.

    Rules:
        1. Different sign class from previous -> EMIT.
        2. Same sign class within minimum_gap_ms -> SUPPRESS (merge/extend duration).
        3. Same sign class after >= minimum_gap_ms -> EMIT (new repetition).

    Attributes:
        minimum_gap_ms: Minimum separation time in milliseconds between two
            distinct events of the same sign class.
    """

    def __init__(self, minimum_gap_ms: float = 800.0) -> None:
        """Initialize EventDeduplicator.

        Args:
            minimum_gap_ms: Minimum gap in milliseconds required to treat the same sign
                as a new occurrence rather than an ongoing continuation.
        """
        self.minimum_gap_ms = max(0.0, minimum_gap_ms)

        self._last_emitted_event: Optional[SignEvent] = None
        self._total_received: int = 0
        self._total_emitted: int = 0
        self._total_suppressed: int = 0

    @property
    def last_emitted_event(self) -> Optional[SignEvent]:
        """Most recent event passed through the deduplicator."""
        return self._last_emitted_event

    @property
    def total_received(self) -> int:
        """Total candidate events presented to deduplicator."""
        return self._total_received

    @property
    def total_emitted(self) -> int:
        """Total events emitted."""
        return self._total_emitted

    @property
    def total_suppressed(self) -> int:
        """Total events suppressed as duplicates."""
        return self._total_suppressed

    @property
    def duplicate_rate(self) -> float:
        """Proportion of received events that were suppressed as duplicates."""
        if self._total_received == 0:
            return 0.0
        return self._total_suppressed / self._total_received

    def process(self, event: SignEvent) -> Optional[SignEvent]:
        """Evaluate a candidate SignEvent for duplicate suppression.

        Args:
            event: The candidate SignEvent.

        Returns:
            SignEvent if allowed through (either fresh sign or separated repetition),
            None if suppressed as a duplicate continuation.
        """
        self._total_received += 1

        if self._last_emitted_event is None:
            # First event ever - emit
            self._last_emitted_event = event
            self._total_emitted += 1
            return event

        if event.class_id != self._last_emitted_event.class_id:
            # Different sign class - always emit
            self._last_emitted_event = event
            self._total_emitted += 1
            return event

        # Same sign class: check temporal gap
        # Time gap between previous event conclusion and this event beginning
        gap_ms = (event.start_time - self._last_emitted_event.end_time) * 1000.0

        if gap_ms >= self.minimum_gap_ms:
            # Sufficient separation has elapsed - new distinct sign occurrence
            self._last_emitted_event = event
            self._total_emitted += 1
            return event

        # Duplicate continuation: suppress event and update last event's boundary
        self._total_suppressed += 1
        if event.end_time > self._last_emitted_event.end_time:
            self._last_emitted_event.end_time = event.end_time
            self._last_emitted_event.duration_ms = round(
                (self._last_emitted_event.end_time - self._last_emitted_event.start_time)
                * 1000.0,
                2,
            )
        return None

    def reset(self) -> None:
        """Reset deduplication state and counters."""
        self._last_emitted_event = None
        self._total_received = 0
        self._total_emitted = 0
        self._total_suppressed = 0
