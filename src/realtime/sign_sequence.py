"""Continuous sign sequence buffer.

Maintains an ordered buffer of validated SignEvent instances, representing
the sequence of discrete sign glosses produced during continuous signing.
"""

from collections import deque
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.realtime.sign_event import SignEvent

logger = logging.getLogger(__name__)


class SignSequenceBuffer:
    """Bounded buffer maintaining chronological sequence of SignEvents.

    Attributes:
        max_events: Maximum number of historical events stored in the buffer.
    """

    def __init__(self, max_events: int = 50) -> None:
        """Initialize SignSequenceBuffer.

        Args:
            max_events: Maximum number of sign events to retain.
        """
        self.max_events = max(1, max_events)
        self._events: deque[SignEvent] = deque(maxlen=self.max_events)

    def __len__(self) -> int:
        """Number of events currently in the buffer."""
        return len(self._events)

    def append_event(self, event: SignEvent) -> None:
        """Append a newly validated sign event to the sequence buffer.

        Args:
            event: The SignEvent to record.
        """
        self._events.append(event)
        logger.debug(
            "Appended sign event: %s [%.2fs - %.2fs]",
            event.label,
            event.start_time,
            event.end_time,
        )

    def get_sequence(self) -> List[SignEvent]:
        """Return list of all SignEvents in chronological order."""
        return list(self._events)

    def get_labels(self) -> List[str]:
        """Return list of gloss/label strings in order."""
        return [evt.label for evt in self._events]

    def last_event(self) -> Optional[SignEvent]:
        """Return the most recently appended SignEvent, or None if buffer is empty."""
        if not self._events:
            return None
        return self._events[-1]

    def clear(self) -> None:
        """Clear all events from the sequence buffer."""
        self._events.clear()

    def format_timeline(self) -> str:
        """Format the sign sequence as a human-readable timeline string."""
        if not self._events:
            return "(Empty sign sequence)"

        lines = []
        for evt in self._events:
            lines.append(f"{evt.label.upper()}")
            lines.append(f"start: {evt.start_time:.1f}s")
            lines.append(f"end: {evt.end_time:.1f}s\n")
        return "\n".join(lines).strip()

    def format_gloss_string(self) -> str:
        """Format as a concatenated gloss sequence (e.g. 'HELP -> WATER -> HOSPITAL')."""
        if not self._events:
            return "(None)"
        return " -> ".join(evt.label.upper() for evt in self._events)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize sequence to list of event dictionaries."""
        return [evt.to_dict() for evt in self._events]

    def save_to_csv(self, file_path: Union[str, Path]) -> None:
        """Write all recorded events in this sequence to CSV format.

        Args:
            file_path: Output CSV file path.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(SignEvent.csv_header().split(","))
            for evt in self._events:
                writer.writerow(evt.to_csv_row().split(","))
        logger.info("Saved %d sign events to %s", len(self._events), path)
