"""Sign event representation for real-time sign detection.

Represents a completed or sufficiently expressed discrete sign event
detected from the temporal prediction stream.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import uuid


@dataclass
class SignEvent:
    """Discrete sign event detected by the real-time pipeline.

    Attributes:
        event_id: Unique identifier for the sign event.
        class_id: Predicted class index.
        label: Human-readable gloss or sign label.
        start_time: Wall-clock timestamp (seconds) when sign expression began.
        end_time: Wall-clock timestamp (seconds) when sign expression completed.
        duration_ms: Duration of the sign event in milliseconds.
        confidence: Average or peak model confidence score over the event window.
        input_quality: Average landmark quality score during the event.
        stability_score: Stability / consensus metric [0.0, 1.0] when event was confirmed.
        source_window_start: Buffer frame index or timestamp corresponding to window start.
        source_window_end: Buffer frame index or timestamp corresponding to window end.
        metadata: Optional dictionary with extra telemetry or model state.
    """

    event_id: str
    class_id: int
    label: str
    start_time: float
    end_time: float
    duration_ms: float
    confidence: float
    input_quality: float
    stability_score: float
    source_window_start: int
    source_window_end: int
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def create(
        cls,
        class_id: int,
        label: str,
        start_time: float,
        end_time: float,
        confidence: float,
        input_quality: float,
        stability_score: float,
        source_window_start: int = 0,
        source_window_end: int = 0,
        event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SignEvent":
        """Factory method computing duration_ms and default event_id."""
        if event_id is None:
            event_id = f"evt_{uuid.uuid4().hex[:8]}"

        duration_ms = max(0.0, (end_time - start_time) * 1000.0)
        return cls(
            event_id=event_id,
            class_id=class_id,
            label=label,
            start_time=round(start_time, 4),
            end_time=round(end_time, 4),
            duration_ms=round(duration_ms, 2),
            confidence=round(confidence, 4),
            input_quality=round(input_quality, 4),
            stability_score=round(stability_score, 4),
            source_window_start=source_window_start,
            source_window_end=source_window_end,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert sign event to dictionary."""
        return asdict(self)

    @classmethod
    def csv_header(cls) -> str:
        """Standard CSV header matching project specification."""
        return (
            "event_id,label,class_id,start_time,end_time,duration_ms,"
            "confidence,input_quality,stability_score,source_window_start,source_window_end"
        )

    def to_csv_row(self) -> str:
        """Format as CSV row matching project specification."""
        return (
            f"{self.event_id},{self.label},{self.class_id},{self.start_time:.4f},"
            f"{self.end_time:.4f},{self.duration_ms:.2f},{self.confidence:.4f},"
            f"{self.input_quality:.4f},{self.stability_score:.4f},"
            f"{self.source_window_start},{self.source_window_end}"
        )
