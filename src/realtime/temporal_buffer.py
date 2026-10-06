"""
SignTalk AI - Temporal Sequence Buffer
Maintains a rolling temporal window of real-time LandmarkFrame instances
for sliding-window inference and spatiotemporal graph modeling.
"""

from typing import List, Optional, Dict, Any, Tuple
from collections import deque
import logging
import numpy as np

from src.realtime.types import LandmarkFrame
from src.data.node_schema import TOTAL_NODES, TARGET_SEQUENCE_LENGTH

logger = logging.getLogger("SignTalk.RealTime.TemporalBuffer")


class TemporalBuffer:
    """
    Fixed-length sliding temporal FIFO buffer accumulating continuous
    LandmarkFrames into model-ready windows of length T.
    """

    def __init__(
        self,
        max_length: int = TARGET_SEQUENCE_LENGTH,
        min_valid_frames: int = 15,
        interpolate_missing_frames: bool = True,
        max_consecutive_interpolated: int = 10
    ):
        """
        Args:
            max_length: Required temporal frame count T for model input (default: 45).
            min_valid_frames: Minimum required valid frames in window for quality check.
            interpolate_missing_frames: Whether to interpolate dropped frame gaps.
            max_consecutive_interpolated: Max dropped frames to bridge via linear interpolation.
        """
        self.max_length = int(max_length)
        self.min_valid_frames = int(min_valid_frames)
        self.interpolate_missing = interpolate_missing_frames
        self.max_consecutive_interpolated = max_consecutive_interpolated

        self._buffer: deque = deque(maxlen=self.max_length)
        self._total_appended = 0

    def append(self, frame: LandmarkFrame) -> None:
        """
        Appends an incoming LandmarkFrame to the buffer.
        If temporal gaps are detected between frame IDs and interpolation is enabled,
        interpolates intermediate frames to protect against ST-GCN accuracy collapse.
        """
        if not isinstance(frame, LandmarkFrame):
            raise TypeError(f"Expected LandmarkFrame, got {type(frame)}")

        if self.interpolate_missing and len(self._buffer) > 0:
            last_frame = self._buffer[-1]
            frame_gap = frame.frame_id - last_frame.frame_id - 1

            if 0 < frame_gap <= self.max_consecutive_interpolated:
                logger.debug(
                    f"Bridging {frame_gap} dropped frame(s) between #{last_frame.frame_id} and #{frame.frame_id}"
                )
                self._interpolate_and_insert_gap(last_frame, frame, frame_gap)

        self._buffer.append(frame)
        self._total_appended += 1

    def _interpolate_and_insert_gap(
        self,
        start_frame: LandmarkFrame,
        end_frame: LandmarkFrame,
        gap_count: int
    ) -> None:
        """Linearly interpolates landmark coordinates across dropped frames."""
        time_step = (end_frame.timestamp - start_frame.timestamp) / (gap_count + 1)

        for step in range(1, gap_count + 1):
            alpha = step / (gap_count + 1)
            interp_raw = (1.0 - alpha) * start_frame.raw_coords + alpha * end_frame.raw_coords
            interp_norm = (1.0 - alpha) * start_frame.normalized_coords + alpha * end_frame.normalized_coords
            interp_vis = (1.0 - alpha) * start_frame.visibility + alpha * end_frame.visibility
            interp_mask = start_frame.mask | end_frame.mask

            interp_lf = LandmarkFrame(
                frame_id=start_frame.frame_id + step,
                timestamp=start_frame.timestamp + step * time_step,
                raw_coords=interp_raw.astype(np.float32),
                normalized_coords=interp_norm.astype(np.float32),
                mask=interp_mask,
                visibility=interp_vis.astype(np.float32),
                modality_stats=start_frame.modality_stats,
                quality=start_frame.quality
            )
            self._buffer.append(interp_lf)
            self._total_appended += 1

    def length(self) -> int:
        """Returns the number of frames currently in the buffer."""
        return len(self._buffer)

    def __len__(self) -> int:
        return self.length()

    def is_ready(self) -> bool:
        """Returns True when buffer has accumulated exactly max_length frames."""
        return len(self._buffer) >= self.max_length

    def get_window(self) -> List[LandmarkFrame]:
        """
        Returns the current temporal window of frames.
        
        Raises:
            ValueError: If buffer does not have sufficient frames.
        """
        if not self.is_ready():
            raise ValueError(
                f"Insufficient frames in buffer: have {len(self._buffer)}, need {self.max_length}"
            )
        return list(self._buffer)

    def get_window_metadata(self) -> Dict[str, Any]:
        """Returns diagnostic metadata about the current buffer window."""
        if len(self._buffer) == 0:
            return {
                "count": 0,
                "is_ready": False,
                "start_time": 0.0,
                "end_time": 0.0,
                "duration_seconds": 0.0,
                "valid_frames": 0,
                "mean_quality": 0.0
            }

        start_time = self._buffer[0].timestamp
        end_time = self._buffer[-1].timestamp
        duration = max(0.0, end_time - start_time)

        valid_count = sum(1 for f in self._buffer if f.quality.is_valid)
        qualities = [f.quality.quality_score for f in self._buffer]
        mean_quality = float(np.mean(qualities)) if qualities else 0.0

        return {
            "count": len(self._buffer),
            "is_ready": self.is_ready(),
            "start_time": round(start_time, 4),
            "end_time": round(end_time, 4),
            "duration_seconds": round(duration, 3),
            "valid_frames": valid_count,
            "mean_quality": round(mean_quality, 4),
            "start_frame_id": self._buffer[0].frame_id,
            "end_frame_id": self._buffer[-1].frame_id
        }

    def is_window_valid(self) -> bool:
        """
        Applies upstream quality gate: checks whether the window contains
        sufficient valid frames and active hand detections.
        """
        if not self.is_ready():
            return False

        meta = self.get_window_metadata()
        if meta["valid_frames"] < self.min_valid_frames:
            return False

        # Active hand requirement (at least 15% of frames have an active hand)
        hand_count = sum(1 for f in self._buffer if f.modality_stats.either_hand_detected)
        if (hand_count / self.max_length) < 0.15:
            return False

        return True

    def clear(self) -> None:
        """Clears all frames from buffer."""
        self._buffer.clear()

    @property
    def total_appended(self) -> int:
        return self._total_appended
