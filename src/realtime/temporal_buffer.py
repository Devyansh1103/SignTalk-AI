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
        max_consecutive_interpolated: int = 10,
        ready_window_sec: Optional[float] = None
    ):
        """
        Args:
            max_length: Required temporal frame count T for model input (default: 45).
            min_valid_frames: Minimum required valid frames in window for quality check.
            interpolate_missing_frames: Whether to interpolate dropped frame gaps.
            max_consecutive_interpolated: Max dropped frames to bridge via linear interpolation.
            ready_window_sec: If set (time-aligned windows), the buffer is ready once it
                spans this many seconds with at least min_valid_frames frames, instead of
                waiting for max_length frames (which takes ~7 s at 6-7 FPS on CPU).
        """
        self.max_length = int(max_length)
        self.min_valid_frames = int(min_valid_frames)
        self.interpolate_missing = interpolate_missing_frames
        self.max_consecutive_interpolated = max_consecutive_interpolated
        self.ready_window_sec = float(ready_window_sec) if ready_window_sec else None

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
        """Returns True when buffer has accumulated max_length frames, or (time-aligned
        mode) when it covers ready_window_sec seconds with at least min_valid_frames frames."""
        n = len(self._buffer)
        if n >= self.max_length:
            return True
        if self.ready_window_sec is None or n < max(2, self.min_valid_frames):
            return False
        span = self._buffer[-1].timestamp - self._buffer[0].timestamp
        return span >= self.ready_window_sec

    def get_time_aligned_window(
        self,
        duration_sec: float = 1.8,
        num_frames: Optional[int] = None,
    ) -> List[LandmarkFrame]:
        """
        Returns exactly `num_frames` frames covering the LAST `duration_sec` seconds,
        picked by timestamp (nearest frame). This makes the model input independent of
        the real camera/processing FPS: the model was trained on ~1.8 s signs sampled at
        25 FPS, so at e.g. 10 FPS the last 1.8 s (18 frames) is stretched to 45 frames
        instead of the window silently covering 4.5 s.

        If less than `duration_sec` of history exists, the whole available span is used.
        """
        n = int(num_frames or self.max_length)
        frames = list(self._buffer)
        if len(frames) < 2:
            return frames
        ts = np.asarray([f.timestamp for f in frames], dtype=np.float64)
        t_end = ts[-1]
        t_start = max(ts[0], t_end - float(duration_sec))
        targets = np.linspace(t_start, t_end, n)
        hi = np.clip(np.searchsorted(ts, targets), 1, len(ts) - 1)
        lo = hi - 1
        pick = np.where(np.abs(targets - ts[lo]) <= np.abs(ts[hi] - targets), lo, hi)
        return [frames[int(i)] for i in pick]

    def get_window(self) -> List[LandmarkFrame]:
        """
        Returns the current temporal window of frames.
        
        Raises:
            ValueError: If buffer does not have sufficient frames.
        """
        if len(self._buffer) < self.max_length:
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

    def is_window_valid(self, min_hand_ratio: float = 0.20) -> bool:
        """
        Applies upstream quality gate: checks whether the window contains
        sufficient valid frames and active hand detections.
        """
        if not self.is_ready():
            return False

        frames = self._gate_frames()
        valid_count = sum(1 for f in frames if f.quality.is_valid)
        # At low FPS a 1.8 s window holds fewer than min_valid_frames frames; require half of them
        if valid_count < min(self.min_valid_frames, int(np.ceil(0.5 * len(frames)))):
            return False

        # Active hand requirement (at least 20% of frames have an active hand detected)
        hand_count = sum(1 for f in frames if f.modality_stats.either_hand_detected)
        if (hand_count / len(frames)) < min_hand_ratio:
            return False

        return True

    def _gate_frames(self) -> List[LandmarkFrame]:
        """Frames the activity gates judge: the same last ready_window_sec seconds the model
        sees in time-aligned mode (otherwise the whole buffer). Judging only the newest frames
        rejected signs exactly when they completed, because the hands were already dropping."""
        frames = list(self._buffer)
        if self.ready_window_sec is None or not frames:
            return frames
        t_start = frames[-1].timestamp - self.ready_window_sec
        return [f for f in frames if f.timestamp >= t_start]

    def has_hand_movement(self, min_velocity: float = 0.018) -> bool:
        """
        Checks whether hands exhibit kinematic signing movement across the window.
        Returns False if hands are completely stationary (e.g. resting on lap/desk).
        """
        if len(self._buffer) < 5:
            return False

        frames = self._gate_frames()
        displacements = []
        for i in range(0, len(frames) - 2, 2):
            f1 = frames[i]
            f2 = frames[i + 2]
            # Hand nodes: 0 through 41
            shared_hands = f1.mask[:42] & f2.mask[:42]
            if np.any(shared_hands):
                d = np.linalg.norm(
                    f2.normalized_coords[:42][shared_hands] - f1.normalized_coords[:42][shared_hands],
                    axis=1
                )
                displacements.append(float(np.mean(d)))

        if not displacements:
            return False
        return float(np.mean(displacements)) >= min_velocity

    def hands_in_signing_space(self, max_y: float = 1.35) -> bool:
        """
        Checks whether detected hands are elevated within the signing space
        (chest/neck/face area, where normalized y relative to mid-shoulder <= max_y).
        Hands resting in lap or on desk typically have y > 1.35.
        """
        if len(self._buffer) < 5:
            return False

        recent_frames = self._gate_frames() if self.ready_window_sec is not None else list(self._buffer)[-10:]
        y_positions = []
        for f in recent_frames:
            valid_hand_indices = np.where(f.mask[:42])[0]
            if len(valid_hand_indices) > 0:
                y_positions.extend(f.normalized_coords[valid_hand_indices, 1].tolist())

        if not y_positions:
            return False
        return float(np.min(y_positions)) <= max_y

    def clear(self) -> None:
        """Clears all frames from buffer."""
        self._buffer.clear()

    @property
    def total_appended(self) -> int:
        return self._total_appended
