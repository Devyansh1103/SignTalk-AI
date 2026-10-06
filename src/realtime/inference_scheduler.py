"""
SignTalk AI - Real-Time Inference Scheduler
Coordinates the temporal sliding cadence, enforces stride pacing,
and implements stale-window dropping policies to prevent queue lag.
"""

import time
import logging
from typing import Optional

from src.realtime.temporal_buffer import TemporalBuffer

logger = logging.getLogger("SignTalk.RealTime.Scheduler")


class InferenceScheduler:
    """
    Manages sliding-window stride cadence and coordinates inference triggers.
    """

    def __init__(
        self,
        stride: int = 5,
        drop_stale_windows: bool = True
    ):
        """
        Args:
            stride: Number of new frames to accumulate before triggering the next inference (default: 5).
            drop_stale_windows: If True, skips intermediate stale windows if inference is currently running.
        """
        self.stride = max(1, int(stride))
        self.drop_stale_windows = drop_stale_windows

        self._frames_since_last_inference = 0
        self._total_frames = 0
        self._inferences_triggered = 0
        self._windows_skipped = 0
        self._is_inferring = False

    def should_infer(self, buffer: TemporalBuffer) -> bool:
        """
        Evaluates whether an inference pass should execute given current buffer state.
        
        Conditions:
          1. Buffer must be full (len == max_length).
          2. Enough new frames have elapsed since last inference (count >= stride).
          3. If drop_stale_windows is enabled and an inference is already running, drops window.
        """
        if not buffer.is_ready():
            return False

        if self._frames_since_last_inference < self.stride:
            return False

        if self._is_inferring and self.drop_stale_windows:
            self._windows_skipped += 1
            logger.debug("Dropped stale sliding window because model inference is currently in flight.")
            return False

        return True

    def on_frame_appended(self) -> None:
        """Invoked each time a new frame is appended to the temporal buffer."""
        self._total_frames += 1
        self._frames_since_last_inference += 1

    def record_inference_start(self) -> None:
        """Marks that a model forward pass has started."""
        self._is_inferring = True
        self._inferences_triggered += 1
        self._frames_since_last_inference = 0

    def record_inference_finish(self) -> None:
        """Marks that a model forward pass has finished."""
        self._is_inferring = False

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def inferences_triggered(self) -> int:
        return self._inferences_triggered

    @property
    def windows_skipped(self) -> int:
        return self._windows_skipped

    @property
    def frames_since_last_inference(self) -> int:
        return self._frames_since_last_inference

    def reset(self) -> None:
        """Resets scheduler state."""
        self._frames_since_last_inference = 0
        self._total_frames = 0
        self._inferences_triggered = 0
        self._windows_skipped = 0
        self._is_inferring = False
