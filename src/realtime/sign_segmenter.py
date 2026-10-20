"""Activity-based sign segmentation for real-time recognition.

The ST-GCN is trained on whole INCLUDE clips (rest -> sign -> rest, resampled to 45 frames).
A sliding 1.8 s window shows the model half-signs and transitions, which it confidently
mislabels, and the same word cannot repeat until the window has slid past it. Instead this
segmenter watches when the hands rise into signing space and drop again, and hands the whole
sign (plus a little rest on each side, like the training clips) to the model exactly once.

Measured on 150 INCLUDE test clips through the live extractor (4:3, 12.5 FPS):
sliding window 69% top-1, activity segment +/-0.5 s 81%.
"""
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from src.realtime.types import LandmarkFrame


def resample_by_time(frames: List[LandmarkFrame], t_start: float, t_end: float, n: int) -> List[LandmarkFrame]:
    """Pick n frames evenly spaced in time over [t_start, t_end] (nearest frame)."""
    ts = np.asarray([f.timestamp for f in frames], dtype=np.float64)
    targets = np.linspace(t_start, t_end, n)
    hi = np.clip(np.searchsorted(ts, targets), 1, len(ts) - 1)
    lo = hi - 1
    pick = np.where(np.abs(targets - ts[lo]) <= np.abs(ts[hi] - targets), lo, hi)
    return [frames[int(i)] for i in pick]


class SignSegmenter:
    """Finds sign boundaries from hand height and returns each finished sign once.

    A frame is active when any detected hand point is in signing space (normalized y, relative
    to mid-shoulder in shoulder widths, <= active_max_y). INCLUDE rest pose sits at y ~2.0,
    signs at y < 1.3.
    """

    def __init__(
        self,
        active_max_y: float = 1.35,
        min_start_frames: int = 2,
        min_active_sec: float = 0.3,
        end_hold_sec: float = 0.5,
        pad_before_sec: float = 0.5,
        pad_after_sec: float = 0.5,
        max_segment_sec: float = 4.0,
        history_sec: float = 7.0,
    ) -> None:
        self.active_max_y = active_max_y
        self.min_start_frames = max(1, min_start_frames)
        self.min_active_sec = min_active_sec
        self.end_hold_sec = end_hold_sec
        self.pad_before_sec = pad_before_sec
        self.pad_after_sec = pad_after_sec
        self.max_segment_sec = max_segment_sec
        self.history_sec = max(history_sec, max_segment_sec + pad_before_sec + pad_after_sec + 1.0)
        self._history: Deque[LandmarkFrame] = deque()
        self.reset()

    @property
    def in_segment(self) -> bool:
        return self._in_segment

    def reset(self) -> None:
        self._history.clear()
        self._in_segment = False
        self._run = 0
        self._run_start = 0.0
        self._seg_start = 0.0
        self._last_active = 0.0

    def is_active(self, frame: LandmarkFrame) -> bool:
        hands = np.where(frame.mask[:42])[0]
        return len(hands) > 0 and float(frame.normalized_coords[hands, 1].min()) <= self.active_max_y

    def update(self, frame: LandmarkFrame) -> Optional[List[LandmarkFrame]]:
        """Add a frame. Returns the frames of a finished sign (start-pad .. end+pad), else None."""
        ts = frame.timestamp
        self._history.append(frame)
        while self._history and self._history[0].timestamp < ts - self.history_sec:
            self._history.popleft()

        active = self.is_active(frame)
        if not self._in_segment:
            if active:
                if self._run == 0:
                    self._run_start = ts
                self._run += 1
                if self._run >= self.min_start_frames:
                    self._in_segment = True
                    self._seg_start = self._run_start
                    self._last_active = ts
            else:
                self._run = 0
            return None

        if active:
            self._last_active = ts
        if ts - self._last_active >= self.end_hold_sec:
            # Hands back at rest: the sign is over
            self._in_segment = False
            self._run = 0
            if self._last_active - self._seg_start < self.min_active_sec:
                return None
            return self._slice(self._seg_start - self.pad_before_sec, self._last_active + self.pad_after_sec)
        if ts - self._seg_start >= self.max_segment_sec:
            # Hands never dropped: classify what we have and start a fresh segment
            seg = self._slice(self._seg_start - self.pad_before_sec, ts)
            self._seg_start = ts
            return seg
        return None

    def _slice(self, t0: float, t1: float) -> Optional[List[LandmarkFrame]]:
        frames = [f for f in self._history if t0 <= f.timestamp <= t1]
        return frames if len(frames) >= 2 else None
