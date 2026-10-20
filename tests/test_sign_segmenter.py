"""Tests for activity-based sign segmentation."""
import numpy as np

from src.realtime.sign_segmenter import SignSegmenter, resample_by_time


class _Frame:
    """Minimal stand-in for LandmarkFrame: only the fields the segmenter reads."""

    def __init__(self, t, hand_y=None):
        self.timestamp = t
        self.mask = np.zeros(93, dtype=bool)
        self.normalized_coords = np.zeros((93, 3), dtype=np.float32)
        if hand_y is not None:
            self.mask[:21] = True
            self.normalized_coords[:21, 1] = hand_y


def _feed(seg, spans, fps=10.0, t0=0.0):
    """spans: list of (seconds, hand_y or None). Returns [(time, segment)] for finished signs."""
    out, t = [], t0
    for dur, y in spans:
        for _ in range(int(round(dur * fps))):
            res = seg.update(_Frame(t, y))
            if res is not None:
                out.append((t, res))
            t += 1.0 / fps
    return out


def test_one_sign_emitted_once_after_hands_drop():
    seg = SignSegmenter(end_hold_sec=0.5, pad_before_sec=0.5, pad_after_sec=0.5)
    out = _feed(seg, [(1.0, 2.0), (1.0, 0.5), (2.0, 2.0)])
    assert len(out) == 1
    t_emit, frames = out[0]
    assert abs(t_emit - 2.5) < 0.11            # 0.5 s after the last active frame (1.9 s)
    assert frames[0].timestamp <= 0.5 + 1e-6   # includes rest before the sign
    assert frames[-1].timestamp >= 2.3        # and rest after it


def test_same_sign_twice_gives_two_segments():
    seg = SignSegmenter(end_hold_sec=0.5)
    out = _feed(seg, [(0.5, 2.0), (0.8, 0.5), (0.7, 2.0), (0.8, 0.5), (1.0, 2.0)])
    assert len(out) == 2


def test_short_twitch_and_missing_hands_ignored():
    seg = SignSegmenter(min_active_sec=0.3)
    assert _feed(seg, [(1.0, None), (0.2, 0.5), (1.0, None), (1.0, 2.0)]) == []


def test_hands_never_drop_forces_segment():
    seg = SignSegmenter(max_segment_sec=2.0)
    out = _feed(seg, [(0.5, 2.0), (5.0, 0.5)])
    assert len(out) == 2


def test_resample_by_time_returns_n_frames_in_order():
    frames = [_Frame(i * 0.1) for i in range(20)]
    win = resample_by_time(frames, 0.0, 1.9, 45)
    ts = [f.timestamp for f in win]
    assert len(win) == 45 and ts[0] == 0.0 and ts[-1] == frames[-1].timestamp
    assert all(a <= b for a, b in zip(ts, ts[1:]))
