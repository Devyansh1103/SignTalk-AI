"""
SignTalk AI - Offline vs. Real-Time Preprocessing Consistency Test
Phase 4 Part 1 Critical Test (Section 29)

Validates that running the identical video sample through:
  1. Offline Phase 2 Pipeline (FrameExtractor -> LandmarkExtractor -> CoordinateNormalizer)
  2. Real-Time Streaming Pipeline (Camera -> FrameProcessor -> LandmarkStream)
produces mathematically consistent landmark representations:
  - Exact node order alignment
  - Identical tensor shapes: (93, 3)
  - Identical detection masks
  - Numerical coordinate divergence within strict floating-point tolerance (<= 1e-4)
"""

import os
import pytest
import cv2
import numpy as np

from src.data.frame_extractor import FrameExtractor
from src.preprocessing.landmark_extractor import LandmarkExtractor as OfflineLandmarkExtractor
from src.preprocessing.normalizer import CoordinateNormalizer as OfflineNormalizer
from src.realtime.landmark_stream import RealTimeLandmarkStream
from src.realtime.realtime_config import RealTimeConfig
from src.realtime.types import FramePacket
from src.data.node_schema import TOTAL_NODES, CHANNELS


@pytest.fixture(scope="module")
def sample_video():
    path = "data/interim/landmark_pilot/pilot_sample_01.mp4"
    if not os.path.exists(path):
        pytest.skip(f"Test sample video not found: {path}")
    return path


def test_offline_vs_realtime_consistency(sample_video):
    """
    Direct comparison between offline pipeline and real-time streaming pipeline
    on identical frames from pilot_sample_01.mp4.
    """
    # -------------------------------------------------------------
    # 1. Setup Pipelines
    # -------------------------------------------------------------
    # Offline Pipeline Components
    offline_extractor = OfflineLandmarkExtractor(enable_face_mesh=False)
    offline_normalizer = OfflineNormalizer(method="torso_scale")

    # Real-Time Pipeline Component (anchor smoothing disabled for 1-to-1 equivalence)
    rt_cfg = RealTimeConfig()
    rt_cfg.processing.flip_horizontal = False  # Consistent orientation with raw video
    rt_cfg.normalization.use_temporal_anchor_smoothing = False
    rt_cfg.runtime.display_preview = False
    rt_stream = RealTimeLandmarkStream(rt_cfg)

    # -------------------------------------------------------------
    # 2. Extract Sample Frames from Video
    # -------------------------------------------------------------
    cap = cv2.VideoCapture(sample_video)
    test_frame_indices = [10, 25, 40]  # Representative frames with active signing
    tested_frames_count = 0

    max_coordinate_diffs = []
    mask_mismatches = 0

    try:
        for f_idx in test_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, bgr_frame = cap.read()
            if not ret or bgr_frame is None:
                continue

            tested_frames_count += 1
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            # --- Offline Pipeline Execution ---
            off_res = offline_extractor.extract_frame_landmarks(rgb_frame)
            off_raw = off_res["coords"]
            off_mask = off_res["mask"]
            off_norm, _ = offline_normalizer.normalize_frame(off_raw, off_mask)

            # --- Real-Time Pipeline Execution ---
            rt_packet = FramePacket(
                frame_id=f_idx,
                timestamp=float(f_idx) / 25.0,
                capture_time=1.0,
                image=bgr_frame,  # Feeds raw BGR as camera does
                width=bgr_frame.shape[1],
                height=bgr_frame.shape[0],
                is_rgb=False,
                is_flipped=False
            )
            rt_lf = rt_stream.process_frame(rt_packet)

            # ---------------------------------------------------------
            # 3. Assertions & Numerical Difference Quantification
            # ---------------------------------------------------------
            # A. Shape Verification
            assert off_norm.shape == (TOTAL_NODES, CHANNELS)
            assert rt_lf.normalized_coords.shape == (TOTAL_NODES, CHANNELS)
            assert off_mask.shape == (TOTAL_NODES,)
            assert rt_lf.mask.shape == (TOTAL_NODES,)

            # B. Mask Agreement
            mask_diff = np.sum(off_mask != rt_lf.mask)
            mask_mismatches += mask_diff
            assert mask_diff == 0, f"Frame {f_idx}: Detection mask mismatch in {mask_diff} nodes"

            # C. Coordinate Agreement on Observed Nodes
            if np.any(off_mask):
                valid_diff = np.abs(off_norm[off_mask] - rt_lf.normalized_coords[off_mask])
                max_diff = float(np.max(valid_diff))
                mean_diff = float(np.mean(valid_diff))
                max_coordinate_diffs.append(max_diff)

                # Tolerance: max L-infinity coordinate difference must be <= 1e-4
                assert max_diff <= 1e-4, (
                    f"Frame {f_idx}: Numerical difference exceeded tolerance. "
                    f"Max diff = {max_diff:.6e}, Mean diff = {mean_diff:.6e}"
                )

    finally:
        cap.release()
        offline_extractor.close()
        rt_stream.close()

    assert tested_frames_count > 0, "No valid test frames could be evaluated from sample video"

    overall_max_diff = max(max_coordinate_diffs) if max_coordinate_diffs else 0.0
    print(f"\n[Consistency Report] Evaluated {tested_frames_count} frames.")
    print(f"Mask Mismatches: {mask_mismatches}")
    print(f"Overall Max Coordinate Difference: {overall_max_diff:.8e} (Tolerance <= 1.0e-4)")
    assert overall_max_diff <= 1e-4
