"""
SignTalk AI - Real-Time Landmark Stream Pipeline
Orchestrates the real-time processing flow:
  Camera Frame Packet
          ↓
  Frame Preprocessor (RGB conversion, flip, resize)
          ↓
  MediaPipe Multimodal Extractor (Hands, Pose, Face)
          ↓
  Deterministic Landmark Fusion (Canonical 93-Node Schema)
          ↓
  Landmark Normalizer (Torso-Scale with Temporal Anchor Smoothing)
          ↓
  Quality Validation & Diagnostic Anomaly Detection
          ↓
  Canonical LandmarkFrame Stream & Optional Visual HUD
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque
import cv2
import numpy as np

from src.data.node_schema import (
    TOTAL_NODES,
    CHANNELS,
    LEFT_HAND_RANGE,
    RIGHT_HAND_RANGE,
    UPPER_POSE_RANGE,
    FACE_CONTOURS_RANGE,
    validate_node_tensor
)
from src.preprocessing.landmark_extractor import LandmarkExtractor
from src.preprocessing.normalizer import CoordinateNormalizer
from src.preprocessing.quality_checker import QualityChecker
from src.realtime.types import (
    FramePacket,
    ModalityDetection,
    QualityReport,
    LandmarkFrame,
    StreamMetrics
)
from src.realtime.realtime_config import RealTimeConfig
from src.realtime.frame_processor import FrameProcessor

logger = logging.getLogger("SignTalk.RealTime.Stream")


class RealTimeLandmarkStream:
    """
    High-performance real-time landmark streaming pipeline converting raw
    camera frames into validated, normalized 93-node spatiotemporal landmark frames.
    """

    def __init__(self, config: Optional[RealTimeConfig] = None):
        self.config = config or RealTimeConfig()

        # 1. Initialize Frame Preprocessor
        self.frame_processor = FrameProcessor(self.config.processing)

        # 2. Initialize MediaPipe Multimodal Extractor
        mp_cfg = self.config.mediapipe
        self.extractor = LandmarkExtractor(
            pose_model_path=mp_cfg.pose_model_path,
            hand_model_path=mp_cfg.hand_model_path,
            face_model_path=mp_cfg.face_model_path,
            pose_confidence=mp_cfg.pose_detection_confidence,
            hand_confidence=mp_cfg.hand_detection_confidence,
            face_confidence=mp_cfg.face_detection_confidence,
            enable_face_mesh=mp_cfg.enable_face
        )

        # 3. Normalizer & Temporal Anchor State
        norm_cfg = self.config.normalization
        self.normalizer = CoordinateNormalizer(
            method=norm_cfg.method,
            min_scale_epsilon=norm_cfg.min_scale_epsilon,
            z_scale_factor=norm_cfg.z_scale_factor
        )
        self.use_anchor_smoothing = norm_cfg.use_temporal_anchor_smoothing
        self.anchor_alpha = float(norm_cfg.anchor_smoothing_alpha)
        self._smoothed_center: Optional[np.ndarray] = None
        self._smoothed_scale: Optional[float] = None

        # 4. Quality Validator
        q_cfg = self.config.quality
        self.quality_checker = QualityChecker(
            weights=q_cfg.weights,
            threshold_good=q_cfg.classification_thresholds.get("good", 0.75),
            threshold_acceptable=q_cfg.classification_thresholds.get("acceptable", 0.50),
            threshold_review=q_cfg.classification_thresholds.get("review", 0.35),
            min_valid_frames=q_cfg.min_valid_frames_in_sequence
        )
        self.jump_threshold = q_cfg.jump_distance_threshold

        # 5. Temporal Continuity State
        self._previous_coords: Optional[np.ndarray] = None
        self._previous_mask: Optional[np.ndarray] = None
        self._previous_lh_detected = False
        self._previous_rh_detected = False

        # 6. Stream Performance Tracking
        self._processed_count = 0
        self._dropped_count = 0
        self._valid_frame_count = 0
        self._warmup_frames = self.config.stream.warmup_frames

        # Latency histories for statistics (excluding warmup)
        self._history_capture_ms: deque = deque(maxlen=200)
        self._history_prep_ms: deque = deque(maxlen=200)
        self._history_extract_ms: deque = deque(maxlen=200)
        self._history_norm_ms: deque = deque(maxlen=200)
        self._history_quality_ms: deque = deque(maxlen=200)
        self._history_total_ms: deque = deque(maxlen=200)

        self._stream_start_time: Optional[float] = None
        self._last_processed_time: Optional[float] = None

        # 7. Privacy-First Optional Debug Recording
        self._record_landmarks = self.config.debug.record_landmarks
        self._record_video = self.config.debug.record_video
        self._recorded_frames: List[Dict[str, Any]] = []
        self._video_writer: Optional[cv2.VideoWriter] = None

        if self._record_landmarks or self._record_video:
            os.makedirs(self.config.debug.record_dir, exist_ok=True)
            logger.warning(
                f"[PRIVACY NOTICE] Debug recording enabled! Storing artifacts to {self.config.debug.record_dir}. "
                "Ensure compliance with privacy standards."
            )

    def process_frame(self, packet: FramePacket) -> LandmarkFrame:
        """
        Executes end-to-end landmark ingestion on an incoming FramePacket.
        
        Args:
            packet: Raw FramePacket from Camera.

        Returns:
            Validated, normalized LandmarkFrame conforming to 93-node contract.
        """
        t_pipeline_start = time.perf_counter()

        # Capture latency (time elapsed between hardware grab and start of pipeline)
        capture_latency_ms = max(0.0, (t_pipeline_start - packet.timestamp) * 1000.0)

        # Step 1: Vision Preprocessing
        t0 = time.perf_counter()
        processed_packet = self.frame_processor.process(packet)
        t_prep_end = time.perf_counter()
        prep_latency_ms = (t_prep_end - t0) * 1000.0

        # Step 2: MediaPipe Landmark Extraction
        t0 = time.perf_counter()
        raw_res = self.extractor.extract_frame_landmarks(processed_packet.image)
        t_extract_end = time.perf_counter()
        extract_latency_ms = (t_extract_end - t0) * 1000.0

        raw_coords = raw_res["coords"]       # (93, 3)
        raw_visibility = raw_res["visibility"] # (93,)
        raw_mask = raw_res["mask"]           # (93,)
        stats = raw_res["stats"]

        modality_detection = ModalityDetection(
            pose_detected=stats.get("pose_detected", False),
            left_hand_detected=stats.get("left_hand_detected", False),
            right_hand_detected=stats.get("right_hand_detected", False),
            face_detected=stats.get("face_detected", False),
            pose_confidence=float(stats.get("pose_confidence", 0.0)),
            left_hand_confidence=float(stats.get("left_hand_confidence", 0.0)),
            right_hand_confidence=float(stats.get("right_hand_confidence", 0.0)),
            face_confidence=float(stats.get("face_confidence", 0.0))
        )

        # Step 3: Normalization (Torso-Scale with Optional Temporal Anchor Smoothing)
        t0 = time.perf_counter()
        norm_coords, norm_meta = self._normalize_realtime_frame(raw_coords, raw_mask)
        t_norm_end = time.perf_counter()
        norm_latency_ms = (t_norm_end - t0) * 1000.0

        # Step 4: Quality Validation & Anomaly Detection
        t0 = time.perf_counter()
        quality_report = self._validate_frame_quality(
            norm_coords=norm_coords,
            mask=raw_mask,
            modality_detection=modality_detection,
            raw_coords=raw_coords
        )
        t_quality_end = time.perf_counter()
        quality_latency_ms = (t_quality_end - t0) * 1000.0

        t_pipeline_end = time.perf_counter()
        total_latency_ms = (t_pipeline_end - packet.timestamp) * 1000.0

        # Create Canonical LandmarkFrame
        landmark_frame = LandmarkFrame(
            frame_id=packet.frame_id,
            timestamp=packet.timestamp,
            raw_coords=raw_coords,
            normalized_coords=norm_coords,
            mask=raw_mask,
            visibility=raw_visibility,
            modality_stats=modality_detection,
            quality=quality_report,
            capture_latency_ms=round(capture_latency_ms, 3),
            preprocessing_latency_ms=round(prep_latency_ms, 3),
            extraction_latency_ms=round(extract_latency_ms, 3),
            normalization_latency_ms=round(norm_latency_ms, 3),
            quality_check_latency_ms=round(quality_latency_ms, 3),
            total_latency_ms=round(total_latency_ms, 3)
        )

        # Step 5: Update Stream Accounting & History
        self._record_frame_metrics(landmark_frame)

        # Step 6: Optional Debug Recording
        if self._record_landmarks:
            self._record_landmark_data(landmark_frame)

        # Update previous state
        self._previous_coords = norm_coords.copy()
        self._previous_mask = raw_mask.copy()
        self._previous_lh_detected = modality_detection.left_hand_detected
        self._previous_rh_detected = modality_detection.right_hand_detected

        return landmark_frame

    def _normalize_realtime_frame(
        self,
        raw_coords: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Normalizes a single real-time frame using torso-scale.
        Applies EMA smoothing to the torso center and scale factor across frames
        to prevent high-frequency jitter caused by slight shoulder detection noise.
        """
        # Node 47 = Left Shoulder, Node 48 = Right Shoulder
        ls_valid = mask[47]
        rs_valid = mask[48]

        if ls_valid and rs_valid:
            p_ls = raw_coords[47]
            p_rs = raw_coords[48]
            curr_center = (p_ls + p_rs) / 2.0
            shoulder_dist = np.linalg.norm(p_ls - p_rs)
            curr_scale = max(float(shoulder_dist), self.config.normalization.min_scale_epsilon)
        elif ls_valid:
            curr_center = raw_coords[47].copy()
            curr_scale = 0.25
        elif rs_valid:
            curr_center = raw_coords[48].copy()
            curr_scale = 0.25
        else:
            # Fallback to valid upper pose joints
            valid_pose = np.where(mask[42:53])[0]
            if len(valid_pose) > 0:
                curr_center = np.mean(raw_coords[42 + valid_pose], axis=0)
                curr_scale = 0.25
            else:
                curr_center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
                curr_scale = 1.0

        # Temporal anchor smoothing (EMA)
        if self.use_anchor_smoothing and self._smoothed_center is not None and self._smoothed_scale is not None:
            c_torso = (1.0 - self.anchor_alpha) * self._smoothed_center + self.anchor_alpha * curr_center
            scale = (1.0 - self.anchor_alpha) * self._smoothed_scale + self.anchor_alpha * curr_scale
        else:
            c_torso = curr_center.copy()
            scale = curr_scale

        # Cache anchor state
        self._smoothed_center = c_torso.copy()
        self._smoothed_scale = float(scale)

        # Apply torso-scale centering across all 93 nodes
        norm_coords = np.zeros_like(raw_coords, dtype=np.float32)
        z_factor = self.config.normalization.z_scale_factor

        for i in range(TOTAL_NODES):
            if mask[i]:
                norm_coords[i, 0] = (raw_coords[i, 0] - c_torso[0]) / scale
                norm_coords[i, 1] = (raw_coords[i, 1] - c_torso[1]) / scale
                norm_coords[i, 2] = (raw_coords[i, 2] - c_torso[2]) / scale * z_factor
            else:
                norm_coords[i] = 0.0

        meta = {
            "center": c_torso.tolist(),
            "scale": float(scale),
            "smoothed": self.use_anchor_smoothing
        }
        return norm_coords, meta

    def _validate_frame_quality(
        self,
        norm_coords: np.ndarray,
        mask: np.ndarray,
        modality_detection: ModalityDetection,
        raw_coords: np.ndarray
    ) -> QualityReport:
        """
        Validates frame-level quality, numerical sanity, missing ratios,
        and temporal displacement jumps.
        """
        warnings = []

        # 1. NaN and Infinite checks
        has_nan_or_inf = bool(np.isnan(norm_coords).any() or np.isinf(norm_coords).any())
        if has_nan_or_inf:
            warnings.append("NaN or Inf values detected in landmark coordinates")

        # 2. Coordinate range verification (raw coordinates in normalized image space)
        valid_raw = raw_coords[mask]
        coord_in_range = True
        if len(valid_raw) > 0:
            if np.any(valid_raw < -0.5) or np.any(valid_raw > 2.0):
                coord_in_range = False
                warnings.append("Raw coordinates outside plausible normalized bounds [-0.5, 2.0]")

        # 3. Missing Landmark Ratio
        total_valid = int(np.sum(mask))
        missing_ratio = float((TOTAL_NODES - total_valid) / TOTAL_NODES)

        # 4. Modality-specific quality score
        stat_dict = {
            "pose_detected": modality_detection.pose_detected,
            "pose_confidence": modality_detection.pose_confidence,
            "left_hand_detected": modality_detection.left_hand_detected,
            "left_hand_confidence": modality_detection.left_hand_confidence,
            "right_hand_detected": modality_detection.right_hand_detected,
            "right_hand_confidence": modality_detection.right_hand_confidence,
            "face_detected": modality_detection.face_detected
        }
        quality_score = self.quality_checker.compute_frame_quality(stat_dict)

        # 5. Temporal discontinuity (jump) detection
        sudden_jumps = False
        max_jump = 0.0
        if self._previous_coords is not None and self._previous_mask is not None:
            mutual_mask = mask & self._previous_mask
            if np.any(mutual_mask):
                displacements = np.linalg.norm(
                    norm_coords[mutual_mask] - self._previous_coords[mutual_mask],
                    axis=1
                )
                max_jump = float(np.max(displacements))
                if max_jump > self.jump_threshold:
                    sudden_jumps = True
                    warnings.append(
                        f"Large temporal jump detected: max joint displacement {max_jump:.3f} > {self.jump_threshold:.3f}"
                    )

        # 6. Chirality & hand identity consistency check
        if self._previous_coords is not None:
            if not self._previous_lh_detected and modality_detection.left_hand_detected and not modality_detection.right_hand_detected:
                if self._previous_rh_detected:
                    warnings.append("Hand chirality switch: right hand disappeared while left hand appeared")

        # Classification decision
        if has_nan_or_inf:
            classification = "REJECT"
            reason = "Numerical NaN or Inf anomaly"
        elif not modality_detection.pose_detected:
            classification = "REJECT"
            reason = "Signer pose not detected"
        elif not modality_detection.either_hand_detected:
            classification = "REVIEW"
            reason = "No active hands detected in signing space"
        elif quality_score >= self.quality_checker.threshold_good:
            classification = "GOOD"
            reason = "Optimal pose and hand tracking"
        elif quality_score >= self.quality_checker.threshold_acceptable:
            classification = "ACCEPTABLE"
            reason = "Acceptable tracking quality"
        elif quality_score >= self.quality_checker.threshold_review:
            classification = "REVIEW"
            reason = "Marginal tracking quality"
        else:
            classification = "REJECT"
            reason = f"Quality score {quality_score:.2f} below threshold"

        is_valid = classification in {"GOOD", "ACCEPTABLE"}

        return QualityReport(
            is_valid=is_valid,
            quality_score=round(quality_score, 4),
            classification=classification,
            missing_ratio=round(missing_ratio, 4),
            has_nan_or_inf=has_nan_or_inf,
            coordinate_in_range=coord_in_range,
            sudden_jumps_detected=sudden_jumps,
            max_jump_distance=round(max_jump, 4),
            rejection_reason=reason,
            warnings=warnings
        )

    def _record_frame_metrics(self, lf: LandmarkFrame):
        """Records latency and throughput statistics."""
        self._processed_count += 1
        now = time.perf_counter()

        if self._stream_start_time is None:
            self._stream_start_time = now
        self._last_processed_time = now

        if lf.quality.is_valid:
            self._valid_frame_count += 1

        # Exclude initial warm-up frames from benchmarking queues
        if self._processed_count > self._warmup_frames:
            self._history_capture_ms.append(lf.capture_latency_ms)
            self._history_prep_ms.append(lf.preprocessing_latency_ms)
            self._history_extract_ms.append(lf.extraction_latency_ms)
            self._history_norm_ms.append(lf.normalization_latency_ms)
            self._history_quality_ms.append(lf.quality_check_latency_ms)
            self._history_total_ms.append(lf.total_latency_ms)

    def _record_landmark_data(self, lf: LandmarkFrame):
        """Saves frame data into internal recording list (if debug enabled)."""
        self._recorded_frames.append({
            "frame_id": lf.frame_id,
            "timestamp": lf.timestamp,
            "raw_coords": lf.raw_coords.copy(),
            "normalized_coords": lf.normalized_coords.copy(),
            "mask": lf.mask.copy(),
            "quality_score": lf.quality.quality_score,
            "classification": lf.quality.classification
        })

    def get_metrics(self) -> StreamMetrics:
        """Returns structured performance and throughput metrics."""
        now = time.perf_counter()
        elapsed = (now - self._stream_start_time) if self._stream_start_time else 0.0

        p_fps = (self._processed_count / elapsed) if elapsed > 0 else 0.0
        valid_ratio = (self._valid_frame_count / max(1, self._processed_count))

        totals = list(self._history_total_ms)
        if totals:
            mean_total = float(np.mean(totals))
            median_total = float(np.median(totals))
            p95_total = float(np.percentile(totals, 95))
            max_total = float(np.max(totals))
        else:
            mean_total = median_total = p95_total = max_total = 0.0

        mean_cap = float(np.mean(self._history_capture_ms)) if self._history_capture_ms else 0.0
        mean_ext = float(np.mean(self._history_extract_ms)) if self._history_extract_ms else 0.0
        mean_norm = float(np.mean(self._history_norm_ms)) if self._history_norm_ms else 0.0
        mean_qual = float(np.mean(self._history_quality_ms)) if self._history_quality_ms else 0.0

        return StreamMetrics(
            total_captured_frames=self._processed_count + self._dropped_count,
            total_processed_frames=self._processed_count,
            total_dropped_frames=self._dropped_count,
            capture_fps=round(p_fps, 2),
            processing_fps=round(p_fps, 2),
            mean_capture_latency_ms=round(mean_cap, 2),
            mean_extraction_latency_ms=round(mean_ext, 2),
            mean_normalization_latency_ms=round(mean_norm, 2),
            mean_quality_latency_ms=round(mean_qual, 2),
            mean_total_latency_ms=round(mean_total, 2),
            median_total_latency_ms=round(median_total, 2),
            p95_total_latency_ms=round(p95_total, 2),
            max_total_latency_ms=round(max_total, 2),
            valid_frame_ratio=round(valid_ratio, 4)
        )

    def render_debug_overlay(
        self,
        bgr_image: np.ndarray,
        lf: LandmarkFrame,
        fps_display: Optional[float] = None
    ) -> np.ndarray:
        """
        Draws skeletal keypoints, connections, and diagnostic HUD overlay
        onto a copy of the camera frame.
        """
        canvas = bgr_image.copy()
        h, w = canvas.shape[:2]

        raw = lf.raw_coords
        mask = lf.mask

        # Helper to convert normalized coordinate (0..1) to pixel coords
        def to_px(pt):
            return int(pt[0] * w), int(pt[1] * h)

        # 1. Draw Upper Pose (Nodes 42..52) - Cyan / Orange
        pose_pairs = [
            (47, 48), (47, 49), (49, 51), (48, 50), (50, 52),
            (42, 43), (42, 44), (43, 45), (44, 46), (42, 47), (42, 48)
        ]
        for u, v in pose_pairs:
            if mask[u] and mask[v]:
                cv2.line(canvas, to_px(raw[u]), to_px(raw[v]), (255, 165, 0), 2, cv2.LINE_AA)

        for i in range(42, 53):
            if mask[i]:
                cv2.circle(canvas, to_px(raw[i]), 4, (0, 255, 255), -1, cv2.LINE_AA)

        # 2. Draw Left Hand (Nodes 0..20) - Green
        lh_chains = [
            [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]
        ]
        for chain in lh_chains:
            for k in range(len(chain) - 1):
                u, v = chain[k], chain[k + 1]
                if mask[u] and mask[v]:
                    cv2.line(canvas, to_px(raw[u]), to_px(raw[v]), (0, 255, 0), 2, cv2.LINE_AA)
        for i in range(0, 21):
            if mask[i]:
                cv2.circle(canvas, to_px(raw[i]), 3, (0, 200, 0), -1, cv2.LINE_AA)

        # 3. Draw Right Hand (Nodes 21..41) - Magenta
        rh_chains = [
            [21, 22, 23, 24, 25], [21, 26, 27, 28, 29], [21, 30, 31, 32, 33],
            [21, 34, 35, 36, 37], [21, 38, 39, 40, 41]
        ]
        for chain in rh_chains:
            for k in range(len(chain) - 1):
                u, v = chain[k], chain[k + 1]
                if mask[u] and mask[v]:
                    cv2.line(canvas, to_px(raw[u]), to_px(raw[v]), (255, 0, 255), 2, cv2.LINE_AA)
        for i in range(21, 42):
            if mask[i]:
                cv2.circle(canvas, to_px(raw[i]), 3, (200, 0, 200), -1, cv2.LINE_AA)

        # 4. Draw Face Contours if present (Nodes 53..92) - Yellow
        for i in range(53, 93):
            if mask[i]:
                cv2.circle(canvas, to_px(raw[i]), 1, (0, 255, 255), -1, cv2.LINE_AA)

        # 5. Draw HUD Overlay Panel (Translucent dark rectangle)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (320, 175), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (10, 10), (320, 175), (80, 80, 80), 1)

        # Quality indicator color
        status_colors = {
            "GOOD": (0, 255, 0),
            "ACCEPTABLE": (0, 215, 255),
            "REVIEW": (0, 140, 255),
            "REJECT": (0, 0, 255)
        }
        q_color = status_colors.get(lf.quality.classification, (200, 200, 200))

        fps_val = fps_display if fps_display is not None else (1000.0 / max(1.0, lf.total_latency_ms))

        lines = [
            (f"FPS: {fps_val:.1f} (Target: {self.config.camera.target_fps:.0f})", (255, 255, 255)),
            (f"Latency: {lf.total_latency_ms:.1f} ms (Ext: {lf.extraction_latency_ms:.1f}ms)", (200, 200, 200)),
            (f"Hands: LH={'Y' if lf.modality_stats.left_hand_detected else 'N'} | RH={'Y' if lf.modality_stats.right_hand_detected else 'N'}", (255, 255, 255)),
            (f"Pose: {'DETECTED' if lf.modality_stats.pose_detected else 'LOST'}", (0, 255, 0) if lf.modality_stats.pose_detected else (0, 0, 255)),
            (f"Quality: {lf.quality.classification} ({lf.quality.quality_score:.2f})", q_color),
            (f"Frame #{lf.frame_id} (Nodes: {int(np.sum(mask))}/93)", (180, 180, 180))
        ]

        y_offset = 32
        for text, col in lines:
            cv2.putText(canvas, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
            y_offset += 24

        return canvas

    def save_recorded_session(self, output_path: Optional[str] = None) -> Optional[str]:
        """Saves collected debug landmark session to an NPZ file."""
        if not self._recorded_frames:
            return None

        out_file = output_path or os.path.join(
            self.config.debug.record_dir,
            f"landmark_stream_debug_{int(time.time())}.npz"
        )
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        frames_data = np.stack([f["normalized_coords"] for f in self._recorded_frames], axis=0)  # (N, 93, 3)
        masks_data = np.stack([f["mask"] for f in self._recorded_frames], axis=0)                # (N, 93)
        timestamps = np.array([f["timestamp"] for f in self._recorded_frames], dtype=np.float64)

        np.savez_compressed(
            out_file,
            data=frames_data.astype(np.float32),
            mask=masks_data.astype(bool),
            timestamps=timestamps,
            num_frames=len(self._recorded_frames)
        )
        logger.info(f"Saved {len(self._recorded_frames)} debug landmark frames to {out_file}")
        return out_file

    def close(self):
        """Releases underlying resources and cleans up MediaPipe pipelines."""
        if hasattr(self, 'extractor') and self.extractor is not None:
            self.extractor.close()
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
