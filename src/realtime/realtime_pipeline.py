"""
SignTalk AI - End-to-End Real-Time Pipeline Orchestrator
Phase 4 Part 4: Integrated AI Inference, Performance Profiling & Linguistic Translation.

Coordinates:
  Camera / Video Ingestion
           ↓
  Frame Preprocessing & Validation
           ↓
  MediaPipe Landmark Extraction (93 Multimodal Nodes)
           ↓
  Torso-Referenced Coordinate Normalization
           ↓
  Quality Validation & Gating
           ↓
  Temporal Rolling Buffer (T=45 frames)
           ↓
  Inference Scheduler (Backpressure & Stride Pacing)
           ↓
  ST-GCN Visual Model Runner [1, 3, 45, 93]
           ↓
  Confidence Filtering (τ=0.65 Gate)
           ↓
  Temporal Majority Smoothing
           ↓
  Sign State Machine (Debounce Transitions)
           ↓
  Event Deduplication (800ms Separation)
           ↓
  Sign Sequence Buffer
           ↓
  Linguistic Translation Layer (English & Hindi Phrases)
           ↓
  Live Transcript & Diagnostic HUD
"""

import os
import time
import enum
import logging
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
import cv2
import numpy as np
import torch

from src.realtime.types import (
    FramePacket,
    LandmarkFrame,
    PredictionResult,
    StreamMetrics
)
from src.realtime.realtime_config import RealTimeConfig
from src.realtime.camera import Camera, CameraError, CameraReadError
from src.realtime.landmark_stream import RealTimeLandmarkStream
from src.realtime.temporal_buffer import TemporalBuffer
from src.realtime.inference_scheduler import InferenceScheduler
from src.realtime.model_runner import STGCNRunner
from src.realtime.confidence_filter import ConfidenceFilter, FilteredPrediction
from src.realtime.prediction_history import PredictionHistory, PredictionRecord
from src.realtime.smoothing import (
    BaseSmoother,
    MajorityVoteSmoother,
    ConfidenceWeightedSmoother,
    TemporalStabilitySmoother,
    SmoothedPrediction
)
from src.realtime.sign_state_machine import SignStateMachine, SignState
from src.realtime.event_deduplicator import EventDeduplicator
from src.realtime.sign_event import SignEvent
from src.realtime.sign_sequence import SignSequenceBuffer
from src.realtime.translator import RealTimeTranslator, TranslationResult
from src.realtime.performance import PerformanceProfiler

logger = logging.getLogger("SignTalk.RealTime.Pipeline")


class PipelineState(str, enum.Enum):
    """Explicit lifecycle states for RealtimePipeline."""
    UNINITIALIZED = "UNINITIALIZED"
    WARMING_UP = "WARMING_UP"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass
class PipelineOutput:
    """
    Standardized, strongly typed output produced per pipeline processing step.
    Serves as the primary data contract for UI overlays and Phase 5 frontend engines.
    """
    frame_id: int
    timestamp: float
    landmark_frame: Optional[LandmarkFrame]
    prediction: Optional[PredictionResult]
    smoothed: Optional[SmoothedPrediction]
    state: str
    latest_event: Optional[SignEvent]
    sign_sequence: List[str]
    translation: Optional[TranslationResult]
    stage_latencies_ms: Dict[str, float]
    fps: float
    pipeline_state: str
    is_valid_input: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serializes output to structured dictionary for JSON / WebSocket APIs."""
        pred_dict = None
        if self.prediction is not None:
            pred_dict = {
                "class_id": self.prediction.class_id,
                "label": self.prediction.label,
                "gloss": self.prediction.gloss,
                "confidence": round(self.prediction.confidence, 4),
                "top_k": self.prediction.top_k,
                "input_quality": round(self.prediction.input_quality, 4),
            }

        trans_dict = None
        if self.translation is not None:
            trans_dict = self.translation.to_dict()

        evt_dict = None
        if self.latest_event is not None:
            evt_dict = self.latest_event.to_dict()

        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "pipeline_state": self.pipeline_state,
            "prediction": pred_dict,
            "smoothed_label": self.smoothed.label if self.smoothed else None,
            "state_machine": self.state,
            "latest_event": evt_dict,
            "sign_sequence": list(self.sign_sequence),
            "translation": trans_dict,
            "stage_latencies_ms": {k: round(v, 3) for k, v in self.stage_latencies_ms.items()},
            "fps": round(self.fps, 2),
            "is_valid_input": self.is_valid_input,
        }


class RealtimePipeline:
    """
    Comprehensive end-to-end real-time orchestrator for SignTalk AI.
    Integrates vision ingestion, landmark processing, spatiotemporal graph inference,
    debounce state tracking, and multilingual sentence translation.
    """

    def __init__(
        self,
        config: Optional[RealTimeConfig] = None,
        camera: Optional[Camera] = None,
        model_runner: Optional[STGCNRunner] = None,
        landmark_stream: Optional[RealTimeLandmarkStream] = None,
        translator: Optional[RealTimeTranslator] = None,
        profiler: Optional[PerformanceProfiler] = None,
    ):
        self.config = config or RealTimeConfig()
        self._state = PipelineState.UNINITIALIZED

        # 1. Performance Profiler
        self.profiler = profiler or PerformanceProfiler(
            history_size=self.config.performance.history_size,
            enabled=self.config.performance.profiling_enabled
        )

        # 2. Camera Ingestion (optional; can be provided or initialized on run)
        self.camera = camera

        # 3. Landmark Stream
        self.landmark_stream = landmark_stream or RealTimeLandmarkStream(self.config)

        # 4. ST-GCN Model Runner
        inf_cfg = self.config.inference
        if model_runner is not None:
            self.model_runner = model_runner
        else:
            self.model_runner = STGCNRunner(
                checkpoint_path=inf_cfg.checkpoint_path,
                device=inf_cfg.device,
                top_k=inf_cfg.top_k,
                warmup_iterations=inf_cfg.warmup_iterations
            )

        # 5. Temporal Buffer
        buf_cfg = self.config.buffer
        self.temporal_buffer = TemporalBuffer(
            max_length=buf_cfg.max_length,
            min_valid_frames=buf_cfg.min_valid_frames,
            interpolate_missing_frames=buf_cfg.interpolate_missing_frames,
            max_consecutive_interpolated=buf_cfg.max_consecutive_interpolated
        )

        # 6. Inference Scheduler
        sched_cfg = self.config.scheduler
        self.scheduler = InferenceScheduler(
            stride=sched_cfg.stride,
            drop_stale_windows=sched_cfg.drop_stale_windows
        )

        # 7. Confidence Filter & History
        self.confidence_filter = ConfidenceFilter(
            threshold=self.config.confidence.threshold,
            uncertain_label=self.config.confidence.uncertain_label,
        )
        self.prediction_history = PredictionHistory(
            max_history=max(20, self.config.smoothing.history_size * 4)
        )

        # 8. Temporal Smoother
        sm_cfg = self.config.smoothing
        if sm_cfg.method == "confidence_weighted":
            self.smoother: BaseSmoother = ConfidenceWeightedSmoother(
                history_size=sm_cfg.history_size,
                decay_factor=sm_cfg.decay_factor,
            )
        elif sm_cfg.method == "temporal_stability":
            self.smoother = TemporalStabilitySmoother(
                min_consecutive=sm_cfg.min_consecutive,
            )
        else:
            self.smoother = MajorityVoteSmoother(
                history_size=sm_cfg.history_size,
                min_votes=sm_cfg.min_votes,
            )

        # 9. State Machine & Deduplication
        self.sign_state_machine = SignStateMachine(
            min_consecutive_predictions=self.config.stability.min_consecutive_predictions,
            min_confidence=self.config.stability.min_confidence,
            min_input_quality=self.config.stability.min_input_quality,
        )
        self.event_deduplicator = EventDeduplicator(
            minimum_gap_ms=self.config.events.minimum_gap_ms
        )
        self.sign_sequence = SignSequenceBuffer(
            max_events=self.config.sequence.max_events
        )

        # 10. Translation Layer
        trans_cfg = self.config.translation
        self.translator = translator or RealTimeTranslator(
            pause_threshold_sec=trans_cfg.pause_threshold_sec,
            min_confidence=trans_cfg.min_confidence,
            max_phrase_signs=trans_cfg.max_phrase_signs
        )

        # Runtime Metrics & State
        self.latest_prediction: Optional[PredictionResult] = None
        self.latest_filtered: Optional[FilteredPrediction] = None
        self.latest_smoothed: Optional[SmoothedPrediction] = None
        self.latest_event: Optional[SignEvent] = None
        self.latest_translation: Optional[TranslationResult] = None
        self.last_step_output: Optional[PipelineOutput] = None

        self._frame_count = 0
        self._last_frame_time = 0.0
        self._fps_estimate = 0.0

        self._state = PipelineState.STOPPED

    @property
    def state(self) -> PipelineState:
        """Returns current pipeline lifecycle state."""
        return self._state

    def initialize(self) -> None:
        """Warms up models, verifies checkpoints, and transitions pipeline to RUNNING."""
        self._state = PipelineState.WARMING_UP
        logger.info("Initializing RealtimePipeline...")

        # Warm up ST-GCN runner if not already warm
        if self.config.inference.warmup_iterations > 0:
            self.model_runner.warmup(self.config.inference.warmup_iterations)

        self._state = PipelineState.RUNNING
        logger.info("RealtimePipeline initialized and RUNNING.")

    def step(self, raw_bgr_frame: Optional[np.ndarray] = None) -> PipelineOutput:
        """
        Executes a single synchronous inference cycle across all stages.
        If raw_bgr_frame is None, grabs a frame from the attached Camera.
        
        Args:
            raw_bgr_frame: Optional OpenCV BGR image array (H, W, 3).
            
        Returns:
            PipelineOutput containing stage artifacts, latencies, and predictions.
        """
        if self._state not in (PipelineState.RUNNING, PipelineState.WARMING_UP):
            if self._state == PipelineState.STOPPED:
                self.initialize()
            elif self._state == PipelineState.PAUSED:
                # Return empty holding output during pause
                return PipelineOutput(
                    frame_id=self._frame_count,
                    timestamp=time.time(),
                    landmark_frame=None,
                    prediction=None,
                    smoothed=self.latest_smoothed,
                    state="PAUSED",
                    latest_event=self.latest_event,
                    sign_sequence=self.sign_sequence.get_labels(),
                    translation=self.latest_translation,
                    stage_latencies_ms={},
                    fps=self._fps_estimate,
                    pipeline_state=self._state.value,
                    is_valid_input=False
                )

        t_e2e_start = time.perf_counter()
        latencies: Dict[str, float] = {}

        # -------------------------------------------------------------
        # Stage 1: Frame Acquisition
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        if raw_bgr_frame is not None:
            img = raw_bgr_frame
            ts = time.time()
        else:
            if self.camera is None:
                raise RuntimeError("No camera attached and no frame supplied to step().")
            packet = self.camera.read()
            img = packet.image
            ts = packet.timestamp
        latencies["capture"] = (time.perf_counter() - t0) * 1000.0
        self.profiler.record_stage("capture", latencies["capture"])

        self._frame_count += 1
        now = time.time()
        if self._last_frame_time > 0:
            dt = now - self._last_frame_time
            if dt > 0:
                self._fps_estimate = 0.85 * self._fps_estimate + 0.15 * (1.0 / dt)
        self._last_frame_time = now

        # Prepare FramePacket
        packet = FramePacket(
            frame_id=self._frame_count,
            timestamp=ts,
            capture_time=now,
            image=img,
            width=img.shape[1],
            height=img.shape[0],
            is_rgb=False,
            is_flipped=self.config.processing.flip_horizontal
        )

        # -------------------------------------------------------------
        # Stage 2: MediaPipe Landmark Extraction & Normalization
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        landmark_frame = self.landmark_stream.process_frame(packet)
        latencies["mediapipe"] = (time.perf_counter() - t0) * 1000.0
        self.profiler.record_stage("mediapipe", latencies["mediapipe"])

        # -------------------------------------------------------------
        # Stage 3: Temporal Buffer Accumulation & Scheduler
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        self.temporal_buffer.append(landmark_frame)
        self.scheduler.on_frame_appended()
        latencies["temporal_buffer"] = (time.perf_counter() - t0) * 1000.0
        self.profiler.record_stage("temporal_buffer", latencies["temporal_buffer"])

        prediction: Optional[PredictionResult] = None

        # -------------------------------------------------------------
        # Stage 4: Sliding Window & ST-GCN Model Inference
        # -------------------------------------------------------------
        if self.scheduler.should_infer(self.temporal_buffer):
            self.scheduler.record_inference_start()
            window = self.temporal_buffer.get_window()

            # Tensor preparation
            t_tp = time.perf_counter()
            x, mask = self.model_runner.window_to_tensor(window)
            latencies["tensor_prep"] = (time.perf_counter() - t_tp) * 1000.0
            self.profiler.record_stage("tensor_prep", latencies["tensor_prep"])

            # Model forward pass
            t_inf = time.perf_counter()
            with torch.inference_mode():
                logits, probs = self.model_runner.predict(x, mask)
            latencies["stgcn_inference"] = (time.perf_counter() - t_inf) * 1000.0
            self.profiler.record_stage("stgcn_inference", latencies["stgcn_inference"])

            self.scheduler.record_inference_finish()

            # Format PredictionResult
            probs_np = probs[0].cpu().numpy()
            logits_np = logits[0].cpu().numpy()
            pred_cid = int(np.argmax(probs_np))
            pred_conf = float(probs_np[pred_cid])

            top_indices = np.argsort(probs_np)[::-1][:self.config.inference.top_k]
            top_k_list = [
                (int(idx), self.model_runner.model.graph_strategy, float(probs_np[idx]))  # placeholder
                for idx in top_indices
            ]
            # Replace placeholder label with canonical label
            from src.data.label_map import get_label, get_gloss, get_translation
            top_k_list = [
                (int(idx), get_label(int(idx)), float(probs_np[idx]))
                for idx in top_indices
            ]

            qualities = [f.quality.quality_score for f in window]
            input_quality = float(np.mean(qualities)) if qualities else 0.0

            prediction = PredictionResult(
                class_id=pred_cid,
                label=get_label(pred_cid),
                gloss=get_gloss(pred_cid),
                translation=get_translation(pred_cid),
                confidence=pred_conf,
                probabilities=probs_np,
                logits=logits_np,
                top_k=top_k_list,
                timestamp=ts,
                window_start_time=window[0].timestamp,
                window_end_time=window[-1].timestamp,
                window_frame_count=len(window),
                inference_start_time=t_inf,
                inference_end_time=time.perf_counter(),
                inference_latency_ms=round(latencies["stgcn_inference"], 2),
                input_quality=round(input_quality, 4),
                is_valid_quality=(sum(1 for f in window if f.quality.is_valid) >= 15),
                buffer_length=len(window),
                device=str(self.model_runner.device)
            )
            self.latest_prediction = prediction

            # ---------------------------------------------------------
            # Stage 5: Confidence Filter & Prediction History
            # ---------------------------------------------------------
            t_cf = time.perf_counter()
            filtered = self.confidence_filter.filter_prediction(prediction)
            self.latest_filtered = filtered

            record = PredictionRecord(
                timestamp=prediction.timestamp,
                window_start=prediction.window_start_time,
                window_end=prediction.window_end_time,
                class_id=filtered.class_id,
                label=filtered.label,
                gloss=filtered.gloss,
                confidence=filtered.confidence,
                input_quality=filtered.input_quality,
                inference_latency_ms=prediction.inference_latency_ms,
                is_valid_quality=prediction.is_valid_quality,
                probabilities=prediction.probabilities,
            )
            self.prediction_history.add(record)
            latencies["confidence_filter"] = (time.perf_counter() - t_cf) * 1000.0
            self.profiler.record_stage("confidence_filter", latencies["confidence_filter"])

            # ---------------------------------------------------------
            # Stage 6: Temporal Smoothing
            # ---------------------------------------------------------
            t_sm = time.perf_counter()
            smoothed = self.smoother.smooth(self.prediction_history)
            self.latest_smoothed = smoothed
            latencies["smoothing"] = (time.perf_counter() - t_sm) * 1000.0
            self.profiler.record_stage("smoothing", latencies["smoothing"])

            # ---------------------------------------------------------
            # Stage 7: State Machine & Event Deduplication
            # ---------------------------------------------------------
            t_fsm = time.perf_counter()
            _, completed_event = self.sign_state_machine.process(smoothed)
            latencies["state_machine"] = (time.perf_counter() - t_fsm) * 1000.0
            self.profiler.record_stage("state_machine", latencies["state_machine"])

            t_dedup = time.perf_counter()
            if completed_event is not None:
                deduped = self.event_deduplicator.process(completed_event)
                if deduped is not None:
                    self.sign_sequence.append_event(deduped)
                    self.latest_event = deduped

                    # -------------------------------------------------
                    # Stage 8: Translation Layer Ingestion
                    # -------------------------------------------------
                    t_trans = time.perf_counter()
                    trans_res = self.translator.process_event(deduped)
                    self.latest_translation = trans_res
                    latencies["translation"] = (time.perf_counter() - t_trans) * 1000.0
                    self.profiler.record_stage("translation", latencies["translation"])

            latencies["deduplicator"] = (time.perf_counter() - t_dedup) * 1000.0
            self.profiler.record_stage("deduplicator", latencies["deduplicator"])

        # Check for pause-based sentence finalization in translation layer
        pause_trans = self.translator.check_pause(ts)
        if pause_trans is not None:
            self.latest_translation = pause_trans

        # Compute End-to-End Latency
        latencies["end_to_end"] = (time.perf_counter() - t_e2e_start) * 1000.0
        self.profiler.record_stage("end_to_end", latencies["end_to_end"])

        output = PipelineOutput(
            frame_id=self._frame_count,
            timestamp=ts,
            landmark_frame=landmark_frame,
            prediction=self.latest_prediction,
            smoothed=self.latest_smoothed,
            state=self.sign_state_machine.state.value,
            latest_event=self.latest_event,
            sign_sequence=self.sign_sequence.get_labels(),
            translation=self.latest_translation,
            stage_latencies_ms=latencies,
            fps=self._fps_estimate,
            pipeline_state=self._state.value,
            is_valid_input=landmark_frame.quality.is_valid
        )
        self.last_step_output = output
        return output

    def render_overlay(
        self,
        bgr_image: np.ndarray,
        output: Optional[PipelineOutput] = None
    ) -> np.ndarray:
        """
        Renders complete debug canvas: landmark skeleton, temporal buffer bar,
        prediction telemetry, FSM status, live sign sequence, natural translation,
        and latency profiling metrics.
        """
        t0 = time.perf_counter()
        out = output or self.last_step_output

        # Base skeleton rendering from LandmarkStream
        lf = out.landmark_frame if out else None
        if lf is not None:
            canvas = self.landmark_stream.render_debug_overlay(bgr_image, lf, self._fps_estimate)
        else:
            canvas = bgr_image.copy()
        h, w = canvas.shape[:2]

        # 1. Temporal Buffer Bar (Bottom-Left)
        buf_len = self.temporal_buffer.length()
        buf_max = self.temporal_buffer.max_length
        buf_ratio = min(1.0, buf_len / max(1, buf_max))

        bar_x, bar_y = 15, h - 35
        bar_w, bar_h = 240, 18
        cv2.rectangle(canvas, (bar_x - 5, bar_y - 20), (bar_x + bar_w + 100, bar_y + bar_h + 5), (20, 20, 20), -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1)

        fill_w = int(bar_w * buf_ratio)
        fill_color = (0, 255, 0) if buf_len >= buf_max else (0, 200, 255)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x + 1, bar_y + 1), (bar_x + fill_w - 1, bar_y + bar_h - 1), fill_color, -1)

        cv2.putText(canvas, f"BUFFER: {buf_len}/{buf_max} ({buf_ratio:.0%})", (bar_x, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{self._fps_estimate:.1f} FPS", (bar_x + bar_w + 10, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # 2. Main HUD Panel (Top-Right)
        panel_w = 400
        panel_h = 270
        panel_x = max(10, w - panel_w - 15)
        panel_y = 10

        overlay = canvas.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (90, 90, 90), 1)

        pred = out.prediction if out else None
        dev_str = pred.device.upper() if pred else "CPU"
        cv2.putText(canvas, f"SIGNTALK REAL-TIME ({dev_str})", (panel_x + 12, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        if pred is not None:
            # Raw Prediction
            conf_color = (0, 255, 0) if pred.confidence >= 0.65 else (0, 215, 255) if pred.confidence >= 0.40 else (0, 140, 255)
            cv2.putText(canvas, f"Raw: {pred.label.upper()} ({pred.confidence:.1%})", (panel_x + 12, panel_y + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, conf_color, 1, cv2.LINE_AA)

            # Smoothed
            sm_lbl = out.smoothed.label.upper() if out.smoothed else "(NONE)"
            cv2.putText(canvas, f"Smoothed: {sm_lbl}", (panel_x + 12, panel_y + 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            # State & Quality
            st_color = (0, 255, 0) if out.state == "ACTIVE" else (0, 200, 255) if out.state == "CANDIDATE" else (180, 180, 180)
            cv2.putText(canvas, f"State: {out.state}", (panel_x + 12, panel_y + 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, st_color, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Quality: {pred.input_quality:.2f}", (panel_x + 200, panel_y + 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

            # Event
            evt_lbl = out.latest_event.label.upper() if out.latest_event else "(NONE)"
            cv2.putText(canvas, f"Event: {evt_lbl}", (panel_x + 12, panel_y + 116),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 50), 1, cv2.LINE_AA)

            # Sequence (glosses)
            seq_str = " -> ".join([s.upper() for s in out.sign_sequence]) if out.sign_sequence else "(NONE)"
            if len(seq_str) > 30:
                seq_str = "..." + seq_str[-27:]
            cv2.putText(canvas, f"Seq: {seq_str}", (panel_x + 12, panel_y + 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

            # Translation (English & Hindi)
            trans_txt = out.translation.text if (out and out.translation) else ""
            if trans_txt:
                cv2.putText(canvas, f"EN: \"{trans_txt}\"", (panel_x + 12, panel_y + 168),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 255, 100), 1, cv2.LINE_AA)

            # Latency breakdown summary
            stats = self.profiler.get_stage_stats("stgcn_inference")
            e2e_stats = self.profiler.get_stage_stats("end_to_end")
            cv2.putText(canvas, f"ST-GCN: {stats['mean']:.1f}ms | Total: {e2e_stats['mean']:.1f}ms",
                        (panel_x + 12, panel_y + 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

            # Top-3 predictions
            top_y = panel_y + 218
            for rank, (cid, lbl, prob) in enumerate(pred.top_k[:2], 1):
                cv2.putText(canvas, f"#{rank} {lbl.upper():<10} {prob:.1%}",
                            (panel_x + 15, top_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1, cv2.LINE_AA)
                top_y += 18
        else:
            cv2.putText(canvas, f"Accumulating window ({buf_len}/{buf_max})...", (panel_x + 12, panel_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Requires {buf_max} frames (1.8s) of context", (panel_x + 12, panel_y + 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

        render_lat = (time.perf_counter() - t0) * 1000.0
        self.profiler.record_stage("display_render", render_lat)
        return canvas

    def pause(self) -> None:
        """Transitions pipeline to PAUSED state."""
        self._state = PipelineState.PAUSED
        logger.info("RealtimePipeline PAUSED.")

    def resume(self) -> None:
        """Resumes pipeline from PAUSED state to RUNNING."""
        self._state = PipelineState.RUNNING
        logger.info("RealtimePipeline RESUMED.")

    def reset(self) -> None:
        """Clears all rolling buffers, debounce state, and sequence transcript."""
        self.temporal_buffer.clear()
        self.prediction_history.clear()
        self.sign_state_machine.reset()
        self.event_deduplicator.reset()
        self.sign_sequence.clear()
        self.translator.reset()
        self.profiler.reset()

        self.latest_prediction = None
        self.latest_filtered = None
        self.latest_smoothed = None
        self.latest_event = None
        self.latest_translation = None
        logger.info("RealtimePipeline RESET complete.")

    def stop(self) -> None:
        """Transitions pipeline to STOPPED state."""
        self._state = PipelineState.STOPPED
        logger.info("RealtimePipeline STOPPED.")

    def close(self) -> None:
        """Releases camera and stream resources."""
        self.stop()
        if self.camera is not None:
            self.camera.stop()
        self.landmark_stream.close()
        self.reset()
        logger.info("RealtimePipeline CLOSED and resources released.")
