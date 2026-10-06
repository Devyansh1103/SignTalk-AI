"""
SignTalk AI - Real-Time Sign Predictor Pipeline
Phase 4 Part 2: Sliding-Window Inference & Temporal Prediction.

Integrates:
  Camera / Video Ingestion
           ↓
  Real-Time Landmark Stream
           ↓
  Temporal Rolling Buffer (T=45)
           ↓
  Inference Scheduler (Stride pacing & stale drop)
           ↓
  ST-GCN Model Runner
           ↓
  Prediction Sinks (Console, CSV Diagnostics, Callbacks)
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any, Callable, Tuple
import cv2
import numpy as np

from src.realtime.types import (
    FramePacket,
    LandmarkFrame,
    PredictionResult,
    StreamMetrics
)
from src.realtime.realtime_config import RealTimeConfig
from src.realtime.camera import Camera
from src.realtime.landmark_stream import RealTimeLandmarkStream
from src.realtime.temporal_buffer import TemporalBuffer
from src.realtime.inference_scheduler import InferenceScheduler
from src.realtime.model_runner import STGCNRunner
from src.realtime.prediction_sink import (
    BasePredictionSink,
    ConsolePredictionSink,
    PredictionLoggerSink,
    CompositePredictionSink
)
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

logger = logging.getLogger("SignTalk.RealTime.Predictor")


class RealTimeSignPredictor:
    """
    Top-level real-time sliding-window sign language predictor.
    Coordinates camera acquisition, landmark streaming, temporal buffering,
    ST-GCN inference scheduling, confidence filtering, temporal smoothing,
    state machine tracking, duplicate suppression, and sign sequence buffering.
    """

    def __init__(
        self,
        config: Optional[RealTimeConfig] = None,
        model_runner: Optional[STGCNRunner] = None,
        landmark_stream: Optional[RealTimeLandmarkStream] = None,
        prediction_sinks: Optional[List[BasePredictionSink]] = None
    ):
        self.config = config or RealTimeConfig()

        # 1. Landmark Stream
        self.landmark_stream = landmark_stream or RealTimeLandmarkStream(self.config)

        # 2. ST-GCN Model Runner
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

        # 3. Temporal Rolling Buffer
        buf_cfg = self.config.buffer
        self.temporal_buffer = TemporalBuffer(
            max_length=buf_cfg.max_length,
            min_valid_frames=buf_cfg.min_valid_frames,
            interpolate_missing_frames=buf_cfg.interpolate_missing_frames,
            max_consecutive_interpolated=buf_cfg.max_consecutive_interpolated
        )

        # 4. Inference Scheduler
        sched_cfg = self.config.scheduler
        self.scheduler = InferenceScheduler(
            stride=sched_cfg.stride,
            drop_stale_windows=sched_cfg.drop_stale_windows
        )

        # 5. Prediction Sinks
        self.composite_sink = CompositePredictionSink()
        if prediction_sinks:
            for s in prediction_sinks:
                self.composite_sink.add_sink(s)
        else:
            self.composite_sink.add_sink(ConsolePredictionSink(verbose=inf_cfg.verbose_console))
            if inf_cfg.record_predictions:
                self.composite_sink.add_sink(
                    PredictionLoggerSink(output_path=inf_cfg.predictions_csv)
                )

        # 6. Phase 4 Part 3: Confidence Filtering, Smoothing, State Machine, & Sequence
        self.confidence_filter = ConfidenceFilter(
            threshold=self.config.confidence.threshold,
            uncertain_label=self.config.confidence.uncertain_label,
        )
        self.prediction_history = PredictionHistory(
            max_history=max(20, self.config.smoothing.history_size * 4)
        )

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

        # Runtime State
        self.latest_prediction: Optional[PredictionResult] = None
        self.latest_filtered: Optional[FilteredPrediction] = None
        self.latest_smoothed: Optional[SmoothedPrediction] = None
        self.latest_event: Optional[SignEvent] = None
        self._prediction_history: List[PredictionResult] = []
        self._inference_latencies_ms: List[float] = []
        self._total_predictions = 0
        self._last_inference_time = 0.0
        self._prediction_fps = 0.0


    def add_sink(self, sink: BasePredictionSink) -> None:
        """Registers an additional prediction sink."""
        self.composite_sink.add_sink(sink)

    def process_frame_packet(self, packet: FramePacket) -> Tuple[LandmarkFrame, Optional[PredictionResult]]:
        """
        Processes a single camera/video FramePacket through the complete pipeline.
        
        Args:
            packet: Captured raw frame packet.
            
        Returns:
            Tuple of:
              - landmark_frame: Processed and normalized landmark frame
              - prediction: PredictionResult if a window was evaluated, else None
        """
        # Step 1: Extract and normalize landmarks
        landmark_frame = self.landmark_stream.process_frame(packet)

        # Step 2: Append to temporal rolling buffer
        self.temporal_buffer.append(landmark_frame)
        self.scheduler.on_frame_appended()

        prediction: Optional[PredictionResult] = None

        # Step 3: Evaluate sliding window readiness
        if self.scheduler.should_infer(self.temporal_buffer):
            prediction = self._execute_inference()

        return landmark_frame, prediction

    def _execute_inference(self) -> PredictionResult:
        """Retrieves current window and executes ST-GCN prediction."""
        self.scheduler.record_inference_start()
        t0 = time.perf_counter()

        window = self.temporal_buffer.get_window()
        prediction = self.model_runner.predict_window(window)

        t_elapsed = (time.perf_counter() - t0) * 1000.0
        self.scheduler.record_inference_finish()

        # Update metrics
        now = time.time()
        if self._last_inference_time > 0:
            dt = now - self._last_inference_time
            if dt > 0:
                self._prediction_fps = 0.8 * self._prediction_fps + 0.2 * (1.0 / dt)
        self._last_inference_time = now

        self.latest_prediction = prediction
        self._prediction_history.append(prediction)
        self._inference_latencies_ms.append(prediction.inference_latency_ms)
        self._total_predictions += 1

        # Phase 4 Part 3: Confidence Filtering
        filtered = self.confidence_filter.filter_prediction(prediction)
        self.latest_filtered = filtered

        # Prediction History
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


        # Temporal Smoothing
        smoothed = self.smoother.smooth(self.prediction_history)
        self.latest_smoothed = smoothed

        # Stability State Machine
        _, completed_event = self.sign_state_machine.process(smoothed)

        # Duplicate Suppression & Sign Sequence Buffer
        if completed_event is not None:
            deduped = self.event_deduplicator.process(completed_event)
            if deduped is not None:
                self.sign_sequence.append_event(deduped)
                self.latest_event = deduped
                if self.config.events.log_events:
                    self.sign_sequence.save_to_csv(self.config.events.events_csv)

        # Publish to sinks
        self.composite_sink.publish(prediction)

        return prediction

    def render_prediction_overlay(
        self,
        bgr_image: np.ndarray,
        landmark_frame: LandmarkFrame,
        fps_display: Optional[float] = None
    ) -> np.ndarray:
        """
        Renders landmark skeleton, tracking status, temporal buffer progress,
        raw prediction, smoothed prediction, state, and continuous sign sequence.
        """
        # 1. First render base skeletal & landmark tracking overlay
        canvas = self.landmark_stream.render_debug_overlay(bgr_image, landmark_frame, fps_display)
        h, w = canvas.shape[:2]

        # 2. Render Temporal Buffer Progress Bar (Bottom-Left)
        buf_len = self.temporal_buffer.length()
        buf_max = self.temporal_buffer.max_length
        buf_ratio = min(1.0, buf_len / max(1, buf_max))

        bar_x, bar_y = 15, h - 35
        bar_w, bar_h = 240, 18

        # Bar background
        cv2.rectangle(canvas, (bar_x - 5, bar_y - 20), (bar_x + bar_w + 90, bar_y + bar_h + 5), (20, 20, 20), -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1)

        # Bar fill (Yellow during fill, Green when full/ready)
        fill_w = int(bar_w * buf_ratio)
        fill_color = (0, 255, 0) if buf_len >= buf_max else (0, 200, 255)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x + 1, bar_y + 1), (bar_x + fill_w - 1, bar_y + bar_h - 1), fill_color, -1)

        buf_text = f"BUFFER: {buf_len}/{buf_max} ({buf_ratio:.0%})"
        cv2.putText(canvas, buf_text, (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        pred_fps_text = f"{self._prediction_fps:.1f} pred/s"
        cv2.putText(canvas, pred_fps_text, (bar_x + bar_w + 10, bar_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # 3. Render Prediction HUD Panel (Top-Right)
        panel_w = 370
        panel_h = 220
        panel_x = max(10, w - panel_w - 15)
        panel_y = 10

        overlay = canvas.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (90, 90, 90), 1)

        pred = self.latest_prediction
        if pred is not None:
            # Color based on confidence
            if pred.confidence >= 0.70:
                conf_color = (0, 255, 0)      # Green
            elif pred.confidence >= 0.40:
                conf_color = (0, 215, 255)    # Yellow
            else:
                conf_color = (0, 140, 255)    # Orange

            # Header
            cv2.putText(
                canvas,
                f"ST-GCN INFERENCE ({pred.device.upper()})",
                (panel_x + 12, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
                cv2.LINE_AA
            )

            # Raw prediction line
            raw_text = f"Raw: {pred.label.upper()} ({pred.confidence:.1%})"
            cv2.putText(canvas, raw_text, (panel_x + 12, panel_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, conf_color, 1, cv2.LINE_AA)

            # Smoothed prediction line
            sm_lbl = self.latest_smoothed.label.upper() if self.latest_smoothed else "(NONE)"
            sm_text = f"Smoothed: {sm_lbl}"
            cv2.putText(canvas, sm_text, (panel_x + 12, panel_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            # State & Quality
            curr_state = self.sign_state_machine.state.value
            state_color = (0, 255, 0) if curr_state == "ACTIVE" else (0, 200, 255) if curr_state == "CANDIDATE" else (180, 180, 180)
            cv2.putText(canvas, f"State: {curr_state}", (panel_x + 12, panel_y + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, state_color, 1, cv2.LINE_AA)

            q_score = pred.input_quality
            cv2.putText(canvas, f"Quality: {q_score:.2f}", (panel_x + 180, panel_y + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

            # Latest Event
            evt_lbl = self.latest_event.label.upper() if self.latest_event else "(NONE)"
            cv2.putText(canvas, f"Event: {evt_lbl}", (panel_x + 12, panel_y + 116), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 50), 1, cv2.LINE_AA)

            # Sequence (glosses)
            seq_str = self.sign_sequence.format_gloss_string()
            if len(seq_str) > 28:
                seq_str = "..." + seq_str[-25:]
            cv2.putText(canvas, f"Seq: {seq_str}", (panel_x + 12, panel_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

            # Top-3 predictions breakdown
            top_y = panel_y + 162
            for rank, (cid, lbl, prob) in enumerate(pred.top_k[:3], 1):
                top_line = f"#{rank} {lbl.upper():<10} {prob:.1%}"
                cv2.putText(
                    canvas,
                    top_line,
                    (panel_x + 15, top_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (200, 200, 200) if rank > 1 else conf_color,
                    1,
                    cv2.LINE_AA
                )
                top_y += 18


        else:
            # Buffer warming / Waiting state
            cv2.putText(
                canvas,
                "ST-GCN INFERENCE",
                (panel_x + 12, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (180, 180, 180),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                canvas,
                f"Accumulating window ({buf_len}/{buf_max})...",
                (panel_x + 12, panel_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 215, 255),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                canvas,
                f"Requires {buf_max} frames (1.8s) of context",
                (panel_x + 12, panel_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (160, 160, 160),
                1,
                cv2.LINE_AA
            )

        return canvas

    def get_inference_metrics(self) -> Dict[str, Any]:
        """Returns diagnostic metrics for inference latency and window throughput."""
        lats = self._inference_latencies_ms
        if lats:
            mean_lat = float(np.mean(lats))
            median_lat = float(np.median(lats))
            p95_lat = float(np.percentile(lats, 95))
            max_lat = float(np.max(lats))
        else:
            mean_lat = median_lat = p95_lat = max_lat = 0.0

        stream_metrics = self.landmark_stream.get_metrics()

        return {
            "total_predictions": self._total_predictions,
            "prediction_fps": round(self._prediction_fps, 2),
            "windows_skipped": self.scheduler.windows_skipped,
            "mean_inference_latency_ms": round(mean_lat, 2),
            "median_inference_latency_ms": round(median_lat, 2),
            "p95_inference_latency_ms": round(p95_lat, 2),
            "max_inference_latency_ms": round(max_lat, 2),
            "stream_capture_fps": stream_metrics.capture_fps,
            "stream_processing_fps": stream_metrics.processing_fps,
            "mean_stream_latency_ms": stream_metrics.mean_total_latency_ms
        }

    def close(self) -> None:
        """Releases all stream, sink, and buffer resources."""
        self.landmark_stream.close()
        self.composite_sink.close()
        self.temporal_buffer.clear()
