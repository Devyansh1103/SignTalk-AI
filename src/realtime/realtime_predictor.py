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

logger = logging.getLogger("SignTalk.RealTime.Predictor")


class RealTimeSignPredictor:
    """
    Top-level real-time sliding-window sign language predictor.
    Coordinates camera acquisition, landmark streaming, temporal buffering,
    ST-GCN inference scheduling, and prediction publishing.
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

        # Runtime State
        self.latest_prediction: Optional[PredictionResult] = None
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
        and current model prediction onto the display frame.
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
        panel_w = 340
        panel_h = 165
        panel_x = max(10, w - panel_w - 15)
        panel_y = 10

        overlay = canvas.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)
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
                (panel_x + 12, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
                cv2.LINE_AA
            )

            # Main Prediction Label
            pred_text = f"{pred.label.upper()}"
            cv2.putText(
                canvas,
                pred_text,
                (panel_x + 12, panel_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                conf_color,
                2,
                cv2.LINE_AA
            )

            # Confidence & Latency
            conf_lat_str = f"Conf: {pred.confidence:.1%} | Latency: {pred.inference_latency_ms:.1f}ms"
            cv2.putText(
                canvas,
                conf_lat_str,
                (panel_x + 12, panel_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (240, 240, 240),
                1,
                cv2.LINE_AA
            )

            # Top-3 predictions breakdown
            top_y = panel_y + 104
            for rank, (cid, lbl, prob) in enumerate(pred.top_k[:3], 1):
                top_line = f"#{rank} {lbl.upper():<10} {prob:.1%}"
                cv2.putText(
                    canvas,
                    top_line,
                    (panel_x + 15, top_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
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
