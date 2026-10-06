"""
SignTalk AI - Real-Time ST-GCN Sign Prediction CLI Entry Point
Phase 4 Part 2: Sliding-Window Inference & Temporal Prediction.

Executes end-to-end real-time sign recognition:
  Camera / Video Ingestion
           ↓
  Real-Time Landmark Stream (MediaPipe + Torso Normalization)
           ↓
  Temporal Buffer (T=45 rolling window)
           ↓
  Inference Scheduler (Stride pacing & stale drop)
           ↓
  ST-GCN Model Runner (SignTalk_STGCN_v1)
           ↓
  Prediction HUD & Console / CSV Output

Usage:
  python scripts/run_realtime_prediction.py
  python scripts/run_realtime_prediction.py --device cpu --stride 5 --top-k 3
  python scripts/run_realtime_prediction.py --video-file data/raw/videos/hello_signer_01_rep1.mp4
  python scripts/run_realtime_prediction.py --benchmark --max-frames 90 --headless
"""

import os
import sys
import time
import argparse
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from src.realtime.types import FramePacket
from src.realtime.camera import Camera, CameraInitializationError, CameraError
from src.realtime.frame_processor import FrameProcessor
from src.realtime.realtime_config import RealTimeConfig
from src.realtime.realtime_predictor import RealTimeSignPredictor
from src.realtime.model_runner import STGCNRunner, ModelRunnerError, ShapeValidationError
from src.realtime.prediction_sink import (
    ConsolePredictionSink,
    PredictionLoggerSink,
    CompositePredictionSink
)
from src.data.node_schema import TARGET_SEQUENCE_LENGTH

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SignTalk.RealTime.PredictionCLI")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SignTalk AI Real-Time Sliding-Window ST-GCN Sign Prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/realtime.yaml",
        help="Path to real-time configuration YAML"
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=None,
        help="Camera device index (overrides config)"
    )
    parser.add_argument(
        "--video-file",
        type=str,
        default=None,
        help="Path to video file to stream through camera pipeline as surrogate"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to ST-GCN checkpoint (.pt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Compute device for ST-GCN inference ('auto', 'cpu', 'cuda')"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Temporal window size T (must match model sequence length: 45)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sliding stride in frames between inference passes (default: 5)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of ranked predictions to report (default: 3)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI preview window"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run in benchmark mode and output performance breakdown"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before terminating"
    )
    parser.add_argument(
        "--record-predictions",
        action="store_true",
        help="Save prediction log to CSV"
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default=None,
        help="Output CSV path for prediction records"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target camera FPS (overrides config)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 1. Load Configuration
    config = RealTimeConfig.from_yaml(args.config)

    # Apply overrides
    if args.camera_id is not None:
        config.camera.device_index = args.camera_id
    if args.fps is not None:
        config.camera.target_fps = args.fps
    if args.device is not None:
        config.inference.device = args.device
    if args.checkpoint is not None:
        config.inference.checkpoint_path = args.checkpoint
    if args.stride is not None:
        config.scheduler.stride = args.stride
    if args.top_k is not None:
        config.inference.top_k = args.top_k
    if args.record_predictions:
        config.inference.record_predictions = True
    if args.predictions_csv is not None:
        config.inference.predictions_csv = args.predictions_csv

    # Window size validation: ST-GCN strictly expects TARGET_SEQUENCE_LENGTH (45)
    if args.window_size is not None:
        if args.window_size != TARGET_SEQUENCE_LENGTH:
            raise ShapeValidationError(
                f"Window size override T={args.window_size} is incompatible with trained "
                f"ST-GCN model sequence length T={TARGET_SEQUENCE_LENGTH}. Overriding with "
                "incompatible sequence length is prohibited without model retraining."
            )
        config.buffer.max_length = args.window_size

    display_preview = not args.headless and config.runtime.display_preview

    logger.info("=" * 60)
    logger.info("  SignTalk AI — Real-Time ST-GCN Sign Prediction (Phase 4 Part 2)")
    logger.info("=" * 60)
    logger.info(f"Model Checkpoint : {config.inference.checkpoint_path}")
    logger.info(f"Inference Device : {config.inference.device}")
    logger.info(f"Sequence Length  : T={config.buffer.max_length} frames (1.8s @ 25 FPS)")
    logger.info(f"Sliding Stride   : {config.scheduler.stride} frames (update every {config.scheduler.stride / config.camera.target_fps * 1000:.0f} ms)")
    logger.info(f"Top-K Ranks      : {config.inference.top_k}")
    logger.info(f"Record Preds CSV : {config.inference.record_predictions}")
    logger.info(f"Display Preview  : {display_preview}")
    logger.info("=" * 60)

    # 2. Initialize Camera Ingestion
    cam_source = args.video_file if args.video_file is not None else config.camera.device_index
    is_video_surrogate = args.video_file is not None

    try:
        camera = Camera(
            device_index=cam_source,
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.target_fps,
            auto_reconnect=False if is_video_surrogate else config.camera.auto_reconnect
        )
        camera.start()
    except CameraInitializationError as e:
        logger.error(f"Failed to initialize camera source: {e}")
        sys.exit(1)

    # 3. Initialize Prediction Sinks
    sinks = [ConsolePredictionSink(show_top_k=True, verbose=args.debug)]
    if config.inference.record_predictions:
        sinks.append(PredictionLoggerSink(output_path=config.inference.predictions_csv))
        logger.info(f"Logging predictions to: {config.inference.predictions_csv}")

    # 4. Initialize Predictor Pipeline
    try:
        predictor = RealTimeSignPredictor(
            config=config,
            prediction_sinks=sinks
        )
    except Exception as e:
        logger.error(f"Failed to initialize RealTimeSignPredictor: {e}", exc_info=True)
        camera.stop()
        sys.exit(1)

    window_name = "SignTalk AI — Real-Time Sign Predictor (ST-GCN)"
    if display_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()
    t_prev = time.perf_counter()
    fps_estimate = config.camera.target_fps

    logger.info("Starting real-time prediction loop. Press 'q' or ESC in preview window to exit.")

    try:
        while True:
            # Capture frame
            packet = camera.read()
            if packet is None:
                if is_video_surrogate:
                    logger.info("End of video stream reached.")
                else:
                    logger.warning("Camera stream disconnected.")
                break

            # Pipeline processing & inference
            landmark_frame, prediction = predictor.process_frame_packet(packet)
            frame_count += 1

            # Estimate FPS
            t_curr = time.perf_counter()
            dt = t_curr - t_prev
            t_prev = t_curr
            if dt > 0:
                fps_estimate = 0.9 * fps_estimate + 0.1 * (1.0 / dt)

            # Display GUI Preview
            if display_preview:
                preview_bgr = FrameProcessor.to_bgr_for_preview(
                    packet.image,
                    is_rgb=packet.is_rgb
                )
                display_frame = predictor.render_prediction_overlay(
                    bgr_image=preview_bgr,
                    landmark_frame=landmark_frame,
                    fps_display=fps_estimate
                )
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    logger.info("User requested exit.")
                    break

            if args.max_frames and frame_count >= args.max_frames:
                logger.info(f"Reached max frame limit: {args.max_frames}")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.error(f"Runtime prediction error: {e}", exc_info=True)
    finally:
        total_time = max(0.001, time.time() - start_time)
        actual_fps = frame_count / total_time

        logger.info("\n" + "=" * 60)
        logger.info("  SignTalk AI — Prediction Run Summary")
        logger.info("=" * 60)
        logger.info(f"Total Frames Processed : {frame_count}")
        logger.info(f"Total Run Time         : {total_time:.2f} s")
        logger.info(f"Effective Ingestion FPS: {actual_fps:.2f} fps")

        metrics = predictor.get_inference_metrics()
        logger.info(f"Total Predictions Made : {metrics['total_predictions']}")
        logger.info(f"Prediction Rate        : {metrics['prediction_fps']:.2f} pred/s")
        logger.info(f"Windows Skipped/Stale  : {metrics['windows_skipped']}")
        logger.info(f"Mean Inference Latency : {metrics['mean_inference_latency_ms']:.2f} ms")
        logger.info(f"Median Inference Lat   : {metrics['median_inference_latency_ms']:.2f} ms")
        logger.info(f"P95 Inference Latency  : {metrics['p95_inference_latency_ms']:.2f} ms")
        logger.info(f"Max Inference Latency  : {metrics['max_inference_latency_ms']:.2f} ms")
        logger.info("=" * 60)

        predictor.close()
        camera.stop()
        if display_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
