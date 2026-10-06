"""
SignTalk AI - Real-Time Landmark Stream CLI Entry Point
Executes the real-time camera ingestion and landmark streaming pipeline.

Usage:
  python scripts/run_realtime_landmarks.py
  python scripts/run_realtime_landmarks.py --camera-id 0 --fps 25
  python scripts/run_realtime_landmarks.py --video-file data/interim/landmark_pilot/pilot_sample_01.mp4
  python scripts/run_realtime_landmarks.py --benchmark --max-frames 100 --headless
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

from src.realtime.camera import Camera, CameraInitializationError, CameraError
from src.realtime.landmark_stream import RealTimeLandmarkStream
from src.realtime.realtime_config import RealTimeConfig
from src.realtime.frame_processor import FrameProcessor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SignTalk.RealTime.Runner")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SignTalk AI Real-Time Camera & Landmark Streaming Pipeline"
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
        help="Optional video file to stream through camera pipeline as surrogate"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Frame width (overrides config)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Frame height (overrides config)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target frame rate (overrides config)"
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
        "--record",
        action="store_true",
        help="Record landmark stream session to disk (privacy warning: disabled by default)"
    )
    parser.add_argument(
        "--threaded",
        action="store_true",
        help="Run camera capture in dedicated background thread (default: True for live webcam, False for video file)"
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Disable horizontal flip preview"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load Configuration
    config = RealTimeConfig.from_yaml(args.config)

    # 2. Apply CLI Overrides
    if args.camera_id is not None:
        config.camera.device_index = args.camera_id
    if args.video_file is not None:
        config.camera.device_index = args.video_file
    if args.width is not None:
        config.camera.width = args.width
        config.processing.target_width = args.width
    if args.height is not None:
        config.camera.height = args.height
        config.processing.target_height = args.height
    if args.fps is not None:
        config.camera.target_fps = args.fps
    if args.headless:
        config.runtime.display_preview = False
    if args.record:
        config.debug.record_landmarks = True
    if args.no_flip:
        config.processing.flip_horizontal = False

    logger.info("Initializing SignTalk AI Real-Time Landmark Stream...")
    logger.info(f"Source Device: {config.camera.device_index}")
    logger.info(f"Target Resolution: {config.camera.width}x{config.camera.height} @ {config.camera.target_fps} FPS")

    # 3. Instantiate Stream & Camera
    try:
        stream = RealTimeLandmarkStream(config)
    except Exception as e:
        logger.error(f"Failed to initialize MediaPipe Landmark Stream: {e}")
        sys.exit(1)

    # Determine threading mode
    if args.threaded:
        use_threading = True
    elif args.video_file is not None:
        use_threading = False
    else:
        use_threading = True

    try:
        camera = Camera(
            device_index=config.camera.device_index,
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.target_fps,
            threaded=use_threading,
            auto_reconnect=config.camera.auto_reconnect,
            loop_video=(args.video_file is not None and not args.benchmark)
        )
        camera.start()
    except CameraInitializationError as e:
        logger.error("=" * 60)
        logger.error("Camera unavailable.")
        logger.error("Please check camera permissions, device connection, or use --video-file.")
        logger.error(f"Details: {e}")
        logger.error("=" * 60)
        stream.close()
        sys.exit(1)

    preview_name = config.runtime.preview_window_name
    display_preview = config.runtime.display_preview

    logger.info("Camera started successfully. Processing frames... Press 'q' or ESC in preview window to exit.")

    frame_count = 0
    start_time = time.perf_counter()

    try:
        while camera.is_running():
            packet = camera.read(timeout=1.0)
            if packet is None:
                # Video file finished or timeout
                if args.video_file is not None:
                    logger.info("End of video stream reached.")
                    break
                continue

            frame_count += 1

            # Ingest frame into Landmark Stream
            landmark_frame = stream.process_frame(packet)

            # Visual Debugging HUD
            if display_preview:
                preview_frame = FrameProcessor.to_bgr_for_preview(
                    packet.image,
                    is_rgb=packet.is_rgb
                )
                hud_frame = stream.render_debug_overlay(preview_frame, landmark_frame)

                cv2.imshow(preview_name, hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("Quit signal received from preview window.")
                    break

            if args.max_frames and frame_count >= args.max_frames:
                logger.info(f"Reached specified max frames ({args.max_frames}). Stopping stream.")
                break

            # Heartbeat logger every 50 frames
            if frame_count % 50 == 0:
                metrics = stream.get_metrics()
                logger.info(
                    f"[Frame #{frame_count:04d}] Latency: {landmark_frame.total_latency_ms:.1f}ms "
                    f"(p50={metrics.median_total_latency_ms:.1f}ms, p95={metrics.p95_total_latency_ms:.1f}ms) | "
                    f"FPS: {metrics.processing_fps:.1f} | Quality: {landmark_frame.quality.classification} "
                    f"({landmark_frame.quality.quality_score:.2f})"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
    finally:
        total_wall_time = time.perf_counter() - start_time
        camera.stop()

        if display_preview:
            cv2.destroyAllWindows()

        # Final Metrics Report
        metrics = stream.get_metrics()
        print("\n" + "=" * 65)
        print("     SIGNTALK AI — REAL-TIME STREAMING BENCHMARK REPORT")
        print("=" * 65)
        print(f"Total Frames Captured:        {metrics.total_captured_frames}")
        print(f"Total Frames Processed:       {metrics.total_processed_frames}")
        print(f"Total Frames Dropped:         {metrics.total_dropped_frames}")
        print(f"Valid Frame Ratio:            {metrics.valid_frame_ratio * 100:.1f}%")
        print(f"Total Session Duration:       {total_wall_time:.2f} s")
        print(f"Effective Processing FPS:     {metrics.processing_fps:.2f} FPS")
        print("-" * 65)
        print("LATENCY BREAKDOWN (Post Warm-Up):")
        print(f"  Camera Queue Latency:       {metrics.mean_capture_latency_ms:.2f} ms")
        print(f"  MediaPipe Extraction:       {metrics.mean_extraction_latency_ms:.2f} ms")
        print(f"  Torso Normalization:        {metrics.mean_normalization_latency_ms:.2f} ms")
        print(f"  Quality Validation:         {metrics.mean_quality_latency_ms:.2f} ms")
        print(f"  Total Latency (Mean):       {metrics.mean_total_latency_ms:.2f} ms")
        print(f"  Total Latency (Median p50): {metrics.median_total_latency_ms:.2f} ms")
        print(f"  Total Latency (p95):        {metrics.p95_total_latency_ms:.2f} ms")
        print(f"  Total Latency (Max):        {metrics.max_total_latency_ms:.2f} ms")
        print("=" * 65 + "\n")

        # Save debug recordings if requested
        if config.debug.record_landmarks:
            saved_file = stream.save_recorded_session()
            if saved_file:
                logger.info(f"Recorded landmark session saved to: {saved_file}")

        stream.close()


if __name__ == "__main__":
    main()
