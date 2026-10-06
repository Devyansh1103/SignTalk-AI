"""
SignTalk AI - Real-Time End-to-End Inference CLI Entry Point
Phase 4 Part 4: Live Camera / Video Stream Recognition & Translation.

Usage:
  python scripts/run_realtime.py
  python scripts/run_realtime.py --camera-id 0 --device cpu
  python scripts/run_realtime.py --video-file data/raw/videos/hello_signer_01_rep1.mp4
  python scripts/run_realtime.py --mode benchmark --max-frames 200 --headless
"""

import os
import sys
import time
import argparse
import logging
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.realtime.realtime_config import RealTimeConfig
from src.realtime.camera import Camera, CameraError
from src.realtime.model_runner import STGCNRunner
from src.realtime.realtime_pipeline import RealtimePipeline, PipelineState


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SignTalk.RealTime.Runner")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SignTalk AI Real-Time End-to-End Sign Recognition & Translation CLI"
    )
    parser.add_argument("--config", type=str, default="configs/realtime.yaml", help="Path to YAML configuration")
    parser.add_argument("--camera-id", type=int, default=None, help="Webcam device index (e.g. 0)")
    parser.add_argument("--video-file", type=str, default=None, help="Path to video file surrogate")
    parser.add_argument("--checkpoint", type=str, default=None, help="ST-GCN model checkpoint (.pt)")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"], help="Inference compute device")
    parser.add_argument("--mode", type=str, default="debug", choices=["debug", "benchmark", "production"], help="Operational mode")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process before exiting")
    parser.add_argument("--headless", action="store_true", help="Run without graphical preview window")
    return parser.parse_args()


def main():
    args = parse_args()
    config = RealTimeConfig.from_yaml(args.config)

    # CLI Overrides
    if args.camera-id is not None:
        config.camera.device_index = args.camera_id
    if args.video_file:
        config.camera.device_index = args.video_file
    if args.checkpoint:
        config.inference.checkpoint_path = args.checkpoint
    if args.device:
        config.inference.device = args.device

    print("=" * 70)
    print("SignTalk AI — Real-Time Sign Recognition & Translation Engine")
    print(f"Source: {config.camera.device_index}")
    print(f"Device: {config.inference.device}")
    print(f"Mode:   {args.mode}")
    print("=" * 70)
    print("Controls:")
    print("  'q' or ESC : Quit application cleanly")
    print("  'r'        : Reset sign sequence & transcript buffer")
    print("  'p'        : Pause / Resume recognition")
    print("=" * 70)

    # Initialize Camera
    camera = Camera(
        device_index=config.camera.device_index,
        width=config.camera.width,
        height=config.camera.height,
        fps=config.camera.target_fps,
        threaded=True,
        loop_video=(args.video_file is not None)
    )

    try:
        camera.start()
    except CameraError as e:
        logger.error(f"Failed to start camera device: {e}")
        return

    # Initialize Pipeline
    runner = STGCNRunner(
        checkpoint_path=config.inference.checkpoint_path,
        device=config.inference.device,
        top_k=config.inference.top_k,
        warmup_iterations=config.inference.warmup_iterations
    )
    pipeline = RealtimePipeline(config=config, camera=camera, model_runner=runner)
    pipeline.initialize()

    frame_idx = 0
    window_name = "SignTalk AI — Live Translation Engine"

    try:
        while True:
            t_start = time.perf_counter()
            out = pipeline.step()
            frame_idx += 1

            if not args.headless:
                # Render diagnostic HUD canvas
                raw_frame = camera._cap.read()[1] if (hasattr(camera, "_cap") and camera._cap is not None) else None
                if raw_frame is None:
                    raw_frame = np.zeros((config.camera.height, config.camera.width, 3), dtype=np.uint8)

                canvas = pipeline.render_overlay(raw_frame, out)
                cv2.imshow(window_name, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    logger.info("Termination key received. Exiting...")
                    break
                elif key == ord('r'):
                    logger.info("Resetting sequence buffer...")
                    pipeline.reset()
                elif key == ord('p'):
                    if pipeline.state == PipelineState.RUNNING:
                        pipeline.pause()
                    else:
                        pipeline.resume()

            if args.max_frames and frame_idx >= args.max_frames:
                logger.info(f"Reached max frames limit ({args.max_frames}). Stopping...")
                break

    except KeyboardInterrupt:
        logger.info("Ctrl+C interrupt received. Stopping cleanly...")
    finally:
        pipeline.close()
        camera.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        print("\nSession complete. Resources safely released.")


if __name__ == "__main__":
    main()
