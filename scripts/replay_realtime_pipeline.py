"""
SignTalk AI - Real-Time Pipeline Replay Harness
Phase 4 Part 4: Deterministic Sequence Replay from Landmark Files or Video.

Usage:
  python scripts/replay_realtime_pipeline.py --input data/processed/sequences/val/seq_0037.npz
  python scripts/replay_realtime_pipeline.py --input data/interim/landmark_pilot/pilot_sample_01.mp4 --device cpu
"""

import os
import sys
import argparse
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.realtime.realtime_config import RealTimeConfig
from src.realtime.realtime_pipeline import RealtimePipeline
from src.realtime.model_runner import STGCNRunner
from src.realtime.types import LandmarkFrame, ModalityDetection, QualityReport
from src.data.node_schema import TOTAL_NODES, TARGET_SEQUENCE_LENGTH


def replay(input_path: str, checkpoint: str = "experiments/stgcn/checkpoints/best_checkpoint.pt", device: str = "cpu"):
    print("=" * 70)
    print("SignTalk AI — Deterministic Pipeline Replay Harness")
    print(f"Input:      {input_path}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Device:     {device}")
    print("=" * 70)

    cfg = RealTimeConfig()
    cfg.inference.checkpoint_path = checkpoint
    cfg.inference.device = device
    cfg.inference.warmup_iterations = 1
    cfg.scheduler.stride = 5

    runner = STGCNRunner(checkpoint_path=checkpoint, device=device, warmup_iterations=1)
    pipeline = RealtimePipeline(config=cfg, model_runner=runner)
    pipeline.initialize()

    if input_path.endswith(".npz"):
        data = np.load(input_path)
        coords = data["data"]  # (3, T, 93)
        mask = data["mask"]    # (1, T, 93)
        coords_t_v_c = np.transpose(coords, (1, 2, 0))
        mask_t_v = mask[0]
        T = coords.shape[1]

        print(f"Replaying {T} frames from NPZ sequence...")
        for rep in range(2):
            for t in range(T):
                f_id = rep * T + t
                lf = LandmarkFrame(
                    frame_id=f_id,
                    timestamp=float(f_id) / 25.0,
                    raw_coords=coords_t_v_c[t].astype(np.float32),
                    normalized_coords=coords_t_v_c[t].astype(np.float32),
                    mask=mask_t_v[t].astype(bool),
                    visibility=np.ones((TOTAL_NODES,), dtype=np.float32),
                    modality_stats=ModalityDetection(True, True, True, False),
                    quality=QualityReport(True, 0.95, "GOOD", 0.0, False, True, False)
                )
                pipeline.temporal_buffer.append(lf)
                pipeline.scheduler.on_frame_appended()
                if pipeline.scheduler.should_infer(pipeline.temporal_buffer):
                    pred = pipeline.model_runner.predict_window(pipeline.temporal_buffer.get_window())
                    filtered = pipeline.confidence_filter.filter_prediction(pred)
                    from src.realtime.prediction_history import PredictionRecord
                    pipeline.prediction_history.add(
                        PredictionRecord(
                            timestamp=pred.timestamp,
                            window_start=pred.window_start_time,
                            window_end=pred.window_end_time,
                            class_id=filtered.class_id,
                            label=filtered.label,
                            gloss=filtered.gloss,
                            confidence=filtered.confidence,
                            input_quality=filtered.input_quality,
                            inference_latency_ms=pred.inference_latency_ms,
                            is_valid_quality=pred.is_valid_quality,
                            probabilities=pred.probabilities,
                        )
                    )
                    smoothed = pipeline.smoother.smooth(pipeline.prediction_history)
                    _, evt = pipeline.sign_state_machine.process(smoothed)
                    if evt is not None:
                        dedup = pipeline.event_deduplicator.process(evt)
                        if dedup is not None:
                            pipeline.sign_sequence.append_event(dedup)
                            pipeline.translator.process_event(dedup)
                            print(f"  [EVENT] Emitted: {dedup.label.upper()} (conf={dedup.confidence:.1%}, dur={dedup.duration_ms:.0f}ms)")

        pipeline.translator.finalize_phrase()
        phrases = pipeline.translator.finalized_phrases
        print("\nFinalized Transcript Phrases:")
        for idx, phr in enumerate(phrases, 1):
            print(f"  Phrase #{idx}: \"{phr['text']}\" (Hindi: \"{phr['hindi_text']}\", conf={phr['confidence']:.2f})")

    pipeline.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input NPZ or MP4 file")
    parser.add_argument("--checkpoint", type=str, default="experiments/stgcn/checkpoints/best_checkpoint.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    replay(args.input, args.checkpoint, args.device)
