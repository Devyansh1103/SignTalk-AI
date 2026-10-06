"""Deterministic landmark replay prediction tool.

Replays recorded landmark sequences (JSON or NPZ format) through the full
ST-GCN temporal inference, confidence filtering, smoothing, state machine,
and continuous sign detection pipeline.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.data.node_schema import TOTAL_NODES, CHANNELS, TARGET_SEQUENCE_LENGTH
from src.realtime.confidence_filter import ConfidenceFilter
from src.realtime.event_deduplicator import EventDeduplicator
from src.realtime.model_runner import STGCNRunner
from src.realtime.prediction_history import PredictionHistory, PredictionRecord
from src.realtime.realtime_config import RealTimeConfig
from src.realtime.sign_event import SignEvent
from src.realtime.sign_sequence import SignSequenceBuffer
from src.realtime.sign_state_machine import SignStateMachine
from src.realtime.smoothing import (
    BaseSmoother,
    ConfidenceWeightedSmoother,
    MajorityVoteSmoother,
    TemporalStabilitySmoother,
)
from src.realtime.temporal_buffer import TemporalBuffer
from src.realtime.types import LandmarkFrame, ModalityDetection, QualityReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ReplayPredictions")


def load_input_landmarks(input_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Loads landmark tensor from either NPZ or JSON file.

    Returns:
        coords: np.ndarray of shape (3, T, 93)
        mask: np.ndarray of shape (1, T, 93) or None
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input landmark file not found: {input_path}")

    if path.suffix.lower() == ".npz":
        npz = np.load(path)
        data = npz["data"]  # (3, T, 93)
        mask = npz["mask"] if "mask" in npz else None
        return data, mask

    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "frames" in raw:
            frames = raw["frames"]
        elif isinstance(raw, list):
            frames = raw
        else:
            raise ValueError(f"Unrecognized JSON structure in {input_path}")

        # Each frame has 'normalized_coords' of shape (93, 3)
        t_len = len(frames)
        coords = np.zeros((3, t_len, TOTAL_NODES), dtype=np.float32)
        for t, f in enumerate(frames):
            nc = np.array(f.get("normalized_coords", np.zeros((TOTAL_NODES, 3))), dtype=np.float32)
            if nc.shape == (TOTAL_NODES, 3):
                coords[:, t, :] = nc.T
            elif nc.shape == (3, TOTAL_NODES):
                coords[:, t, :] = nc
        return coords, None

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Expected .npz or .json")


def replay_sequence(
    input_path: str,
    config: Optional[RealTimeConfig] = None,
    checkpoint_path: Optional[str] = None,
    stride: int = 5,
    repeat: int = 2,
    output_events_path: Optional[str] = None,
) -> List[SignEvent]:
    """Deterministically replays a landmark sequence through the pipeline.

    Args:
        input_path: Path to NPZ or JSON landmark sequence.
        config: RealTimeConfig instance (loaded from configs/realtime.yaml by default).
        checkpoint_path: Path to ST-GCN checkpoint.
        stride: Stride frames between window inferences.
        repeat: Number of repetitions of sequence to simulate sustained continuous gesture.
        output_events_path: Optional file to save emitted SignEvents.

    Returns:
        List of emitted SignEvents in order.
    """
    cfg = config or RealTimeConfig.from_yaml()
    ckpt = checkpoint_path or cfg.inference.checkpoint_path

    coords, mask = load_input_landmarks(input_path)
    c, base_frames, v = coords.shape
    if repeat > 1:
        coords = np.tile(coords, (1, repeat, 1))
    c, total_frames, v = coords.shape
    logger.info("Loaded sequence from %s: shape %s (%d total frames, %d repeats)", input_path, coords.shape, total_frames, repeat)


    # Initialize model runner
    model_runner = STGCNRunner(checkpoint_path=ckpt, device="cpu", warmup_iterations=0)

    # Pipeline stages
    conf_filter = ConfidenceFilter(
        confidence_threshold=cfg.confidence.threshold,
        min_input_quality=cfg.stability.min_input_quality,
    )
    history = PredictionHistory(max_history=cfg.smoothing.history_size * 4)

    if cfg.smoothing.method == "confidence_weighted":
        smoother: BaseSmoother = ConfidenceWeightedSmoother(
            history_size=cfg.smoothing.history_size,
            recency_decay=cfg.smoothing.decay_factor,
        )
    elif cfg.smoothing.method == "temporal_stability":
        smoother = TemporalStabilitySmoother(
            min_consecutive=cfg.smoothing.min_consecutive,
        )
    else:
        smoother = MajorityVoteSmoother(
            history_size=cfg.smoothing.history_size,
            min_votes=cfg.smoothing.min_votes,
        )

    state_machine = SignStateMachine(
        min_consecutive_predictions=cfg.stability.min_consecutive_predictions,
        min_confidence=cfg.stability.min_confidence,
        min_input_quality=cfg.stability.min_input_quality,
    )
    deduplicator = EventDeduplicator(minimum_gap_ms=cfg.events.minimum_gap_ms)
    sequence_buffer = SignSequenceBuffer(max_events=cfg.sequence.max_events)

    window_size = cfg.buffer.max_length  # 45

    # Simulate streaming frame by frame into temporal buffer
    temp_buf = TemporalBuffer(max_length=window_size)
    fps = 25.0

    step_idx = 0
    for frame_idx in range(total_frames):
        raw_c = coords[:, frame_idx, :].T  # (93, 3)
        lf = LandmarkFrame(
            frame_id=frame_idx,
            timestamp=frame_idx / fps,
            raw_coords=raw_c.astype(np.float32),
            normalized_coords=raw_c.astype(np.float32),
            mask=np.ones(TOTAL_NODES, dtype=bool),
            visibility=np.ones(TOTAL_NODES, dtype=np.float32),
            modality_stats=ModalityDetection(
                pose_detected=True,
                left_hand_detected=True,
                right_hand_detected=True,
                face_detected=False,
            ),

            quality=QualityReport(
                is_valid=True,
                quality_score=0.85,
                classification="GOOD",
                missing_ratio=0.10,
                has_nan_or_inf=False,
                coordinate_in_range=True,
                sudden_jumps_detected=False,
            ),

        )
        temp_buf.append(lf)

        # Trigger sliding inference every `stride` frames once buffer is ready
        if temp_buf.is_ready() and (frame_idx % stride == 0 or frame_idx == total_frames - 1):
            window = temp_buf.get_window()
            prediction = model_runner.predict_window(window)

            filtered = conf_filter.filter(prediction)
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
            )
            history.append(record)

            smoothed = smoother.smooth(history)
            _, completed = state_machine.process(smoothed)

            logger.info(
                "Step %d (frame %d): raw=%s(%.2f), filtered=%s, smoothed=%s(%.2f, stable=%s), state=%s, completed=%s",
                step_idx,
                frame_idx,
                prediction.label,
                prediction.confidence,
                filtered.label,
                smoothed.label,
                smoothed.confidence,
                smoothed.is_stable,
                state_machine.state.value,
                completed.label if completed else None,
            )

            if completed is not None:
                deduped = deduplicator.process(completed)
                if deduped is not None:
                    sequence_buffer.append_event(deduped)
            step_idx += 1


    # End-of-stream flush
    flush_evt = state_machine.force_end()
    if flush_evt is not None:
        deduped = deduplicator.process(flush_evt)
        if deduped is not None:
            sequence_buffer.append_event(deduped)

    events = sequence_buffer.get_sequence()
    logger.info("Replay finished. Total emitted events: %d", len(events))
    logger.info("Sequence glosses: %s", sequence_buffer.format_gloss_string())

    if output_events_path:
        sequence_buffer.save_to_csv(output_events_path)

    return events


def main():
    parser = argparse.ArgumentParser(description="Deterministic Landmark Replay Tool")
    parser.add_argument("--input", required=True, help="Path to input NPZ or JSON landmark sequence")
    parser.add_argument("--config", default="configs/realtime.yaml", help="Path to realtime.yaml")
    parser.add_argument("--checkpoint", default=None, help="ST-GCN checkpoint path")
    parser.add_argument("--stride", type=int, default=5, help="Sliding inference stride frames")
    parser.add_argument("--repeat", type=int, default=2, help="Number of repetitions to simulate continuous gesture")
    parser.add_argument("--output-events", default=None, help="Path to save events CSV")

    args = parser.parse_args()
    cfg = RealTimeConfig.from_yaml(args.config)
    replay_sequence(
        input_path=args.input,
        config=cfg,
        checkpoint_path=args.checkpoint,
        stride=args.stride,
        repeat=args.repeat,
        output_events_path=args.output_events,
    )



if __name__ == "__main__":
    main()
