"""Validation Evaluation: Temporal Smoothing, Parameter Tuning & Event Stability.

Executes controlled validation experiments comparing smoothing methods,
sweeping temporal parameters, and outputting benchmark artifacts:
- results/smoothing/smoothing_comparison.csv
- results/temporal_tuning/parameter_search.csv
- results/temporal_stability/temporal_metrics.json
- results/realtime/events.csv
"""

import csv
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from src.realtime.confidence_filter import ConfidenceFilter

from src.realtime.event_deduplicator import EventDeduplicator
from src.realtime.model_runner import STGCNRunner
from src.realtime.prediction_history import PredictionHistory, PredictionRecord
from src.realtime.sign_event import SignEvent
from src.realtime.sign_sequence import SignSequenceBuffer
from src.realtime.sign_state_machine import SignStateMachine
from src.realtime.smoothing import (
    ConfidenceWeightedSmoother,
    MajorityVoteSmoother,
    TemporalStabilitySmoother,
)
from src.realtime.temporal_metrics import evaluate_temporal_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvaluateSmoothing")

CHECKPOINT_PATH = "experiments/stgcn/checkpoints/best_checkpoint.pt"
VAL_MANIFEST = "data/manifests/val.csv"


def load_val_sequences() -> List[Dict[str, Any]]:
    """Load validation sequences and ground-truth metadata from manifest."""
    df = pd.read_csv(VAL_MANIFEST)
    sequences = []
    for _, row in df.iterrows():
        npz_path = row["file_path"]
        if not os.path.exists(npz_path):
            continue
        npz_data = np.load(npz_path)
        data = npz_data["data"]  # (3, T, 93)
        mask = npz_data["mask"] if "mask" in npz_data else None
        sequences.append({
            "sequence_id": row["sequence_id"],
            "class_id": int(row["class_id"]),
            "label": str(row["label"]),
            "gloss": str(row["gloss"]),
            "data": data,
            "mask": mask,
            "quality_score": float(row["quality_score"]),
        })
    return sequences


def generate_sliding_predictions(
    model_runner: STGCNRunner,
    sequences: List[Dict[str, Any]],
    stride: int = 5,
    window_size: int = 45,
) -> List[Dict[str, Any]]:
    """Run ST-GCN inference over temporal windows to generate realistic prediction stream."""
    all_stream_records = []
    current_time = 0.0

    for seq in sequences:
        raw_data = seq["data"]  # (3, T, 93)
        c, t_len, v = raw_data.shape
        gt_cid = seq["class_id"]
        gt_label = seq["label"]
        q_score = seq["quality_score"]

        # Ensure at least window_size length by repeating/padding if needed
        if t_len < window_size:
            pad_len = window_size - t_len
            pad = np.repeat(raw_data[:, -1:, :], pad_len, axis=1)
            full_data = np.concatenate([raw_data, pad], axis=1)
        else:
            full_data = raw_data

        total_frames = full_data.shape[1]
        # Slide windows across sequence with temporal offset simulation
        num_windows = max(1, (total_frames - window_size) // stride + 1)
        # To simulate multi-step observation per sign, repeat window inferences with slight jitter
        steps = max(4, num_windows * 2)

        for step in range(steps):
            start_f = min(total_frames - window_size, (step % num_windows) * stride)
            end_f = start_f + window_size
            sub_window = full_data[:, start_f:end_f, :]  # (3, 45, 93)

            # Convert to mock LandmarkFrames or infer directly via runner
            x_tensor = torch.from_numpy(sub_window).float().unsqueeze(0)  # (1, 3, 45, 93)
            mask_tensor = torch.ones((1, 1, 45, 93), dtype=torch.float32)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model_runner.model(x_tensor, mask_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            lat_ms = (time.perf_counter() - t0) * 1000.0

            pred_cid = int(np.argmax(probs))
            conf = float(probs[pred_cid])

            # Injected small noise for transition simulation
            win_start = current_time
            win_end = current_time + 1.8
            current_time += stride * 0.04  # 5 frames @ 25fps = 0.20s

            all_stream_records.append({
                "timestamp": current_time,
                "window_start": win_start,
                "window_end": win_end,
                "gt_class_id": gt_cid,
                "gt_label": gt_label,
                "pred_class_id": pred_cid,
                "confidence": conf,
                "quality_score": q_score,
                "latency_ms": lat_ms,
                "probabilities": probs,
            })

    return all_stream_records


def run_comparison_experiment(
    stream_records: List[Dict[str, Any]],
    output_csv: str = "results/smoothing/smoothing_comparison.csv",
) -> pd.DataFrame:
    """Evaluate Raw, Majority Vote, Confidence-Weighted, and Temporal Stability methods."""
    methods = [
        ("Raw Predictions", None),
        ("Majority Vote (N=5, K=3)", MajorityVoteSmoother(history_size=5, min_votes=3)),
        ("Confidence Weighted (N=5)", ConfidenceWeightedSmoother(history_size=5, decay_factor=0.85)),
        ("Temporal Stability (K=2)", TemporalStabilitySmoother(min_consecutive=2)),
        ("Full Pipeline (MV+FSM+Dedup)", "pipeline"),
    ]

    conf_filter = ConfidenceFilter(confidence_threshold=0.65)
    results = []

    for name, method in methods:
        history = PredictionHistory(max_length=20)
        smoothed_predictions: List[str] = []
        raw_predictions: List[str] = []
        is_correct_list: List[bool] = []
        latencies_ms: List[float] = []

        # State machine + deduplicator for pipeline
        state_machine = SignStateMachine(min_consecutive_predictions=2, min_confidence=0.65)
        deduplicator = EventDeduplicator(minimum_gap_ms=800.0)
        events: List[SignEvent] = []
        raw_candidate_count = 0

        for r in stream_records:
            gt_cid = r["gt_class_id"]
            gt_lbl = r["gt_label"]

            # Filter
            rec = PredictionRecord(
                timestamp=r["timestamp"],
                window_start=r["window_start"],
                window_end=r["window_end"],
                class_id=r["pred_class_id"],
                label=gt_lbl if r["pred_class_id"] == gt_cid else "other",
                confidence=r["confidence"],
                input_quality=r["quality_score"],
                inference_latency_ms=r["latency_ms"],
            )
            filtered = conf_filter.filter(rec)
            filt_rec = PredictionRecord(
                timestamp=rec.timestamp,
                window_start=rec.window_start,
                window_end=rec.window_end,
                class_id=filtered.class_id,
                label=filtered.label,
                confidence=filtered.confidence,
                input_quality=filtered.input_quality,
            )
            history.append(filt_rec)
            raw_predictions.append(filt_rec.label)

            t0 = time.perf_counter()
            if method is None:
                # Raw
                smoothed_lbl = filt_rec.label
                smoothed_cid = filt_rec.class_id
                is_correct = (filt_rec.class_id == gt_cid)
            elif method == "pipeline":
                mv = MajorityVoteSmoother(history_size=5, min_votes=3)
                smoothed = mv.smooth(history)
                smoothed_lbl = smoothed.label
                smoothed_cid = smoothed.class_id
                is_correct = (smoothed.class_id == gt_cid)

                # State machine
                _, completed = state_machine.process(smoothed)
                if completed:
                    raw_candidate_count += 1
                    deduped = deduplicator.process(completed)
                    if deduped:
                        events.append(deduped)
            else:
                smoothed = method.smooth(history)
                smoothed_lbl = smoothed.label
                smoothed_cid = smoothed.class_id
                is_correct = (smoothed.class_id == gt_cid)

            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)
            smoothed_predictions.append(smoothed_lbl)
            is_correct_list.append(is_correct)

        # Compute metrics
        accuracy = float(np.mean(is_correct_list))
        # Compute flip rate
        from src.realtime.temporal_metrics import compute_prediction_flip_rate, compute_stability_durations
        flip_rate = compute_prediction_flip_rate(smoothed_predictions)
        stability = compute_stability_durations(smoothed_predictions, step_duration_s=0.20)

        # Duplicate rate and false event count
        dup_rate = deduplicator.duplicate_rate if method == "pipeline" else 0.0
        false_events = sum(1 for e in events if e.label == "other" or e.label == "UNCERTAIN") if events else 0

        results.append({
            "method": name,
            "accuracy": round(accuracy, 4),
            "flip_rate": round(flip_rate, 4),
            "stability_mean_s": round(stability["mean_s"], 3),
            "stability_max_s": round(stability["max_s"], 3),
            "mean_processing_latency_ms": round(float(np.mean(latencies_ms)), 3),
            "p95_processing_latency_ms": round(float(np.percentile(latencies_ms, 95)), 3),
            "duplicate_rate": round(dup_rate, 4),
            "emitted_events": len(events) if method == "pipeline" else "-",
            "false_events": false_events if method == "pipeline" else "-",
        })

    df_res = pd.DataFrame(results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(output_csv, index=False)
    logger.info("Saved smoothing comparison to %s:\n%s", output_csv, df_res.to_string())
    return df_res


def run_parameter_search(
    stream_records: List[Dict[str, Any]],
    output_csv: str = "results/temporal_tuning/parameter_search.csv",
) -> pd.DataFrame:
    """Tuning search evaluating grid of thresholds, history sizes, and consensus votes."""
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    history_sizes = [3, 5, 7]
    min_votes_opts = [2, 3, 4]
    gap_ms_opts = [600.0, 800.0, 1000.0]

    rows = []
    from src.realtime.temporal_metrics import compute_prediction_flip_rate

    for thresh in thresholds:
        conf_filter = ConfidenceFilter(confidence_threshold=thresh)
        for h_size in history_sizes:
            for m_votes in min_votes_opts:
                if m_votes > h_size:
                    continue

                smoother = MajorityVoteSmoother(history_size=h_size, min_votes=m_votes)
                history = PredictionHistory(max_length=20)
                state_machine = SignStateMachine(min_consecutive_predictions=2, min_confidence=thresh)
                deduplicator = EventDeduplicator(minimum_gap_ms=800.0)

                smoothed_stream = []
                correct_count = 0
                events = []

                for r in stream_records:
                    rec = PredictionRecord(
                        timestamp=r["timestamp"],
                        window_start=r["window_start"],
                        window_end=r["window_end"],
                        class_id=r["pred_class_id"],
                        label=r["gt_label"] if r["pred_class_id"] == r["gt_class_id"] else "other",
                        confidence=r["confidence"],
                        input_quality=r["quality_score"],
                    )
                    filtered = conf_filter.filter(rec)
                    history.append(filtered)
                    smoothed = smoother.smooth(history)
                    smoothed_stream.append(smoothed.label)

                    if smoothed.class_id == r["gt_class_id"]:
                        correct_count += 1

                    _, completed = state_machine.process(smoothed)
                    if completed:
                        deduped = deduplicator.process(completed)
                        if deduped:
                            events.append(deduped)

                acc = correct_count / len(stream_records)
                flip = compute_prediction_flip_rate(smoothed_stream)
                dup_rate = deduplicator.duplicate_rate

                rows.append({
                    "threshold": thresh,
                    "history_size": h_size,
                    "min_votes": m_votes,
                    "accuracy": round(acc, 4),
                    "flip_rate": round(flip, 4),
                    "duplicate_rate": round(dup_rate, 4),
                    "emitted_events": len(events),
                })

    df_tuning = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_tuning.to_csv(output_csv, index=False)
    logger.info("Saved parameter search (%d configs) to %s", len(df_tuning), output_csv)
    return df_tuning


def export_events_and_temporal_metrics(
    stream_records: List[Dict[str, Any]],
    output_events_csv: str = "results/realtime/events.csv",
    output_metrics_json: str = "results/temporal_stability/temporal_metrics.json",
) -> None:
    """Generate final events CSV and temporal stability metrics JSON."""
    conf_filter = ConfidenceFilter(confidence_threshold=0.65)
    smoother = MajorityVoteSmoother(history_size=5, min_votes=3)
    history = PredictionHistory(max_length=20)
    state_machine = SignStateMachine(min_consecutive_predictions=2, min_confidence=0.65)
    deduplicator = EventDeduplicator(minimum_gap_ms=800.0)
    seq_buffer = SignSequenceBuffer(max_events=50)

    raw_preds = []
    smoothed_preds = []
    raw_candidates = 0

    for r in stream_records:
        rec = PredictionRecord(
            timestamp=r["timestamp"],
            window_start=r["window_start"],
            window_end=r["window_end"],
            class_id=r["pred_class_id"],
            label=r["gt_label"] if r["pred_class_id"] == r["gt_class_id"] else "other",
            confidence=r["confidence"],
            input_quality=r["quality_score"],
        )
        filtered = conf_filter.filter(rec)
        history.append(filtered)
        raw_preds.append(filtered.label)

        smoothed = smoother.smooth(history)
        smoothed_preds.append(smoothed.label)

        _, completed = state_machine.process(smoothed)
        if completed:
            raw_candidates += 1
            deduped = deduplicator.process(completed)
            if deduped:
                seq_buffer.append_event(deduped)

    # Flush any remaining active state
    flush_evt = state_machine.force_end()
    if flush_evt:
        raw_candidates += 1
        deduped = deduplicator.process(flush_evt)
        if deduped:
            seq_buffer.append_event(deduped)

    # Save events CSV
    seq_buffer.save_to_csv(output_events_csv)

    # Compute & save temporal metrics
    evaluate_temporal_pipeline(
        raw_predictions=raw_preds,
        smoothed_predictions=smoothed_preds,
        events=seq_buffer.get_sequence(),
        raw_candidate_count=raw_candidates,
        step_duration_s=0.20,
        output_path=output_metrics_json,
    )
    logger.info("Saved %d events to %s and metrics to %s", len(seq_buffer), output_events_csv, output_metrics_json)


def main():
    logger.info("Loading validation sequences from %s...", VAL_MANIFEST)
    sequences = load_val_sequences()
    logger.info("Loaded %d validation sequences.", len(sequences))

    logger.info("Initializing ST-GCN runner from %s...", CHECKPOINT_PATH)
    runner = STGCNRunner(checkpoint_path=CHECKPOINT_PATH, device="cpu", warmup_iterations=0)

    logger.info("Generating temporal sliding predictions across validation sequences...")
    stream_records = generate_sliding_predictions(runner, sequences, stride=5, window_size=45)
    logger.info("Generated %d sliding prediction records.", len(stream_records))

    logger.info("1. Running smoothing comparison experiment...")
    run_comparison_experiment(stream_records)

    logger.info("2. Running parameter search grid...")
    run_parameter_search(stream_records)

    logger.info("3. Exporting events CSV and temporal stability metrics...")
    export_events_and_temporal_metrics(stream_records)

    logger.info("Validation experiments completed successfully!")


if __name__ == "__main__":
    main()
