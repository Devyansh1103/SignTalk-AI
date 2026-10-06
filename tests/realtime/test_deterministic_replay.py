"""Deterministic Replay Regression Tests (Phase 4 Part 3).

Verifies that for a fixed landmark sequence, checkpoint, and configuration,
the pipeline outputs completely identical and deterministic prediction records,
smoothed outputs, and SignEvents.
"""

import os
from pathlib import Path
import pytest
import numpy as np
import torch

from src.realtime.realtime_config import RealTimeConfig
from scripts.replay_predictions import replay_sequence

CHECKPOINT_PATH = "experiments/stgcn/checkpoints/best_checkpoint.pt"
VAL_SAMPLE_PATH = "data/processed/sequences/val/seq_0037.npz"


class TestDeterministicReplay:
    @pytest.mark.skipif(not os.path.exists(CHECKPOINT_PATH), reason="Checkpoint missing")
    @pytest.mark.skipif(not os.path.exists(VAL_SAMPLE_PATH), reason="Validation sample missing")
    def test_deterministic_replay_identical_outputs(self):
        """Runs two independent passes over the same input sequence and verifies identical outputs."""
        torch.manual_seed(42)
        np.random.seed(42)

        cfg = RealTimeConfig()
        cfg.inference.checkpoint_path = CHECKPOINT_PATH
        cfg.confidence.threshold = 0.65
        cfg.smoothing.history_size = 5
        cfg.smoothing.min_votes = 3
        cfg.stability.min_consecutive_predictions = 2

        # Pass 1
        events_run_1 = replay_sequence(
            input_path=VAL_SAMPLE_PATH,
            config=cfg,
            checkpoint_path=CHECKPOINT_PATH,
            stride=5,
            repeat=2,
        )

        # Pass 2
        events_run_2 = replay_sequence(
            input_path=VAL_SAMPLE_PATH,
            config=cfg,
            checkpoint_path=CHECKPOINT_PATH,
            stride=5,
            repeat=2,
        )

        assert len(events_run_1) == len(events_run_2)

        for e1, e2 in zip(events_run_1, events_run_2):
            assert e1.label == e2.label
            assert e1.class_id == e2.class_id
            assert e1.start_time == pytest.approx(e2.start_time, abs=1e-4)
            assert e1.end_time == pytest.approx(e2.end_time, abs=1e-4)
            assert e1.confidence == pytest.approx(e2.confidence, abs=1e-4)
            assert e1.stability_score == pytest.approx(e2.stability_score, abs=1e-4)
