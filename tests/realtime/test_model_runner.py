"""
Unit tests for STGCNRunner (Phase 4 Part 2).
Validates model initialization, checkpoint loading, device management,
warm-up behavior, and inference execution mode.
"""

import pytest
import os
import torch
import numpy as np

from src.realtime.model_runner import STGCNRunner, ModelRunnerError, ShapeValidationError
from src.realtime.types import LandmarkFrame, ModalityDetection, QualityReport, PredictionResult
from src.data.node_schema import TOTAL_NODES, CHANNELS, TARGET_SEQUENCE_LENGTH
from src.data.label_map import NUM_CLASSES


CHECKPOINT_PATH = "experiments/stgcn/checkpoints/best_checkpoint.pt"


def make_valid_frame(frame_id: int) -> LandmarkFrame:
    return LandmarkFrame(
        frame_id=frame_id,
        timestamp=frame_id * 0.04,
        raw_coords=np.zeros((TOTAL_NODES, CHANNELS), dtype=np.float32),
        normalized_coords=np.zeros((TOTAL_NODES, CHANNELS), dtype=np.float32),
        mask=np.ones((TOTAL_NODES,), dtype=bool),
        visibility=np.ones((TOTAL_NODES,), dtype=np.float32),
        modality_stats=ModalityDetection(True, True, True, False),
        quality=QualityReport(True, 0.95, "GOOD", 0.0, False, True, False)
    )


class TestSTGCNRunner:
    @pytest.mark.skipif(not os.path.exists(CHECKPOINT_PATH), reason="Checkpoint missing")
    def test_checkpoint_loading_and_eval_mode(self):
        runner = STGCNRunner(checkpoint_path=CHECKPOINT_PATH, device="cpu", warmup_iterations=1)
        assert runner.model is not None
        assert not runner.model.training  # Must be in eval() mode

        # Gradients must be disabled
        for param in runner.model.parameters():
            assert not param.requires_grad

    def test_missing_checkpoint_raises_filenotfound(self):
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            STGCNRunner(checkpoint_path="non_existent/path/model.pt", device="cpu", warmup_iterations=0)

    def test_device_resolution(self):
        runner_cpu = STGCNRunner(checkpoint_path=CHECKPOINT_PATH, device="cpu", warmup_iterations=0)
        assert runner_cpu.device == torch.device("cpu")

        runner_auto = STGCNRunner(checkpoint_path=CHECKPOINT_PATH, device="auto", warmup_iterations=0)
        assert runner_auto.device in [torch.device("cpu"), torch.device("cuda")]

    def test_cuda_unavailable_raises_clear_error(self, monkeypatch):
        # Force cuda.is_available to return False
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ModelRunnerError, match="CUDA device requested"):
            STGCNRunner(checkpoint_path=CHECKPOINT_PATH, device="cuda", warmup_iterations=0)

    @pytest.mark.skipif(not os.path.exists(CHECKPOINT_PATH), reason="Checkpoint missing")
    def test_raw_prediction_shape_and_probability(self):
        runner = STGCNRunner(checkpoint_path=CHECKPOINT_PATH, device="cpu", warmup_iterations=0)
        dummy_x = torch.zeros((1, CHANNELS, TARGET_SEQUENCE_LENGTH, TOTAL_NODES), dtype=torch.float32)
        dummy_mask = torch.ones((1, 1, TARGET_SEQUENCE_LENGTH, TOTAL_NODES), dtype=torch.float32)

        logits, probs = runner.predict(dummy_x, dummy_mask)
        assert logits.shape == (1, NUM_CLASSES)
        assert probs.shape == (1, NUM_CLASSES)

        # Probabilities must sum to ~1.0
        prob_sum = float(torch.sum(probs).item())
        assert np.isclose(prob_sum, 1.0, atol=1e-4)

    @pytest.mark.skipif(not os.path.exists(CHECKPOINT_PATH), reason="Checkpoint missing")
    def test_predict_window_returns_prediction_result(self):
        runner = STGCNRunner(checkpoint_path=CHECKPOINT_PATH, device="cpu", top_k=3, warmup_iterations=0)
        window = [make_valid_frame(i) for i in range(TARGET_SEQUENCE_LENGTH)]

        pred = runner.predict_window(window)
        assert isinstance(pred, PredictionResult)
        assert 0 <= pred.class_id < NUM_CLASSES
        assert len(pred.label) > 0
        assert 0.0 <= pred.confidence <= 1.0
        assert len(pred.top_k) == 3
        assert pred.inference_latency_ms >= 0.0
        assert pred.window_frame_count == TARGET_SEQUENCE_LENGTH
