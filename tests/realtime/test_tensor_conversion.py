"""
Unit tests for ST-GCN input tensor conversion (Phase 4 Part 2).
Validates conversion from rolling window of LandmarkFrames to ST-GCN input tensors
[1, 3, 45, 93], dimension reordering (T, V, C) -> (1, C, T, V), and strict shape validation.
"""

import pytest
import numpy as np
import torch

from src.realtime.model_runner import STGCNRunner, ShapeValidationError, ModelRunnerError
from src.realtime.types import LandmarkFrame, ModalityDetection, QualityReport
from src.data.node_schema import TOTAL_NODES, CHANNELS, TARGET_SEQUENCE_LENGTH


def make_valid_frame(frame_id: int) -> LandmarkFrame:
    coords = np.zeros((TOTAL_NODES, CHANNELS), dtype=np.float32)
    norm = np.zeros((TOTAL_NODES, CHANNELS), dtype=np.float32)
    # Fill in distinguishable values to verify dimension reordering
    norm[:, 0] = 1.0  # X channel
    norm[:, 1] = 2.0  # Y channel
    norm[:, 2] = 3.0  # Z channel

    mask = np.ones((TOTAL_NODES,), dtype=bool)
    vis = np.ones((TOTAL_NODES,), dtype=np.float32)

    return LandmarkFrame(
        frame_id=frame_id,
        timestamp=frame_id * 0.04,
        raw_coords=coords,
        normalized_coords=norm,
        mask=mask,
        visibility=vis,
        modality_stats=ModalityDetection(True, True, True, False),
        quality=QualityReport(True, 0.95, "GOOD", 0.0, False, True, False)
    )


class TestTensorConversion:
    @pytest.fixture
    def runner(self):
        # Instantiate STGCNRunner on CPU without loading heavy weights if possible, or using best_checkpoint
        return STGCNRunner(checkpoint_path="experiments/stgcn/checkpoints/best_checkpoint.pt", device="cpu", warmup_iterations=0)

    def test_correct_dimensions_and_reordering(self, runner):
        window = [make_valid_frame(i) for i in range(TARGET_SEQUENCE_LENGTH)]
        x, mask = runner.window_to_tensor(window)

        # Expected shape: [1, 3, 45, 93]
        assert x.shape == (1, 3, TARGET_SEQUENCE_LENGTH, TOTAL_NODES)
        assert mask.shape == (1, 1, TARGET_SEQUENCE_LENGTH, TOTAL_NODES)
        assert x.dtype == torch.float32
        assert mask.dtype == torch.float32

        # Verify channel mapping:
        # C=0 was set to 1.0, C=1 to 2.0, C=2 to 3.0
        assert torch.all(x[0, 0, :, :] == 1.0)
        assert torch.all(x[0, 1, :, :] == 2.0)
        assert torch.all(x[0, 2, :, :] == 3.0)

    def test_incorrect_sequence_length_raises(self, runner):
        # 30 frames instead of 45
        short_window = [make_valid_frame(i) for i in range(30)]
        with pytest.raises(ShapeValidationError, match="Window length mismatch"):
            runner.window_to_tensor(short_window)

    def test_incorrect_node_count_raises(self, runner):
        # Frame with wrong node count (e.g. 50 instead of 93)
        window = [make_valid_frame(i) for i in range(TARGET_SEQUENCE_LENGTH)]
        bad_frame = make_valid_frame(10)
        bad_frame.normalized_coords = np.zeros((50, 3), dtype=np.float32)
        window[10] = bad_frame

        with pytest.raises(ShapeValidationError, match="invalid normalized_coords shape"):
            runner.window_to_tensor(window)

    def test_invalid_tensor_shape_in_predict_raises(self, runner):
        # Invalid sequence length in tensor
        bad_x = torch.zeros((1, 3, 20, 93), dtype=torch.float32)
        with pytest.raises(ShapeValidationError, match="Invalid input tensor shape"):
            runner.predict(bad_x)

    def test_nan_values_raise_error(self, runner):
        window = [make_valid_frame(i) for i in range(TARGET_SEQUENCE_LENGTH)]
        # Inject NaN into frame
        bad_coords = np.zeros((TOTAL_NODES, CHANNELS), dtype=np.float32)
        bad_coords[0, 0] = np.nan
        bad_frame = LandmarkFrame(
            frame_id=5,
            timestamp=0.2,
            raw_coords=bad_coords,
            normalized_coords=bad_coords,
            mask=np.ones((TOTAL_NODES,), dtype=bool),
            visibility=np.ones((TOTAL_NODES,), dtype=np.float32),
            modality_stats=ModalityDetection(True, True, True, False),
            quality=QualityReport(False, 0.0, "REJECT", 1.0, True, False, False)
        )
        window[5] = bad_frame

        with pytest.raises(ModelRunnerError, match="NaN or Infinite"):
            runner.window_to_tensor(window)


def make_raw_frame(frame_id: int, hand_visible: bool = True, aspect_y_scale: float = 1.0) -> LandmarkFrame:
    raw = np.zeros((TOTAL_NODES, CHANNELS), dtype=np.float32)
    raw[47] = [0.6, 0.5, 0.0]   # left shoulder
    raw[48] = [0.4, 0.5, 0.0]   # right shoulder -> center (0.5, 0.5), width 0.2
    raw[0:21] = [0.5 + 0.001 * frame_id, 0.3, 0.0]  # left hand drifting right
    mask = np.zeros((TOTAL_NODES,), dtype=bool)
    mask[47:49] = True
    mask[0:21] = hand_visible
    return LandmarkFrame(
        frame_id=frame_id,
        timestamp=frame_id * 0.04,
        raw_coords=raw,
        normalized_coords=np.zeros((TOTAL_NODES, CHANNELS), dtype=np.float32),
        mask=mask,
        visibility=np.ones((TOTAL_NODES,), dtype=np.float32),
        modality_stats=ModalityDetection(hand_visible, False, True, False),
        quality=QualityReport(True, 0.95, "GOOD", 0.0, False, True, False),
        aspect_y_scale=aspect_y_scale,
    )


class TestOfflinePostprocess:
    @pytest.fixture
    def runner(self):
        return STGCNRunner(checkpoint_path="experiments/stgcn/checkpoints/best_checkpoint.pt", device="cpu", warmup_iterations=0)

    def test_matches_training_normalization(self, runner):
        window = [make_raw_frame(i) for i in range(TARGET_SEQUENCE_LENGTH)]
        x, mask = runner.window_to_tensor(window, offline_postprocess=True)
        # Hand x at frame 0: (0.5 - 0.5) / 0.2 = 0; y: (0.3 - 0.5) / 0.2 = -1
        assert np.isclose(x[0, 0, 0, 0].item(), 0.0, atol=1e-5)
        assert np.isclose(x[0, 1, 0, 0].item(), -1.0, atol=1e-5)

    def test_bridges_short_hand_gaps(self, runner):
        window = [make_raw_frame(i, hand_visible=not (10 <= i < 14)) for i in range(TARGET_SEQUENCE_LENGTH)]
        x, mask = runner.window_to_tensor(window, offline_postprocess=True)
        assert mask[0, 0, 12, 0].item() == 1.0
        # Linearly interpolated x at frame 12 lies between frames 9 and 14
        assert np.isclose(x[0, 0, 12, 0].item(), (0.001 * 12) / 0.2, atol=1e-4)

    def test_aspect_correction_scales_vertical_offsets(self, runner):
        window = [make_raw_frame(i, aspect_y_scale=4.0 / 3.0) for i in range(TARGET_SEQUENCE_LENGTH)]
        x, _ = runner.window_to_tensor(window, offline_postprocess=True)
        # Both hand and shoulders scale by 4/3, so hand-to-shoulder y offset grows by 4/3
        assert np.isclose(x[0, 1, 0, 0].item(), -4.0 / 3.0, atol=1e-4)
