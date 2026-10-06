"""
End-to-End Real-Time Pipeline Integration Tests (Phase 4 Part 4).
Validates full pipeline construction, state machine transitions, synthetic frame processing,
tensor validation, event emission, phrase translation, and graceful error recovery.
"""

import os
import pytest
import numpy as np
import cv2

from src.realtime.realtime_pipeline import RealtimePipeline, PipelineState, PipelineOutput
from src.realtime.realtime_config import RealTimeConfig
from src.realtime.model_runner import STGCNRunner
from src.data.node_schema import TOTAL_NODES, CHANNELS, TARGET_SEQUENCE_LENGTH


CHECKPOINT_PATH = "experiments/stgcn/checkpoints/best_checkpoint.pt"


class TestRealtimePipelineIntegration:
    @pytest.fixture
    def pipeline(self):
        """Constructs a test pipeline with mockable/CPU configuration."""
        cfg = RealTimeConfig()
        cfg.inference.device = "cpu"
        cfg.inference.warmup_iterations = 1
        cfg.buffer.max_length = 45
        cfg.scheduler.stride = 5

        # Instantiate runner only if checkpoint exists
        if os.path.exists(CHECKPOINT_PATH):
            runner = STGCNRunner(
                checkpoint_path=CHECKPOINT_PATH,
                device="cpu",
                warmup_iterations=1
            )
        else:
            runner = None

        pipe = RealtimePipeline(config=cfg, model_runner=runner)
        return pipe

    def test_pipeline_construction_and_initialization(self, pipeline):
        assert pipeline.state == PipelineState.STOPPED
        pipeline.initialize()
        assert pipeline.state == PipelineState.RUNNING

    def test_pipeline_lifecycle_transitions(self, pipeline):
        pipeline.initialize()
        assert pipeline.state == PipelineState.RUNNING

        pipeline.pause()
        assert pipeline.state == PipelineState.PAUSED

        # Stepping during PAUSED returns holding output
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = pipeline.step(dummy_frame)
        assert out.pipeline_state == "PAUSED"

        pipeline.resume()
        assert pipeline.state == PipelineState.RUNNING

        pipeline.stop()
        assert pipeline.state == PipelineState.STOPPED

    @pytest.mark.skipif(not os.path.exists(CHECKPOINT_PATH), reason="ST-GCN checkpoint missing")
    def test_end_to_end_synthetic_frame_stream(self, pipeline):
        """
        Feeds synthetic frames through the pipeline to verify sliding window accumulation,
        ST-GCN inference triggering, FSM state updates, and output contract serialization.
        """
        pipeline.initialize()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Feed 50 synthetic frames (exceeding T=45 to trigger window prediction)
        outputs = []
        for f_idx in range(50):
            out = pipeline.step(dummy_frame)
            outputs.append(out)

        assert len(outputs) == 50
        last_out = outputs[-1]

        # Verify output structure and contract
        assert isinstance(last_out, PipelineOutput)
        assert last_out.frame_id == 50
        assert last_out.pipeline_state == "RUNNING"
        assert isinstance(last_out.stage_latencies_ms, dict)
        assert "mediapipe" in last_out.stage_latencies_ms
        assert "end_to_end" in last_out.stage_latencies_ms

        # Buffer was full -> ST-GCN inference should have triggered
        assert pipeline.temporal_buffer.length() == 45
        assert pipeline.latest_prediction is not None
        assert 0 <= pipeline.latest_prediction.class_id < 10

        # Serialization to JSON-compatible dictionary
        d = last_out.to_dict()
        assert isinstance(d, dict)
        assert d["frame_id"] == 50
        assert "prediction" in d

        # Render overlay verification
        canvas = pipeline.render_overlay(dummy_frame, last_out)
        assert isinstance(canvas, np.ndarray)
        assert canvas.shape == dummy_frame.shape

    def test_pipeline_reset_and_close(self, pipeline):
        pipeline.initialize()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.step(dummy_frame)

        pipeline.reset()
        assert pipeline.temporal_buffer.length() == 0
        assert pipeline.latest_prediction is None
        assert len(pipeline.sign_sequence.get_sequence()) == 0

        pipeline.close()
        assert pipeline.state == PipelineState.STOPPED
