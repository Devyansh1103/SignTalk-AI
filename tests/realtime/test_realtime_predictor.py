"""
Unit & Integration tests for RealTimeSignPredictor pipeline (Phase 4 Part 2).
Validates end-to-end flow from FramePacket ingestion to sliding temporal predictions,
custom sink publication, HUD overlay rendering, and metrics aggregation.
"""

import pytest
import os
import numpy as np

from src.realtime.types import FramePacket, PredictionResult
from src.realtime.realtime_config import RealTimeConfig
from src.realtime.realtime_predictor import RealTimeSignPredictor
from src.realtime.model_runner import STGCNRunner
from src.realtime.prediction_sink import BasePredictionSink
from src.data.node_schema import TARGET_SEQUENCE_LENGTH


class MockCollectorSink(BasePredictionSink):
    """Sink that collects all emitted predictions for assertion."""
    def __init__(self):
        self.predictions = []

    def publish(self, prediction: PredictionResult) -> None:
        self.predictions.append(prediction)


CHECKPOINT_PATH = "experiments/stgcn/checkpoints/best_checkpoint.pt"


class TestRealTimePredictor:
    @pytest.mark.skipif(not os.path.exists(CHECKPOINT_PATH), reason="Checkpoint missing")
    def test_end_to_end_packet_ingestion(self):
        cfg = RealTimeConfig()
        cfg.buffer.max_length = TARGET_SEQUENCE_LENGTH  # 45
        cfg.scheduler.stride = 5
        cfg.inference.device = "cpu"
        cfg.inference.checkpoint_path = CHECKPOINT_PATH
        cfg.inference.warmup_iterations = 0

        collector = MockCollectorSink()
        predictor = RealTimeSignPredictor(config=cfg, prediction_sinks=[collector])

        # Generate 55 synthetic frame packets (black BGR frames)
        h, w = 480, 640
        bgr = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(55):
            packet = FramePacket(
                frame_id=i,
                timestamp=i * 0.04,
                capture_time=i * 0.04,
                image=bgr,
                width=w,
                height=h
            )
            lf, pred = predictor.process_frame_packet(packet)
            assert lf.frame_id == i

            # Before frame 44 (45 frames accumulated), no prediction
            if i < 44:
                assert pred is None
            elif i == 44:
                # 45th frame triggers inference!
                assert pred is not None
                assert isinstance(pred, PredictionResult)
            elif i in (49, 54):
                # Stride of 5 triggers at 49 and 54
                assert pred is not None

        # Verify collector collected predictions
        assert len(collector.predictions) == 3

        # Test overlay rendering
        dummy_packet = FramePacket(
            frame_id=100,
            timestamp=4.0,
            capture_time=4.0,
            image=bgr,
            width=w,
            height=h
        )
        lf, _ = predictor.process_frame_packet(dummy_packet)
        overlay = predictor.render_prediction_overlay(bgr, lf, fps_display=25.0)
        assert overlay.shape == bgr.shape
        assert overlay.dtype == np.uint8

        # Test metrics
        metrics = predictor.get_inference_metrics()
        assert metrics["total_predictions"] == 3
        assert metrics["mean_inference_latency_ms"] >= 0.0

        predictor.close()
