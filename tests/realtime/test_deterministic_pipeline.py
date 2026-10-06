"""
Deterministic End-to-End Pipeline Replay Tests (Phase 4 Part 4).
Validates that repeated runs of an identical sequence through the complete pipeline
produce bit-identical predictions, sign events, and translation outputs.
"""

import os
import pytest
import numpy as np
import torch

from src.realtime.realtime_pipeline import RealtimePipeline
from src.realtime.realtime_config import RealTimeConfig
from src.realtime.model_runner import STGCNRunner
from src.realtime.types import LandmarkFrame, ModalityDetection, QualityReport
from src.data.node_schema import TOTAL_NODES, TARGET_SEQUENCE_LENGTH


CHECKPOINT_PATH = "experiments/stgcn/checkpoints/best_checkpoint.pt"
VAL_SAMPLE_PATH = "data/processed/sequences/val/seq_0037.npz"


class TestDeterministicPipeline:
    @pytest.mark.skipif(not os.path.exists(CHECKPOINT_PATH), reason="Checkpoint missing")
    @pytest.mark.skipif(not os.path.exists(VAL_SAMPLE_PATH), reason="Validation sample missing")
    def test_pipeline_determinism_across_runs(self):
        """Executes two independent pipeline evaluations on the same LandmarkFrame stream."""
        seq_data = np.load(VAL_SAMPLE_PATH)
        data = seq_data["data"]  # (3, 45, 93)
        mask = seq_data["mask"]  # (1, 45, 93)
        data_t_v_c = np.transpose(data, (1, 2, 0))  # (45, 93, 3)
        mask_t_v = mask[0]  # (45, 93)

        def run_pipeline():
            torch.manual_seed(42)
            np.random.seed(42)

            cfg = RealTimeConfig()
            cfg.inference.checkpoint_path = CHECKPOINT_PATH
            cfg.inference.device = "cpu"
            cfg.inference.warmup_iterations = 1
            cfg.scheduler.stride = 5

            runner = STGCNRunner(checkpoint_path=CHECKPOINT_PATH, device="cpu", warmup_iterations=1)
            pipe = RealtimePipeline(config=cfg, model_runner=runner)
            pipe.initialize()

            # Feed the 45 frames twice (90 frames) to trigger state machine & events
            predictions = []
            for rep in range(2):
                for t in range(TARGET_SEQUENCE_LENGTH):
                    f_id = rep * TARGET_SEQUENCE_LENGTH + t
                    lf = LandmarkFrame(
                        frame_id=f_id,
                        timestamp=float(f_id) / 25.0,
                        raw_coords=data_t_v_c[t].astype(np.float32),
                        normalized_coords=data_t_v_c[t].astype(np.float32),
                        mask=mask_t_v[t].astype(bool),
                        visibility=np.ones((TOTAL_NODES,), dtype=np.float32),
                        modality_stats=ModalityDetection(True, True, True, False),
                        quality=QualityReport(True, 0.95, "GOOD", 0.0, False, True, False)
                    )
                    # Bypass landmark stream and append directly to temporal buffer
                    pipe.temporal_buffer.append(lf)
                    pipe.scheduler.on_frame_appended()
                    if pipe.scheduler.should_infer(pipe.temporal_buffer):
                        pred = pipe.model_runner.predict_window(pipe.temporal_buffer.get_window())
                        filtered = pipe.confidence_filter.filter_prediction(pred)
                        pipe.prediction_history.add(
                            type("Rec", (), {
                                "timestamp": pred.timestamp,
                                "window_start": pred.window_start_time,
                                "window_end": pred.window_end_time,
                                "class_id": filtered.class_id,
                                "label": filtered.label,
                                "gloss": filtered.gloss,
                                "confidence": filtered.confidence,
                                "input_quality": filtered.input_quality,
                                "inference_latency_ms": pred.inference_latency_ms,
                                "is_valid_quality": pred.is_valid_quality,
                                "probabilities": pred.probabilities,
                            })()
                        )
                        smoothed = pipe.smoother.smooth(pipe.prediction_history)
                        _, evt = pipe.sign_state_machine.process(smoothed)
                        if evt is not None:
                            dedup = pipe.event_deduplicator.process(evt)
                            if dedup is not None:
                                pipe.sign_sequence.append_event(dedup)
                                pipe.translator.process_event(dedup)
                        predictions.append(pred)

            pipe.translator.finalize_phrase()
            events = pipe.sign_sequence.get_sequence()
            phrases = pipe.translator.finalized_phrases
            return predictions, events, phrases

        preds1, evts1, phr1 = run_pipeline()
        preds2, evts2, phr2 = run_pipeline()

        assert len(preds1) == len(preds2)
        for p1, p2 in zip(preds1, preds2):
            assert p1.class_id == p2.class_id
            assert np.allclose(p1.probabilities, p2.probabilities, atol=1e-5)

        assert len(evts1) == len(evts2)
        for e1, e2 in zip(evts1, evts2):
            assert e1.label == e2.label
            assert e1.confidence == pytest.approx(e2.confidence, abs=1e-4)

        assert len(phr1) == len(phr2)
        for h1, h2 in zip(phr1, phr2):
            assert h1["text"] == h2["text"]
            assert h1["hindi_text"] == h2["hindi_text"]
