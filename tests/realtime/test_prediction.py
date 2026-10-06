"""
Unit tests for structured prediction logic and top-k output (Phase 4 Part 2).
Validates argmax, confidence assignment, top-k ordering, and serializability.
"""

import pytest
import numpy as np

from src.realtime.types import PredictionResult
from src.data.label_map import get_label, get_gloss, get_translation, NUM_CLASSES


class TestPrediction:
    def test_prediction_result_fields_and_top_k(self):
        # Create synthetic probability distribution where class 3 is top
        probs = np.array([0.05, 0.02, 0.03, 0.70, 0.05, 0.05, 0.02, 0.03, 0.03, 0.02], dtype=np.float32)
        logits = np.log(probs + 1e-7)

        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        assert pred_id == 3
        assert np.isclose(confidence, 0.70)

        top_indices = np.argsort(probs)[::-1][:3]
        top_k = [(int(idx), get_label(int(idx)), float(probs[idx])) for idx in top_indices]

        res = PredictionResult(
            class_id=pred_id,
            label=get_label(pred_id),
            gloss=get_gloss(pred_id),
            translation=get_translation(pred_id),
            confidence=confidence,
            probabilities=probs,
            logits=logits,
            top_k=top_k,
            timestamp=100.0,
            window_start_time=98.2,
            window_end_time=100.0,
            window_frame_count=45,
            inference_start_time=100.0,
            inference_end_time=100.02,
            inference_latency_ms=20.0,
            input_quality=0.88,
            is_valid_quality=True,
            buffer_length=45,
            device="cpu"
        )

        assert res.class_id == 3
        assert res.label == "happy"
        assert np.isclose(res.confidence, 0.70, atol=1e-4)
        assert len(res.top_k) == 3
        assert res.top_k[0][0] == 3
        assert res.top_k[0][1] == "happy"

        # Check serialization
        d = res.to_dict()
        assert d["class_id"] == 3
        assert d["label"] == "happy"
        assert np.isclose(d["confidence"], 0.70, atol=1e-4)
        assert len(d["top_k"]) == 3

        console_str = res.format_console()
        assert "HAPPY" in console_str
        assert "70.0%" in console_str
