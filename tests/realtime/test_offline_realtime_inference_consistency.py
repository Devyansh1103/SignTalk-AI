"""
Offline vs. Real-Time Model Inference Consistency Test (Phase 4 Part 2).
Validates that running an identical sequence through:
  1. Offline dataset loading + direct ST-GCN evaluation
  2. Real-time LandmarkFrame sequence + STGCNRunner.predict_window()
produces strictly identical logits, softmax probabilities, and predicted classes.
"""

import pytest
import os
import numpy as np
import torch

from src.realtime.model_runner import STGCNRunner
from src.realtime.types import LandmarkFrame, ModalityDetection, QualityReport
from src.data.node_schema import TOTAL_NODES, CHANNELS, TARGET_SEQUENCE_LENGTH
from src.data.label_map import get_label


CHECKPOINT_PATH = "experiments/stgcn/checkpoints/best_checkpoint.pt"
TEST_SEQ_PATH = "data/processed/sequences/test/seq_0049.npz"


class TestOfflineRealtimeInferenceConsistency:
    @pytest.mark.skipif(not os.path.exists(CHECKPOINT_PATH), reason="Checkpoint missing")
    @pytest.mark.skipif(not os.path.exists(TEST_SEQ_PATH), reason="Test sequence missing")
    def test_inference_numerical_and_class_consistency(self):
        # 1. Load ground truth offline test sample
        seq_data = np.load(TEST_SEQ_PATH)
        offline_data = seq_data["data"]      # Shape: (3, 45, 93)
        offline_mask = seq_data["mask"]      # Shape: (1, 45, 93)
        ground_truth_label = int(seq_data["label"])

        # 2. Run Offline Inference directly through model
        runner = STGCNRunner(checkpoint_path=CHECKPOINT_PATH, device="cpu", warmup_iterations=1)

        x_offline = torch.from_numpy(offline_data).unsqueeze(0).float()
        mask_offline = torch.from_numpy(offline_mask).unsqueeze(0).float()

        with torch.no_grad():
            logits_offline, probs_offline = runner.predict(x_offline, mask_offline)

        pred_class_offline = int(torch.argmax(probs_offline, dim=-1).item())
        probs_offline_np = probs_offline.cpu().numpy()[0]
        logits_offline_np = logits_offline.cpu().numpy()[0]

        # 3. Simulate Real-Time Ingestion: unpack into 45 LandmarkFrame instances
        # offline_data has shape (3, 45, 93) -> transposed to (45, 93, 3) for each frame
        data_t_v_c = np.transpose(offline_data, (1, 2, 0))  # (45, 93, 3)
        mask_t_v = offline_mask[0]                         # (45, 93)

        realtime_window = []
        for t in range(TARGET_SEQUENCE_LENGTH):
            frame_norm = data_t_v_c[t].astype(np.float32)  # (93, 3)
            frame_mask = mask_t_v[t].astype(bool)         # (93,)

            lf = LandmarkFrame(
                frame_id=t,
                timestamp=t * (1.0 / 25.0),
                raw_coords=frame_norm.copy(),
                normalized_coords=frame_norm,
                mask=frame_mask,
                visibility=np.ones((TOTAL_NODES,), dtype=np.float32),
                modality_stats=ModalityDetection(True, True, True, False),
                quality=QualityReport(True, 0.95, "GOOD", 0.0, False, True, False)
            )
            realtime_window.append(lf)

        # 4. Run Real-Time Predict Window
        realtime_pred = runner.predict_window(realtime_window)

        # 5. Verify Invariants
        # A. Class identity matches
        assert realtime_pred.class_id == pred_class_offline, (
            f"Class mismatch: Real-time predicted {realtime_pred.class_id} ({realtime_pred.label}), "
            f"Offline predicted {pred_class_offline} ({get_label(pred_class_offline)})"
        )

        # B. Prediction matches ground truth on seq_0049 ('hello' -> class 0)
        assert realtime_pred.class_id == ground_truth_label

        # C. Logits match down to numerical precision
        max_logit_diff = np.max(np.abs(realtime_pred.logits - logits_offline_np))
        assert max_logit_diff < 1e-5, f"Logits differ by {max_logit_diff}"

        # D. Softmax probabilities match down to numerical precision
        max_prob_diff = np.max(np.abs(realtime_pred.probabilities - probs_offline_np))
        assert max_prob_diff < 1e-5, f"Probabilities differ by {max_prob_diff}"

        # E. Top-1 confidence matches
        assert np.isclose(realtime_pred.confidence, float(probs_offline_np[pred_class_offline]), atol=1e-5)
