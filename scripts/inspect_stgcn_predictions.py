#!/usr/bin/env python3
"""
SignTalk AI: ST-GCN Sample Prediction Inspection Script.

Analyzes individual sample predictions on the isolated test set:
  - Sequence ID, True Label, Predicted Label, Confidence Probability
  - Correct vs. Incorrect classification breakdown
  - Identifies High-Confidence Errors and Low-Confidence Predictions
  - Correlates performance with signer identity and sequence quality
Saves detailed inspection CSV to experiments/stgcn/metrics/sample_predictions.csv.
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.device import get_device
from src.models.stgcn import SignSTGCN
from src.data.dataloader import create_sequence_dataloader
from src.training.trainer import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description="Inspect ST-GCN sample-level predictions.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/stgcn/checkpoints/best_checkpoint.pt",
        help="Path to trained checkpoint."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/test.csv",
        help="Path to test manifest."
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    device, _ = get_device("auto")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    graph_cfg = config.get("graph", {})

    # Reconstruct Model
    model = SignSTGCN(
        in_channels=dataset_cfg.get("in_channels", 3),
        num_classes=dataset_cfg.get("num_classes", 10),
        num_nodes=dataset_cfg.get("num_nodes", 93),
        sequence_length=dataset_cfg.get("sequence_length", 45),
        graph_strategy=graph_cfg.get("strategy", "spatial"),
        block_channels=model_cfg.get("block_channels", [64, 64, 128, 128, 256, 256]),
        block_strides=model_cfg.get("block_strides", [1, 1, 2, 1, 2, 1]),
        temporal_kernel_size=model_cfg.get("temporal_kernel_size", 9),
        dropout=model_cfg.get("dropout", 0.3),
        residual=model_cfg.get("residual", True),
        use_learnable_edge_weights=model_cfg.get("use_learnable_edge_weights", True)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    loader = create_sequence_dataloader(
        manifest_path=args.manifest,
        split="test",
        batch_size=4,
        shuffle=False,
        filter_rejects=False,
        augment=False,
        include_velocity=dataset_cfg.get("include_velocity", False)
    )

    # Class names
    class_names = [f"Class_{i}" for i in range(dataset_cfg.get("num_classes", 10))]
    vocab_path = config.get("paths", {}).get("vocabulary_file", "assets/vocabularies/mvp_10.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as vf:
            vocab = json.load(vf)
            c2g = vocab.get("class_id_to_gloss", {})
            class_names = [c2g.get(str(i), f"Class_{i}") for i in range(dataset_cfg.get("num_classes", 10))]

    trainer = ModelTrainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        config=config,
        class_names=class_names
    )

    _, metrics, y_true, y_pred, y_probs, metadata = trainer.evaluate(loader)

    records = []
    for i in range(len(y_true)):
        true_c = int(y_true[i])
        pred_c = int(y_pred[i])
        probs = y_probs[i]
        conf = float(probs[pred_c])
        meta = metadata[i] if i < len(metadata) else {}

        true_label = class_names[true_c] if true_c < len(class_names) else str(true_c)
        pred_label = class_names[pred_c] if pred_c < len(class_names) else str(pred_c)
        is_correct = (true_c == pred_c)

        rec = {
            "sequence_id": meta.get("sequence_id", f"sample_{i}"),
            "source_recording_id": meta.get("source_recording_id", "N/A"),
            "signer_id": meta.get("signer_id", "unknown"),
            "quality_status": meta.get("quality_status", "UNKNOWN"),
            "quality_score": meta.get("quality_score", 0.0),
            "true_class_id": true_c,
            "true_label": true_label,
            "pred_class_id": pred_c,
            "pred_label": pred_label,
            "confidence": round(conf, 4),
            "correct": is_correct,
            "error_type": "None" if is_correct else ("High_Confidence_Error" if conf >= 0.50 else "Standard_Error")
        }
        records.append(rec)

    df_inspect = pd.DataFrame(records)

    print("\n============================================================")
    print("      ST-GCN SAMPLE PREDICTION INSPECTION (TEST SET)")
    print("============================================================")
    print(f"{'Seq ID':<10} {'Signer':<10} {'Quality':<12} {'True Label':<12} {'Pred Label':<12} {'Conf':<8} {'Status'}")
    print("-" * 75)

    for _, r in df_inspect.iterrows():
        status_str = "CORRECT" if r["correct"] else f"FAIL ({r['error_type']})"
        print(f"{r['sequence_id']:<10} {r['signer_id']:<10} {r['quality_status']:<12} {r['true_label']:<12} {r['pred_label']:<12} {r['confidence']:<8.4f} {status_str}")

    print("------------------------------------------------------------")
    correct_count = df_inspect["correct"].sum()
    total_count = len(df_inspect)
    print(f"Total Correct: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")

    hi_errs = df_inspect[df_inspect["error_type"] == "High_Confidence_Error"]
    print(f"High-Confidence Errors (Conf >= 0.50): {len(hi_errs)}")
    print("============================================================\n")

    metrics_dir = config.get("paths", {}).get("metrics_dir", "experiments/stgcn/metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    out_csv = os.path.join(metrics_dir, "sample_predictions.csv")
    df_inspect.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Saved sample prediction log to: {out_csv}")


if __name__ == "__main__":
    main()
