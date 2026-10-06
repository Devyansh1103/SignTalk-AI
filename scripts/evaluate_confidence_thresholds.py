"""
SignTalk AI - Validation-Set Confidence Threshold Optimization
Phase 4 Part 3: Confidence, Smoothing & Continuous Sign Detection.

Evaluates ST-GCN confidence calibration across candidate rejection thresholds
tau in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
strictly on the validation partition (data/manifests/val.csv).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.realtime.model_runner import STGCNRunner
from src.data.label_map import get_label, NUM_CLASSES


def evaluate_validation_thresholds(
    val_manifest_path: str = "data/manifests/val.csv",
    checkpoint_path: str = "experiments/stgcn/checkpoints/best_checkpoint.pt",
    output_csv: str = "results/confidence_thresholds.csv"
):
    print("=" * 65)
    print("  SignTalk AI — Validation Confidence Threshold Search")
    print("=" * 65)

    runner = STGCNRunner(checkpoint_path=checkpoint_path, device="cpu", warmup_iterations=2)
    df = pd.read_csv(val_manifest_path)

    records = []
    print(f"\nProcessing {len(df)} validation sequences...")

    for _, row in df.iterrows():
        seq_id = row["sequence_id"]
        true_cid = int(row["class_id"])
        npz_path = row["file_path"]

        data = np.load(npz_path)
        x = torch.from_numpy(data["data"]).unsqueeze(0).float()
        mask = torch.from_numpy(data["mask"]).unsqueeze(0).float()

        with torch.no_grad():
            logits, probs = runner.predict(x, mask)

        probs_np = probs.cpu().numpy()[0]
        pred_cid = int(np.argmax(probs_np))
        conf = float(probs_np[pred_cid])
        correct = (pred_cid == true_cid)

        records.append({
            "sequence_id": seq_id,
            "true_class_id": true_cid,
            "true_label": row["label"],
            "pred_class_id": pred_cid,
            "pred_label": get_label(pred_cid),
            "confidence": conf,
            "is_correct": correct,
            "quality_score": float(row["quality_score"]),
            "valid_frames": int(row["valid_frames"])
        })
        print(f"  [{seq_id}] true={row['label']:<10} pred={get_label(pred_cid):<10} conf={conf:.4f} match={correct}")

    pred_df = pd.DataFrame(records)
    thresholds = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    results = []

    print("\n" + "=" * 78)
    print(f"{'Threshold (tau)':<15} | {'Accepted':<9} | {'Rejected':<9} | {'Rej Rate':<9} | {'Acc Prec':<9} | {'False Acc':<9} | {'F1':<6}")
    print("-" * 78)

    for tau in thresholds:
        accepted = pred_df[pred_df["confidence"] >= tau]
        rejected = pred_df[pred_df["confidence"] < tau]

        num_accepted = len(accepted)
        num_rejected = len(rejected)
        rejection_rate = num_rejected / len(pred_df)

        if num_accepted > 0:
            true_positives = int(accepted["is_correct"].sum())
            false_positives = num_accepted - true_positives
            accepted_accuracy = true_positives / num_accepted  # Precision among accepted
        else:
            true_positives = 0
            false_positives = 0
            accepted_accuracy = 0.0

        # False negatives are correct predictions that were rejected
        false_negatives = int(rejected["is_correct"].sum())

        precision = (true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0.0
        recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results.append({
            "threshold": tau,
            "num_accepted": num_accepted,
            "num_rejected": num_rejected,
            "rejection_rate": round(rejection_rate, 4),
            "accepted_accuracy": round(accepted_accuracy, 4),
            "false_acceptances": false_positives,
            "false_rejections": false_negatives,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        })

        print(
            f"{tau:<15.2f} | {num_accepted:<9} | {num_rejected:<9} | {rejection_rate:>8.1%}  | {accepted_accuracy:>8.1%}  | {false_positives:<9} | {f1:.4f}"
        )

    print("=" * 78)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    print(f"\nSaved threshold analysis to: {output_csv}")
    return pred_df, res_df


if __name__ == "__main__":
    evaluate_validation_thresholds()
