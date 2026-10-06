"""
SignTalk AI: Complete End-to-End Pipeline Validation Script.

Executes and logs the full pipeline:
  Landmarks [C, T, V]
      ↓
  ST-GCN
      ↓
  Feature Projection
      ↓
  Transformer Decoder
      ↓
  Token Sequence
      ↓
  Confidence Thresholding
      ↓
  Natural Language Output
"""

import os
import sys
import argparse
import json
import time
from typing import Optional, Dict, Any, List

sys.path.insert(0, os.getcwd())

import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.sign_sequence_dataset import SignSequenceDataset
from src.data.collate import sign_sequence_collate_fn
from src.models.sign_translation_model import SignTranslationModel
from src.nlp.tokenizer import SignLanguageTokenizer
from src.inference.translation_pipeline import SignTranslationPipeline
from src.utils.device import resolve_device


def run_full_pipeline_validation(
    config_path: str = "configs/transformer.yaml",
    manifest_path: Optional[str] = None
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg.get("device", "auto"))
    paths = cfg["paths"]
    manifest = manifest_path or paths["test_manifest"]
    checkpoint_file = os.path.join(paths["checkpoint_dir"], "best_checkpoint.pt")
    metrics_dir = paths["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)

    print("=" * 80)
    print("SignTalk AI — Complete End-to-End Pipeline Validation")
    print(f"Manifest: {manifest}")
    print(f"Checkpoint: {checkpoint_file}")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Initialize Pipeline Components
    tokenizer = SignLanguageTokenizer(vocab_path=cfg["transformer"]["vocabulary_file"])
    model = SignTranslationModel(
        stgcn_config=cfg.get("stgcn", {}),
        transformer_config=cfg.get("transformer", {}),
        freeze_stgcn=True
    ).to(device)

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_file}")

    ckpt = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    conf_thresh = cfg.get("decoding", {}).get("confidence_threshold", 0.50)
    pipeline = SignTranslationPipeline(
        model=model,
        tokenizer=tokenizer,
        confidence_threshold=conf_thresh,
        device=device
    )

    # 2. Run over all samples
    dataset = SignSequenceDataset(manifest_path=manifest, augment=False)
    pipeline_records = []

    print(f"\nProcessing {len(dataset)} validation sequences...\n")
    print(f"{'Sample ID':<12} | {'Ground Truth':<12} | {'Pred Gloss':<14} | {'Translation':<14} | {'Conf':<6} | {'Status':<12} | {'Latency':<8}")
    print("-" * 88)

    for i in range(len(dataset)):
        sample = dataset[i]
        x = sample["x"]
        mask = sample["mask"]
        meta = sample["metadata"]

        res = pipeline.translate_sequence(x, mask=mask)

        gt_gloss = meta["gloss"]
        gt_trans = meta["translation"]
        pred_gloss = res["predicted_gloss"]
        pred_trans = res["predicted_translation"]
        conf = res["confidence"]
        status = res["status"]
        lat = res["latency"]["total_ms"]

        is_match = (pred_gloss == gt_gloss)

        print(
            f"{meta['sequence_id']:<12} | {gt_gloss:<12} | {pred_gloss:<14} | {pred_trans:<14} | {conf:<6.4f} | {status:<12} | {lat:<8.2f}ms "
            f"{'[OK]' if is_match else '[MISMATCH]'}"
        )

        pipeline_records.append({
            "sample_id": meta["sequence_id"],
            "signer_id": meta["signer_id"],
            "ground_truth_gloss": gt_gloss,
            "ground_truth_translation": gt_trans,
            "predicted_gloss": pred_gloss,
            "predicted_translation": pred_trans,
            "primary_token_id": res["primary_token_id"],
            "confidence": conf,
            "is_uncertain": res["is_uncertain"],
            "status": status,
            "is_correct": is_match,
            "latency_ms": lat
        })

    # Summary Statistics
    df = pd.DataFrame(pipeline_records)
    acc = float((df["predicted_gloss"] == df["ground_truth_gloss"]).mean())
    accepted_df = df[df["status"] == "ACCEPTED"]
    accepted_acc = float((accepted_df["predicted_gloss"] == accepted_df["ground_truth_gloss"]).mean()) if len(accepted_df) > 0 else 0.0

    print("-" * 88)
    print(f"Overall Accuracy:                  {acc*100:.2f}%")
    print(f"Confidence-Gated Accepted Count:   {len(accepted_df)} / {len(df)} ({len(accepted_df)/len(df)*100:.1f}%)")
    print(f"Precision on Accepted Signs:       {accepted_acc*100:.2f}%")
    print(f"Mean Pipeline Latency:             {df['latency_ms'].mean():.2f} ms")
    print("=" * 88)

    out_csv = os.path.join(metrics_dir, "pipeline_validation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved complete pipeline validation records to: {out_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Full Translation Pipeline")
    parser.add_argument("--config", type=str, default="configs/transformer.yaml")
    parser.add_argument("--manifest", type=str, default=None)
    args = parser.parse_args()
    run_full_pipeline_validation(args.config, args.manifest)
