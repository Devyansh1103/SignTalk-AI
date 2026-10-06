#!/usr/bin/env python3
"""
SignTalk AI: Baseline Prediction CLI.

Ingests a single sequence file (.npz) or raw test sample and emits
predicted sign class, confidence probability, and top-3 candidates.
"""

import sys
import os
import argparse
import json
import numpy as np
import torch

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.device import get_device
from src.models.baseline import SignBaselineModel
from src.data.stgcn_tensor import landmarks_to_stgcn_tensor


def main():
    parser = argparse.ArgumentParser(description="Run baseline inference on a sequence file.")
    parser.add_argument("--sequence", type=str, required=True, help="Path to .npz sequence file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/baseline/checkpoints/best_checkpoint.pt",
        help="Path to trained checkpoint."
    )
    args = parser.parse_args()

    if not os.path.exists(args.sequence):
        print(f"[ERROR] Sequence file not found: {args.sequence}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    device, _ = get_device("auto")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    model_cfg = config["model"]

    # Reconstruct Model
    model = SignBaselineModel(
        in_channels=model_cfg.get("in_channels", 3),
        num_nodes=model_cfg.get("num_nodes", 93),
        sequence_length=model_cfg.get("sequence_length", 45),
        proj_dim=model_cfg.get("proj_dim", 128),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.3),
        bidirectional=model_cfg.get("bidirectional", True),
        num_classes=model_cfg.get("num_classes", 10),
        pooling_type=model_cfg.get("pooling_type", "mean_max")
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load sequence
    with np.load(args.sequence) as arc:
        data = arc["data"]
        mask = arc["mask"] if "mask" in arc else None
        true_label = arc["label"] if "label" in arc else None

    tensor_x, tensor_mask = landmarks_to_stgcn_tensor(data, mask)
    tensor_x = tensor_x.unsqueeze(0).to(device)      # [1, 3, 45, 93]
    tensor_mask = tensor_mask.unsqueeze(0).to(device) # [1, 1, 45, 93]

    # Class names
    class_names = [f"Class_{i}" for i in range(10)]
    vocab_path = "assets/vocabularies/mvp_10.json"
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as vf:
            vocab = json.load(vf)
            c2g = vocab.get("class_id_to_gloss", {})
            class_names = [c2g.get(str(i), f"Class_{i}") for i in range(10)]

    with torch.no_grad():
        logits = model(tensor_x, tensor_mask)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    top_label = class_names[top_idx]

    top3_indices = np.argsort(probs)[-3:][::-1]

    print("\n============================================================")
    print("           SIGNTALK AI: BASELINE INFERENCE")
    print("============================================================")
    print(f"Sequence File:    {args.sequence}")
    if true_label is not None:
        print(f"True Class:       {class_names[int(true_label)]} (ID: {true_label})")
    print(f"Predicted Class:  {top_label} (ID: {top_idx})")
    print(f"Confidence:       {top_prob:.4f} ({top_prob * 100:.1f}%)")
    print("------------------------------------------------------------")
    print("Top-3 Candidates:")
    for rank, idx in enumerate(top3_indices, start=1):
        print(f"  {rank}. {class_names[idx]:<12} (ID: {idx:>2}) — {probs[idx]*100:.2f}%")
    print("============================================================\n")


if __name__ == "__main__":
    main()
