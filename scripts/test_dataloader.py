#!/usr/bin/env python3
"""
SignTalk AI: Sequence DataLoader Validation Script.

Validates that PyTorch DataLoaders for train, val, and test splits
correctly yield batched tensors [B, C, T, V], auxiliary masks,
decoder tokens, and metadata without NaNs or Infs.
"""

import sys
import os
import argparse
import torch

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataloader import create_sequence_dataloader


def test_split_loader(split_name: str, manifest_path: str, batch_size: int = 4):
    print(f"\n--- Testing DataLoader for Split: '{split_name.upper()}' ---")
    loader = create_sequence_dataloader(
        manifest_path=manifest_path,
        split=split_name,
        batch_size=batch_size,
        num_workers=0
    )
    dataset_len = len(loader.dataset)
    print(f"Dataset samples for '{split_name}': {dataset_len}")

    if dataset_len == 0:
        print(f"[WARNING] Split '{split_name}' dataset is empty (check filtering).")
        return True

    batch_count = 0
    total_samples = 0

    for i, batch in enumerate(loader):
        x = batch["x"]
        mask = batch["mask"]
        labels = batch["label"]
        gloss_ids = batch["gloss_id"]
        dec_in = batch["decoder_input"]
        dec_tgt = batch["decoder_target"]
        meta = batch["metadata"]

        B, C, T, V = x.shape
        total_samples += B
        batch_count += 1

        # Assertions
        assert x.dtype == torch.float32, f"Expected float32 for x, got {x.dtype}"
        assert mask.dtype == torch.float32, f"Expected float32 for mask, got {mask.dtype}"
        assert labels.dtype == torch.int64, f"Expected int64 for labels, got {labels.dtype}"
        assert C == 3, f"Expected C=3 channels, got {C}"
        assert T == 45, f"Expected T=45 frames, got {T}"
        assert V == 93, f"Expected V=93 nodes, got {V}"
        assert mask.shape == (B, 1, 45, 93), f"Unexpected mask shape: {mask.shape}"
        assert dec_in.shape == (B, 2), f"Unexpected decoder_input shape: {dec_in.shape}"
        assert dec_tgt.shape == (B, 2), f"Unexpected decoder_target shape: {dec_tgt.shape}"

        # Finite checks
        assert torch.all(torch.isfinite(x)), "Non-finite values (NaN/Inf) detected in x!"
        assert torch.all(torch.isfinite(mask)), "Non-finite values detected in mask!"

        if i == 0:
            print(f"  First Batch Shape [B, C, T, V]: {list(x.shape)}")
            print(f"  Mask Shape [B, 1, T, V]:       {list(mask.shape)}")
            print(f"  Labels:                        {labels.tolist()}")
            print(f"  Gloss Token IDs:               {gloss_ids.tolist()}")
            print(f"  Decoder Inputs:                {dec_in.tolist()}")
            print(f"  Sample 0 Sequence ID:          {meta[0]['sequence_id']} ({meta[0]['label']})")

    print(f"[PASSED] Processed {batch_count} batches ({total_samples} samples) successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test PyTorch Sequence DataLoader.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/sequence_manifest.csv",
        help="Path to master sequence manifest."
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    args = parser.parse_args()

    print("\n============================================================")
    print("      SIGNTALK AI: PYTORCH DATALOADER VALIDATION")
    print("============================================================")
    print(f"Manifest: {args.manifest}")

    all_passed = True
    for split in ["train", "val", "test"]:
        passed = test_split_loader(split, args.manifest, args.batch_size)
        all_passed = all_passed and passed

    print("\n============================================================")
    if all_passed:
        print("          ALL DATALOADER CHECKS PASSED")
    else:
        print("          SOME DATALOADER CHECKS FAILED")
    print("============================================================\n")


if __name__ == "__main__":
    main()
