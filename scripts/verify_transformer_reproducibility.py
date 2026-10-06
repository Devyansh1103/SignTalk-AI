"""
SignTalk AI: Transformer Reproducibility Verification Script.

Executes two identical twin training runs from random seed 42 to verify
strict numerical determinism in the Transformer translation layer:
  - Verifies identical parameter initialization
  - Verifies exact loss and metric values across epochs
  - Verifies zero numerical drift
  - Saves report to experiments/transformer/metrics/reproducibility_report.json
"""

import os
import sys
import json
import time

sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.sign_sequence_dataset import SignSequenceDataset
from src.data.collate import sign_sequence_collate_fn
from src.models.sign_translation_model import SignTranslationModel
from src.utils.reproducibility import set_seed


def run_reproducibility_test(num_epochs: int = 3):
    print("=" * 65)
    print("SignTalk AI — Transformer Reproducibility Verification")
    print("=" * 65)

    def run_experiment(run_id: int):
        set_seed(42)
        device = torch.device("cpu")

        train_ds = SignSequenceDataset(manifest_path="data/manifests/train.csv", augment=False)
        loader = DataLoader(train_ds, batch_size=4, shuffle=False, collate_fn=sign_sequence_collate_fn)

        model = SignTranslationModel(
            stgcn_config={"sequence_length": 45, "num_nodes": 93, "in_channels": 3},
            transformer_config={"embed_dim": 64, "vocab_size": 14, "num_heads": 4, "encoder_layers": 1, "decoder_layers": 1, "ff_dim": 128},
            freeze_stgcn=True,
            stgcn_checkpoint="experiments/stgcn/checkpoints/best_checkpoint.pt"
        ).to(device)

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        epoch_losses = []
        for ep in range(num_epochs):
            model.train()
            ep_loss = 0.0
            for batch in loader:
                x = batch["x"].to(device)
                dec_in = batch["decoder_input"].to(device)
                dec_tgt = batch["decoder_target"].to(device)

                optimizer.zero_grad()
                logits = model(x, dec_in)
                loss = criterion(logits.view(-1, 14), dec_tgt.view(-1))
                loss.backward()
                optimizer.step()
                ep_loss += loss.item() * x.size(0)

            epoch_losses.append(ep_loss / len(train_ds))

        # Final weights
        weights = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        return epoch_losses, weights

    print("\nExecuting Run 1 (Seed 42)...")
    losses1, weights1 = run_experiment(1)

    print("\nExecuting Run 2 (Seed 42)...")
    losses2, weights2 = run_experiment(2)

    loss_diff = max(abs(l1 - l2) for l1, l2 in zip(losses1, losses2))
    param_diff = max((w1 - w2).abs().max().item() for w1, w2 in zip(weights1, weights2))

    print("\n" + "=" * 65)
    print("REPRODUCIBILITY AUDIT RESULTS:")
    print("=" * 65)
    print(f"Run 1 Epoch Losses: {[round(l, 6) for l in losses1]}")
    print(f"Run 2 Epoch Losses: {[round(l, 6) for l in losses2]}")
    print(f"Max Loss Difference:      {loss_diff:.2e}")
    print(f"Max Parameter Difference: {param_diff:.2e}")

    is_deterministic = (loss_diff == 0.0) and (param_diff == 0.0)
    print(f"Strict Determinism:       {'CONFIRMED' if is_deterministic else 'FAILED'}")
    print("=" * 65)

    report = {
        "seed": 42,
        "epochs": num_epochs,
        "run1_losses": losses1,
        "run2_losses": losses2,
        "max_loss_difference": loss_diff,
        "max_parameter_difference": param_diff,
        "is_strictly_deterministic": is_deterministic,
        "pytorch_version": torch.__version__,
        "python_version": sys.version
    }

    out_file = "experiments/transformer/metrics/reproducibility_report.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved reproducibility report to: {out_file}")
    return is_deterministic


if __name__ == "__main__":
    run_reproducibility_test()
