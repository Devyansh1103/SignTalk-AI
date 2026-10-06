"""
SignTalk AI: Small Subset Overfit Diagnostic for Transformer.

Validates model optimization mechanics on a tiny 4-sample dataset:
  - Verifies target shifting [BOS, y1] -> [y1, EOS]
  - Verifies attention masks and causal decoding
  - Verifies gradient flow through FeatureProjection and SignLanguageTransformer
  - Confirms capability to reach 100% token and sequence exact match accuracy
"""

import os
import sys
import time

sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.sign_sequence_dataset import SignSequenceDataset
from src.data.collate import sign_sequence_collate_fn
from src.models.sign_translation_model import SignTranslationModel
from src.evaluation.sequence_metrics import compute_token_accuracy, compute_sequence_accuracy
from src.utils.reproducibility import set_seed


def run_overfit_test(num_epochs: int = 35):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("SignTalk AI — Small-Subset Overfit Diagnostic")
    print("=" * 65)

    # 1. Load small 4-sample subset
    full_dataset = SignSequenceDataset(manifest_path="data/manifests/train.csv")
    subset_indices = [0, 1, 2, 3]  # 4 distinct samples
    subset = Subset(full_dataset, subset_indices)
    loader = DataLoader(subset, batch_size=4, shuffle=False, collate_fn=sign_sequence_collate_fn)

    sample_meta = [full_dataset[i]["metadata"] for i in subset_indices]
    print(f"\nTraining on {len(subset)} fixed samples:")
    for idx, m in enumerate(sample_meta):
        print(f"  Sample {idx+1}: {m['sequence_id']} | Gloss: {m['gloss']} | Translation: {m['translation']}")

    # 2. Instantiate Model
    stgcn_ckpt = "experiments/stgcn/checkpoints/best_checkpoint.pt"
    model = SignTranslationModel(
        stgcn_config={"sequence_length": 45, "num_nodes": 93, "in_channels": 3},
        transformer_config={"embed_dim": 128, "vocab_size": 14, "num_heads": 4, "dropout": 0.0},
        freeze_stgcn=True,
        stgcn_checkpoint=stgcn_ckpt
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.005,
        weight_decay=0.0
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"\nStarting Overfit Optimization ({num_epochs} Epochs)...")
    converged_epoch = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            dec_in = batch["decoder_input"].to(device)
            dec_tgt = batch["decoder_target"].to(device)

            optimizer.zero_grad()
            logits = model(x=x, target_tokens=dec_in, mask=mask)
            B, S, V = logits.shape
            loss = criterion(logits.view(-1, V), dec_tgt.view(-1))
            loss.backward()
            optimizer.step()

        # Evaluate on the 4 samples
        model.eval()
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            token_acc = compute_token_accuracy(preds, dec_tgt, pad_idx=0)
            seq_acc = compute_sequence_accuracy(preds, dec_tgt, pad_idx=0)

            # Autoregressive generation check
            gen_tokens, _ = model.generate(x=x, mask=mask, max_length=5)
            gen_no_bos = gen_tokens[:, 1:]
            gen_seq_acc = compute_sequence_accuracy(gen_no_bos, dec_tgt, pad_idx=0)

        if epoch % 5 == 0 or epoch == 1 or (seq_acc == 1.0 and converged_epoch is None):
            print(
                f"Epoch {epoch:02d}/{num_epochs:02d} | "
                f"Loss: {loss.item():.4f} | "
                f"Token Acc: {token_acc*100:.1f}% | "
                f"Seq Acc (TF): {seq_acc*100:.1f}% | "
                f"Gen Seq Acc: {gen_seq_acc*100:.1f}%"
            )

        if token_acc == 1.0 and seq_acc == 1.0 and gen_seq_acc == 1.0 and converged_epoch is None:
            converged_epoch = epoch

    print("\n" + "=" * 65)
    if converged_epoch is not None:
        print(f"SUCCESS: Small subset 100% memorized by Epoch {converged_epoch}!")
    else:
        print(f"Final Epoch Loss: {loss.item():.4f}, Seq Acc: {seq_acc*100:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    run_overfit_test()
