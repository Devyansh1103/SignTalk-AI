"""
SignTalk AI: Transformer Smoke Test.

Verifies end-to-end integration before full training:
  1. Tokenizer initialization and vocabulary mapping
  2. Batch collation from actual dataset
  3. ST-GCN visual feature extraction
  4. Transformer forward pass with teacher forcing
  5. CrossEntropyLoss calculation
  6. Backward gradient propagation through Transformer layers
  7. Checkpoint generation in experiments/transformer/smoke_test/
"""

import os
import sys
import shutil

# Ensure workspace root is in python path
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.nlp.tokenizer import SignLanguageTokenizer
from src.data.sign_sequence_dataset import SignSequenceDataset
from src.data.collate import sign_sequence_collate_fn
from src.models.sign_translation_model import SignTranslationModel


def run_smoke_test():
    smoke_dir = "experiments/transformer/smoke_test"
    os.makedirs(smoke_dir, exist_ok=True)
    print("=" * 65)
    print("Starting SignTalk AI Transformer Smoke Test")
    print("=" * 65)

    # 1. Tokenizer test
    print("\n[Step 1] Initializing Tokenizer...")
    tokenizer = SignLanguageTokenizer(vocab_path="assets/vocabularies/mvp_10.json")
    assert tokenizer.vocab_size == 14
    tokens = tokenizer.encode("HELLO", add_special_tokens=True)
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    assert decoded == "HELLO"
    print(f"  Tokenizer OK: Vocab size = {tokenizer.vocab_size}, Encoded HELLO -> {tokens}")

    # 2. Batch construction
    print("\n[Step 2] Building Dataset & DataLoader...")
    dataset = SignSequenceDataset(manifest_path="data/manifests/train.csv")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=sign_sequence_collate_fn)
    batch = next(iter(loader))
    print(f"  Batch OK: x={list(batch['x'].shape)}, decoder_input={list(batch['decoder_input'].shape)}")

    # 3. Model instantiation & ST-GCN checkpoint loading
    print("\n[Step 3] Instantiating SignTranslationModel...")
    ckpt_path = "experiments/stgcn/checkpoints/best_checkpoint.pt"
    model = SignTranslationModel(
        stgcn_config={"sequence_length": 45, "num_nodes": 93, "in_channels": 3},
        transformer_config={"embed_dim": 128, "vocab_size": 14, "num_heads": 4},
        freeze_stgcn=True,
        stgcn_checkpoint=ckpt_path
    )
    print("  Model instantiated and ST-GCN initialized.")

    # 4. Forward pass
    print("\n[Step 4] Executing Forward Pass...")
    x = batch["x"]
    mask = batch["mask"]
    dec_in = batch["decoder_input"]
    dec_tgt = batch["decoder_target"]

    logits = model(x=x, target_tokens=dec_in, mask=mask)
    print(f"  Forward Pass OK: Logits shape = {list(logits.shape)}")
    assert logits.shape == (4, 2, 14)

    # 5. Loss calculation
    print("\n[Step 5] Calculating Token CrossEntropyLoss...")
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss = criterion(logits.view(-1, 14), dec_tgt.view(-1))
    print(f"  Loss OK: Value = {loss.item():.4f}")
    assert not torch.isnan(loss) and loss.item() > 0.0

    # 6. Backward pass
    print("\n[Step 6] Executing Backward Pass & Gradient Flow Verification...")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss.backward()

    # Verify gradients in Transformer lm_head and projection
    assert model.transformer.lm_head.weight.grad is not None
    assert model.transformer.lm_head.weight.grad.norm().item() > 0.0
    assert model.feature_projection.projection.weight.grad is not None
    # Verify ST-GCN is frozen (no gradients)
    for p in model.stgcn.parameters():
        assert p.grad is None or p.grad.norm().item() == 0.0

    optimizer.step()
    print("  Backward Pass OK: Non-zero gradients verified in language layer, ST-GCN frozen.")

    # 7. Checkpoint save
    print("\n[Step 7] Testing Checkpoint Serialization...")
    smoke_ckpt_file = os.path.join(smoke_dir, "smoke_checkpoint.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item()
    }, smoke_ckpt_file)
    assert os.path.exists(smoke_ckpt_file)
    print(f"  Checkpoint saved successfully: {smoke_ckpt_file} ({os.path.getsize(smoke_ckpt_file)/1e6:.2f} MB)")

    print("\n" + "=" * 65)
    print("SMOKE TEST PASSED 100% WITH ALL CHECKS VERIFIED!")
    print("=" * 65)


if __name__ == "__main__":
    run_smoke_test()
