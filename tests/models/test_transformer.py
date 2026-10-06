"""
Unit tests for SignLanguageTransformer and SignTransformer.
"""

import pytest
import torch
from src.models.sign_language_transformer import SignLanguageTransformer
from src.models.sign_transformer import SignTransformer


def test_transformer_forward():
    model = SignLanguageTransformer(
        vocab_size=14,
        embed_dim=64,
        num_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        ff_dim=128
    )

    visual_features = torch.randn(2, 12, 64)
    target_tokens = torch.tensor([[2, 4], [2, 5]], dtype=torch.long)

    logits = model(visual_features, target_tokens)
    assert logits.shape == (2, 2, 14)
    assert not torch.isnan(logits).any()


def test_transformer_generate():
    model = SignLanguageTransformer(
        vocab_size=14,
        embed_dim=64,
        num_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        ff_dim=128
    )

    visual_features = torch.randn(2, 12, 64)
    tokens, confs = model.generate(visual_features, max_length=5)
    assert tokens.shape[0] == 2
    assert confs.shape[0] == 2
    assert (tokens[:, 0] == model.bos_idx).all()


def test_transformer_alias():
    assert SignTransformer is SignLanguageTransformer
