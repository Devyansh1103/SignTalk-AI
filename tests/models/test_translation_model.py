"""
Unit tests for SignTranslationModel end-to-end integration.
"""

import pytest
import torch
from src.models.sign_translation_model import SignTranslationModel


def test_sign_translation_model_forward():
    stgcn_cfg = {
        "in_channels": 3,
        "num_classes": 10,
        "num_nodes": 93,
        "sequence_length": 45,
        "block_channels": [32, 32, 64, 64, 128, 128],
        "block_strides": [1, 1, 2, 1, 2, 1],
        "temporal_kernel_size": 5
    }
    tf_cfg = {
        "embed_dim": 64,
        "vocab_size": 14,
        "num_heads": 4,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "ff_dim": 128
    }

    model = SignTranslationModel(
        stgcn_config=stgcn_cfg,
        transformer_config=tf_cfg,
        freeze_stgcn=True
    )

    x = torch.randn(2, 3, 45, 93)
    target_tokens = torch.tensor([[2, 4], [2, 5]], dtype=torch.long)

    logits = model(x, target_tokens)
    assert logits.shape == (2, 2, 14)
    assert not torch.isnan(logits).any()


def test_sign_translation_model_generate():
    stgcn_cfg = {
        "in_channels": 3,
        "num_classes": 10,
        "num_nodes": 93,
        "sequence_length": 45,
        "block_channels": [32, 32, 64, 64, 128, 128],
        "block_strides": [1, 1, 2, 1, 2, 1],
        "temporal_kernel_size": 5
    }
    tf_cfg = {
        "embed_dim": 64,
        "vocab_size": 14,
        "num_heads": 4,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "ff_dim": 128
    }

    model = SignTranslationModel(
        stgcn_config=stgcn_cfg,
        transformer_config=tf_cfg,
        freeze_stgcn=True
    )

    x = torch.randn(2, 3, 45, 93)
    tokens, confs = model.generate(x, max_length=4)
    assert tokens.shape[0] == 2
    assert confs.shape[0] == 2
    assert (tokens[:, 0] == 2).all()  # <BOS>
