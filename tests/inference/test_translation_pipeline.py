"""
Unit tests for SignTranslationPipeline.
"""

import pytest
import torch
import numpy as np
from src.models.sign_translation_model import SignTranslationModel
from src.nlp.tokenizer import SignLanguageTokenizer
from src.inference.translation_pipeline import SignTranslationPipeline


def test_translation_pipeline_execution():
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
    tokenizer = SignLanguageTokenizer(vocab_path="assets/vocabularies/mvp_10.json")

    pipeline = SignTranslationPipeline(
        model=model,
        tokenizer=tokenizer,
        confidence_threshold=0.10,
        device="cpu"
    )

    dummy_seq = np.random.randn(3, 45, 93).astype(np.float32)
    result = pipeline.translate_sequence(dummy_seq)

    assert "predicted_gloss" in result
    assert "predicted_translation" in result
    assert "confidence" in result
    assert "latency" in result
    assert result["latency"]["total_ms"] > 0.0
