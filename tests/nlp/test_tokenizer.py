"""
Unit tests for SignLanguageTokenizer and TextNormalizer.
"""

import pytest
from src.nlp.text_normalizer import TextNormalizer
from src.nlp.tokenizer import SignLanguageTokenizer


def test_text_normalizer():
    normalizer = TextNormalizer(lowercase=False, strip_punctuation=True)
    text = "  Hello,   World!  "
    clean = normalizer.normalize(text)
    assert clean == "Hello World"

    gloss = normalizer.normalize_gloss("thank you")
    assert gloss == "THANK_YOU"


def test_tokenizer_encoding_decoding():
    tokenizer = SignLanguageTokenizer(vocab_path="assets/vocabularies/mvp_10.json")
    assert tokenizer.vocab_size == 14
    assert tokenizer.pad_id == 0
    assert tokenizer.unk_id == 1
    assert tokenizer.bos_id == 2
    assert tokenizer.eos_id == 3

    # Test encoding
    tokens = tokenizer.encode("HELLO", add_special_tokens=True)
    assert tokens == [2, 4, 3]  # <BOS>, HELLO (4), <EOS>

    # Test decoding
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    assert decoded == "HELLO"

    # Test translation lookup
    translation = tokenizer.decode_to_translation([4])
    assert translation == "Hello"

    # Test unknown token fallback
    unk_tokens = tokenizer.encode("UNKNOWN_SIGN_123", add_special_tokens=False)
    assert unk_tokens == [1]
