# -*- coding: utf-8 -*-
"""Tests for the synthetic gloss -> sentence corpus (src/nlp/gloss_grammar.py)."""
from src.data.label_map import CANONICAL_CLASSES
from src.models.gloss_translator import detokenize, tokenize_sentence
from src.nlp import gloss_grammar as G


def _vocab():
    classes = [CANONICAL_CLASSES[i] for i in sorted(CANONICAL_CLASSES)]
    return [c.label for c in classes], {c.label: c.translation for c in classes}, {c.label: c.hindi or c.translation for c in classes}


def test_hindi_agreement():
    assert G.hi_adj("अच्छा", "f", False) == "अच्छी"
    assert G.hi_adj("अच्छा", "m", True) == "अच्छे"
    assert G.hi_adj("नया", "f", False) == "नई"
    assert G.hi_adj("खुश", "f", True) == "खुश"
    assert G.hi_poss("मेर", "m", True) == "मेरे"
    assert G.hi_copula("f", False, 1) == "थी"


def test_corpus_uses_only_vocabulary_and_covers_every_word():
    labels, en, hi = _vocab()
    pairs = G.build_corpus(labels, en, hi, n=3000, seed=0)
    assert all(t in set(labels) for g, _, _ in pairs for t in g)
    assert {g[0] for g, _, _ in pairs if len(g) == 1} == set(labels)
    assert len({g for g, _, _ in pairs}) == len(pairs)


def test_known_sentences():
    labels, en, hi = _vocab()
    pairs = {g: (e, h) for g, e, h in G.build_corpus(labels, en, hi, n=3000, seed=0)}
    assert pairs[("hello", "teacher")] == ("Hello, teacher!", "नमस्ते, शिक्षक!")
    assert pairs[("time", "house")] == ("Time to go home.", "घर जाने का समय है।")


def test_tokenize_roundtrip():
    for s in ["Hello, how are you?", "मेरी माँ बीमार हैं।", "My T-shirt is blue."]:
        assert detokenize(tokenize_sentence(s)) == s


def test_wear_without_a_wear_sign():
    import random
    r = random.Random(0)
    seen = [G.p_wear(r) for _ in range(200)]
    assert all("wearing" in en and "पहन" in hi for _, en, hi in seen)
    # a colour before the garment is worn; after it, it stays the "is <colour>" pattern
    assert all(g[-1] in G.CLOTHES for g, _, _ in seen)
