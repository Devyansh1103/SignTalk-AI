# Tokenization Strategy: SignTalk AI

**Document ID:** STAI-P2P3-017  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead NLP Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Tokenization Architecture

In sign language processing, tokenization operates across two linguistic domains:
1. **Sign Gloss Tokenization:** Discrete morphological signs mapped to unique vocabulary indices.
2. **Spoken Language Wordpiece / Subword Tokenization:** Natural language translation text decomposed using subword algorithms (e.g. Byte-Pair Encoding or SentencePiece).

---

## 2. Active Gloss Tokenizer Implementation

For the active isolated sign vocabulary, SignTalk AI implements a deterministic lexical tokenizer utilizing [`assets/vocabularies/mvp_10.json`](file:///d:/SignAI/assets/vocabularies/mvp_10.json):

### 2.1 Special Tokens
- `<PAD>` (Index 0): Padding token for batch alignment.
- `<UNK>` (Index 1): Out-of-vocabulary or unverified gesture token.
- `<BOS>` (Index 2): Beginning-of-sequence token initiating autoregressive decoding.
- `<EOS>` (Index 3): End-of-sequence token terminating autoregressive generation.

### 2.2 Vocabulary Tokens
- Indices 4 through 13 represent the 10 canonical lexical signs (`HELLO`, `THANK_YOU`, `GOOD`, `HAPPY`, `MONDAY`, `CAR`, `BIRD`, `HOUSE`, `TIME`, `TEACHER`).
- Total vocabulary size: $V_{lex} = 14$ tokens.

### 2.3 Tokenization Function
```python
def tokenize_gloss(class_id: int, max_len: int = 3) -> Dict[str, torch.Tensor]:
    """Maps class ID to autoregressive Transformer target tensors."""
    token_id = class_id + 4
    decoder_input = torch.tensor([2, token_id], dtype=torch.long)  # [<BOS>, TOKEN]
    decoder_target = torch.tensor([token_id, 3], dtype=torch.long) # [TOKEN, <EOS>]
    attention_mask = torch.tensor([1, 1], dtype=torch.long)
    return {
        "decoder_input": decoder_input,
        "decoder_target": decoder_target,
        "attention_mask": attention_mask
    }
```

---

## 3. Spoken Language Tokenization (Future Continuous Translation)

When continuous sentence translations (e.g. ISLTranslate) are connected:
- **Tokenizer:** Pre-trained HuggingFace tokenizer (e.g. `google/flan-t5-base` or `facebook/mbart-large-50`).
- **Target Vocabulary:** Subword vocabulary covering English and regional Indian languages (Hindi, Tamil).
- **Current Status:** Marked as **Pending Continuous Data Video Sourcing**.
