# SignTalk AI — Tokenizer Design

**Document ID:** `DOC-P3P3-TOKENIZER-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** COMPLETE & DOCUMENTED  

---

## 1. Architectural Role

The tokenizer connects neural sequence models to discrete symbolic linguistic entities (sign glosses and English translations). It manages encoding of target strings into token IDs, decoding of predicted logits into strings, vocabulary lookup, and special token management.

---

## 2. Vocabulary Specification

The vocabulary is version-locked in `assets/vocabularies/mvp_10.json` (Version `1.0.0`):
- **Total Vocabulary Size:** $|\mathcal{V}| = 14$

### A. Special Control Tokens (IDs 0–3)
- `<PAD>` (ID 0): Sequence padding token. Ignored in loss computation and masked out in attention mechanisms.
- `<UNK>` (ID 1): Out-of-vocabulary fallback token for unrecognized signs or novel inputs.
- `<BOS>` (ID 2): Beginning-of-sequence prompt token supplied to initialize autoregressive decoding.
- `<EOS>` (ID 3): End-of-sequence termination token emitted to stop decoding.

### B. Lexical Sign Gloss Tokens (IDs 4–13)
| Token ID | Class ID | Canonical Gloss | Target Translation | Semantic Domain |
| :---: | :---: | :--- | :--- | :--- |
| **4** | 0 | `HELLO` | Hello | Greetings |
| **5** | 1 | `THANK_YOU` | Thank you | Greetings |
| **6** | 2 | `GOOD` | Good | Adjectives |
| **7** | 3 | `HAPPY` | Happy | Adjectives |
| **8** | 4 | `MONDAY` | Monday | Days and Time |
| **9** | 5 | `CAR` | Car | Transport |
| **10** | 6 | `BIRD` | Bird | Animals |
| **11** | 7 | `HOUSE` | House | Home |
| **12** | 8 | `TIME` | Time | Days and Time |
| **13** | 9 | `TEACHER` | Teacher | Jobs |

---

## 3. Evaluation Protocol Compliance

1. **No Data Leakage:** The vocabulary is strictly constructed from canonical training class schemas. No test-set specific words or novel tokens are injected.
2. **Fixed Vocabulary:** Lexical IDs map 1-to-1 with categorical indices shifted by the 4 special tokens ($\text{token\_id} = \text{class\_id} + 4$).
3. **Decoupled Translation:** The tokenizer retains translation dictionaries permitting direct decoding from token IDs into natural English.
