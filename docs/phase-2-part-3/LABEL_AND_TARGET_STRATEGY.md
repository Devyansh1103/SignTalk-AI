# Label and Target Strategy: SignTalk AI

**Document ID:** STAI-P2P3-009  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead NLP & ML Data Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Multi-Level Target Hierarchy

Sign language translation spans multiple levels of linguistic representation. To ensure strict linguistic rigor and prevent the fabrication of synthetic supervisory signals, SignTalk AI defines four distinct target tiers:

| Tier Level | Representation | Description & Semantics | Example Representation | Supported in Current Dataset |
| :---: | :--- | :--- | :---: | :---: |
| **Level 1** | **Class ID** | Zero-indexed contiguous integer identifier for ST-GCN cross-entropy classification | `0` | **Yes (Active)** |
| **Level 2** | **Sign Label** | Canonical lower-case lexical sign name | `"hello"` | **Yes (Active)** |
| **Level 3** | **Linguistic Gloss** | Standard uppercase sign language gloss representation | `"HELLO"` | **Yes (Active)** |
| **Level 4** | **Natural Language Translation** | English spoken/written text translation equivalent | `"Hello"` | **Yes (Lexical Equivalent)** |

---

## 2. Active 10-Class Vocabulary Mapping Matrix

The 10 vocabulary classes validated in the Phase 2 dataset are mapped deterministically across all four tiers:

| Class ID | Canonical Label (Level 2) | Linguistic Gloss (Level 3) | Natural Language Equivalent (Level 4) | Semantic Category | Hands Used |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 0 | `hello` | `HELLO` | `"Hello"` | Greetings | 1 |
| 1 | `thankyou` | `THANK_YOU` | `"Thank you"` | Greetings | 1 |
| 2 | `good` | `GOOD` | `"Good"` | Adjectives | 2 |
| 3 | `happy` | `HAPPY` | `"Happy"` | Adjectives | 2 |
| 4 | `monday` | `MONDAY` | `"Monday"` | Days & Time | 1 |
| 5 | `car` | `CAR` | `"Car"` | Transportation | 2 |
| 6 | `bird` | `BIRD` | `"Bird"` | Animals | 1 or 2 |
| 7 | `house` | `HOUSE` | `"House"` | Home | 2 |
| 8 | `time` | `TIME` | `"Time"` | Days & Time | 2 |
| 9 | `teacher` | `TEACHER` | `"Teacher"` | Jobs | 2 |

---

## 3. Supervision Integrity Rules

1. **No Syntactic Hallucination:** A class label representing a single lexical noun or adjective (e.g. `house` or `happy`) is NEVER transformed into an elaborate English sentence (e.g. *"This is a big house"* or *"I feel very happy"*). Such hallucinations corrupt training gradients and invalidate translation benchmarks.
2. **Explicit Gloss vs. Text Separation:** Glosses represent sign morphemes in capital letters with underscores for multi-word tokens (e.g. `THANK_YOU`), whereas natural language text follows standard English casing and orthography.
3. **Preservation of Source Annotations:** If a source dataset specifies annotations using alternative naming conventions (e.g. `storeorshop`, `smalllittle`), the raw string is retained in `source_label` while being mapped to `canonical_label` via a formal normalization dictionary.
