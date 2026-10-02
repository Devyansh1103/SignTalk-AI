# Transformer Target Preparation: SignTalk AI

**Document ID:** STAI-P2P3-016  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead NLP Engineer & Research Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Objective & Target Interface Definition

The SignTalk AI architecture pairs a spatial-temporal graph encoder (ST-GCN) with a sequence decoder (Transformer/NLP) to translate gestural representations into linguistic outputs. This document establishes the target tensor interface for sequence-to-sequence modeling.

```
Landmark Sequence Tensor X ∈ R^{B × C × T × V}
                │
                ▼
        [ST-GCN Encoder]
                │
                ▼
Temporal Feature Embeddings H ∈ R^{B × T' × D}
                │
                ▼ (Cross-Attention)
      [Transformer Decoder] ◄── Autoregressive Target Tokens Y_{in}
                │
                ▼
Output Logits P(w_m | w_{<m}, H) ──► Cross-Entropy Loss vs. Y_{target}
```

---

## 2. Target Representation Under Current Isolated Supervision

### 2.1 Linguistic Reality
As mandated by Section 47 (Research Honesty Rule), the current benchmark dataset comprises isolated sign recordings from AI4Bharat INCLUDE-50.
- **True Target:** Each sequence corresponds to an isolated sign gloss (e.g. `HELLO`, `TIME`, `HOUSE`).
- **No Sentence Fabrication:** We strictly do NOT invent natural language sentence targets (e.g. *"The teacher is standing near the house"*) because such pairings lack ground-truth supervision.

### 2.2 Autoregressive Target Sequence Formatting
For compatibility with standard Transformer decoders (e.g. BART, T5, or custom encoder-decoder models), the single isolated gloss is represented as a structured autoregressive sequence:

$$\mathbf{y}_{input} = [\text{<BOS>}, \text{GLOSS\_ID}], \quad \mathbf{y}_{target} = [\text{GLOSS\_ID}, \text{<EOS>}]$$

| Sequence Type | Tensor Content | Length | Description |
| :--- | :--- | :---: | :--- |
| **Decoder Input Tokens** | `[2, gloss_token_id]` | 2 | Initiates generation with `<BOS>` token |
| **Decoder Target Labels** | `[gloss_token_id, 3]` | 2 | Computes cross-entropy loss up to `<EOS>` |
| **Attention Mask** | `[1, 1]` | 2 | Full causal attention |

---

## 3. Standard Vocabulary Mapping (`assets/vocabularies/mvp_10.json`)

| Class ID | Canonical Label | Gloss Token | Gloss Token ID | Decoder Input Sequence | Decoder Target Sequence |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 0 | `hello` | `HELLO` | 4 | `[2, 4]` | `[4, 3]` |
| 1 | `thankyou` | `THANK_YOU` | 5 | `[2, 5]` | `[5, 3]` |
| 2 | `good` | `GOOD` | 6 | `[2, 6]` | `[6, 3]` |
| 3 | `happy` | `HAPPY` | 7 | `[2, 7]` | `[7, 3]` |
| 4 | `monday` | `MONDAY` | 8 | `[2, 8]` | `[8, 3]` |
| 5 | `car` | `CAR` | 9 | `[2, 9]` | `[9, 3]` |
| 6 | `bird` | `BIRD` | 10 | `[2, 10]` | `[10, 3]` |
| 7 | `house` | `HOUSE` | 11 | `[2, 11]` | `[11, 3]` |
| 8 | `time` | `TIME` | 12 | `[2, 12]` | `[12, 3]` |
| 9 | `teacher` | `TEACHER` | 13 | `[2, 13]` | `[13, 3]` |

---

## 4. Extensibility to Future Continuous Translation

When continuous video sequences (e.g. from ISLTranslate or ISL-CSLTR) are introduced:
1. Target tokens will expand to full sentence sequences: `[<BOS>, token_1, token_2, ..., token_M, <EOS>]`.
2. The Transformer decoder interface remains completely unchanged, accepting variable-length token sequences with padding masks $\mathbf{M}_{pad}$.
