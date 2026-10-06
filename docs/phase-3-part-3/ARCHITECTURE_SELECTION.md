# SignTalk AI — Architecture Selection (Phase 3 Part 3)

**Document ID:** `DOC-P3P3-ARCH-SELECT-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** COMPLETE & APPROVED  

---

## 1. Candidate Architectural Approaches

We evaluate four structural candidates connecting the visual spatial-temporal graph encoder to a language representation:

### OPTION A: ST-GCN Direct Classification Head
- **Architecture:** `ST-GCN` $\to$ Global Average Pooling $\to$ Linear $(256 \to 10)$.
- **Pros:** Minimal latency, direct gradient path, evaluated in Phase 3 Part 2.
- **Cons:** Strictly isolated; does not support sequential autoregressive token decoding or linguistic context.

### OPTION B: ST-GCN + Transformer Encoder-Only Gloss Classifier
- **Architecture:** `ST-GCN` $\to$ Temporal Feature Sequence $\to$ Transformer Encoder $\to$ Pooling $\to$ Linear $(D_{\text{model}} \to |\mathcal{V}|)$.
- **Pros:** Adds cross-frame self-attention over the temporal sequence; simple training loop.
- **Cons:** Cannot perform autoregressive decoding, token generation, or variable-length sentence prediction.

### OPTION C: ST-GCN + Transformer Encoder-Decoder (Selected Architectural Foundation)
- **Architecture:** `ST-GCN` $\to$ Feature Projection $\to$ Positional Encoding $\to$ Transformer Encoder $\to$ Transformer Decoder (Cross-Attention) $\to$ Vocabulary Projection.
- **Pros:**
  - Implements the complete intended sequence-to-sequence translation paradigm.
  - Supports autoregressive token generation with teacher forcing ($[<BOS>, y_1] \to [y_1, <EOS>]$).
  - Directly scales to multi-word sentence decoding when continuous datasets are integrated.
  - Fully decoupled visual encoder and language decoder for freeze vs fine-tuning ablations.
- **Cons:** Slightly higher computational overhead than direct classification; requires careful learning rate scheduling and teacher-forcing masking.

### OPTION D: Pipeline (ST-GCN $\to$ Gloss ID $\to$ External Pretrained LLM)
- **Architecture:** `ST-GCN` classifies sign $\to$ string lookup $\to$ Prompt to GPT/Llama/T5.
- **Pros:** Fluent surface English generation for simple prompts.
- **Cons:** Massive parameter footprint ($>1\text{B}$ params); introduces third-party dependency; unfeasible for real-time edge execution; ignores spatial-temporal visual embeddings.

---

## 2. Selection Rationale

| Evaluation Criterion | Option A (Direct) | Option B (Enc-Only) | Option C (Enc-Dec) | Option D (Pipeline+LLM) |
| :--- | :---: | :---: | :---: | :---: |
| **Supports Current Supervision** | Yes | Yes | **Yes** | Yes |
| **Autoregressive Extensibility** | No | No | **Yes** | No (External) |
| **End-to-End Differentiable** | Yes | Yes | **Yes** | No |
| **Inference Latency** | Lowest | Low | **Moderate** | Very High |
| **Architectural Completeness** | Minimal | Partial | **Full** | Fragmented |

**Decision:** We implement **Option C** (`SignTranslationModel` with `SignSTGCN` visual encoder + `FeatureProjection` + `SignLanguageTransformer` encoder-decoder). The decoder operates on the canonical token vocabulary ($|\mathcal{V}| = 14$), handling both isolated token sequence generation now and multi-token sentence generation in future continuous releases.
