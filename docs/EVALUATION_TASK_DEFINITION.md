# SignTalk AI — Evaluation Task Definition

**Document ID:** `DOC-P3P4-TASK-DEF-001`  
**Phase:** Phase 3 — Part 4 (Model Evaluation, Ablation & Final Model Selection)  
**System Title:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Technical Subtitle:** A Spatial-Temporal Graph and Transformer-Based Sign-to-Text System  
**Date:** October 2026  
**Status:** FORMALLY DEFINED & BOUNDED  

---

## 1. Task Taxonomy and Landscape

Sign language artificial intelligence systems span four canonical task formulations:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TASK TAXONOMY IN SIGN LANGUAGE AI                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Level A: Isolated Sign Classification                                       │
│   Input: Pre-segmented landmark sequence [B, C, T, V]                       │
│   Target: Single categorical sign class y ∈ {1, ..., K}                     │
│   Metrics: Top-1/Top-3 Accuracy, Precision, Recall, Macro/Weighted F1       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Level B: Continuous Sign Recognition (CSLR)                                 │
│   Input: Continuous, unsegmented signing stream X_{1:T}                     │
│   Target: Ordered sequence of sign glosses (g_1, g_2, ..., g_M)             │
│   Metrics: Word Error Rate (WER), Character Error Rate (CER)                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Level C: Gloss-to-Text Translation                                          │
│   Input: Gloss sequence (g_1, ..., g_M) in sign language grammar            │
│   Target: Natural language sentence (w_1, ..., w_N) in spoken grammar       │
│   Metrics: BLEU-1 to BLEU-4, ROUGE-L, chrF, Semantic Similarity             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Level D: End-to-End Sign-to-Text Translation                                │
│   Input: Continuous video / landmarks X_{1:T}                               │
│   Target: Natural language sentence (w_1, ..., w_N)                         │
│   Metrics: Sentence BLEU, ROUGE-L, METEOR, chrF, Human Quality Rating       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Actual Supported Task for SignTalk AI Prototype

### Grounded Task Classification
Based on an audit of `data/manifests/` and the underlying 60 video sequences from INCLUDE-50:
- **Supported Task:** **Level A (Isolated Sign Classification) & Level A+ (Single-Token Translation Alignment)**.
- Each sequence in `data/processed/sequences/` represents a single isolated sign recording padded/interpolated to $T=45$ frames.
- Supervision consists of:
  1. `class_id`: Integer label $\in \{0, 1, \dots, 9\}$.
  2. `label`: Canonical label name (`hello`, `thankyou`, `good`, `happy`, `monday`, `car`, `bird`, `house`, `time`, `teacher`).
  3. `gloss`: Upper-case gloss (`HELLO`, `THANK_YOU`, `GOOD`, `HAPPY`, `MONDAY`, `CAR`, `BIRD`, `HOUSE`, `TIME`, `TEACHER`).
  4. `translation`: English spoken phrase ("Hello", "Thank you", "Good", "Happy", "Monday", "Car", "Bird", "House", "Time", "Teacher").

### Formal Mathematical Formulation

#### 1. Isolated Classification (Baseline & ST-GCN)
Given a spatiotemporal landmark tensor $\mathbf{X} \in \mathbb{R}^{B \times C \times T \times V}$ ($C=3, T=45, V=93$) and optional validity mask $\mathbf{M} \in \{0, 1\}^{B \times 1 \times T \times V}$:
$$\hat{y} = \arg\max_{c \in \{0, \dots, 9\}} P(y = c \mid \mathbf{X}, \mathbf{M})$$

#### 2. Sequence-Decoded Translation Alignment (SignTranslationModel)
Given spatiotemporal visual features $\mathbf{Z}_{\text{visual}} = \text{ST-GCN}(\mathbf{X}) \in \mathbb{R}^{B \times T' \times D}$ where $T'=12, D=256$:
The autoregressive Transformer decoder generates token sequence $\hat{\mathbf{y}} = (y_0, y_1, y_2)$:
$$y_0 = \langle\text{BOS}\rangle, \quad y_1 = \text{token\_id}, \quad y_2 = \langle\text{EOS}\rangle$$
Conditioned on:
$$P(\mathbf{y} \mid \mathbf{Z}_{\text{visual}}) = \prod_{s=1}^{S} P(y_s \mid y_{<s}, \mathbf{Z}_{\text{visual}})$$
where vocabulary size $|\mathcal{V}| = 14$ ($\langle\text{PAD}\rangle, \langle\text{UNK}\rangle, \langle\text{BOS}\rangle, \langle\text{EOS}\rangle$ and 10 lexical glosses).

---

## 3. Explicit Boundaries and Scientific Integrity Disclaimers

> [!IMPORTANT]
> **Scientific Integrity Rule:**
> - The dataset does **NOT** contain continuous sentence recordings.
> - The dataset does **NOT** contain complex syntactic phrases or multi-word continuous discourse.
> - Therefore, reporting 4-gram BLEU (BLEU-4) as a measure of "fluent multi-word sentence translation" is scientifically meaningless because reference sequences are single-gloss phrases (length $\le 2$ words). BLEU-1 and ROUGE-L reflect lexical and unigram concordance, while BLEU-4 is mathematically 0.0 for 1-to-2 token sequences.
> - We strictly **DO NOT** claim Level D continuous end-to-end machine translation.
> - The Transformer layer functions as an **autoregressive sequence generator and linguistic alignment interface** verifying that the visual-to-language bridge functions correctly before scaling to sentence-level datasets in future work.

---

## 4. Evaluated Candidate Models & Metrics Matrix

| Model Architecture | Task Formulation | Primary Metrics | Secondary / Diagnostic Metrics |
| :--- | :--- | :--- | :--- |
| **Baseline (BiLSTM)** | Isolated Classification | Top-1 Accuracy, Macro F1, Weighted F1 | Top-3 Accuracy, Per-Class F1, Latency, Params |
| **ST-GCN (Graph Conv)** | Isolated Classification | Top-1 Accuracy, Macro F1, Weighted F1 | Top-3 Accuracy, Per-Class F1, ECE, Latency, Params |
| **ST-GCN + Transformer** | Sequence & Translation Alignment | Token Accuracy, Sequence Exact Match, Macro F1 | BLEU-1, BLEU-2, ROUGE-L, Latency, Params |

---

## 5. Decision Gate for Phase 4 Real-Time Inference

For a model to be selected for Phase 4 real-time deployment:
1. **Accuracy Threshold:** Test Top-1 Accuracy $\ge 75\%$ on the test partition.
2. **Macro F1 Threshold:** Macro F1 $\ge 0.70$.
3. **Latency Bound:** Mean end-to-end CPU inference latency $\le 50\text{ ms}$ for real-time responsiveness ($\ge 20\text{ FPS}$).
4. **Parameter Footprint:** Model size $\le 50\text{ MB}$ for lightweight edge or container deployment.
5. **Architectural Stability:** Well-conditioned logits, calibrated confidence, and documented failure boundaries.
