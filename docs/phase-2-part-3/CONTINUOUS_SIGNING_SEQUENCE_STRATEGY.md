# Continuous Signing Sequence Strategy: SignTalk AI

**Document ID:** STAI-P2P3-007  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead NLP & Research Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Overview & Problem Definition

Continuous Sign Language Translation (CSLT) involves mapping an unsegmented stream of continuous signing gestures to spoken language glosses and grammatical sentences. Unlike isolated signs, continuous signing exhibits:
- Strong inter-sign coarticulation (the preparation phase of sign $k$ blends with the retraction phase of sign $k-1$).
- Variable signing tempo and prosodic pauses.
- Simultaneous manual and non-manual articulation (facial expressions indicating question vs. statement).

---

## 2. Theoretical Taxonomy of Continuous Supervision

We evaluate five supervision cases across potential continuous sign datasets:

| Supervision Case | Description | Ground Truth Available | Applicable Modeling Strategy |
| :--- | :--- | :--- | :--- |
| **Case A** | Explicit Sign Boundaries | Timestamps $[t_{start}, t_{end}]$ for every gloss | Direct temporal slicing into sign tokens |
| **Case B** | Approximate Gloss Boundaries | Weak timestamps or coarse keyframes | Semi-supervised boundary refinement (HMM / Viterbi) |
| **Case C** | Sentence Glosses (Unbounded) | Ordered sequence of glosses $(g_1, g_2, \dots, g_M)$ | Connectionist Temporal Classification (CTC) alignment |
| **Case D** | Sentence Text Only (No Glosses) | Spoken language translation text only | End-to-End Sequence-to-Sequence (ST-GCN + Transformer) |
| **Case E** | No Continuous Video Acquired | Isolated signs only (Current local active state) | Sign-level modeling + real-time sliding window buffer |

---

## 3. Status of Current Dataset & Future Roadmap

### 3.1 Current Local Benchmark Status (Case E)
As established in [`DATA_SEMANTICS_ANALYSIS.md`](file:///d:/SignAI/docs/phase-2-part-3/DATA_SEMANTICS_ANALYSIS.md), the active benchmark dataset comprises isolated sign recordings from AI4Bharat INCLUDE-50.
- **Explicit Boundary Prohibition:** We strictly DO NOT fabricate temporal sign boundaries or concatenate isolated clips to mimic continuous signing.
- **Architectural Preparedness:** While our training targets currently operate at the isolated sign/gloss level, our sequence pipeline generates standardized $T=45$ temporal slices that are identical to the sliding-window buffers emitted by a continuous real-time video stream.

### 3.2 Strategy for Candidate Continuous Datasets (ISLTranslate & ISL-CSLTR)
When continuous video corpora are integrated in future phases:
1. **Case C / Case D Approach (CTC & Attention):**
   - Temporal feature maps will be extracted from continuous video via ST-GCN: $\mathbf{H} = \text{ST-GCN}(\mathbf{X}) \in \mathbb{R}^{T' \times D}$.
   - A CTC loss will align unsegmented frame sequences with gloss sequences:
     $$\mathcal{L}_{CTC} = -\ln P(\mathbf{g} \mid \mathbf{H})$$
   - A Transformer decoder with cross-attention will translate the continuous representations into natural language text:
     $$\mathcal{L}_{Trans} = -\sum_{m=1}^{M} \ln P(w_m \mid w_{<m}, \mathbf{H})$$
2. **Online Segmentation Pipeline:** An online activity detector (monitoring hand velocity $\|\mathbf{v}_{wrist}\|_2$ and energy) will delineate sign stroke boundaries during continuous webcam streaming.
