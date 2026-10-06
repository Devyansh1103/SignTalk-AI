# SignTalk AI — Dataset Limitations & Future Continuous Corpus Requirements

**Document ID:** `DOC-P3P3-DATA-LIMIT-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** COMPLETE & DOCUMENTED  

---

## 1. Current Dataset State & Boundary

The current finalized sequence dataset (`signTalk-seq-v1.0.0`) contains 60 validated sequences representing 10 lexical sign categories performed by 3 signers under controlled settings.

### What CAN Be Trained Today
1. **Visual Representation Learning:** Spatial graph convolutions over body, hands, and facial landmarks.
2. **Spatiotemporal Temporal Dynamics:** Temporal modeling over 45-frame sequence intervals.
3. **Isolated Sign Classification & Single-Token Translation:** Mapping kinematic trajectories to canonical gloss tokens (`HELLO`) and English lexical translations ("Hello").
4. **Extensible Translation Layer Architecture:** The end-to-end interface connecting ST-GCN visual feature extraction to an autoregressive Transformer encoder-decoder.

### What CANNOT Be Trained Today
1. **Continuous Sentence Translation:** Translating continuous, unsegmented signing into multi-word English sentences (e.g., "The teacher is coming on Monday").
2. **Co-articulation & Sign Transitions:** Modeling movement epenthesis between adjacent signs.
3. **Spatial Grammar & Non-Manual Signals:** Advanced ISL syntax utilizing spatial indexing, eyebrow furrowing, head tilts, and mouth morphemes for interrogative or conditional grammar.

---

## 2. Requirements for the Next Dataset Expansion

To transition from isolated sign translation to true continuous sentence-level Indian Sign Language translation, subsequent dataset collection must satisfy:

| Dimension | Current Dataset (`v1.0.0`) | Required Continuous Corpus (`v2.0.0`) |
| :--- | :--- | :--- |
| **Signing Modality** | Isolated Signs | Continuous Natural Signing |
| **Linguistic Structure** | Single lexical words | Full sentences & conversational dialogues |
| **Annotation Level** | Single class ID & gloss | Sequence gloss transcripts + English translations |
| **Vocabulary Scale** | 10 vocabulary classes | $\ge 250$ to $1,000$ sign lemmas |
| **Sequence Duration** | Fixed 45 frames (1.5 seconds) | Variable 60 to 300 frames (2 to 10 seconds) |
| **Signer Diversity** | 3 adult signers | $\ge 20$ diverse native/fluent signers |
| **Grammar Diversity** | Lexical citations | Declaratives, questions, imperatives, spatial indexing |
| **Contextual Domains** | General greeting/noun pilot | Healthcare, education, emergency, daily commerce |

---

## 3. Explicit Research Integrity Declaration

SignTalk AI explicitly records that claims of sentence-level natural language translation are deferred until continuous multi-sign corpora with paired linguistic supervision are collected and benchmarked. The current phase establishes the validated architectural and modular foundation for this future milestone.
