# Data Semantics Analysis: SignTalk AI

**Document ID:** STAI-P2P3-002  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Research Engineer & Lead ML Data Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Objective & Scope

Before constructing sequence representations or training pipelines, the semantic properties of the source data must be formally categorized. This document analyzes what the acquired dataset represents linguistically, temporally, and structurally to prevent false architectural assumptions.

---

## 2. Source Dataset Semantic Classification

Every source dataset within the SignTalk AI registry is classified according to the taxonomy below:
- **Category A:** Isolated Sign (Gloss / Word level)
- **Category B:** Continuous Signing (Continuous motion, no sentence boundaries)
- **Category C:** Sentence-Level Signing (Full grammatical sentences)
- **Category D:** Gloss-Level Continuous Signing (Sentences with continuous gloss boundaries)
- **Category E:** Sign-to-Text Paired Data (Continuous video with aligned natural language translation)
- **Category F:** Other (Lexical reference, fingerspelling, phonology)

| Dataset Identifier | Registered Role | Semantic Category | Granularity | Ground-Truth Supervision Available |
| :--- | :--- | :---: | :--- | :--- |
| **INCLUDE-50** | Primary Benchmark (Active) | **Category A** | Isolated Word / Gloss | Class ID, Gloss Label, Signer ID, Repetition |
| **ISLTranslate** | Linguistic Expansion (Registered) | **Category E** | Continuous Sentence | English Sentence Text, Metadata (No local aligned video) |
| **ISL-CSLTR** | Candidate Continuous | **Category C** | Continuous Phrase | Phrase-level video (Unbounded glosses) |

---

## 3. Detailed Semantic Analysis of Active Dataset (INCLUDE-50 Subset)

### 3.1 Sample and Recording Semantics
- **One Recording = One Isolated Sign:** Each video recording captures exactly one isolated Indian Sign Language gesture performed in isolation (e.g. `hello`, `car`, `thankyou`).
- **Absence of Coarticulation:** Because tokens are recorded in isolation, there is no inter-sign coarticulation, transitional hand movement between consecutive grammatical words, or continuous non-manual clause boundary markers.
- **Recording Structure:**
  1. *Preparation Phase (Frames 1–10):* Hands rise from rest position into signing space.
  2. *Stroke / Nucleus Phase (Frames 11–35):* Active morphological execution of the sign (dominant articulator movement).
  3. *Retraction Phase (Frames 36–55):* Hands return toward neutral resting position.

### 3.2 Label Semantics
- **Supervision Level:** Level 1 (Class ID: 0–9) and Level 2 (Sign Label / Gloss: e.g. `THANKYOU`).
- **Absence of Natural Language Syntax:** The labels are isolated lexical items, NOT natural language sentences.
- **Syntactic Limitation:** A label such as `time` or `teacher` cannot be treated as an English sentence like *"What is the time?"* or *"The teacher entered the room"* without fabricating linguistic targets.

### 3.3 Signer & Dialectal Characteristics
- **Signers:** Recorded by 3 deaf participants (`signer_01`, `signer_02`, `signer_03`) at St. Louis School for the Deaf, Adyar, Chennai.
- **Dialect:** Southern Indian Sign Language dialectal conventions.
- **Visual Framing:** Upper body frontal viewpoint, neutral backdrop, controlled indoor illumination.

---

## 4. Research Honesty Statement

> [!IMPORTANT]
> **Authoritative Research Honesty Finding:**  
> The current dataset supports isolated sign-level sequence modeling. Continuous sentence-level sign language translation requires additional continuous signing data with appropriate temporal and linguistic supervision.
>
> We strictly reject the practice of artificially concatenating isolated video clips to synthesize "pseudo-continuous" sentences. Such synthetic concatenation introduces unnatural jump discontinuities, eliminates genuine coarticulation, and produces invalid evaluation metrics.

---

## 5. Architectural Implications for Phase 2 Part 3

1. **Sign-Level Sequence Modeling:** Each valid recording will be processed as a unified sign-level temporal sequence of length $T = 45$ frames.
2. **ST-GCN Input Interface:** The spatial-temporal graph network will model the isolated gesture trajectory across the 93 skeletal nodes.
3. **Transformer & NLP Interface:** The Transformer target representation will map to discrete lexical sign tokens (gloss IDs), providing the exact architectural foundation required when continuous sentence data is introduced in future iterations.
