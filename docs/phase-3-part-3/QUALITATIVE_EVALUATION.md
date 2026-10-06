# SignTalk AI — Qualitative Translation Evaluation Framework

**Document ID:** `DOC-P3P3-QUAL-EVAL-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** SPECIFIED & DOCUMENTED  

---

## 1. Overview and Human Evaluation Stance

Automatic evaluation metrics (BLEU, ROUGE, Word Error Rate) measure n-gram lexical overlap between model hypotheses and reference strings. In sign language translation, however, such metrics frequently fail to capture grammatical adequacy, spatial reference preservation, and conceptual meaning.

> [!IMPORTANT]
> **Human Evaluation Declaration:** In Phase 3 Part 3, **no formal human deaf/hard-of-hearing or certified ISL interpreter evaluation study was conducted**. All reported numbers are automatic test metrics. The qualitative framework below defines the standard protocol established for subsequent human validation in later deployment phases.

---

## 2. Qualitative Evaluation Dimensions

When human evaluations are executed, translations are scored on a 5-point Likert scale across four core dimensions:

### 1. Meaning Preservation (Semantic Fidelity)
- **Score 5:** Complete preservation of intended communicative intent without omission.
- **Score 3:** Core intent understandable, but secondary semantic qualifiers lost.
- **Score 1:** Complete alteration or contradiction of the signed meaning.

### 2. Grammaticality & Naturalness
- **Score 5:** Fluent, grammatical, natural spoken English sentence.
- **Score 3:** Minor grammatical lapses (e.g., tense, preposition) that do not hinder comprehension.
- **Score 1:** Fragmented or incomprehensible word salad.

### 3. Sign Omission (Deletions)
- Detects whether essential informational elements expressed in the visual gesture (e.g., negative headshakes, spatial direction) were omitted from the translated text.

### 4. Hallucination (Insertions)
- Detects whether the language decoder generated fluent English words that had no semantic basis in the visual input.

---

## 3. Application to Current Isolated Dataset

For isolated sign translation ($N=1$ gloss per sequence):
- The translation task is a deterministic 1-to-1 semantic projection (`HELLO` $\to$ "Hello", `THANK_YOU` $\to$ "Thank you").
- In this regime, hallucination risk is low, and semantic preservation corresponds strictly to correct lexical identification.
- As continuous datasets with multi-word ISL syntax are introduced, human qualitative scoring will become the primary benchmark of communicative utility.
