# SignTalk AI — Language Supervision Audit

**Document ID:** `DOC-P3P3-SUPERVISION-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** COMPLETE & VERIFIED  

---

## 1. Audit Objective

This document audits the linguistic and supervisory signals present in the current dataset (`signTalk-seq-v1.0.0`) to establish what learning signals are genuinely available for model optimization, preventing hallucinated tasks or unsupported translation claims.

---

## 2. Linguistic Signal Availability Table

| Signal | Available? | Source File / Column | Annotation Quality | Usable for Phase 3 Part 3? |
| :--- | :---: | :--- | :--- | :---: |
| **Categorical Sign Class** | **YES** | `data/manifests/*.csv` (`label` 0–9) | 100% verified isolated classes | **YES** (Standard classification & token target) |
| **Canonical Gloss ID** | **YES** | `data/processed/label_mapping.csv` (`gloss_id` 4–13) | Standardized uppercase sign tokens | **YES** (Vocabulary index for Transformer decoder) |
| **Canonical Gloss String** | **YES** | `data/manifests/*.csv` (`gloss`) | Verified MediaPipe/INCLUDE-50 mappings | **YES** (Token text decoding & evaluation) |
| **English Translation** | **YES** | `data/manifests/*.csv` (`translation`) | Single-word/phrase lexical mapping | **YES** (Lexical lookup and token decoding) |
| **Special Tokens** | **YES** | `assets/vocabularies/mvp_10.json` | `<PAD>` (0), `<UNK>` (1), `<BOS>` (2), `<EOS>` (3) | **YES** (Autoregressive sequence conditioning) |
| **Gloss Sequences (Multi-Sign)**| **NO** | N/A | Absent in isolated sign recordings | **NO** (Requires continuous signing dataset) |
| **Sentence-Level English Text** | **NO** | N/A | Absent in isolated sign recordings | **NO** (Requires continuous paired corpora) |
| **Sub-Sign Frame Alignments** | **NO** | N/A | No per-frame phonetic/gloss boundaries | **NO** (Weakly-supervised or CTC needed for CSLR) |
| **Signer Metadata** | **YES** | `data/manifests/*.csv` (`signer_id`) | Strict signer-disjoint splits maintained | **YES** (Generalization verification) |
| **Validity / Missingness Mask** | **YES** | `data/processed/sequences/*/*.npz` (`mask`) | Real MediaPipe landmark visibility | **YES** (Attention key padding masks) |

---

## 3. Explicit Boundary Declaration

As evidenced in the table above, the current dataset release contains **isolated sign tokens**. It does **not** contain continuous multi-sign sentence transcripts or complex syntactic alignments.

Therefore:
1. We implement the complete, extensible **ST-GCN $\to$ Feature Projection $\to$ Transformer Encoder-Decoder $\to$ Vocabulary Projection** architecture.
2. We train and validate the Transformer model on the isolated token sequence interface ($[<BOS>, \text{gloss\_id}] \to [\text{gloss\_id}, <EOS>]$).
3. We explicitly record that continuous sentence-level machine translation requires expanded continuous ISL corpora.
