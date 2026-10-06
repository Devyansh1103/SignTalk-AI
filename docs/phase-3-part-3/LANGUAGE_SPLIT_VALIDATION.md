# SignTalk AI — Language Split Validation

**Document ID:** `DOC-P3P3-SPLIT-VAL-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** AUDITED & VERIFIED  

---

## 1. Audit Objective

A common pitfall in sequence-to-sequence translation models is lexical, recording, or signer leakage across train, validation, and test partitions. This audit confirms that the strict partition boundaries established in Phase 2 Part 3 and maintained in Phase 3 Part 1 & 2 remain strictly enforced during Transformer training.

---

## 2. Partition Independence Verification

| Partition | Sequence Count | Source Recordings | Signer IDs Represented | Classes Represented | Overlap with Test Set |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 36 | 36 distinct video takes | `signer_01`, `signer_02`, `signer_03` | 10 classes | **0% (Disjoint recordings)** |
| **VAL** | 12 | 12 distinct video takes | `signer_01`, `signer_02`, `signer_03` | 10 classes | **0% (Disjoint recordings)** |
| **TEST** | 12 | 12 distinct video takes | `signer_01`, `signer_02`, `signer_03` | 10 classes | **0% (Disjoint recordings)** |

### Strict Guarantees
1. **Zero Window Leakage:** Every sequence is extracted from an independent, non-overlapping video take. No rolling or sliding window spans partition borders.
2. **Zero Text / Gloss Leakage:** Target tokens are evaluated on held-out visual recordings.
3. **No Test Decision Making:** Hyperparameters and checkpoint selection rely strictly on validation split performance. Test data is evaluated only once on the finalized model checkpoint.
