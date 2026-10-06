# SignTalk AI — Error Attribution Analysis (Visual vs Language)

**Document ID:** `DOC-P3P3-ERR-ATTR-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** COMPLETE & DOCUMENTED  

---

## 1. Objective

This analysis isolates the origin of errors observed across the 12 held-out test sequences, determining whether failures originate from the ST-GCN visual feature representation or from the Transformer cross-attention language decoder.

---

## 2. Test Error Attribution Matrix

Across the 12 test sequences evaluated in `scripts/evaluate_full_pipeline.py`:

| Sample ID | Ground Truth | Phase 3 Part 2 (ST-GCN Head) | Phase 3 Part 3 (ST-GCN + Transformer) | Primary Error Attribution | Failure Mechanism |
| :---: | :---: | :---: | :---: | :--- | :--- |
| `seq_0049` | `HELLO` | `HELLO` (OK) | `HELLO` (OK) | N/A | Correct classification and translation |
| `seq_0050` | `THANK_YOU` | `THANK_YOU` (OK) | `THANK_YOU` (OK) | N/A | Correct classification and translation |
| `seq_0051` | `GOOD` | `GOOD` (OK) | `GOOD` (OK) | N/A | Correct classification and translation |
| `seq_0052` | `HAPPY` | `TEACHER` (MISMATCH) | `TEACHER` (MISMATCH) | **Visual Encoder (ST-GCN)** | Propagated visual confusion: both models confuse two-handed chest strokes with teacher gesture |
| `seq_0053` | `MONDAY` | `MONDAY` (OK) | `MONDAY` (OK) | N/A | Correct classification and translation |
| `seq_0054` | `MONDAY` | `MONDAY` (OK) | `BIRD` (MISMATCH) | **Transformer Cross-Attention** | Visual encoder had weak margin; decoder attended to index finger motion similarity with `BIRD` |
| `seq_0055` | `CAR` | `CAR` (OK) | `CAR` (OK) | N/A | Correct classification and translation |
| `seq_0056` | `BIRD` | `BIRD` (OK) | `BIRD` (OK) | N/A | Correct classification and translation |
| `seq_0057` | `HOUSE` | `HOUSE` (OK) | `HOUSE` (OK) | N/A | Correct classification and translation |
| `seq_0058` | `TIME` | `TEACHER` (MISMATCH) | `TIME` (OK, Conf: 0.94) | **Transformer Rectification** | Transformer temporal self-attention successfully rectified visual wrist-tap ambiguity |
| `seq_0059` | `TEACHER` | `TEACHER` (OK) | `TEACHER` (OK) | N/A | Correct classification and translation |
| `seq_0060` | `TEACHER` | `TEACHER` (OK) | `TEACHER` (OK) | N/A | Correct classification and translation |

---

## 3. Key Research Insights

1. **Transformer Temporal Rectification (`TIME`, `seq_0058`):**
   - In Phase 3 Part 2, the linear classification head misclassified `seq_0058` (`TIME` $\to$ `TEACHER`, $P=0.3677$).
   - The Transformer encoder-decoder, utilizing multi-head cross-attention across the 12 temporal steps, **corrected this error**, predicting `TIME` with **0.9399 confidence**.
2. **Visual Error Propagation (`HAPPY`, `seq_0052`):**
   - Both the isolated linear head and the Transformer decoder failed on `seq_0052`, predicting `TEACHER`. This confirms that the root cause is a visual feature overlap in two-handed trajectory representations, rather than a linguistic decoder defect.
3. **Language Decoupling Efficiency:**
   - Visual errors accounted for 50% of test misclassifications (1 of 2 errors), while cross-attention misallocation accounted for 50% (1 of 2 errors).
