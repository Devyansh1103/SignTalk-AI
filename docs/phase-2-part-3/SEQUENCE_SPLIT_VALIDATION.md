# Sequence Split Validation Specification: SignTalk AI

**Document ID:** STAI-P2P3-020  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** ML Data Engineer & Research Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Split Isolation Mandate

In accordance with Section 0 (Rules 7, 8, 9), data leakage across evaluation boundaries is strictly prohibited:
- **Rule 7:** No moving processed samples between train/val/test after sequence generation.
- **Rule 8:** No splitting individual frames from the same video across partitions.
- **Rule 9:** Sequences originating from the same source recording must NEVER cross evaluation boundaries.

---

## 2. Automated Partition Isolation Audit

The automated leakage validator [`src/data/split_validator.py`](file:///d:/SignAI/src/data/split_validator.py) checks three set intersections:

$$\mathcal{S}_{train} \cap \mathcal{S}_{val} = \emptyset$$
$$\mathcal{S}_{train} \cap \mathcal{S}_{test} = \emptyset$$
$$\mathcal{S}_{val} \cap \mathcal{S}_{test} = \emptyset$$

where $\mathcal{S}$ denotes the set of unique source recording IDs and raw video filenames.

### 2.1 Partition Quantities
- **Train Split:** 36 sequences ($60.0\%$) originating from 36 distinct video recordings.
- **Validation Split:** 12 sequences ($20.0\%$) originating from 12 distinct video recordings.
- **Test Split:** 12 sequences ($20.0\%$) originating from 12 distinct video recordings.
- **Overlap:** Exactly $0$ shared recordings ($0.0\%$ leakage).

---

## 3. Signer Stratification vs. Signer Independence Analysis

### 3.1 Signer Stratification in Active Partition
In the active 60-sequence dataset:
- Every signer (`signer_01`, `signer_02`, `signer_03`) performs distinct video recordings.
- The 60 recordings are partitioned:
  - `signer_01`: 12 train, 4 val, 4 test ($20$ total).
  - `signer_02`: 12 train, 4 val, 4 test ($20$ total).
  - `signer_03`: 12 train, 4 val, 4 test ($20$ total).
- **Result:** Video recordings are $100\%$ isolated (zero video leakage), with equal demographic representation across all three splits.

### 3.2 Protocol for Signer-Independent Benchmarking
For cross-signer generalization evaluation in Phase 3/4:
- Train: `signer_01` and `signer_02` ($40$ videos).
- Test: Unseen `signer_03` ($20$ videos).
- The `SplitValidator` provides full programmatic support for both signer-stratified and strictly signer-disjoint partitions.
