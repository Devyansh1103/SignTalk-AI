# Post-Sequence Class Distribution: SignTalk AI

**Document ID:** STAI-P2P3-021  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** ML Data Engineer & Research Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Class Distribution Analysis

Following canonical sequence generation, we audit the dataset to ensure temporal windowing and standardization did not alter class balance or skew vocabulary representations.

### 1.1 Complete Class Breakdown Across Splits

| Class ID | Canonical Label | Train Count | Val Count | Test Count | Total Sequences | Class Percentage | Balance Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `hello` | 4 | 1 | 1 | 6 | 10.0% | $100\%$ Balanced |
| 1 | `thankyou` | 3 | 2 | 1 | 6 | 10.0% | $100\%$ Balanced |
| 2 | `good` | 4 | 1 | 1 | 6 | 10.0% | $100\%$ Balanced |
| 3 | `happy` | 4 | 1 | 1 | 6 | 10.0% | $100\%$ Balanced |
| 4 | `monday` | 3 | 1 | 2 | 6 | 10.0% | $100\%$ Balanced |
| 5 | `car` | 4 | 1 | 1 | 6 | 10.0% | $100\%$ Balanced |
| 6 | `bird` | 3 | 2 | 1 | 6 | 10.0% | $100\%$ Balanced |
| 7 | `house` | 4 | 1 | 1 | 6 | 10.0% | $100\%$ Balanced |
| 8 | `time` | 4 | 1 | 1 | 6 | 10.0% | $100\%$ Balanced |
| 9 | `teacher` | 3 | 1 | 2 | 6 | 10.0% | $100\%$ Balanced |
| **Total** | **All 10 Classes** | **36** | **12** | **12** | **60** | **100.0%** | **Perfect Uniformity** |

---

## 2. Temporal Metrics per Vocabulary Class

| Class Label | Mean Frame Count | Min Frames | Max Frames | Mean Duration (s) | Sequence Length $T$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `hello` | 50.0 | 45 | 55 | 2.00 s | 45 frames |
| `thankyou` | 50.0 | 45 | 55 | 2.00 s | 45 frames |
| `good` | 50.0 | 45 | 55 | 2.00 s | 45 frames |
| `happy` | 50.0 | 45 | 55 | 2.00 s | 45 frames |
| `monday` | 49.7 | 45 | 55 | 1.99 s | 45 frames |
| `car` | 50.0 | 45 | 55 | 2.00 s | 45 frames |
| `bird` | 50.0 | 45 | 55 | 2.00 s | 45 frames |
| `house` | 50.0 | 45 | 55 | 2.00 s | 45 frames |
| `time` | 49.3 | 45 | 55 | 1.97 s | 45 frames |
| `teacher` | 50.0 | 45 | 55 | 2.00 s | 45 frames |

---

## 3. Comparison with Phase 2 Part 1

- In Phase 2 Part 1, the raw dataset established exactly 6 samples per class across all 10 classes.
- Sequence construction produced exactly 6 sequences per class across all 10 classes ($0\%$ class attrition).
- No class was eliminated, overrepresented, or diluted.
