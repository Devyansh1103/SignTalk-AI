# Class Imbalance Analysis: SignTalk AI

**Document ID:** STAI-P3P1-016  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 3 — Part 1 (Baseline Model & Training Infrastructure)  
**Author:** ML Data Engineer & Research Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Class Distribution Across Partitions

The canonical sequence dataset `signTalk-seq-v1.0.0` was constructed from 60 video instances across 10 vocabulary classes. We examine class representation across splits:

| Class ID | Canonical Label | Master Manifest | Train Split (Raw) | Train Split (Filtered) | Val Split | Test Split | Support Balance |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `hello` | 6 | 4 | 4 | 1 | 1 | High |
| 1 | `thankyou` | 6 | 3 | 2 | 2 | 1 | Moderate |
| 2 | `good` | 6 | 4 | 3 | 1 | 1 | High |
| 3 | `happy` | 6 | 4 | 2 | 1 | 1 | Moderate |
| 4 | `monday` | 6 | 3 | 3 | 1 | 2 | Moderate |
| 5 | `car` | 6 | 4 | 4 | 1 | 1 | High |
| 6 | `bird` | 6 | 3 | 2 | 2 | 1 | Moderate |
| 7 | `house` | 6 | 4 | 3 | 1 | 1 | High |
| 8 | `time` | 6 | 4 | 3 | 1 | 1 | High |
| 9 | `teacher` | 6 | 3 | 2 | 1 | 2 | Moderate |
| **Total** | **All 10 Classes** | **60** | **36** | **28** | **12** | **12** | **Uniform Baseline** |

---

## 2. Evaluation Metric Implications

1. **Why Accuracy is Insufficient:** In small validation and test splits (12 samples each), correct predictions on majority classes can mask zero recall on classes with single instances.
2. **Mandatory Reporting of Macro Metrics:** To ensure balanced evaluation, all experiment runs report:
   - **Macro F1:** Arithmetic mean of individual class F1-scores, penalizing models that fail on minority classes.
   - **Per-Class Precision, Recall, and Support:** Full breakdown identifying exact failure classes.
   - **Normalized Confusion Matrix:** Row-normalized to display per-class recall probabilities directly.
