# Sequence Dataset Quality Report: SignTalk AI

**Document ID:** STAI-P2P3-043  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction & Dataset Finalization)  
**Dataset Version:** `signTalk-seq-v1.0.0`  
**Author:** Lead ML Data Engineer & Research Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Executive Dataset Summary

| Dimension | Specification & Measured Result | Verification Reference |
| :--- | :---: | :--- |
| **Dataset Version** | `signTalk-seq-v1.0.0` | [`data/manifests/sequence_manifest.csv`](file:///d:/SignAI/data/manifests/sequence_manifest.csv) |
| **Language Focus** | Indian Sign Language (ISL) | AI4Bharat INCLUDE-50 Subset |
| **Total Source Recordings** | 60 raw video files | `data/raw/videos/*.mp4` |
| **Total Canonical Sequences** | 60 sequences | `data/processed/sequences/**/*.npz` |
| **Total Rejected Sequences** | 16 sequences ($26.7\%$) | [`data/manifests/rejected_sequences.csv`](file:///d:/SignAI/data/manifests/rejected_sequences.csv) |
| **Total Vocabulary Classes** | 10 classes | `assets/vocabularies/mvp_10.json` |
| **Total Signers** | 3 deaf participants | `signer_01`, `signer_02`, `signer_03` |
| **Train Sequences** | 36 sequences ($60.0\%$) | `data/manifests/train.csv` |
| **Validation Sequences** | 12 sequences ($20.0\%$) | `data/manifests/val.csv` |
| **Test Sequences** | 12 sequences ($20.0\%$) | `data/manifests/test.csv` |
| **Native Sequence Duration** | $\mu = 49.90\text{ frames}$ ($1.996\text{ s}$), $\sigma = 4.96$ | $\min = 45$, $\text{median} = 49$, $\max = 55$ |
| **Standardized Sequence Length**| $T = 45\text{ frames}$ ($1.80\text{ s @ 25 FPS}$) | $100\%$ standardized |
| **Padding Overhead** | $0.0\%$ | Primary canonical mode |
| **Mean Missing Joint Ratio** | $21.4\%$ unobserved/interpolated | Resting hand / face contours |
| **Data Leakage Status** | **ZERO LEAKAGE (PASSED)** | Checked via `SplitValidator` |

---

## 2. Quality Classification Breakdown

| Classification Category | Sequence Count | Percentage | Operational Policy |
| :--- | :---: | :---: | :--- |
| **`GOOD`** ($Q \ge 0.75$) | 3 | 5.0% | High-confidence, complete tracking across all modalities |
| **`ACCEPTABLE`** ($Q \ge 0.50$) | 36 | 60.0% | Standard isolated sign sequences suitable for model training |
| **`REVIEW`** ($0.35 \le Q < 0.50$) | 5 | 8.3% | Marginal detection coverage; flagged for diagnostic evaluation |
| **`REJECT`** ($Q < 0.35$) | 16 | 26.7% | Excluded from training DataLoader (`filter_rejects=True`); preserved for adversarial stress tests |

---

## 3. Linguistic Supervision Availability

- **Isolated Sign Glosses:** **AVAILABLE** (10 canonical glosses: `HELLO`, `THANK_YOU`, `GOOD`, `HAPPY`, `MONDAY`, `CAR`, `BIRD`, `HOUSE`, `TIME`, `TEACHER`).
- **Natural Language Translations:** **AVAILABLE (LEXICAL EQUIVALENT)** (`"Hello"`, `"Thank you"`, `"Good"`, etc.).
- **Continuous Sentence Data:** **MISSING LOCAL ALIGNED VIDEO** (31,117 sentences registered in metadata; video collection pending).
- **Background / Idle Gesture Frames:** **MISSING IN BENCHMARK** (Mitigated via kinematic energy gating).

---

## 4. Class & Signer Balance Verification

- **Class Representation:** Exactly 6 sequences per class across all 10 classes ($100\%$ balanced).
- **Signer Representation:** Exactly 20 sequences per signer across all 3 signers ($100\%$ balanced).
- **Zero Attrition:** No classes or signers were eliminated during sequence generation.
