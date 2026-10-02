# Sliding-Window Duplication Analysis: SignTalk AI

**Document ID:** STAI-P2P3-023  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** ML Data Engineer & Research Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Context & Motivation

When temporal sequence datasets employ overlapping sliding windows ($W = 45$, stride $S = 5$), multiple sequences are extracted from a single continuous recording. This document analyzes window duplication, temporal autocorrelation, and establishes rules to prevent spurious evaluation inflation.

---

## 2. Empirical Window Yield Analysis

For the validated dataset ($N \in [45, 55]$ frames), the theoretical yield under sliding window parameters ($W = 45$, $S = 5$) is:

$$K = \left\lfloor \frac{N - 45}{5} \right\rfloor + 1$$

| Native Frame Count | Video Instances | Windows per Video ($K$) | Total Windows | Adjacent Window Overlap |
| :---: | :---: | :---: | :---: | :---: |
| **45 frames** | 30 | 1 window | 30 | None ($0.0\%$) |
| **53 frames** | 3 | 2 windows | 6 | $40 / 45 = 88.89\%$ |
| **55 frames** | 27 | 3 windows | 81 | $40 / 45 = 88.89\%$ |
| **Total** | **60 videos** | **Mean: 1.95** | **117 windows** | Weighted Mean: $69.2\%$ |

---

## 3. Autocorrelation & Data Duplication Risks

1. **High Temporal Autocorrelation:** Adjacent windows generated with stride $S = 5$ share $40$ out of $45$ frames ($88.9\%$ identical landmark data).
2. **False Generalization Risk:** If sliding windows from the same video are randomly assigned to train and test sets, the model achieves near-$100\%$ test accuracy simply by memorizing identical frames.
3. **Class Skewing:** Because $55$-frame videos produce 3 windows while $45$-frame videos produce only 1, unconstrained sliding windowing distorts class and repetition balance ($81$ windows from repetition 2 vs $30$ from repetition 1).

---

## 4. Architectural Mitigation & Final Decision

To ensure absolute scientific validity:
- **Canonical Dataset (Primary Mode):** The official benchmark sequence dataset utilizes **Mode 1: Canonical Full-Utterance Windowing** ($1$ sequence per recording = $60$ canonical sequences). This guarantees $0\%$ duplicate windowing, perfect class balance, and zero autocorrelation across evaluation partitions.
- **Sliding-Window Inference Benchmark (Secondary Mode):** Overlapping sliding windows ($117$ windows) are restricted strictly to real-time inference latency simulation and temporal jitter robustness benchmarking. They are never mixed across partition boundaries.
