# Validation-Based Confidence Threshold Selection

**Project:** SignTalk AI: Real-Time Indian Sign Language Translation Platform  
**Architecture:** Spatial-Temporal Graph Convolutional Network (`SignTalk_STGCN_v1`)  
**Phase:** Phase 4 — Part 3: Confidence Filtering, Smoothing & Continuous Sign Detection  
**Evaluation Dataset:** Validation Split (`data/manifests/val.csv`, $N=12$ sequences)

---

## 1. Executive Summary & Objective

In real-time inference, raw classification argmax predictions from sliding-window spatiotemporal graphs can produce spurious outputs when presented with incomplete gestures, inter-sign transitions, or poor landmark visibility. 

Phase 3 established that ST-GCN exhibits an **Expected Calibration Error (ECE) of 0.2713**, indicating that raw softmax probabilities cannot be directly interpreted as true Bayesian posterior probabilities of correctness. Therefore, arbitrary thresholds (e.g. naive 0.50 or 0.90) must not be assumed without empirical validation.

This document records the empirical threshold selection experiment conducted strictly on the **validation partition** (`data/manifests/val.csv`), adhering to the core principle: **no optimization on the final test set**.

---

## 2. Experimental Methodology

The validation manifest contains 12 representative isolated sign sequences spanning the MVP-10 vocabulary. Each sequence was evaluated against the certified checkpoint [`best_checkpoint.pt`](file:///d:/SignAI/experiments/stgcn/checkpoints/best_checkpoint.pt) across confidence thresholds $\tau \in [0.50, 0.90]$ in steps of 0.05.

### Evaluation Metrics
For each threshold $\tau$:
- **Accepted Predictions:** Predictions with model confidence $\ge \tau$.
- **Rejection Rate:** Proportion of total predictions falling below threshold $\tau$ (labeled `UNCERTAIN`).
- **Accepted Accuracy (Precision):** Correct predictions among accepted candidates:
  $$\text{Precision} = \frac{\text{True Accepted}}{\text{Total Accepted}}$$
- **False Acceptance Rate (FAR):** Inaccurate predictions that exceeded threshold $\tau$.
- **False Rejection Rate (FRR):** Accurate predictions incorrectly rejected because confidence $< \tau$.

---

## 3. Empirical Validation Results

The table below summarizes empirical results across the evaluated threshold grid (from [`results/confidence_thresholds.csv`](file:///d:/SignAI/results/confidence_thresholds.csv)):

| Threshold $\tau$ | Accepted Count | Rejected Count | Rejection Rate (%) | True Accepted | False Accepted | Accepted Precision (%) | False Rejections |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | 12 | 0 | 0.0% | 11 | 1 | 91.7% | 0 |
| **0.55** | 12 | 0 | 0.0% | 11 | 1 | 91.7% | 0 |
| **0.60** | 10 | 2 | 16.7% | 9 | 1 | 90.0% | 2 |
| **0.65** | 10 | 2 | 16.7% | 9 | 1 | 90.0% | 2 |
| **0.70** | 10 | 2 | 16.7% | 9 | 1 | 90.0% | 2 |
| **0.75** | 8 | 4 | 33.3% | 7 | 1 | 87.5% | 4 |
| **0.80** | 6 | 6 | 50.0% | 6 | 0 | 100.0% | 5 |
| **0.85** | 6 | 6 | 50.0% | 6 | 0 | 100.0% | 5 |
| **0.90** | 4 | 8 | 66.7% | 4 | 0 | 100.0% | 7 |

---

## 4. Key Findings & Trade-Off Analysis

1. **Raw Validation Performance:**
   At threshold $\tau = 0.50$, 11 of 12 validation sequences were correctly identified (91.7% raw accuracy). The only classification error occurred on `seq_0045` (`bird`), which was misclassified as `time` with confidence score 0.7908.
2. **The High-Precision Regime ($\tau \ge 0.80$):**
   At $\tau = 0.80$, false acceptances drop to zero, yielding 100% precision among accepted predictions. However, this comes at the cost of rejecting 50% of valid sequences (5 false rejections), including valid signs whose landmark quality was acceptable but non-ideal.
3. **The Balanced Operational Regime ($\tau = 0.65$):**
   At $\tau = 0.65$, 83.3% of signs are accepted with 90.0% precision, while maintaining low rejection (16.7%). Only 2 valid signs with marginal landmark coverage are flagged as `UNCERTAIN` for user re-signing.
4. **Decoupling Landmark Quality from Model Confidence:**
   Confidence scores must never be blindly multiplied by landmark quality (e.g. $0.90 \times 0.60 = 0.54$). Both values are maintained as separate gates:
   - `model_confidence >= 0.65`
   - `input_quality >= 0.40`

---

## 5. Selected Threshold & Recommendation

Based on validation evidence and the requirements of real-time Indian Sign Language recognition:
- **Selected Confidence Threshold:** $\mathbf{\tau = 0.65}$
- **Low-Confidence Tag:** `UNCERTAIN`
- **Landmark Quality Gate:** $\mathbf{q_{\text{min}} = 0.40}$

Predictions with confidence $< 0.65$ are marked as `UNCERTAIN` without terminating the streaming pipeline, feeding into the stability evaluation state machine to avoid triggering false sign events.
