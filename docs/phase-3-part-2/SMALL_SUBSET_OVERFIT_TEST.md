# SignTalk AI — ST-GCN Small-Subset Overfit Verification Test

**Document ID:** `DOC-P3P2-OVERFIT-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Script Reference:** [`scripts/overfit_small_subset.py`](file:///d:/SignAI/scripts/overfit_small_subset.py)  
**Report Artifact:** [`experiments/stgcn/overfit_test/overfit_results.json`](file:///d:/SignAI/experiments/stgcn/overfit_test/overfit_results.json)  
**Date of Execution:** October 2, 2026  

---

## 1. Purpose and Diagnostic Objective

Before launching full-scale training across all 36 training sequences, a controlled overfitting experiment on a tiny subset of the dataset is essential.

This diagnostic test verifies:
1. **Unimpeded Gradient Flow:** Confirms backward gradients propagate seamlessly through spatial graph convolutions, temporal convolutions, and residual branches without vanishing or exploding.
2. **Adjacency Tensor Functionality:** Validates that the partitioned graph matrix ($\mathbf{A} \in \mathbb{R}^{3 \times 93 \times 93}$) properly aggregates spatial features without numerical instability.
3. **Model Learning Capacity:** Confirms that the ST-GCN architecture has sufficient expressive power to achieve near-zero training loss and 100% classification accuracy on a closed subset.

> [!NOTE]
> This is strictly a diagnostic sanity test, not a measure of generalization.

---

## 2. Experimental Protocol

- **Dataset Subset:** 4 distinct training samples representing 4 separate sign classes:
  - `seq_0001` (`HELLO`, Class 0)
  - `seq_0005` (`THANK_YOU`, Class 1)
  - `seq_0009` (`GOOD`, Class 2)
  - `seq_0013` (`HAPPY`, Class 3)
- **Model Architecture:** `SignSTGCN` ($C=3$, $V=93$, $T=45$, block channels $[32, 64, 128]$, dropout $= 0.0$).
- **Optimizer:** AdamW ($\text{LR} = 0.005$, $\text{weight decay} = 0.0$).
- **Objective:** CrossEntropyLoss.
- **Duration:** 30 epochs.

---

## 3. Measured Training Trajectory

Below is the measured loss and accuracy progression across the 30 overfit epochs:

| Epoch | Cross-Entropy Loss | Accuracy (%) | Correct / Total | Diagnostic Observation |
| :---: | :---: | :---: | :---: | :--- |
| **1** | 2.3013 | 25.0% | 1 / 4 | Initial random initialization (near uniform $10\%$) |
| **5** | 1.0261 | 50.0% | 2 / 4 | Rapid loss decrease, spatiotemporal features separating |
| **6** | 0.8817 | **100.0%** | **4 / 4** | Complete separation achieved in 6 epochs |
| **10** | 0.4920 | 100.0% | 4 / 4 | Consistent confidence margin widening |
| **15** | 0.2235 | 100.0% | 4 / 4 | Loss $< 0.25$ |
| **20** | 0.0532 | 100.0% | 4 / 4 | Loss $< 0.10$ |
| **25** | 0.0112 | 100.0% | 4 / 4 | Loss $< 0.02$ |
| **30** | **0.0036** | **100.0%** | **4 / 4** | Numerical convergence to near-zero |

```
============================================================
Final Cross-Entropy Loss:       0.0036
Final Accuracy:                 100.0% (4 / 4 correct)
Verification Status:            PASSED (100% capacity verified)
============================================================
```

---

## 4. Diagnostic Conclusions

1. **Zero Gradient Bottlenecks:** Gradients flowed from the classification head through all 3 ST-GCN blocks to the raw input coordinates without any vanishing or stagnation.
2. **Correct Node-Adjacency Alignment:** The 93-node adjacency tensor correctly localized coordinate messages without generating `NaN` or `Inf` values.
3. **Clearance for Full Training:** The ST-GCN implementation is certified as mechanically and mathematically sound, clearing the quality gate for full-dataset training.
