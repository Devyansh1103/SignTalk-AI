# SignTalk AI — ST-GCN Error Analysis & Diagnostics

**Document ID:** `DOC-P3P2-ERROR-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Evaluated Data:** Isolated Test Set (`data/manifests/test.csv`, $N=12$)  
**Prediction Log:** [`experiments/stgcn/metrics/sample_predictions.csv`](file:///d:/SignAI/experiments/stgcn/metrics/sample_predictions.csv)  
**Confusion Matrix:** [`experiments/stgcn/plots/confusion_matrix.png`](file:///d:/SignAI/experiments/stgcn/plots/confusion_matrix.png)  

---

## 1. Executive Summary

Of the 12 test sequences evaluated on the best ST-GCN checkpoint:
- **10 sequences were correctly classified ($83.33\%$)**.
- **2 sequences resulted in misclassifications ($16.67\%$)**.
- **0 High-Confidence Errors occurred** (neither error had confidence $\ge 0.50$).

This represents a major reduction in error rate compared to the Phase 3 Part 1 baseline model (which produced 7 errors and 1 high-confidence failure).

---

## 2. In-Depth Dissection of the Two Misclassifications

Both errors in the test set occurred on sequences performed by `signer_02`:

### Error 1: `seq_0052` — `HAPPY` $\to$ `TEACHER`
- **True Label:** `HAPPY` (Class 3)
- **Predicted Label:** `TEACHER` (Class 9)
- **Softmax Confidence:** **0.4530** (45.30%)
- **Runner-Up Probability:** `HAPPY` (0.3412 / 34.12%)
- **Margin:** $11.18\text{ percentage points}$
- **Signer / Quality:** `signer_02` | `ACCEPTABLE` (score: 0.6167)
- **Kinematic & Morphological Cause:**
  - In Indian Sign Language, `HAPPY` is executed by brushing the palm upward against the chest.
  - `TEACHER` is executed by extending open hands forward and downward from the chin/chest level.
  - Both signs share an overlapping spatial bounding volume in the upper chest. In `seq_0052`, `signer_02` performed `HAPPY` with a brisk vertical stroke that truncated early, causing the temporal convolution filters ($K_t=9$) to confuse the stroke termination with the downward release of `TEACHER`.
  - Crucially, the model's confidence was diffuse ($45.3\%$), correctly reflecting uncertainty between the two candidate glosses.

### Error 2: `seq_0058` — `TIME` $\to$ `TEACHER`
- **True Label:** `TIME` (Class 8)
- **Predicted Label:** `TEACHER` (Class 9)
- **Softmax Confidence:** **0.3677** (36.77%)
- **Runner-Up Probability:** `TIME` (0.3105 / 31.05%)
- **Margin:** $5.72\text{ percentage points}$
- **Signer / Quality:** `signer_02` | `ACCEPTABLE` (score: 0.5133)
- **Kinematic & Morphological Cause:**
  - `TIME` is an asymmetric, localized sign: the dominant index finger taps the non-dominant wrist twice.
  - In `seq_0058`, the non-dominant wrist remained stationary at the lower boundary of the camera frame, leading to intermittent landmark tracking dropouts on the non-dominant wrist.
  - Without a grounded wrist target, the localized tapping motion resembled the hand retraction stroke of `TEACHER`. The model produced a very low-confidence prediction ($36.77\%$), with `TIME` closely trailing at $31.05\%$.

---

## 3. High-Confidence Error Elimination

| Architecture | Total Errors | High-Confidence Errors ($\ge 0.50$) | High-Confidence Error Rate |
| :--- | :---: | :---: | :---: |
| **Baseline Model (BiGRU)** | 7 | 1 (`seq_0050`, Conf: 0.6444) | 14.3% of errors |
| **Graph Model (ST-GCN)** | **2** | **0** | **0.0% of errors** |

In ST-GCN, every single high-confidence prediction ($\ge 0.50$) was **100% correct**:
- `seq_0049` (`HELLO`): 0.9987 (Correct)
- `seq_0050` (`THANK_YOU`): 0.5516 (Correct)
- `seq_0051` (`GOOD`): 0.9929 (Correct)
- `seq_0053` (`MONDAY`): 0.7730 (Correct)
- `seq_0054` (`MONDAY`): 0.6963 (Correct)
- `seq_0055` (`CAR`): 0.9527 (Correct)
- `seq_0056` (`BIRD`): 0.6850 (Correct)
- `seq_0057` (`HOUSE`): 0.6067 (Correct)
- `seq_0059` (`TEACHER`): 0.5260 (Correct)
- `seq_0060` (`TEACHER`): 0.7827 (Correct)

---

## 4. Key Takeaways for Deployment & Post-Processing

1. **Confidence Thresholding Gating:**
   - Setting a prediction confidence gating threshold of $\theta_{\text{conf}} = 0.50$ (as specified in the Phase 1 blueprint) achieves **$100.0\%$ precision** on the test set: all 10 accepted predictions are true positives, while the 2 ambiguous predictions are routed to user clarification or rejected as unconfident.
2. **Targeted Data Augmentation for Subtle Signs:**
   - Signs like `TIME` and `HAPPY` benefit from temporal speed jittering ($0.8\times - 1.2\times$) and subtle vertical translation augmentation during training to prevent slight signing elevation shifts from causing confusion with `TEACHER`.
