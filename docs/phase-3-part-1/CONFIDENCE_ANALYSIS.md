# SignTalk AI — Prediction Confidence & Calibration Analysis

**Document ID:** `DOC-P3P1-CONFIDENCE-001`  
**Phase:** Phase 3 — Part 1 (Baseline Model & Training Infrastructure)  
**Model:** `SignBaselineModel` (2-Layer BiGRU)  
**Checkpoint:** `experiments/baseline/checkpoints/best_checkpoint.pt` (Epoch 25)  
**Evaluation Set:** Test Set ($N=12$ sequences, isolated signers)  
**Inspection Log:** [`experiments/baseline/metrics/sample_predictions.csv`](file:///d:/SignAI/experiments/baseline/metrics/sample_predictions.csv)  

---

## 1. Overview and Purpose

In real-time sign language recognition, raw softmax probabilities output by neural networks are frequently misaligned with true empirical classification accuracy. Standard cross-entropy minimization encourages overconfidence, especially on small vocabulary datasets.

This analysis inspects:
1. The distribution of softmax prediction probabilities on the isolated test set.
2. The prevalence of **High-Confidence Errors** (predictions with confidence $\ge 0.50$ that are incorrect).
3. The prevalence of **Low-Confidence Correct Predictions** (true signs correctly identified but with diffuse probability mass).
4. An empirical calculation of **Expected Calibration Error (ECE)** and reliability binning.

> [!IMPORTANT]
> The baseline model outputs raw softmax probabilities. These probabilities are **uncalibrated** and are presented here strictly for error diagnostic purposes. No post-hoc temperature scaling or Platt scaling has been applied in this phase.

---

## 2. Sample-by-Sample Confidence Log

The following table details the exact inference outputs produced by the baseline checkpoint across all 12 test sequences:

| Sequence ID | Signer ID | Quality Status | True Gloss | Predicted Gloss | Softmax Confidence | Outcome | Diagnostic Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `seq_0049` | signer_02 | ACCEPTABLE | HELLO | HELLO | **0.3994** | CORRECT | Low-Confidence Correct |
| `seq_0050` | signer_01 | ACCEPTABLE | THANK_YOU | HAPPY | **0.6444** | FAIL | **High-Confidence Error** |
| `seq_0051` | signer_03 | ACCEPTABLE | GOOD | HELLO | **0.3529** | FAIL | Low-Confidence Error |
| `seq_0052` | signer_02 | ACCEPTABLE | HAPPY | HAPPY | **0.2914** | CORRECT | Low-Confidence Correct |
| `seq_0053` | signer_01 | REJECT | MONDAY | HAPPY | **0.4653** | FAIL | Low-Confidence Error |
| `seq_0054` | signer_03 | ACCEPTABLE | MONDAY | MONDAY | **0.4554** | CORRECT | Low-Confidence Correct |
| `seq_0055` | signer_02 | REJECT | CAR | CAR | **0.4755** | CORRECT | Low-Confidence Correct |
| `seq_0056` | signer_01 | ACCEPTABLE | BIRD | CAR | **0.3186** | FAIL | Low-Confidence Error |
| `seq_0057` | signer_03 | ACCEPTABLE | HOUSE | HOUSE | **0.3798** | CORRECT | Low-Confidence Correct |
| `seq_0058` | signer_02 | ACCEPTABLE | TIME | CAR | **0.3843** | FAIL | Low-Confidence Error |
| `seq_0059` | signer_01 | REJECT | TEACHER | CAR | **0.3812** | FAIL | Low-Confidence Error |
| `seq_0060` | signer_03 | ACCEPTABLE | TEACHER | HELLO | **0.3565** | FAIL | Low-Confidence Error |

---

## 3. High-Confidence Error Investigation

Across the 12 evaluation samples, exactly **1 High-Confidence Error** occurred:
- **Sample:** `seq_0050`
- **Signer:** `signer_01`
- **True Label:** `THANK_YOU` (Class 1)
- **Predicted Label:** `HAPPY` (Class 3)
- **Softmax Confidence:** **0.6444** (64.44%)
- **Next Runner-Up:** `THANK_YOU` (0.1821)

### Visual and Kinematic Breakdown:
- The sign `THANK_YOU` in Indian Sign Language involves an open flat palm extending forward from the chin/chest level.
- The sign `HAPPY` involves an open hand patting or brushing upward across the chest.
- Because the baseline model flattens all 93 spatial landmarks into a 279-dimensional vector without preserving the structural edge topology of the hand joints, it struggles to differentiate a forward-extending wrist trajectory (`THANK_YOU`) from an upward-brushing chest trajectory (`HAPPY`). The dominant spatial motion in both signs involves high chest-region activity, causing the recurrent hidden state to converge decisively on `HAPPY`.

---

## 4. Confidence Calibration & Reliability Analysis

To evaluate model calibration, predictions are binned by confidence into 3 intervals: $[0.2, 0.4)$, $[0.4, 0.6)$, and $[0.6, 0.8)$.

### Calibration Bin Table

| Confidence Bin $B_m$ | Bin Range | Sample Count ($|B_m|$) | Accuracy $\text{acc}(B_m)$ | Mean Confidence $\text{conf}(B_m)$ | Absolute Calibration Gap $|\text{acc} - \text{conf}|$ | Weight ($|B_m| / N$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bin 1** | $[0.2, 0.4)$ | 8 | $3 / 8 = 0.3750$ (37.5%) | 0.3580 (35.8%) | **0.0170** (1.70%) | 0.6667 |
| **Bin 2** | $[0.4, 0.6)$ | 3 | $2 / 3 = 0.6667$ (66.7%) | 0.4654 (46.5%) | **0.2013** (20.13%) | 0.2500 |
| **Bin 3** | $[0.6, 0.8)$ | 1 | $0 / 1 = 0.0000$ (0.0%) | 0.6444 (64.4%) | **0.6444** (64.44%) | 0.0833 |

### Expected Calibration Error (ECE) Formula:
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

$$\text{ECE} = (0.6667 \times 0.0170) + (0.2500 \times 0.2013) + (0.0833 \times 0.6444)$$
$$\text{ECE} = 0.01133 + 0.05033 + 0.05370 = \mathbf{0.1154} \quad (\mathbf{11.54\%})$$

---

## 5. Key Findings and Implications

1. **Diffuse Probabilities on Unambiguous Signs:**
   - Correctly classified signs (`HELLO`, `HAPPY`, `MONDAY`, `CAR`, `HOUSE`) had confidence scores between $0.29$ and $0.48$. Given a 10-class random uniform baseline of $0.10$, a confidence of $0.35$ represents clear non-random concentration, but reveals significant uncertainty across candidate classes.
2. **Attractor Class Dynamics:**
   - Misclassifications disproportionately landed on three broad macro-motion classes: `CAR` (3 errors), `HELLO` (2 errors), and `HAPPY` (2 errors).
   - In total, 7 of the 12 samples were classified as either `CAR` or `HAPPY` or `HELLO`. This indicates that when the temporal GRU cannot extract decisive landmark trajectory nuances, it defaults to the highest-variance kinetic classes in the training partition.
3. **No Temperature Scaling Recommended for Baseline:**
   - Since the baseline is strictly an empirical point of comparison for the upcoming ST-GCN, temperature calibration or loss scaling would artificially alter baseline behavior. The uncalibrated ECE of $11.54\%$ serves as the reference benchmark for Phase 3 Part 2.
