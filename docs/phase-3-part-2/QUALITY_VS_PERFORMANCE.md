# SignTalk AI — Sequence Quality vs. Performance Analysis (ST-GCN)

**Document ID:** `DOC-P3P2-ANALYSIS-QUAL-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Evaluated Manifest:** [`data/manifests/test.csv`](file:///d:/SignAI/data/manifests/test.csv)  
**Prediction Source:** [`experiments/stgcn/metrics/sample_predictions.csv`](file:///d:/SignAI/experiments/stgcn/metrics/sample_predictions.csv)  

---

## 1. Overview and Hypotheses

During Phase 2 Part 2, every landmark sequence was evaluated by automated quality filtering heuristics:
- **`ACCEPTABLE`:** Valid hand detection in $\ge 8$ frames and overall tracking score $\ge 0.35$.
- **`REJECT`:** Severe occlusion, marginal tracking, or active hand detection in $< 8$ frames.

The baseline model in Phase 3 Part 1 struggled on `REJECT` samples ($33.3\%$ accuracy), as vectorizing the coordinates caused the model to lose orientation when hand coordinates vanished.

This analysis evaluates how the spatiotemporal graph network handles sequences with differing landmark detection quality.

---

## 2. Empirical Performance Comparison by Quality Tier

| Quality Status | Defining Criteria | Sample Count ($N$) | ST-GCN Correct | ST-GCN Accuracy | Baseline Accuracy | Measured $\Delta$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`ACCEPTABLE`** | Hand frames $\ge 8$, Quality $\ge 0.35$ | 9 | 7 | **77.8%** (7 / 9) | 44.4% (4 / 9) | **+33.4 pp** |
| **`REJECT`** | Hand frames $< 8$, Quality $< 0.35$ | 3 | 3 | **100.0%** (3 / 3) | 33.3% (1 / 3) | **+66.7 pp** |

---

## 3. Detailed Inspection of Marginal Quality Samples

The three test sequences flagged as `REJECT` were:

1. **`seq_0053` (`MONDAY`, `signer_01`):**
   - *Landmark Issue:* Active hand frames $= 0$ (both hands dropped due to motion blur).
   - *Baseline Result:* Failed completely (predicted `HAPPY`).
   - *ST-GCN Result:* **CORRECT (`MONDAY`, Confidence: 0.7730)**.
   - *Kinematic Mechanism:* While hand coordinates were zeroed out, the upper pose arm skeleton (nodes 47-52: shoulders, elbows, and wrists) captured the rhythmic lateral oscillation characteristic of `MONDAY`. The spatiotemporal graph convolutions successfully propagated these arm kinematics into the classification head.
2. **`seq_0055` (`CAR`, `signer_02`):**
   - *Landmark Issue:* Hand frames $= 5$ (partial tracking).
   - *Baseline Result:* Correct (`CAR`, Confidence: 0.4755).
   - *ST-GCN Result:* **CORRECT (`CAR`, Confidence: 0.9527)**.
   - *Kinematic Mechanism:* Confident bimanual steering motion modeled across the shoulder and elbow joints.
3. **`seq_0059` (`TEACHER`, `signer_01`):**
   - *Landmark Issue:* Hand frames $= 1$ (near-total hand loss).
   - *Baseline Result:* Failed (predicted `CAR`).
   - *ST-GCN Result:* **CORRECT (`TEACHER`, Confidence: 0.5260)**.
   - *Kinematic Mechanism:* Downward arm sweeping trajectory resolved via elbow and shoulder joints.

---

## 4. Key Takeaways

1. **Kinematic Redundancy:** The full 93-node multimodal graph provides anatomical redundancy. When distal finger articulators are momentarily occluded, the proximal arm skeleton (elbows and shoulders) maintains coarse gesture trajectory.
2. **Robustness Over Baseline:** ST-GCN demonstrated far higher resilience to missing hand landmarks than the flattened baseline model, achieving 3 / 3 accuracy on the degraded test sequences.
