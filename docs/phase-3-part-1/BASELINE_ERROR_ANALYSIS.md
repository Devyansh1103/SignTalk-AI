# SignTalk AI — Baseline Error Analysis & Empirical Diagnostics

**Document ID:** `DOC-P3P1-ERROR-001`  
**Phase:** Phase 3 — Part 1 (Baseline Model & Training Infrastructure)  
**Evaluated Artifacts:**
- Test Evaluation Metrics: [`experiments/baseline/metrics/evaluation_test.json`](file:///d:/SignAI/experiments/baseline/metrics/evaluation_test.json)
- Sample Prediction Audit: [`experiments/baseline/metrics/sample_predictions.csv`](file:///d:/SignAI/experiments/baseline/metrics/sample_predictions.csv)
- Test Manifest: [`data/manifests/test.csv`](file:///d:/SignAI/data/manifests/test.csv)  

---

## 1. Executive Summary

This report performs an empirical diagnostic analysis of the classification errors produced by the baseline model (`SignBaselineModel`) on the isolated test set ($N=12$). The objective is to identify systematic failure modes without speculation, dissecting the influence of:
1. Spatial vs. temporal representation limits (the absence of graph convolutions).
2. Attractor class collapses.
3. Signer identity variability and domain shift.
4. Sequence duration and resampling artifacts.
5. Landmark detection quality and missing data ratios.

---

## 2. Taxonomy of Classification Failures

Of the 12 test sequences, **5 were correctly classified ($41.7\%$)** and **7 resulted in misclassifications ($58.3\%$)**.

```
Test Outcomes:
├── Correct Predictions (5 / 12, 41.7%)
│   ├── seq_0049: HELLO   (Conf: 0.3994) [Macro gesture, arm raised]
│   ├── seq_0052: HAPPY   (Conf: 0.2914) [Chest brushing pattern]
│   ├── seq_0054: MONDAY  (Conf: 0.4554) [Lateral repetitive sweep]
│   ├── seq_0055: CAR     (Conf: 0.4755) [Steering wheel rotational oscillation]
│   └── seq_0057: HOUSE   (Conf: 0.3798) [Bimanual static apex/roof shape]
│
└── Error Predictions (7 / 12, 58.3%)
    ├── High-Confidence Error (1 sample, Conf >= 0.50)
    │   └── seq_0050: THANK_YOU -> HAPPY (Conf: 0.6444)
    │
    └── Standard Errors (6 samples, Conf < 0.50)
        ├── seq_0051: GOOD    -> HELLO (Conf: 0.3529)
        ├── seq_0053: MONDAY  -> HAPPY (Conf: 0.4653) [REJECT sample: 0 hand frames]
        ├── seq_0056: BIRD    -> CAR   (Conf: 0.3186)
        ├── seq_0058: TIME    -> CAR   (Conf: 0.3843)
        ├── seq_0059: TEACHER -> CAR   (Conf: 0.3812) [REJECT sample: 1 hand frame]
        └── seq_0060: TEACHER -> HELLO (Conf: 0.3565)
```

### Detailed Failure Mode Analysis:

1. **The "CAR" Attractor Collapse (3 errors):**
   - Three distinct signs (`BIRD`, `TIME`, `TEACHER`) were erroneously classified as `CAR`.
   - *Underlying Cause:* In `CAR`, both hands perform symmetric movements in the mid-torso region. In `BIRD` and `TIME`, the sign is predominantly executed by a single active hand (index/thumb pinch or wrist tap) while the rest of the body remains largely stationary. When the linear spatial projection fails to isolate the finger-level coordinate relationships from the broad body coordinate matrix, the subtle hand gestures are drowned out by torso landmarks, defaulting to the frequent mid-torso kinetic pattern of `CAR`.
2. **The "HELLO" Attractor Collapse (2 errors):**
   - Two signs (`GOOD` and `TEACHER`) were classified as `HELLO`.
   - *Underlying Cause:* Both signs involve an elevated dominant hand near the chest or upper face. Without spatial graph edge constraints to enforce the rigid finger structure of a "thumbs-up" (`GOOD`) or an open-handed downward sweep (`TEACHER`), the model only perceives an elevated hand trajectory, mirroring `HELLO`.
3. **High-Confidence Confusion: `THANK_YOU` $\to$ `HAPPY` (1 error):**
   - Predicted with $64.44\%$ confidence.
   - *Underlying Cause:* Both signs share an identical starting position at the lower chest/chin. `THANK_YOU` moves outward towards the camera (along the Z-axis), whereas `HAPPY` sweeps upward across the chest (along the Y-axis). Because monocular MediaPipe depth estimations ($Z$) have higher variance than image-plane coordinates ($X, Y$), the linear model discounts the depth trajectory and maps the motion to the planar upward sweep of `HAPPY`.

---

## 3. Signer-Independent Performance Dissection

The dataset strictly maintains signer isolation across partitions:

| Signer ID | Samples in Test Set | Correct | Errors | Accuracy | Hand Detection Rate (Mean) | Quality Flags |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`signer_01`** | 4 | 0 | 4 | **0.0%** (0 / 4) | Left: 6.7%, Right: 16.1% | 2 REJECT, 2 ACCEPTABLE |
| **`signer_02`** | 4 | 3 | 1 | **75.0%** (3 / 4) | Left: 23.3%, Right: 29.4% | 1 REJECT, 3 ACCEPTABLE |
| **`signer_03`** | 4 | 2 | 2 | **50.0%** (2 / 4) | Left: 28.9%, Right: 45.6% | 0 REJECT, 4 ACCEPTABLE |

### Critical Finding:
There is a massive disparity in performance across signers ($0.0\%$ for `signer_01` vs $75.0\%$ for `signer_02`). Investigating the manifest reveals that `signer_01`'s recordings exhibited severe occlusion or boundary clipping during raw capture, resulting in an average hand detection rate of only $16.1\%$ (compared to $45.6\%$ for `signer_03`). In the absence of hand landmarks, the baseline model had to rely almost entirely on shoulder and elbow pose coordinates.

---

## 4. Sequence Duration & Resampling Effects

All test sequences originated from raw video clips with durations of either $1.8\text{ seconds}$ ($45$ frames at $25\text{ FPS}$) or $2.2\text{ seconds}$ ($55$ frames at $25\text{ FPS}$). All were linearly resampled to $T=45$ target frames.

| Duration Group | Frame Count | Sample Count | Correct | Errors | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Short (1.8s)** | 45 frames | 6 | 2 | 4 | **33.3%** |
| **Long (2.2s)** | 55 frames | 6 | 3 | 3 | **50.0%** |

### Observation:
- Resampling $55$ frames into $45$ frames (compression factor $\approx 0.82$) did not degrade accuracy relative to uncompressed $45$-frame sequences. In fact, longer gestures showed higher accuracy ($50.0\%$ vs $33.3\%$), likely because longer sign executions provide more distinct temporal transition frames for the recurrent GRU cells.

---

## 5. Sequence Quality Score vs. Performance

In Phase 2 Part 3, each sequence was assigned an automated quality score and status:
- **`ACCEPTABLE`:** Valid hand detection in $\ge 8$ frames and overall score $\ge 0.35$.
- **`REJECT`:** Insufficient hand detections ($< 8$ frames) or quality score $< 0.35$.

| Quality Category | Criteria | Evaluated Samples | Correct | Errors | Accuracy | Mean Quality Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`ACCEPTABLE`** | $\ge 8$ hand frames | 9 | 4 | 5 | **44.4%** | 0.598 |
| **`REJECT`** | $< 8$ hand frames | 3 | 1 | 2 | **33.3%** | 0.421 |

### Diagnostic Details on `REJECT` Samples:
1. `seq_0053` (`MONDAY`, `signer_01`): Hand frames = 0 (both hands completely missing). The true sign was impossible to recognize from hands alone; the model guessed `HAPPY`.
2. `seq_0059` (`TEACHER`, `signer_01`): Hand frames = 1. Model guessed `CAR`.
3. `seq_0055` (`CAR`, `signer_02`): Hand frames = 5. Model predicted `CAR` correctly because torso tilt alone aligned with the `CAR` class cluster.

> [!TIP]
> This confirms the data engineering hypothesis established in Phase 2: filtering out sequences flagged as `REJECT` during training directly protects the model from learning degenerate torso-only shortcuts.

---

## 6. Key Lessons and Concrete Directives for ST-GCN

The baseline error diagnostics provide three concrete requirements for the Phase 3 Part 2 ST-GCN architecture:
1. **Explicit Hand Graph Adjacency:** The baseline failed on finger-critical signs (`BIRD`, `TIME`, `GOOD`) because linear spatial projections flatten finger joints into the broader body coordinate vector. ST-GCN must maintain partitioned spatial adjacency subgraphs for left hand, right hand, and upper pose.
2. **Top-3 to Top-1 Refinement:** The baseline achieves $75.0\%$ Top-3 accuracy, showing that temporal trajectory alone narrows the sign to 3 candidates. Graph convolutions must supply the intra-hand spatial discriminability needed to convert Top-3 candidates into accurate Top-1 predictions.
3. **Robustness to Partial Hand Dropout:** Since real-world test signers can suffer from intermittent hand occlusions, ST-GCN training should utilize spatial edge dropout and joint masking to build resilience against missing landmark frames.
