# SignTalk AI — Baseline Test Evaluation Results

**Document ID:** `DOC-P3P1-RESULTS-001`  
**Phase:** Phase 3 — Part 1 (Baseline Model & Training Infrastructure)  
**Model:** `SignBaselineModel` (2-Layer Bidirectional GRU)  
**Evaluated Checkpoint:** `experiments/baseline/checkpoints/best_checkpoint.pt` (Epoch 25)  
**Evaluation Set:** Isolated Test Manifest (`data/manifests/test.csv`, $N=12$ sequences)  
**Hardware:** Intel/AMD x86_64 CPU (PyTorch 2.13.0)  
**Evaluation Date:** October 2, 2026  

---

## 1. Executive Summary

This report documents the official, unadjusted experimental results obtained by evaluating the trained baseline model on the isolated test set. In accordance with the Phase 3 Evaluation Protocol ([`EVALUATION_PROTOCOL.md`](file:///d:/SignAI/docs/phase-3-part-1/EVALUATION_PROTOCOL.md)), the test partition was evaluated **exactly once** using the checkpoint selected by peak validation macro-F1 (`best_checkpoint.pt`). No hyperparameters were adjusted in response to test feedback.

```
========================================================================================
                              OFFICIAL BASELINE TEST METRICS
========================================================================================
Test Loss (Cross-Entropy):          1.8946
Top-1 Accuracy:                     0.4167 (41.67%)  [5 / 12 correct]
Top-3 Accuracy:                     0.7500 (75.00%)  [9 / 12 in top 3]
Macro Precision:                    0.2917 (29.17%)
Macro Recall:                       0.4500 (45.00%)
Macro F1-Score:                     0.3067 (30.67%)
Weighted F1-Score:                  0.3111 (31.11%)
Total Evaluated Samples:            12
========================================================================================
```

---

## 2. Per-Class Performance Breakdown

The table below details precision, recall, F1-score, and sample support across each of the 10 vocabulary classes:

| Class ID | Gloss Name | Precision | Recall | F1-Score | Support (N) | Performance Classification |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | `HELLO` | 0.3333 | 1.0000 | 0.5000 | 1 | Moderate Recognition (Broad attractor) |
| **1** | `THANK_YOU` | 0.0000 | 0.0000 | 0.0000 | 1 | Missed (Confused with HAPPY) |
| **2** | `GOOD` | 0.0000 | 0.0000 | 0.0000 | 1 | Missed (Confused with HELLO) |
| **3** | `HAPPY` | 0.3333 | 1.0000 | 0.5000 | 1 | Moderate Recognition (Broad attractor) |
| **4** | `MONDAY` | 1.0000 | 0.5000 | 0.6667 | 2 | Strong Recognition (Distinct trajectory) |
| **5** | `CAR` | 0.2500 | 1.0000 | 0.4000 | 1 | Moderate Recognition (Broad attractor) |
| **6** | `BIRD` | 0.0000 | 0.0000 | 0.0000 | 1 | Missed (Confused with CAR) |
| **7** | `HOUSE` | 1.0000 | 1.0000 | 1.0000 | 1 | Perfect Recognition (Bimanual roof shape) |
| **8** | `TIME` | 0.0000 | 0.0000 | 0.0000 | 1 | Missed (Confused with CAR) |
| **9** | `TEACHER` | 0.0000 | 0.0000 | 0.0000 | 2 | Missed (Confused with CAR and HELLO) |

---

## 3. Test Set Confusion Matrix

The 10-class test set confusion matrix (Rows = True Class, Columns = Predicted Class) is:

```
True \ Pred      HELLO   THANK_YOU   GOOD   HAPPY   MONDAY   CAR   BIRD   HOUSE   TIME   TEACHER
-------------------------------------------------------------------------------------------------
HELLO (0)          1         0        0       0       0       0      0       0      0       0
THANK_YOU (1)      0         0        0       1       0       0      0       0      0       0
GOOD (2)           1         0        0       0       0       0      0       0      0       0
HAPPY (3)          0         0        0       1       0       0      0       0      0       0
MONDAY (4)         0         0        0       1       1       0      0       0      0       0
CAR (5)            0         0        0       0       0       1      0       0      0       0
BIRD (6)           0         0        0       0       0       1      0       0      0       0
HOUSE (7)          0         0        0       0       0       0      0       1      0       0
TIME (8)           0         0        0       0       0       1      0       0      0       0
TEACHER (9)        1         0        0       0       0       1      0       0      0       0
-------------------------------------------------------------------------------------------------
Pred Totals:       3         0        0       2       1       4      0       1      0       0
```

Visual confusion matrix heatmap saved to:  
[`experiments/baseline/plots/confusion_matrix.png`](file:///d:/SignAI/experiments/baseline/plots/confusion_matrix.png)

---

## 4. Inference Latency & Throughput Benchmark

Inference performance was measured over 200 iterations on standard CPU hardware without GPU acceleration:

### Batch Size 1 (Simulated Streaming Real-Time Window)

| Pipeline Stage | Mean Latency | Median Latency | P95 Latency | Relative % |
| :--- | :--- | :--- | :--- | :--- |
| **Tensor Collation & Transfer** | 0.010 ms | 0.010 ms | 0.012 ms | 0.04% |
| **BiGRU Model Inference** | 24.357 ms | 24.498 ms | 28.962 ms | 99.49% |
| **Softmax & Top-1/Top-3 Post-Processing** | 0.117 ms | 0.057 ms | 0.077 ms | 0.47% |
| **Total Pipeline Latency** | **24.483 ms** | **24.606 ms** | **29.053 ms** | **100.0%** |
| **Inference Throughput** | **40.8 seq/sec** | — | — | — |

### Batch Size 8 (Batched Offline Inference)

| Metric | Measured Value | Unit |
| :--- | :--- | :--- |
| **Total Mean Latency per Batch** | **46.073** | ms |
| **Total Median Latency per Batch** | **40.457** | ms |
| **P95 Latency per Batch** | **67.906** | ms |
| **Batched Throughput** | **173.6** | sequences / sec |

> [!NOTE]
> The single-sequence inference latency of **24.48 ms** equates to ~40.8 FPS on a CPU. This confirms that the baseline model comfortably meets standard 30 FPS real-time processing constraints, providing a rapid execution reference.

---

## 5. Architectural Conclusions & Justification for ST-GCN

1. **Macro vs. Micro Kinematics:**
   - Signs with large, distinct bimanual spatial forms (`HOUSE` with both hands forming a sloped apex, F1=1.00) or unique repetitive lateral sweeps (`MONDAY`, F1=0.67) are captured effectively by the BiGRU.
   - Signs relying on subtle finger-joint configurations (`BIRD` with index-thumb pinching, `GOOD` with thumb extension, `TIME` with index tapping wrist) completely fail and collapse into generic motion attractors (`CAR`, `HELLO`).
2. **Top-3 Accuracy Headroom:**
   - Top-1 accuracy is $41.67\%$, but Top-3 accuracy jumps to **$75.00\%$** ($9 / 12$ samples). This proves that the recurrent network learns meaningful temporal dynamics that place the correct sign in the top rank candidates, but lacks the fine-grained spatial discrimination needed to isolate the top-1 prediction.
3. **The Empirical Case for ST-GCN:**
   - These findings establish the exact scientific baseline for Phase 3 Part 2. The ST-GCN architecture is specifically designed to overcome this precise bottleneck by constructing spatial adjacency graphs over finger phalanges and joints.
