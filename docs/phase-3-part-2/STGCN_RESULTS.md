# SignTalk AI — Official ST-GCN Measured Results Report

**Document ID:** `DOC-P3P2-RESULTS-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Evaluation Set:** Isolated Test Manifest (`data/manifests/test.csv`, $N=12$)  
**Evaluated Checkpoint:** `experiments/stgcn/checkpoints/best_checkpoint.pt` (Epoch 34)  
**Hardware Environment:** Intel/AMD x86_64 Multi-Core CPU (PyTorch 2.13.0)  
**Evaluation Date:** October 2, 2026  

---

## 1. Verified Experimental Measurements

All metrics reported below were experimentally measured under the strict zero-leakage test protocol without post-hoc tuning or estimation:

| Specification / Metric | Empirically Measured Value |
| :--- | :--- |
| **Dataset Release Version** | `signTalk-seq-v1.0.0` |
| **Landmark Schema Version** | `93-node-v1` (21 LH, 21 RH, 11 Pose, 40 Face) |
| **Vocabulary Size ($K$)** | 10 Classes (`mvp_10.json`) |
| **ST-GCN Configuration** | 6 Blocks ($[64, 64, 128, 128, 256, 256]$), $K=3$ Spatial Configuration |
| **Total Model Parameters** | **2,137,818** |
| **Trainable Parameters** | **2,137,818** (100.0%) |
| **Non-Trainable Parameters**| **0** (0.0%) |
| **Checkpoint File Size** | **25.88 MB** ($25,876,189$ bytes) |
| **Training Epochs Completed**| **54 Epochs** (Early stopping triggered, patience 20) |
| **Best Validation Epoch** | **Epoch 34** |
| **Best Validation Accuracy** | **0.9167** (91.67%) |
| **Best Validation Macro F1** | **0.9333** (93.33%) |
| **Test Cross-Entropy Loss** | **0.6645** |
| **Test Top-1 Accuracy** | **0.8333** (83.33%, 10 / 12 correct) |
| **Test Top-3 Accuracy** | **0.9167** (91.67%, 11 / 12 in top 3) |
| **Test Macro Precision** | **0.7500** (75.00%) |
| **Test Macro Recall** | **0.8000** (80.00%) |
| **Test Macro F1-Score** | **0.7667** (76.67%) |
| **Test Weighted F1-Score** | **0.7778** (77.78%) |
| **CPU Latency (B=1, Mean)** | **105.884 ms** |
| **CPU Latency (B=1, Median)**| **89.905 ms** |
| **CPU Latency (B=1, P95)** | **166.936 ms** |
| **CPU Throughput (B=1)** | **9.4 sequences / second** |
| **High-Confidence Errors** | **0 (Zero)** |

---

## 2. Per-Class Test Performance Table

| Class ID | Gloss Name | Precision | Recall | F1-Score | Support | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | `HELLO` | 1.0000 | 1.0000 | 1.0000 | 1 | Perfect Recognition |
| **1** | `THANK_YOU` | 1.0000 | 1.0000 | 1.0000 | 1 | Perfect Recognition |
| **2** | `GOOD` | 1.0000 | 1.0000 | 1.0000 | 1 | Perfect Recognition |
| **3** | `HAPPY` | 0.0000 | 0.0000 | 0.0000 | 1 | Missed (Confused with TEACHER) |
| **4** | `MONDAY` | 1.0000 | 1.0000 | 1.0000 | 2 | Perfect Recognition (2 / 2) |
| **5** | `CAR` | 1.0000 | 1.0000 | 1.0000 | 1 | Perfect Recognition |
| **6** | `BIRD` | 1.0000 | 1.0000 | 1.0000 | 1 | Perfect Recognition |
| **7** | `HOUSE` | 1.0000 | 1.0000 | 1.0000 | 1 | Perfect Recognition |
| **8** | `TIME` | 0.0000 | 0.0000 | 0.0000 | 1 | Missed (Confused with TEACHER) |
| **9** | `TEACHER` | 0.5000 | 1.0000 | 0.6667 | 2 | Detected (2 / 2, 2 false positives) |

---

## 3. Confusion Matrix (Test Set)

```
True \ Pred      HELLO   THANK_YOU   GOOD   HAPPY   MONDAY   CAR   BIRD   HOUSE   TIME   TEACHER
-------------------------------------------------------------------------------------------------
HELLO (0)          1         0        0       0       0       0      0       0      0       0
THANK_YOU (1)      0         1        0       0       0       0      0       0      0       0
GOOD (2)           0         0        1       0       0       0      0       0      0       0
HAPPY (3)          0         0        0       0       0       0      0       0      0       1
MONDAY (4)         0         0        0       0       2       0      0       0      0       0
CAR (5)            0         0        0       0       0       1      0       0      0       0
BIRD (6)           0         0        0       0       0       0      1       0      0       0
HOUSE (7)          0         0        0       0       0       0      0       1      0       0
TIME (8)           0         0        0       0       0       0      0       0      0       1
TEACHER (9)        0         0        0       0       0       0      0       0      0       2
-------------------------------------------------------------------------------------------------
Pred Totals:       1         1        1       0       2       1      1       1      0       4
```
Visual heatmap: [`experiments/stgcn/plots/confusion_matrix.png`](file:///d:/SignAI/experiments/stgcn/plots/confusion_matrix.png)
Training curves: [`experiments/stgcn/plots/training_curves.png`](file:///d:/SignAI/experiments/stgcn/plots/training_curves.png)
Evaluation JSON: [`experiments/stgcn/metrics/evaluation_test.json`](file:///d:/SignAI/experiments/stgcn/metrics/evaluation_test.json)
