# SignTalk AI — Empirical Baseline vs. ST-GCN Comparative Report

**Document ID:** `DOC-P3P2-COMPARISON-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Evaluation Set:** Isolated Test Manifest (`data/manifests/test.csv`, $N=12$ sequences)  
**Evaluated Checkpoints:**
- Baseline: `experiments/baseline/checkpoints/best_checkpoint.pt` (Epoch 25)
- ST-GCN: `experiments/stgcn/checkpoints/best_checkpoint.pt` (Epoch 34)  
**Hardware Environment:** Intel/AMD x86_64 CPU (PyTorch 2.13.0)  

---

## 1. Executive Summary & Objective Metrics Table

This report documents the empirical comparison between the Phase 3 Part 1 temporal baseline (`SignBaselineModel`) and the Phase 3 Part 2 Spatial-Temporal Graph Convolutional Network (`SignSTGCN`). Both models were trained, validated, and tested under the exact same data partitions, random seed (42), and isolated test evaluation protocol.

| Evaluation Metric | Baseline Model (`SignBaselineModel`) | Graph Model (`SignSTGCN`) | Absolute Difference ($\Delta$) | Relative Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Architectural Family** | 2-Layer Bidirectional GRU | 6-Block ST-GCN ($K=3$) | Structural Shift | — |
| **Spatial Graph Adjacency** | None (Flattened $[B, T, 279]$) | Physical Bones ($[B, 3, T, 93]$) | Inductive Bias Added | — |
| **Total Parameter Count** | **597,898** | **2,137,818** | $+1,539,920$ | $3.58\times$ |
| **Trainable Parameters** | **597,898** (100.0%) | **2,137,818** (100.0%) | $+1,539,920$ | $3.58\times$ |
| **Checkpoint Size** | **7.21 MB** ($7,209,887$ bytes) | **25.88 MB** ($25,876,189$ bytes) | $+18.67\text{ MB}$ | $3.59\times$ |
| **Test Top-1 Accuracy** | **0.4167** (41.67%) | **0.8333** (83.33%) | **+41.66 pp** | **2.00×** |
| **Test Top-3 Accuracy** | **0.7500** (75.00%) | **0.9167** (91.67%) | **+16.67 pp** | **1.22×** |
| **Test Macro Precision** | **0.2917** (29.17%) | **0.7500** (75.00%) | **+45.83 pp** | **2.57×** |
| **Test Macro Recall** | **0.4500** (45.00%) | **0.8000** (80.00%) | **+35.00 pp** | **1.78×** |
| **Test Macro F1-Score** | **0.3067** (30.67%) | **0.7667** (76.67%) | **+46.00 pp** | **2.50×** |
| **Test Weighted F1-Score** | **0.3111** (31.11%) | **0.7778** (77.78%) | **+46.67 pp** | **2.50×** |
| **Test Cross-Entropy Loss** | **1.8946** | **0.6645** | **-1.2301** | **0.35×** |
| **CPU Latency (B=1, Mean)** | **24.483 ms** | **105.884 ms** | $+81.401\text{ ms}$ | $4.33\times$ |
| **CPU Latency (B=1, Median)**| **24.606 ms** | **89.905 ms** | $+65.299\text{ ms}$ | $3.65\times$ |
| **CPU Latency (B=1, P95)** | **29.053 ms** | **166.936 ms** | $+137.883\text{ ms}$ | $5.75\times$ |
| **CPU Throughput (B=1)** | **40.8 seq/sec** | **9.4 seq/sec** | $-31.4\text{ seq/sec}$ | $0.23\times$ |
| **High-Confidence Errors** | **1 sample** (Conf: 0.6444) | **0 samples** | $-1\text{ error}$ | Complete elimination |

---

## 2. Per-Class Performance Comparison

The table below contrasts the per-class F1-scores achieved by both architectures on the identical test samples:

| Class ID | Gloss Name | Baseline F1 | ST-GCN F1 | Absolute $\Delta$ | Measured Discrepancy Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | `HELLO` | 0.5000 | **1.0000** | $+0.5000$ | Baseline suffered false-positive attractor inflation; ST-GCN achieved perfect precision and recall. |
| **1** | `THANK_YOU` | 0.0000 | **1.0000** | $+1.0000$ | Baseline failed completely (confused with HAPPY); ST-GCN correctly resolved the forward extension. |
| **2** | `GOOD` | 0.0000 | **1.0000** | $+1.0000$ | Baseline failed completely (confused with HELLO); ST-GCN resolved thumb articulation with 99.29% confidence. |
| **3** | `HAPPY` | 0.5000 | **0.0000** | $-0.5000$ | Single test sample misclassified by ST-GCN as TEACHER. |
| **4** | `MONDAY` | 0.6667 | **1.0000** | $+0.3333$ | Baseline missed the REJECT-quality sample; ST-GCN correctly recognized both test samples (2/2). |
| **5** | `CAR` | 0.4000 | **1.0000** | $+0.6000$ | Baseline suffered severe false-positive attractor inflation (4 predictions); ST-GCN isolated true CAR. |
| **6** | `BIRD` | 0.0000 | **1.0000** | $+1.0000$ | Baseline failed completely (confused with CAR); ST-GCN resolved index-thumb pinching. |
| **7** | `HOUSE` | 1.0000 | **1.0000** | $0.0000$ | Both architectures recognized bimanual roof geometry perfectly. |
| **8** | `TIME` | 0.0000 | **0.0000** | $0.0000$ | Single test sample confused with TEACHER by ST-GCN. |
| **9** | `TEACHER` | 0.0000 | **0.6667** | $+0.6667$ | Baseline failed both samples (0/2); ST-GCN correctly classified both samples (2/2 detected). |

---

## 3. Detailed Error Pattern Contrasts

1. **Resolution of Subtle Hand Configurations:**
   - On the baseline model, three distinct fine-grained signs (`GOOD`, `BIRD`, `THANK_YOU`) had F1-scores of $0.0000$.
   - On ST-GCN, all three signs achieved **$1.0000$ F1-score**. The inclusion of spatial graph convolutions over the 21 hand nodes allowed the network to differentiate subtle finger joint angle patterns that were drowned out by the baseline's flattened projection.
2. **Attractor Class Dissolution:**
   - In the baseline model, `CAR` acted as an attractor for 4 out of 12 test predictions (3 false positives).
   - In ST-GCN, `CAR` was predicted exactly once (1 true positive, 0 false positives, 100% precision).
3. **High-Confidence Error Elimination:**
   - The baseline exhibited 1 high-confidence failure (`seq_0050`, `THANK_YOU` $\to$ `HAPPY`, confidence $0.6444$).
   - ST-GCN produced **0 high-confidence errors** across the test set. All correct predictions had high confidence ($52.6\%$ to $99.9\%$), while the 2 misclassifications were low-confidence ($36.8\%$ and $45.3\%$).

---

## 4. Latency vs. Accuracy Trade-Off Analysis

- **Accuracy & F1 Advantage:** ST-GCN achieved an increase of **$+41.66\text{ percentage points}$** in top-1 accuracy ($83.33\%$ vs $41.67\%$) and **$+46.00\text{ percentage points}$** in macro F1 ($76.67\%$ vs $30.67\%$).
- **Computational Cost:** ST-GCN latency is **$105.88\text{ ms}$** per sequence on a CPU compared to **$24.48\text{ ms}$** for the baseline model.
- **Application Viability:** At $\sim 106\text{ ms}$ on CPU, ST-GCN executes in under $120\text{ ms}$, comfortably fitting within human conversational real-time response budgets ($< 250\text{ ms}$) even before GPU acceleration or ONNX Runtime quantization.
