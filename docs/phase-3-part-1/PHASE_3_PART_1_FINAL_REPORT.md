# Phase 3 Part 1 Final Report

**Project:** SignTalk AI — Real-Time Indian Sign Language Translation and Communication Platform  
**Document ID:** `REPORT-P3P1-FINAL-001`  
**Phase:** Phase 3 — Part 1 (Baseline Model & Training Infrastructure)  
**Author:** Lead Computer Vision, ML Data, and Research Engineering Team  
**Date of Completion:** October 2, 2026  
**Status:** COMPLETE & AUDITED  

---

## 1. Objective

The primary objective of Phase 3 Part 1 was to establish a scientifically sound, reproducible machine-learning baseline and a modular training/evaluation infrastructure before developing the final Spatial-Temporal Graph Convolutional Network (ST-GCN) architecture.

The baseline answers the foundational research question:
> *"How well can Indian Sign Language sequences be recognized purely using temporal recurrent sequence modeling without spatial graph convolutions?"*

This phase guarantees an unbiased, quantitative baseline reference ($41.67\%$ Top-1 Accuracy, $0.3067$ Macro-F1) against which future graph-convolutional representations will be evaluated.

---

## 2. Dataset Version

The experiments in this phase were executed exclusively on the frozen sequence dataset constructed in Phase 2 Part 3:
- **Dataset Release Identifier:** `signTalk-seq-v1.0.0`
- **Landmark Topology Schema:** `93-node-v1` (Pose: 33, Left Hand: 21, Right Hand: 21, Face Outline: 18)
- **Coordinate Normalization:** Torso-scaled, mid-hip centered, invariant to camera distance
- **Total Sequence Count:** 60 standardized sequence archives (`.npz`)
- **Partitions:**
  - **Train:** 36 sequences (`data/manifests/train.csv`)
  - **Validation:** 12 sequences (`data/manifests/val.csv`)
  - **Test:** 12 sequences (`data/manifests/test.csv`)
- **Signer Isolation:** Strict zero-leakage partition assignment across `signer_01`, `signer_02`, and `signer_03`.
- **Vocabulary:** 10 core MVP sign classes (`assets/vocabularies/mvp_10.json`):
  1. `HELLO`
  2. `THANK_YOU`
  3. `GOOD`
  4. `HAPPY`
  5. `MONDAY`
  6. `CAR`
  7. `BIRD`
  8. `HOUSE`
  9. `TIME`
  10. `TEACHER`

---

## 3. Baseline Architecture

The primary baseline model implemented is the **2-Layer Bidirectional Gated Recurrent Unit (BiGRU) with Linear Spatial Projection** (`SignBaselineModel`, defined in [`src/models/baseline.py`](file:///d:/SignAI/src/models/baseline.py)):

1. **Spatial Linear Projection:** Flattens spatial coordinates $C \times V = 3 \times 93 = 279$ and projects each frame into a $128$-dimensional latent embedding via Linear + LayerNorm + Dropout ($p=0.3$).
2. **Temporal Bidirectional Recurrence:** 2 stacked BiGRU layers with hidden state dimension $128$ in each direction ($256$ bidirectional channels per time step).
3. **Dual Temporal Pooling:** Concatenates global temporal average pooling ($256$) and global temporal maximum pooling ($256$), producing a robust $512$-dimensional fixed-length representation.
4. **Classifier Head:** Multi-layer perceptron consisting of Linear($512 \to 128$), LayerNorm, ReLU, Dropout ($p=0.3$), and Linear($128 \to 10$) producing raw class logits.
5. **Total Trainable Parameters:** **597,898**.

---

## 4. Input Representation

The baseline strictly ingests the finalized Phase 2 sequence tensor format:
- **Tensor Shape:** $[B, C, T, V] = [B, 3, 45, 93]$
  - $B$: Batch size
  - $C = 3$: Normalized Cartesian coordinates $[x, y, z]$
  - $T = 45$: Temporal sequence frames (fixed $1.8\text{s}$ window at $25\text{ FPS}$)
  - $V = 93$: Anatomical landmark nodes
- **Internal Conversion:** Inside the model forward pass, tensors are permuted and reshaped to $[B, T, C \cdot V] = [B, 45, 279]$ without altering disk storage or dataset loaders.
- **Validity Mask:** Optional boolean mask $[B, 1, T, V]$ is accepted and forwarded to support zero-padded sequence masking.

---

## 5. Training Configuration

The experiment was governed by [`configs/baseline.yaml`](file:///d:/SignAI/configs/baseline.yaml):
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$)
- **Initial Learning Rate:** $1.0 \times 10^{-3}$
- **Weight Decay:** $1.0 \times 10^{-4}$
- **Learning Rate Scheduler:** CosineAnnealingLR ($T_{\max} = 100, \eta_{\min} = 1.0 \times 10^{-5}$)
- **Gradient Clipping:** Max norm $1.0$
- **Batch Size:** 8
- **Maximum Epochs:** 100
- **Early Stopping:** Patience 15 epochs monitored on `val_macro_f1`
- **Random Seed:** 42

---

## 6. Hardware

- **Operating System:** Windows 11 AMD64
- **Processor:** Intel / AMD x86_64 Multi-Core CPU
- **Accelerator Detection:** CUDA not available in current host process; automatic fallback to CPU successfully triggered via [`src/utils/device.py`](file:///d:/SignAI/src/utils/device.py).
- **Execution Runtime:** Python 3.13.12 (Miniconda), PyTorch 2.13.0+cpu.
- **Total Training Duration:** 32.29 seconds (40 epochs).

---

## 7. Training Procedure

1. Deterministic seeds initialized across Python, NumPy, and PyTorch via `set_seed(42)`.
2. Model parameters initialized with Xavier uniform linear weights and orthogonal recurrent transition matrices.
3. Multi-class Cross-Entropy Loss minimized with batch shuffling enabled.
4. Gradients clipped at norm $1.0$ to prevent recurrent gradient explosions.
5. Learning rate adjusted after every epoch according to the cosine decay schedule.
6. Checkpoint updated at each new validation macro-F1 peak.
7. Training terminated automatically at epoch 40 due to early stopping (peak at epoch 25).

---

## 8. Validation Procedure

Validation was executed at the conclusion of every epoch on the isolated validation split ($N=12$ sequences from `data/manifests/val.csv`):
- Data augmentation disabled (`augment=False`).
- Loss, Top-1 accuracy, and macro-F1 monitored without gradient computation (`torch.no_grad()`).
- Best validation performance achieved at **Epoch 25**:
  - Validation Loss: **1.7766**
  - Validation Macro F1: **0.2667**
  - Validation Top-1 Accuracy: **0.3333**
- Checkpoints persisted to `experiments/baseline/checkpoints/best_checkpoint.pt`.

---

## 9. Test Procedure

In compliance with [`EVALUATION_PROTOCOL.md`](file:///d:/SignAI/docs/phase-3-part-1/EVALUATION_PROTOCOL.md):
- The test partition (`data/manifests/test.csv`, $N=12$) was completely isolated during training and hyperparameter selection.
- Evaluated **exactly once** using `best_checkpoint.pt`.
- No hyperparameter tuning, loss weight modifications, or architecture tweaks were made in response to test set results.

---

## 10. Metrics

The evaluation engine ([`src/evaluation/classification_metrics.py`](file:///d:/SignAI/src/evaluation/classification_metrics.py)) computes:
- Multi-class Accuracy (Top-1 and Top-3)
- Macro Precision, Recall, and F1-Score (unweighted class mean)
- Weighted F1-Score (weighted by class support)
- Confusion Matrix (integer count and row-normalized percentages)
- Sample-level Softmax Confidence probabilities

---

## 11. Baseline Results

Measured performance on the isolated test set ($N=12$):

| Metric | Measured Value | Percentage |
| :--- | :--- | :--- |
| **Test Cross-Entropy Loss** | **1.8946** | — |
| **Top-1 Accuracy** | **0.4167** | **41.67%** (5 / 12 correct) |
| **Top-3 Accuracy** | **0.7500** | **75.00%** (9 / 12 in top 3) |
| **Macro Precision** | **0.2917** | **29.17%** |
| **Macro Recall** | **0.4500** | **45.00%** |
| **Macro F1-Score** | **0.3067** | **30.67%** |
| **Weighted F1-Score** | **0.3111** | **31.11%** |

### Per-Class Test Breakdown:

| Class ID | Gloss | Precision | Recall | F1-Score | Support | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 0 | `HELLO` | 0.3333 | 1.0000 | 0.5000 | 1 | Recognized |
| 1 | `THANK_YOU` | 0.0000 | 0.0000 | 0.0000 | 1 | Missed |
| 2 | `GOOD` | 0.0000 | 0.0000 | 0.0000 | 1 | Missed |
| 3 | `HAPPY` | 0.3333 | 1.0000 | 0.5000 | 1 | Recognized |
| 4 | `MONDAY` | 1.0000 | 0.5000 | 0.6667 | 2 | Recognized |
| 5 | `CAR` | 0.2500 | 1.0000 | 0.4000 | 1 | Recognized |
| 6 | `BIRD` | 0.0000 | 0.0000 | 0.0000 | 1 | Missed |
| 7 | `HOUSE` | 1.0000 | 1.0000 | 1.0000 | 1 | Perfect (1.00) |
| 8 | `TIME` | 0.0000 | 0.0000 | 0.0000 | 1 | Missed |
| 9 | `TEACHER` | 0.0000 | 0.0000 | 0.0000 | 2 | Missed |

---

## 12. Confusion Analysis

Test confusion analysis revealed two primary structural attractor classes:
1. **`CAR` Attractor:** 4 samples predicted as `CAR` (1 true `CAR`, 1 `BIRD`, 1 `TIME`, 1 `TEACHER`). Because `CAR` contains high-energy bimanual oscillation at chest level, unstructured linear projections aggregate all general mid-body hand movements into this cluster.
2. **`HELLO` Attractor:** 3 samples predicted as `HELLO` (1 true `HELLO`, 1 `GOOD`, 1 `TEACHER`). Upper-body hand elevations without structural finger connectivity defaulted to the single-hand elevated wave pattern of `HELLO`.
3. **`HOUSE` Isolation:** `HOUSE` achieved 100% precision and recall (F1=1.00) because its bimanual static roof configuration creates a unique bilateral spatial coordinate signature easily resolved even by flattened projections.

---

## 13. Error Analysis

Sample-level inspection ([`BASELINE_ERROR_ANALYSIS.md`](file:///d:/SignAI/docs/phase-3-part-1/BASELINE_ERROR_ANALYSIS.md)) revealed:
- **High-Confidence Error:** Exactly 1 sample (`seq_0050`, `THANK_YOU` $\to$ `HAPPY`, confidence 0.6444) due to depth trajectory ambiguity along the camera optical axis.
- **Signer Disparity:** Test accuracy was $0.0\%$ for `signer_01` (due to low raw landmark coverage of $16.1\%$), $75.0\%$ for `signer_02`, and $50.0\%$ for `signer_03`.
- **Quality Flags:** `ACCEPTABLE` sequences achieved $44.4\%$ accuracy, while `REJECT` sequences (where hand detections were missing) achieved $33.3\%$ accuracy.
- **Calibration Error:** Expected Calibration Error (ECE) was **$11.54\%$**.

---

## 14. Model Size

From [`MODEL_SIZE_REPORT.md`](file:///d:/SignAI/docs/phase-3-part-1/MODEL_SIZE_REPORT.md):
- **Total Parameters:** **597,898**
- **Trainable Parameters:** **597,898** (100.0%)
- **Model Weights (fp32):** **2.28 MB**
- **Checkpoint Footprint (`best_checkpoint.pt`):** **7.21 MB** ($7,209,887$ bytes, including model weights, AdamW optimizer moments, scheduler state, and configuration dictionary).

---

## 15. Inference Benchmark

Benchmarked over 200 iterations on CPU ([`BASELINE_RESULTS.md`](file:///d:/SignAI/docs/phase-3-part-1/BASELINE_RESULTS.md)):
- **Batch Size 1 (Real-Time Stream Window):**
  - Preprocessing Latency: Mean $0.010$ ms (P95: $0.012$ ms)
  - BiGRU Inference Latency: Mean $24.357$ ms (P95: $28.962$ ms)
  - Postprocessing Latency: Mean $0.117$ ms (P95: $0.077$ ms)
  - **Total Latency:** **24.483 ms** (Median: $24.606$ ms, P95: $29.053$ ms)
  - **Throughput:** **40.8 sequences/sec** (~40.8 FPS)
- **Batch Size 8 (Batched Inference):**
  - Mean Latency per Batch: $46.073$ ms (Throughput: $173.6$ seq/sec)

---

## 16. Reproducibility

From [`REPRODUCIBILITY_TEST.md`](file:///d:/SignAI/docs/phase-3-part-1/REPRODUCIBILITY_TEST.md):
- Dual independent training runs initialized with seed 42 yielded **0.00e+00 floating-point discrepancy** across all epoch training and validation losses.
- Validation predictions matched **100% bit-for-bit** across all verification epochs.
- Deterministic behavior confirmed under [`src/utils/reproducibility.py`](file:///d:/SignAI/src/utils/reproducibility.py).

---

## 17. Limitations

1. **Absence of Spatial Graph Priors:** By vectorizing 93 landmarks into 279 values, the model ignores physical bone adjacencies, leading to confusion among signs differing primarily in finger configurations.
2. **Dataset Scale:** The MVP pilot contains 60 sequences. While sufficient for baseline infrastructure verification, scaling data volume in subsequent phases is critical.
3. **Monocular Z-Coordinate Noise:** Forward motions (`THANK_YOU`) are susceptible to depth noise from monocular MediaPipe extraction.

---

## 18. Lessons for ST-GCN

1. **Top-3 to Top-1 Bridge:** The baseline achieves $75.0\%$ Top-3 accuracy. Spatial graph convolutions are specifically needed to provide the spatial resolution to distinguish the top-1 candidate from the top-3 pool.
2. **Partitioned Graph Adjacency:** Separate subgraphs must be maintained for the left hand ($21$ nodes), right hand ($21$ nodes), and upper pose ($33$ nodes) to prevent hand features from being drowned out by torso coordinates.
3. **Reusable Training Engine:** The `ModelTrainer` coordinator in [`src/training/trainer.py`](file:///d:/SignAI/src/training/trainer.py) successfully handled checkpointing, early stopping, and metric logging, and is ready for ST-GCN without modification.

---

## 19. Open Issues

- Real-world occlusions in `signer_01` recordings indicate that spatial graph dropout and joint-masking augmentations should be explored during ST-GCN training.
- GPU acceleration can be engaged when running on GPU-equipped environments via the existing automatic device selector (`src/utils/device.py`).

---

## 20. Readiness for Phase 3 Part 2

Phase 3 Part 1 has fulfilled all quality gates:
- [x] Implementation audit completed
- [x] Baseline model implemented and tested
- [x] Reusable training engine implemented
- [x] Checkpointing and early stopping verified
- [x] Zero-leakage evaluation protocol preserved
- [x] Test metrics measured and documented ($41.67\%$ accuracy, $0.3067$ macro F1)
- [x] Inference benchmark measured ($24.48$ ms latency, $40.8$ FPS)
- [x] Unit test suite passing ($13/13$ tests pass)
- [x] Deterministic reproducibility verified ($0.00e+00$ diff)
- [x] Complete technical documentation compiled

**The SignTalk AI project is fully prepared and scientifically positioned for Phase 3 Part 2 (ST-GCN Architecture & Spatial Graph Modeling).**
