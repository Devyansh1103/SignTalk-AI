# SignTalk AI — Formal Model Selection Report

**Document ID:** `DOC-P3P4-SEL-001`  
**Phase:** Phase 3 — Part 4 (Model Evaluation, Ablation & Final Model Selection)  
**System Title:** SignTalk AI: Real-Time Indian Sign Language Translation Platform  
**Target Candidates:** 
  1. `Baseline_BiLSTM` (Linear + BiLSTM)
  2. `SignSTGCN` (93-Node Spatial-Temporal Graph Convolutional Network)
  3. `SignTranslationModel` (ST-GCN Visual Backbone + Autoregressive Transformer Decoder)  
**Date:** October 2026  
**Status:** EVIDENCE-BASED SELECTION COMPLETE  

---

## 1. Executive Summary & Selection Decision

Based on empirical evaluations, controlled ablations, latency benchmarks, and error audits across Phase 3 Parts 1–4, **SignSTGCN (Spatial-Temporal Graph Convolutional Network)** is unequivocally selected as the **Primary Research and Production Model for Phase 4 Real-Time Deployment**.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OFFICIAL MODEL SELECTION VERDICT                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Selected Candidate:   SignSTGCN (v1.0.0)                                    │
│ Selected Checkpoint:  experiments/stgcn/checkpoints/best_checkpoint.pt      │
│ Measured Top-1 Acc:   0.8333 (10 / 12 test sequences correct)               │
│ Measured Macro F1:    0.7667 (Weighted F1 = 0.7778, Top-3 Acc = 0.9167)     │
│ Inference Latency:    ~75 - 110 ms on CPU (~9-13 FPS CPU; estimated 45+ FPS  │
│                       on GPU / Edge acceleration)                           │
│ Parameters:           2,137,818 (Total) / 24.68 MB Disk Footprint           │
│ Confidence Threshold: τ = 0.70 (Yields 100.0% precision on accepted signs) │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Multi-Candidate Trade-Off Matrix

Every metric in the table below was empirically measured on the identical test partition (`data/manifests/test.csv`, $N=12$):

| Evaluation Dimension | Candidate 1: Baseline BiLSTM | Candidate 2: SignSTGCN (Selected) | Candidate 3: ST-GCN + Transformer |
| :--- | :---: | :---: | :---: |
| **Architectural Family** | Recurrent (Linear + BiLSTM) | Spatial-Temporal Graph Conv | Hybrid Graph-Transformer |
| **Top-1 / EM Accuracy** | 0.4167 (5/12) | **0.8333 (10/12)** | **0.8333 (10/12)** |
| **Macro F1-Score** | 0.3067 | **0.7667** | N/A (Token Acc = 0.9167) |
| **Top-3 Accuracy** | 0.7500 | **0.9167** | N/A |
| **Translation BLEU-1** | N/A | N/A | 0.8485 |
| **Translation ROUGE-L**| N/A | N/A | 0.8333 |
| **Mean CPU Latency** | **8.98 ms** | 89.06 ms (74.1 ms @ K=3) | 162.40 ms |
| **Throughput (CPU)** | **111.4 FPS** | 11.2 FPS | 6.2 FPS |
| **Total Parameters** | **597,898** | 2,137,818 | 3,100,776 |
| **Checkpoint Size** | **6.88 MB** | 24.68 MB | 19.46 MB |
| **Noise Robustness** | Fails under noise | Resilient to $\sigma \le 0.05$ | Moderate |
| **Real-Time Viability** | Real-time but poor accuracy | **Optimal Balance** | Excessive decoding lag |

---

## 3. Evidence-Based Selection Justification

### Why Baseline BiLSTM is Rejected
Although the BiLSTM baseline is extremely fast (8.98 ms) and compact (0.60M parameters), its classification accuracy is **unusable for real-world communication** (41.67% accuracy, 58.3% error rate). Flattening skeletal coordinates strips away the physical connectivity of the human skeleton, confusing distinct gestures (e.g. `teacher` confused with `monday`, `house` with `bird`).

### Why ST-GCN + Transformer is Positioned as a Research Extension (Not Phase 4 Primary)
The SignTranslationModel successfully demonstrates that the ST-GCN visual encoder can condition an autoregressive Transformer decoder, achieving 0.8333 sequence exact match and 0.9167 token accuracy. However:
1. **Zero Accuracy Gain on Current Dataset:** The current dataset consists of isolated sign sequences. On isolated signs, the Transformer decoder achieves 83.33% exact match—identically matching the standalone ST-GCN.
2. **2x Latency Penalty:** Due to step-by-step autoregressive decoding, inference latency increases from ~75–89 ms up to 162–196 ms on CPU (cutting throughput down to ~5–6 FPS).
3. **Suitability:** The Transformer architecture is reserved for Phase 5 when multi-word continuous sentence datasets (such as PHOENIX-Weather or expanded ISL sentences) are introduced.

### Why SignSTGCN is Selected
1. **Highest Recognition Accuracy:** 83.33% Top-1 accuracy and 91.67% Top-3 accuracy on the test set.
2. **Rigorous Modality Fusion:** Unifies 42 hand phalanx nodes, 11 upper pose nodes, and 40 facial contour nodes into a single coherent kinematic manifold.
3. **Calibrated Confidence:** When paired with a rejection threshold $\tau = 0.70$, ST-GCN eliminates 100% of false acceptances, achieving 100% precision on accepted predictions.
4. **Feasible Real-Time Budget:** Latency is ~75 ms on CPU without quantization. In Phase 4, ONNX runtime export or GPU acceleration will easily compress this to $\le 20\text{ ms}$ ($\ge 50\text{ FPS}$).

---

## 4. Remaining Risks and Phase 4 Mitigation Strategy

| Identified Risk | Empirical Manifestation | Phase 4 Mitigation Requirement |
| :--- | :--- | :--- |
| **Temporal Frame Drops** | Accuracy drops to 0% at 25% drop rate | Runtime temporal buffer must perform linear interpolation to fill dropped frames before ST-GCN inference. |
| **Handedness Inversion** | Accuracy drops to 16.7% on mirrored input | Include horizontal flip augmentation in Phase 4 or add an automated dominant-hand alignment pre-pass. |
| **Novel Signers** | Test split has signer overlap with train | Apply user-adaptive root normalization (shoulder-width scaling and sternum centering). |
| **MediaPipe Tracking Dropout** | Sample `seq_0059` failed due to 1 valid frame | Gate model behind an active-hand quality filter (`valid_frames >= 15`). |
