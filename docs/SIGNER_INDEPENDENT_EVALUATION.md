# SignTalk AI — Signer-Independent Generalization Analysis

**Document ID:** `DOC-P3P4-SIGNER-GEN-001`  
**Phase:** Phase 3 — Part 4 (Model Evaluation & Model Selection)  
**System Title:** SignTalk AI: Real-Time Indian Sign Language Translation Platform  
**Date:** October 2026  
**Status:** AUDITED & CONSTRAINED  

---

## 1. Executive Summary & Core Finding

A foundational requirement for robust sign language recognition is **signer-independent generalization**: the ability of a trained neural architecture to accurately transcribe gestures from a novel signer whose anthropometric proportions, signing cadence, and idiosyncratic movement dynamics were completely absent during model training.

> [!WARNING]
> **Audit Finding on Current Dataset Partitioning:**  
> In the current `signTalk-seq-v1.0.0` dataset, the train, validation, and test splits are partitioned by **recording repetitions** across classes, **NOT by signer identity**.
> 
> $$\text{Train Signers} \cap \text{Test Signers} = \{\text{signer\_01}, \text{signer\_02}, \text{signer\_03}\} \neq \emptyset$$
> 
> Therefore, test set performance reflects **signer-dependent (or signer-overlapping) generalization**. True zero-shot signer independence cannot be scientifically claimed on the standard test set.

---

## 2. Exhaustive Split Audit by Signer Identity

An inspection of `data/manifests/` across the 60 sequences produces the following distribution:

| Partition | Total Sequences | `signer_01` Sequences | `signer_02` Sequences | `signer_03` Sequences | Signer Overlap with Train |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 36 | 13 (36.1%) | 11 (30.6%) | 12 (33.3%) | Self |
| **Validation** | 12 | 4 (33.3%) | 4 (33.3%) | 4 (33.3%) | 100% (3/3 signers) |
| **Test** | 12 | 4 (33.3%) | 4 (33.3%) | 4 (33.3%) | 100% (3/3 signers) |
| **Total** | **60** | **21** | **19** | **20** | - |

### Repetition Isolation
Although signers overlap, individual recording instances (repetitions) are strictly separated:
- For example:
  - `hello_signer_01_rep1` is in `train`
  - `hello_signer_01_rep2` is in `train`
  - `hello_signer_02_rep1` is in `val`
  - `hello_signer_02_rep2` is in `test`
No exact identical video clip or temporal sequence appears in more than one partition.

---

## 3. Empirical Implications

1. **Why ST-GCN Excels:**  
   The ST-GCN achieves 83.33% Top-1 accuracy on the test partition because its graph convolutions capture invariant topological bone-link connectivity. However, having seen `signer_01`, `signer_02`, and `signer_03` in training mitigates inter-subject variance (e.g., forearm lengths, shoulder span).
2. **Generalization Risk in Real-World Deployment (Phase 4):**  
   When a user in Phase 4 tests the webcam pipeline who is neither `signer_01`, `signer_02`, nor `signer_03`, accuracy is anticipated to experience a degradation of approximately $10\%\text{--}25\%$ based on literature benchmarks for isolated sign recognition on unseen signers.

---

## 4. Proposed Leave-One-Signer-Out (LOSO) Protocol for Phase 4 / Next Dataset

To formally benchmark true signer independence in future phases, the following 3-fold cross-validation protocol is formally specified:

| Fold | Training Signers | Held-Out Test Signer | Purpose |
| :---: | :---: | :---: | :--- |
| **Fold 1** | `signer_02`, `signer_03` (40 seqs) | `signer_01` (20 seqs) | Zero-shot evaluation on Signer 1 |
| **Fold 2** | `signer_01`, `signer_03` (39 seqs) | `signer_02` (21 seqs) | Zero-shot evaluation on Signer 2 |
| **Fold 3** | `signer_01`, `signer_02` (40 seqs) | `signer_03` (20 seqs) | Zero-shot evaluation on Signer 3 |

### Implementation Guardrails
- In Phase 4, the runtime inference pipeline must apply **subject-adaptive root normalization** (dividing by the shoulder width and centering at the nose/mid-shoulder) to minimize body scale variations across novel signers.
