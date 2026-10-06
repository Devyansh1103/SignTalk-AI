# SignTalk AI — Phase 3 Part 4 Implementation Audit

**Document ID:** `DOC-P3P4-AUDIT-001`  
**Phase:** Phase 3 — Part 4 (Model Evaluation, Ablation & Final Model Selection)  
**System Title:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Technical Subtitle:** A Spatial-Temporal Graph and Transformer-Based Sign-to-Text System  
**Audit Date:** October 2026  
**Auditor:** Antigravity AI Engineering Team  
**Evaluation Target:** Models, Pipelines, Datasets, and Checkpoints developed across Phase 3 Parts 1–3  

---

## 1. Executive Summary

This comprehensive audit inspects the complete codebase of the SignTalk AI project prior to conducting rigorous evaluation, controlled ablations, benchmark analysis, and model selection for Phase 4 handoff. 

All claims are grounded strictly in the verified contents of the repository:
- All three planned model checkpoints (`baseline`, `stgcn`, `transformer`) are trained, saved, and structurally valid.
- The dataset manifests and preprocessed sequence representations exist under `data/manifests/` and `data/processed/sequences/`.
- The dataset supports **Isolated Sign Classification and Single-Token Translation Alignment** across 10 vocabulary classes. It does **NOT** support continuous, unconstrained sign language translation sentences.
- 75 automated unit and regression tests pass across data processing, model layers, and inference interfaces.
- The evaluation infrastructure from Parts 1–3 consists of fragmented scripts. Part 4 establishes a unified, reproducible evaluation engine (`src/evaluation/`), central configs (`configs/`), controlled ablation runners, error categorizers, robustness testers, confidence calibrators, and latency benchmarks.

---

## 2. Component Implementation Status

| Component | Status | Location | Notes |
| :--- | :---: | :--- | :--- |
| **Data Ingestion & Manifests** | Implemented | `data/manifests/` | 60 sequences total (36 train, 12 val, 12 test). Stratified by recording/class. |
| **Landmark Schema** | Implemented | `src/preprocessing/` | 93-node multimodal schema (Left Hand: 21, Right Hand: 21, Upper Pose: 11, Face Contours: 40). |
| **Sequence Preprocessing** | Implemented | `src/data/` | Linear temporal interpolation to $T=45$ frames, root normalization, missing landmark masks. |
| **Baseline Architecture** | Implemented | `src/models/baseline.py` | Linear spatial projection (128) + 2-layer BiLSTM (128) + mean-max pooling + FC (10). |
| **ST-GCN Architecture** | Implemented | `src/models/stgcn.py` | 6 ST-GCN blocks (64->64->128->128->256->256) with spatial configuration partitioning ($K=3$). |
| **Skeletal Graph Topology** | Implemented | `src/models/graph.py` | 93 nodes, 103 undirected anatomical skeletal edges + self loops. $K=3$ spatial, $K=2$ distance, $K=1$ uniform. |
| **Transformer NLP Layer** | Implemented | `src/models/sign_translation_model.py` | Linear temporal projection + 2-layer Transformer decoder with cross-attention to ST-GCN features. |
| **Tokenizer & Normalizer** | Implemented | `src/nlp/` | Character/token-level tokenizer with vocabulary size $|\mathcal{V}|=14$ (special tokens: PAD, UNK, BOS, EOS + 10 glosses). |
| **Evaluation Metrics** | Partially Implemented | `src/evaluation/` | Top-1/Top-3, Macro/Weighted F1 in `classification_metrics.py`; BLEU, ROUGE-L, Token Acc in `sequence_metrics.py`. Needs unified interface. |
| **Confidence Calibration** | Partially Implemented | `scripts/inspect_*.py` | Confidence scores logged, but formal ECE, Brier score, and threshold tuning are missing. |
| **Controlled Ablations** | Missing | `scripts/`, `src/evaluation/` | Planned in Part 2 docs, but no automated modular ablation framework exists. |
| **Robustness Suite** | Missing | `scripts/`, `src/evaluation/` | No automated perturbed input evaluation (jitter, missing frames, noise, scale). |
| **Central Experiment Config** | Missing | `configs/` | Individual configs exist (`baseline.yaml`, `stgcn.yaml`, `transformer.yaml`), but unified evaluation/ablation configs are absent. |
| **Model Registry** | Missing | `models/` | No centralized `models/model_registry.yaml` registering candidates and metrics. |
| **Phase 4 Handoff Package** | Missing | `docs/` | Requires formal evidence-based selection and contract definition. |

---

## 3. Checkpoint & Artifact Availability

All model checkpoints are present in the filesystem and loadable with PyTorch 2.13.0+cpu:

```text
experiments/
├── baseline/checkpoints/
│   ├── best_checkpoint.pt   (7.2 MB, epoch 48, val_acc=0.5833)
│   └── latest_checkpoint.pt (7.2 MB)
├── stgcn/checkpoints/
│   ├── best_checkpoint.pt   (25.9 MB, epoch 34, val_acc=0.8333)
│   └── latest_checkpoint.pt (25.9 MB)
└── transformer/checkpoints/
    ├── best_checkpoint.pt   (20.4 MB, epoch 35, val_loss=0.5312)
    └── latest_checkpoint.pt (20.4 MB)
```

Preprocessed sequences exist at:
- `data/processed/sequences/train/*.npz` (36 files)
- `data/processed/sequences/val/*.npz` (12 files)
- `data/processed/sequences/test/*.npz` (12 files)

Each `.npz` archive contains:
- `landmarks`: `[3, 45, 93]` Float32 tensor
- `mask`: `[1, 45, 93]` Bool/Float32 tensor
- `label`: Integer class ID
- `gloss`: String gloss
- `translation`: String translation

---

## 4. Dataset, Split & Signer Audit

### Dataset Identity
- **Dataset Version:** `signTalk-seq-v1.0.0`
- **Origin:** Curated subset of INCLUDE-50 Indian Sign Language dataset
- **Vocabulary Size:** 10 classes
- **Vocabulary Classes:**
  1. `hello` (HELLO / "Hello")
  2. `thankyou` (THANK_YOU / "Thank you")
  3. `good` (GOOD / "Good")
  4. `happy` (HAPPY / "Happy")
  5. `monday` (MONDAY / "Monday")
  6. `car` (CAR / "Car")
  7. `bird` (BIRD / "Bird")
  8. `house` (HOUSE / "House")
  9. `time` (TIME / "Time")
  10. `teacher` (TEACHER / "Teacher")

### Partition Breakdown
- **Train:** 36 sequences (3 to 4 sequences per class)
- **Validation:** 12 sequences (1 to 2 sequences per class)
- **Test:** 12 sequences (1 to 2 sequences per class)
- **Total:** 60 sequences

### Critical Finding: Signer Overlap Across Splits
An exhaustive audit of the `signer_id` column in `data/manifests/` reveals:
- **Train Signers:** `signer_01`, `signer_02`, `signer_03`
- **Validation Signers:** `signer_01`, `signer_02`, `signer_03`
- **Test Signers:** `signer_01`, `signer_02`, `signer_03`

$$\text{Train Signers} \cap \text{Test Signers} = \{\text{signer\_01}, \text{signer\_02}, \text{signer\_03}\} \neq \emptyset$$

**Audit Conclusion:**
The current partition is a **stratified recording/repetition split**, NOT an unseen-signer split. Although different repetitions (e.g. rep 1 vs rep 2) are isolated into separate splits preventing identical clip leakage, the model has seen all 3 signers during training. Unseen-signer generalization cannot be claimed from the standard test split alone. A dedicated leave-one-signer-out evaluation protocol must be designed and documented.

---

## 5. Architectural Comparison Matrix

| Property | Baseline (Phase 3 Part 1) | ST-GCN (Phase 3 Part 2) | ST-GCN + Transformer (Part 3) |
| :--- | :--- | :--- | :--- |
| **Input Shape** | $[B, 3, 45, 93]$ | $[B, 3, 45, 93]$ | $[B, 3, 45, 93]$ |
| **Spatial Modeling** | Flattened Linear Projection ($3 \times 93 = 279 \to 128$) | 93-Node Graph Convolutions with learnable mask | ST-GCN Backbone ($T'=12, D=256$) |
| **Temporal Modeling**| 2-layer BiLSTM ($hidden=128$) | 1D Temporal Convolutions ($kernel=9$, stride 1 or 2) | ST-GCN temporal conv + Transformer Self-Attention |
| **Decoding** | Mean-Max Pooling + Linear | Global Avg Pooling + Linear | Autoregressive Transformer Decoder |
| **Parameters** | 895,370 (~0.90 M) | 3,116,938 (~3.12 M) | 4,960,014 (~4.96 M) |
| **Checkpoint Size** | 7.2 MB | 25.9 MB | 20.4 MB |
| **Measured Test Acc**| **0.4167** (5/12) | **0.8333** (10/12) | **0.8333** (10/12 EM sequence) |
| **Macro F1** | **0.3067** | **0.7667** | **0.8303** |

---

## 6. Reproducibility & Environment Audit

- **Operating System:** Windows 10/11
- **Python Version:** 3.13.12 (Anaconda)
- **PyTorch Version:** 2.13.0+cpu (CPU execution mode; CUDA is not available on host)
- **Random Seed:** Standardized to `42` across data generation, dataloaders, and model training.
- **Git Tracking:** Active git repository with commit history.
- **Unit Test Status:** 75 passed unit tests in `tests/`.

---

## 7. Known Limitations & Gaps to Address in Part 4

1. **Dataset Size:** 60 total sequences across 10 classes is a prototype-scale dataset. While sufficient for architectural comparison and proof-of-concept validation, generalization bounds must be stated with scientific caution.
2. **Device Dependency:** Benchmarking on CPU must be clearly identified as CPU-only latency ($2.13.0+\text{cpu}$) and not confused with GPU-accelerated latency.
3. **Absence of Central Evaluation Framework:** Previously, baseline, ST-GCN, and transformer evaluations ran in separate ad-hoc scripts. Part 4 must unify all evaluations under `src/evaluation/`.
4. **Controlled Ablations Pending:** Modality ablations (hand vs hand+pose vs full), graph topology ablations (uniform vs distance vs spatial), and temporal window ablations need explicit, automated execution and measurement.
5. **Robustness Testing Pending:** Synthetic perturbation tests (noise, dropouts, temporal scaling) must be executed to stress-test model degradation curves.

---

## 8. Audit Verdict

**Readiness for Phase 3 Part 4:** **APPROVED WITH CONSTRAINTS**  
All prerequisites, data files, checkpoints, and models are present and sound. Proceed immediately to define the evaluation task, establish the central configuration, construct the evaluation pipeline, run all experiments, and assemble the Phase 4 handoff package.
