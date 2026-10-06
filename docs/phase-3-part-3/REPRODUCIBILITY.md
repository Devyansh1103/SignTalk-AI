# SignTalk AI — Reproducibility and Environment Specification

**Document ID:** `DOC-P3P3-REPRO-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** VERIFIED & DETERMINISTIC  

---

## 1. System and Hardware Environment

| Dimension | Specification |
| :--- | :--- |
| **Operating System** | Windows 11 Enterprise (AMD64) |
| **Python Version** | Python 3.13.12 (Miniconda) |
| **PyTorch Version** | PyTorch 2.13.0+cpu |
| **TorchVision / NumPy** | NumPy 2.2.3, Scikit-learn 1.6.1 |
| **CUDA Available** | False (Deterministic CPU Execution) |
| **Processor** | AMD64 Family 25 Model 80 Stepping 0 (8 physical cores, 16 threads) |

---

## 2. Software Versions & Artefact Hashes

- **Dataset Version:** `signTalk-seq-v1.0.0`
- **ST-GCN Checkpoint:** `experiments/stgcn/checkpoints/best_checkpoint.pt` (Epoch 34, 25.88 MB)
- **Transformer Checkpoint:** `experiments/transformer/checkpoints/best_checkpoint.pt` (Epoch 4, 19.46 MB)
- **Vocabulary Version:** `assets/vocabularies/mvp_10.json` (Version 1.0.0, 14 tokens)
- **Random Seed:** `42` (`torch.manual_seed(42)`, `np.random.seed(42)`, `torch.use_deterministic_algorithms(False)`)

---

## 3. Empirical Twin-Run Verification

The reproducibility verification script (`scripts/verify_transformer_reproducibility.py`) executed two independent runs from seed 42:
- Run 1 Epoch Losses: `[2.388151, 1.582870, 1.227320]`
- Run 2 Epoch Losses: `[2.388151, 1.582870, 1.227320]`
- **Maximum Loss Discrepancy:** **`0.00e+00`**
- **Maximum Parameter Tensor Discrepancy:** **`0.00e+00`**
- **Strict Determinism Status:** **CONFIRMED**
