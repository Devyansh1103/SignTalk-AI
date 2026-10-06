# SignTalk AI — Reproducibility & Determinism Verification Test

**Document ID:** `DOC-P3P1-REPRO-001`  
**Phase:** Phase 3 — Part 1 (Baseline Model & Training Infrastructure)  
**Script Reference:** [`scripts/verify_reproducibility.py`](file:///d:/SignAI/scripts/verify_reproducibility.py)  
**Configuration Reference:** [`src/utils/reproducibility.py`](file:///d:/SignAI/src/utils/reproducibility.py)  
**Report Artifact:** [`experiments/baseline/metrics/reproducibility_report.json`](file:///d:/SignAI/experiments/baseline/metrics/reproducibility_report.json)  
**Date of Execution:** October 2, 2026  

---

## 1. Objective and Determinism Standard

Scientific reproducibility requires that identical training scripts, configurations, and random seeds produce bit-level or floating-point identical loss trajectories, parameter states, and evaluation outputs across independent runs.

In deep learning systems, nondeterminism commonly originates from:
1. Python built-in hash randomization (`PYTHONHASHSEED`).
2. NumPy pseudo-random number generator state.
3. PyTorch CPU and CUDA PRNG state.
4. Non-deterministic atomic floating-point accumulation in GPU cuDNN convolution and reduction kernels.
5. Asynchronous multi-threaded DataLoader worker processes.

The objective of this test was to execute two separate training runs (`Run A` and `Run B`) initialized with seed `42`, evaluating losses and predictions after each epoch to verify deterministic reproducibility.

---

## 2. Experimental Verification Protocol

The verification test was executed via [`scripts/verify_reproducibility.py`](file:///d:/SignAI/scripts/verify_reproducibility.py) with the following specifications:
- **Seed:** `42` enforced via `set_seed(42)` in [`src/utils/reproducibility.py`](file:///d:/SignAI/src/utils/reproducibility.py).
- **Environment:** CPU execution on Windows 11 AMD64 (PyTorch 2.13.0).
- **Training Epochs:** 5 full epochs per run.
- **Model Architecture:** `SignBaselineModel` (2-layer BiGRU, 597,898 parameters).
- **Data Loaders:** Identical dataset splits (`train.csv`, `val.csv`), batch size 8.

---

## 3. Measured Trajectory Comparison

Below is the verbatim comparison logged across all 5 verification epochs:

| Epoch | Train Loss (Run A) | Train Loss (Run B) | Absolute Diff ($\Delta_{\text{train}}$) | Val Loss (Run A) | Val Loss (Run B) | Absolute Diff ($\Delta_{\text{val}}$) | Val Predictions Exact Match |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2.460203 | 2.460203 | **0.00e+00** | 2.329057 | 2.329057 | **0.00e+00** | **TRUE (100%)** |
| **2** | 2.355871 | 2.355871 | **0.00e+00** | 2.280832 | 2.280832 | **0.00e+00** | **TRUE (100%)** |
| **3** | 2.307392 | 2.307392 | **0.00e+00** | 2.270294 | 2.270294 | **0.00e+00** | **TRUE (100%)** |
| **4** | 2.352280 | 2.352280 | **0.00e+00** | 2.243364 | 2.243364 | **0.00e+00** | **TRUE (100%)** |
| **5** | 2.205622 | 2.205622 | **0.00e+00** | 2.220578 | 2.220578 | **0.00e+00** | **TRUE (100%)** |

```
============================================================
Maximum Float Discrepancy:      0.00e+00
Prediction Match Rate:          100.0% (12 / 12 samples per epoch)
Verification Status:            PASSED (PERFECT DETERMINISM)
============================================================
```

---

## 4. Hardware and Platform-Specific Caveats

While CPU execution on PyTorch 2.13.0 achieved bit-for-bit exact reproducibility ($\Delta = 0.00$), the following constraints apply when scaling to GPU clusters:
1. **CUDA Atomic Adds:** When training future ST-GCN models on CUDA hardware, certain backward kernels (e.g. `torch.bmm`, atomic scatter/gather in graph convolutions) can exhibit minute floating-point rounding divergence on the order of $\le 10^{-7}$ across different GPU architectures.
2. **cuDNN Determinism Flags:** [`src/utils/reproducibility.py`](file:///d:/SignAI/src/utils/reproducibility.py) explicitly sets `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` whenever CUDA is detected, minimizing nondeterminism at a slight cost to CUDA kernel autotuning speed.
3. **Multi-worker Dataloading:** If `num_workers > 0` is enabled in future large-scale training, worker seeds must be set using PyTorch's `worker_init_fn` to prevent identical data loader shuffles across processes.
