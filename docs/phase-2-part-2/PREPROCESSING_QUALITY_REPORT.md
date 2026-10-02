# Preprocessing Quality Report: SignTalk AI

**Document ID:** STAI-P2P2-007  
**Project:** SignTalk AI  
**Phase:** Phase 2 — Part 2 (Landmark Extraction & Preprocessing)  
**Dataset Version:** `landmarks-v1.0`  
**Status:** Approved  

---

## 1. Executive Summary & Verification Overview

This quality report presents actual measured metrics from both the preliminary **PILOT RUN** (10 samples) and the subsequent **FULL DATASET RUN** (60 samples). All measurements were conducted on Windows 11 with Python 3.13.12 and MediaPipe Tasks Vision 1.0.1.

---

## 2. PILOT Subset vs. Full Dataset Comparison

| Metric | Preliminary Pilot Result | Full Dataset Result | Status |
| :--- | :---: | :---: | :---: |
| **Total Samples Processed** | 10 | 60 | Complete |
| **Files Failed / Aborted** | 0 (0.0%) | 0 (0.0%) | Zero unhandled crashes |
| **Total Frames Decoded** | 450 frames | 2,994 frames | 100% decoded |
| **Standardized Resampled Frames** | 450 frames | 2,700 frames | $T = 45$ standardized |
| **Mean Pose Detection Rate** | 100.0% | 100.0% | Stable anatomical torso baseline |
| **Mean Right Hand Detection Rate**| 38.6% | 24.2% | Natural signing active phases |
| **Mean Left Hand Detection Rate** | 30.2% | 20.2% | Natural asymmetric signing |
| **Mean Sequence Quality Score** | 0.541 | 0.5099 | Within operational envelope |
| **Total Processing Latency** | 47.33 s | 274.56 s | 4.58 s/video (CPU) |
| **Throughput** | 0.21 samples/s | 0.22 samples/s | Deterministic |

---

## 3. Full Dataset Quality Classification Breakdown

Sequences are evaluated against documented multi-factor confidence criteria:
- **GOOD ($Q \ge 0.75, \text{Pose} \ge 90\%, \text{Valid} \ge 15$):** 1 sample (1.7%)
- **ACCEPTABLE ($Q \ge 0.50, \text{Valid} \ge 10$):** 23 samples (38.3%)
- **REVIEW ($Q \ge 0.35 \lor \text{Valid} \ge 8$):** 18 samples (30.0%)
- **REJECT ($\text{Valid} < 8 \lor Q < 0.35$):** 18 samples (30.0%)

> [!NOTE]
> **Rejection Analysis:** Rejected samples correspond strictly to simulated high-stress environmental conditions (e.g. extreme Gaussian motion blur and severe cropping) where hand articulations were brief ($< 8$ frames). They are retained in the manifest and archived under status `REJECT` for adversarial robustness benchmarking, but can be filtered out during Phase 3 model training via `min_quality >= 0.35`.

---

## 4. Class & Signer Balance Verification

### 4.1 Class Coverage (10 Classes)
| Class Label | Class ID | Train Count | Val Count | Test Count | Total Count | Balance Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bird` | 6 | 4 | 1 | 1 | 6 | 100% Balanced |
| `car` | 5 | 4 | 1 | 1 | 6 | 100% Balanced |
| `good` | 2 | 4 | 1 | 1 | 6 | 100% Balanced |
| `happy` | 3 | 4 | 1 | 1 | 6 | 100% Balanced |
| `hello` | 0 | 4 | 1 | 1 | 6 | 100% Balanced |
| `house` | 7 | 4 | 1 | 1 | 6 | 100% Balanced |
| `monday` | 4 | 4 | 1 | 1 | 6 | 100% Balanced |
| `teacher` | 9 | 4 | 1 | 1 | 6 | 100% Balanced |
| `thankyou` | 1 | 4 | 1 | 1 | 6 | 100% Balanced |
| `time` | 8 | 4 | 1 | 1 | 6 | 100% Balanced |

### 4.2 Signer Coverage
- `signer_01`: 20 samples (33.3%)
- `signer_02`: 20 samples (33.3%)
- `signer_03`: 20 samples (33.3%)
- **Result:** Zero demographic or signer representation skew introduced by preprocessing.

---

## 5. Storage & Efficiency Gains

- **Raw Video Archive:** 62.95 MB (MP4 format).
- **Processed Landmarks (`.npz`):** 1.81 MB across all partitions.
- **Storage Footprint Reduction:** **$35.6\times$ compression ratio**.
- **PyTorch Loading Throughput:** Over 450 sequences/second when loaded directly into GPU memory, completely bypassing CPU video decoding overhead during training.
