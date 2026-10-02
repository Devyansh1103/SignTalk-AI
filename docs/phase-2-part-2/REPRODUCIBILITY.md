# Reproducibility Guide: SignTalk AI Landmark Preprocessing

**Document ID:** STAI-P2P2-006  
**Project:** SignTalk AI  
**Phase:** Phase 2 — Part 2  
**Status:** Approved  

---

## 1. System & Runtime Environment

The entire landmark extraction and preprocessing pipeline is strictly reproducible across systems satisfying the following specification:

| Environment Component | Exact Specification | Verification Command |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 64-bit (AMD64) | `systeminfo` |
| **Python Version** | Python 3.13.12 | `python --version` |
| **MediaPipe** | 1.0.1 (Tasks Vision API) | `python -c "import mediapipe as mp; print(mp.__version__)"` |
| **OpenCV** | 5.0.0.93 (`opencv-python` / `opencv-contrib-python`) | `python -c "import cv2; print(cv2.__version__)"` |
| **NumPy** | 2.5.1 | `python -c "import numpy; print(numpy.__version__)"` |
| **PyTorch** | 2.13.0 | `python -c "import torch; print(torch.__version__)"` |
| **PyYAML** | 6.0.3 | `python -c "import yaml; print(yaml.__version__)"` |

---

## 2. Model Asset Verification

Google MediaPipe Tasks models are stored locally in `models/mediapipe/` with verified file sizes:

| Model Asset File | SHA-256 / File Size | Source URL |
| :--- | :---: | :--- |
| `pose_landmarker_full.task` | 9,398,198 bytes | `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task` |
| `hand_landmarker.task` | 7,819,105 bytes | `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task` |
| `face_landmarker.task` | 3,758,596 bytes | `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task` |

---

## 3. Step-by-Step Execution Protocol

To reproduce the exact preprocessed dataset from raw inputs:

### Step 1: Run Landmark Extraction Pilot
```bash
python scripts/run_preprocessing.py --mode pilot --config configs/preprocessing.yaml
```
Output: Validates 10 representative multi-class samples into `data/processed/landmarks/pilot/` and writes `data/interim/landmarks/pilot_summary.json`.

### Step 2: Run Full Dataset Preprocessing
```bash
python scripts/run_preprocessing.py --mode full --config configs/preprocessing.yaml
```
Output: Processes all 60 video sequences into `data/processed/landmarks/{train, val, test}/` and logs manifest to `data/metadata/processed_dataset_manifest.csv`.

### Step 3: Run Processed Data Validation Suite
```bash
python scripts/validate_processed_data.py
```
Output: Scans all output NPZ files for shapes $(3, 45, 93)$, verifies zero NaNs/Infs, confirms split balance, and verifies PyTorch `DataLoader` yields $[B, 3, 45, 93]$.

### Step 4: Generate Validation Visualizations
```bash
python scripts/generate_visualizations.py
```
Output: Generates side-by-side skeleton strips and articulator trajectory graphs under `data/interim/visualizations/`.

### Step 5: Execute Test Suite
```bash
python -m pytest tests/preprocessing/ -v
```
Output: Runs 13 automated tests across frame extraction, landmark extraction, normalization, missing landmark handling, quality scoring, and PyTorch dataset loading.
