# SignTalk AI — Phase 3 Part 1: Baseline Model & Training Infrastructure

**Document ID:** `DOC-P3P1-README-001`  
**Phase:** Phase 3 — Part 1 (Baseline Model & Training Infrastructure)  
**Status:** COMPLETE  

---

## 1. Purpose

The objective of Phase 3 Part 1 is to establish a rigorous, reproducible machine-learning baseline and training infrastructure for the SignTalk AI Indian Sign Language recognition platform before implementing complex Spatial-Temporal Graph Convolutional Networks (ST-GCN).

The baseline model provides an empirical reference point answering:
> *"How well can Indian Sign Language sequences be classified without spatial graph convolutions?"*

---

## 2. Selected Baseline Architecture

The primary baseline is a **2-Layer Bidirectional Gated Recurrent Unit (BiGRU)** with linear spatial coordinate projection and dual temporal pooling:

- **Input Dimension:** $[B, 3, 45, 93]$ ($B$ batches, $C=3$ spatial channels $[x, y, z]$, $T=45$ temporal frames, $V=93$ anatomical landmarks).
- **Spatial Flattening:** Reshaped to $[B, 45, 279]$ ($3 \times 93 = 279$ coordinates per frame).
- **Linear Feature Projection:** Linear layer mapping $279 \to 128$ dimensions, followed by LayerNorm and Dropout ($p=0.3$).
- **Recurrent Core:** 2-layer Bidirectional GRU (Hidden size $128 \times 2 = 256$ features per time step).
- **Temporal Pooling:** Concatenated Temporal Mean Pooling ($256$) and Temporal Max Pooling ($256$) producing a fixed $512$-dimensional sequence descriptor.
- **Classifier Head:** Multi-layer Perceptron ($512 \to 128 \to 10$) with LayerNorm, ReLU, Dropout, and linear logits output.
- **Total Parameters:** **597,898** (100% trainable).

---

## 3. Environment & Setup

Ensure the active Conda/Python environment is activated with required dependencies:

```bash
# Verify Python and PyTorch installation
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

Verify dataset manifests and sequence archives:
```bash
ls data/manifests/
# Expected: train.csv (36 seqs), val.csv (12 seqs), test.csv (12 seqs)
```

---

## 4. Configuration

The training run is controlled via [`configs/baseline.yaml`](file:///d:/SignAI/configs/baseline.yaml).

Key parameters:
```yaml
experiment_name: "baseline_bigru_v1"
seed: 42
device: "auto"

dataset:
  version: "signTalk-seq-v1.0.0"
  num_classes: 10
  sequence_length: 45
  num_nodes: 93
  in_channels: 3

model:
  proj_dim: 128
  hidden_dim: 128
  num_layers: 2
  dropout: 0.3
  bidirectional: true
  pooling_type: "mean_max"

training:
  batch_size: 8
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "AdamW"
  scheduler: "CosineAnnealingLR"
  early_stopping_patience: 15
  early_stopping_metric: "val_macro_f1"
```

---

## 5. Execution Commands

### 1. Training the Baseline Model
```bash
python scripts/train_baseline.py --config configs/baseline.yaml
```
- Trains the model with deterministic seed initialization.
- Validates after every epoch on the validation partition (`val.csv`).
- Automatically manages early stopping and saves checkpoints:
  - `experiments/baseline/checkpoints/best_checkpoint.pt`
  - `experiments/baseline/checkpoints/latest_checkpoint.pt`

### 2. Evaluating on the Isolated Test Set
```bash
python scripts/evaluate_baseline.py \
  --checkpoint experiments/baseline/checkpoints/best_checkpoint.pt \
  --split test \
  --manifest data/manifests/test.csv
```
- Evaluates top-1/top-3 accuracy, macro/weighted precision, recall, and F1.
- Saves evaluation JSON: `experiments/baseline/metrics/evaluation_test.json`.

### 3. Inspecting Sample Predictions
```bash
python scripts/inspect_predictions.py \
  --checkpoint experiments/baseline/checkpoints/best_checkpoint.pt \
  --manifest data/manifests/test.csv
```
- Produces sample-by-sample diagnostic table and CSV: `experiments/baseline/metrics/sample_predictions.csv`.

### 4. Single-Sequence Inference (CLI)
```bash
python scripts/predict_baseline.py \
  --checkpoint experiments/baseline/checkpoints/best_checkpoint.pt \
  --sequence data/processed/sequences/test/seq_0049.npz
```

### 5. Plotting Training Curves
```bash
python scripts/plot_training_history.py \
  --history experiments/baseline/metrics/training_history.json \
  --output experiments/baseline/plots/training_curves.png
```

### 6. Generating Confusion Matrix Heatmap
```bash
python scripts/generate_confusion_matrix.py \
  --metrics experiments/baseline/metrics/evaluation_test.json \
  --output experiments/baseline/plots/confusion_matrix.png
```

### 7. Inference Latency Benchmark
```bash
python scripts/benchmark_inference.py --runs 200
```

### 8. Reproducibility Test
```bash
python scripts/verify_reproducibility.py
```

---

## 6. Directory and Artifact Structure

```
experiments/baseline/
├── checkpoints/
│   ├── best_checkpoint.pt           # Selected by peak val_macro_f1 (Epoch 25, 7.21 MB)
│   └── latest_checkpoint.pt         # Last training epoch (Epoch 40, 7.21 MB)
├── logs/
│   └── training.log                 # Epoch-by-epoch training and validation logs
├── metrics/
│   ├── training_history.json        # Trajectory of loss, accuracy, and macro F1
│   ├── metrics.csv                  # Tabular metrics across all training epochs
│   ├── evaluation_val.json          # Validation set evaluation report
│   ├── evaluation_test.json         # Isolated test set evaluation report
│   ├── sample_predictions.csv       # Per-sample test predictions, confidence, errors
│   ├── inference_benchmark.json     # CPU latency (mean, median, P95) benchmark
│   └── reproducibility_report.json  # Bit-for-bit duplicate training run audit
└── plots/
    ├── training_curves.png          # Tri-panel Loss, Accuracy, and Macro F1 curves
    └── confusion_matrix.png         # 10x10 annotated confusion matrix heatmap
```

---

## 7. Measured Baseline Performance Reference

| Metric | Measured Baseline Value |
| :--- | :--- |
| **Top-1 Test Accuracy** | **41.67%** (5 / 12 correct) |
| **Top-3 Test Accuracy** | **75.00%** (9 / 12 samples) |
| **Test Macro F1** | **0.3067** (30.67%) |
| **Test Weighted F1** | **0.3111** (31.11%) |
| **Single-Window Latency (CPU)** | **24.48 ms** (~40.8 FPS) |
| **Total Parameters** | **597,898** |

---

## 8. Reproducibility Guarantee

Deterministic execution is enforced via [`src/utils/reproducibility.py`](file:///d:/SignAI/src/utils/reproducibility.py). Dual independent 5-epoch training runs with seed `42` demonstrated **0.00e+00 float discrepancy** across training/validation losses and 100% identical sample predictions.
