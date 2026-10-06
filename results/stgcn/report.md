# SignTalk AI — Evaluation Report: SignSTGCN

- **Evaluated Checkpoint:** `experiments/stgcn/checkpoints/best_checkpoint.pt`
- **Evaluation Split:** `test`
- **Sample Count:** `12`
- **Model Size:** `24.68 MB`
- **Total Parameters:** `2,137,818`

---

## 1. Primary Classification Performance

| Metric | Measured Value |
| :--- | :---: |
| **Top-1 Accuracy** | **0.8333** |
| **Macro Precision** | 0.7500 |
| **Macro Recall** | 0.8000 |
| **Macro F1-Score** | **0.7667** |
| **Weighted F1-Score** | 0.7778 |
| **Top-3 Accuracy** | 0.9167 |
| **Cross-Entropy Loss** | 0.6645 |

## 2. Inference Benchmark & Latency

| Metric | Value | Budget / Target |
| :--- | :---: | :---: |
| **Mean Latency** | `105.69 ms` | $\le 40.0\text{ ms}$ (25 FPS) |
| **Median Latency** | `104.56 ms` | - |
| **P95 Latency** | `164.0 ms` | $\le 60.0\text{ ms}$ |
| **Throughput** | `9.46 FPS` | $\ge 20.0\text{ FPS}$ |
| **Meets Real-Time Budget** | `False` | Required for Edge |

## 3. Confidence & Calibration Diagnostics

- **Expected Calibration Error (ECE):** `0.2713`
- **Brier Score:** `0.2756`
- **Correct Predictions Mean Confidence:** `0.7566`
- **Incorrect Predictions Mean Confidence:** `0.4103`

## 4. Top Confused Sign Pairs

| True Sign | Predicted Sign | Error Count | Probable Contributing Factor |
| :--- | :--- | :---: | :--- |
| **happy** | **teacher** | 1 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **time** | **teacher** | 1 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |

## 5. Failure Attribution Breakdown

| Failure Category | Count | Proportion |
| :--- | :---: | :---: |
| `SEMANTIC_KINEMATIC_SIMILARITY` | 2 | 100.0% |
