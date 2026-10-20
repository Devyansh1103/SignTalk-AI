# SignTalk AI — Evaluation Report: SignSTGCN

- **Evaluated Checkpoint:** `experiments/stgcn/checkpoints/best_checkpoint.pt`
- **Evaluation Split:** `test`
- **Sample Count:** `332`
- **Model Size:** `25.29 MB`
- **Total Parameters:** `2,171,485`

---

## 1. Primary Classification Performance

| Metric | Measured Value |
| :--- | :---: |
| **Top-1 Accuracy** | **0.8675** |
| **Macro Precision** | 0.8813 |
| **Macro Recall** | 0.8700 |
| **Macro F1-Score** | **0.8598** |
| **Weighted F1-Score** | 0.8595 |
| **Top-3 Accuracy** | 0.9428 |
| **Cross-Entropy Loss** | 0.5913 |

## 2. Inference Benchmark & Latency

| Metric | Value | Budget / Target |
| :--- | :---: | :---: |
| **Mean Latency** | `14.09 ms` | $\le 40.0\text{ ms}$ (25 FPS) |
| **Median Latency** | `12.39 ms` | - |
| **P95 Latency** | `27.17 ms` | $\le 60.0\text{ ms}$ |
| **Throughput** | `70.97 FPS` | $\ge 20.0\text{ FPS}$ |
| **Meets Real-Time Budget** | `True` | Required for Edge |

## 3. Confidence & Calibration Diagnostics

- **Expected Calibration Error (ECE):** `0.0308`
- **Brier Score:** `0.2099`
- **Correct Predictions Mean Confidence:** `0.8937`
- **Incorrect Predictions Mean Confidence:** `0.6258`

## 4. Top Confused Sign Pairs

| True Sign | Predicted Sign | Error Count | Probable Contributing Factor |
| :--- | :--- | :---: | :--- |
| **dress** | **clothing** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **blue** | **yellow** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **loud** | **table** | 1 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **loud** | **truck** | 1 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **ugly** | **second** | 1 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |

## 5. Failure Attribution Breakdown

| Failure Category | Count | Proportion |
| :--- | :---: | :---: |
| `SEMANTIC_KINEMATIC_SIMILARITY` | 37 | 84.1% |
| `SEMANTIC_OVERCONFIDENT_CONFUSION` | 7 | 15.9% |
