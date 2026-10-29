# SignTalk AI — Evaluation Report: SignSTGCN

- **Evaluated Checkpoint:** `experiments/stgcn/checkpoints/best_checkpoint.pt`
- **Evaluation Split:** `test`
- **Sample Count:** `645`
- **Model Size:** `26.15 MB`
- **Total Parameters:** `2,202,068`

---

## 1. Primary Classification Performance

| Metric | Measured Value |
| :--- | :---: |
| **Top-1 Accuracy** | **0.8760** |
| **Macro Precision** | 0.9072 |
| **Macro Recall** | 0.8801 |
| **Macro F1-Score** | **0.8759** |
| **Weighted F1-Score** | 0.8726 |
| **Top-3 Accuracy** | 0.9550 |
| **Cross-Entropy Loss** | 0.5722 |

## 2. Inference Benchmark & Latency

| Metric | Value | Budget / Target |
| :--- | :---: | :---: |
| **Mean Latency** | `3.38 ms` | $\le 40.0\text{ ms}$ (25 FPS) |
| **Median Latency** | `3.36 ms` | - |
| **P95 Latency** | `3.59 ms` | $\le 60.0\text{ ms}$ |
| **Throughput** | `295.69 FPS` | $\ge 20.0\text{ FPS}$ |
| **Meets Real-Time Budget** | `True` | Required for Edge |

## 3. Confidence & Calibration Diagnostics

- **Expected Calibration Error (ECE):** `0.0352`
- **Brier Score:** `0.1842`
- **Correct Predictions Mean Confidence:** `0.9425`
- **Incorrect Predictions Mean Confidence:** `0.6734`

## 4. Top Confused Sign Pairs

| True Sign | Predicted Sign | Error Count | Probable Contributing Factor |
| :--- | :--- | :---: | :--- |
| **dress** | **clothing** | 3 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **husband** | **wife** | 3 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **sick** | **poor** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **clothing** | **dress** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **red** | **pink** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |

## 5. Failure Attribution Breakdown

| Failure Category | Count | Proportion |
| :--- | :---: | :---: |
| `SEMANTIC_KINEMATIC_SIMILARITY` | 60 | 75.0% |
| `SEMANTIC_OVERCONFIDENT_CONFUSION` | 20 | 25.0% |
