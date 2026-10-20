# SignTalk AI — Evaluation Report: SignSTGCN

- **Evaluated Checkpoint:** `experiments/stgcn/checkpoints/latest_checkpoint.pt`
- **Evaluation Split:** `test`
- **Sample Count:** `332`
- **Model Size:** `25.06 MB`
- **Total Parameters:** `2,171,485`

---

## 1. Primary Classification Performance

| Metric | Measured Value |
| :--- | :---: |
| **Top-1 Accuracy** | **0.8193** |
| **Macro Precision** | 0.8426 |
| **Macro Recall** | 0.8274 |
| **Macro F1-Score** | **0.8120** |
| **Weighted F1-Score** | 0.8045 |
| **Top-3 Accuracy** | 0.9458 |
| **Cross-Entropy Loss** | 0.6903 |

## 2. Inference Benchmark & Latency

| Metric | Value | Budget / Target |
| :--- | :---: | :---: |
| **Mean Latency** | `46.53 ms` | $\le 40.0\text{ ms}$ (25 FPS) |
| **Median Latency** | `45.77 ms` | - |
| **P95 Latency** | `48.14 ms` | $\le 60.0\text{ ms}$ |
| **Throughput** | `21.49 FPS` | $\ge 20.0\text{ FPS}$ |
| **Meets Real-Time Budget** | `False` | Required for Edge |

## 3. Confidence & Calibration Diagnostics

- **Expected Calibration Error (ECE):** `0.0329`
- **Brier Score:** `0.2546`
- **Correct Predictions Mean Confidence:** `0.8619`
- **Incorrect Predictions Mean Confidence:** `0.5406`

## 4. Top Confused Sign Pairs

| True Sign | Predicted Sign | Error Count | Probable Contributing Factor |
| :--- | :--- | :---: | :--- |
| **dress** | **clothing** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **red** | **pink** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **blue** | **white** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **brown** | **doctor** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **colour** | **red** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |

## 5. Failure Attribution Breakdown

| Failure Category | Count | Proportion |
| :--- | :---: | :---: |
| `SEMANTIC_KINEMATIC_SIMILARITY` | 55 | 91.7% |
| `SEMANTIC_OVERCONFIDENT_CONFUSION` | 5 | 8.3% |
