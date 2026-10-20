# SignTalk AI — Evaluation Report: SignSTGCN

- **Evaluated Checkpoint:** `best_checkpoint.pt`
- **Evaluation Split:** `test`
- **Sample Count:** `332`
- **Model Size:** `25.29 MB`
- **Total Parameters:** `2,171,485`

---

## 1. Primary Classification Performance

| Metric | Measured Value |
| :--- | :---: |
| **Top-1 Accuracy** | **0.8554** |
| **Macro Precision** | 0.8819 |
| **Macro Recall** | 0.8605 |
| **Macro F1-Score** | **0.8501** |
| **Weighted F1-Score** | 0.8452 |
| **Top-3 Accuracy** | 0.9548 |
| **Cross-Entropy Loss** | 0.5551 |

## 2. Inference Benchmark & Latency

| Metric | Value | Budget / Target |
| :--- | :---: | :---: |
| **Mean Latency** | `78.22 ms` | $\le 40.0\text{ ms}$ (25 FPS) |
| **Median Latency** | `75.45 ms` | - |
| **P95 Latency** | `84.32 ms` | $\le 60.0\text{ ms}$ |
| **Throughput** | `12.78 FPS` | $\ge 20.0\text{ FPS}$ |
| **Meets Real-Time Budget** | `False` | Required for Edge |

## 3. Confidence & Calibration Diagnostics

- **Expected Calibration Error (ECE):** `0.03`
- **Brier Score:** `0.2104`
- **Correct Predictions Mean Confidence:** `0.8996`
- **Incorrect Predictions Mean Confidence:** `0.6076`

## 4. Top Confused Sign Pairs

| True Sign | Predicted Sign | Error Count | Probable Contributing Factor |
| :--- | :--- | :---: | :--- |
| **dress** | **clothing** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **shoes** | **tshirt** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **pink** | **red** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **colour** | **red** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **loud** | **cat** | 1 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |

## 5. Failure Attribution Breakdown

| Failure Category | Count | Proportion |
| :--- | :---: | :---: |
| `SEMANTIC_KINEMATIC_SIMILARITY` | 41 | 85.4% |
| `SEMANTIC_OVERCONFIDENT_CONFUSION` | 7 | 14.6% |
