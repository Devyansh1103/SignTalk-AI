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
| **Top-1 Accuracy** | **0.8614** |
| **Macro Precision** | 0.8930 |
| **Macro Recall** | 0.8652 |
| **Macro F1-Score** | **0.8563** |
| **Weighted F1-Score** | 0.8538 |
| **Top-3 Accuracy** | 0.9548 |
| **Cross-Entropy Loss** | 0.6313 |

## 2. Inference Benchmark & Latency

| Metric | Value | Budget / Target |
| :--- | :---: | :---: |
| **Mean Latency** | `3.71 ms` | $\le 40.0\text{ ms}$ (25 FPS) |
| **Median Latency** | `3.56 ms` | - |
| **P95 Latency** | `4.29 ms` | $\le 60.0\text{ ms}$ |
| **Throughput** | `269.21 FPS` | $\ge 20.0\text{ FPS}$ |
| **Meets Real-Time Budget** | `True` | Required for Edge |

## 3. Confidence & Calibration Diagnostics

- **Expected Calibration Error (ECE):** `0.03`
- **Brier Score:** `0.2005`
- **Correct Predictions Mean Confidence:** `0.9265`
- **Incorrect Predictions Mean Confidence:** `0.624`

## 4. Top Confused Sign Pairs

| True Sign | Predicted Sign | Error Count | Probable Contributing Factor |
| :--- | :--- | :---: | :--- |
| **dress** | **clothing** | 3 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **happy** | **radio** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **red** | **pink** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **blue** | **pencil** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **loud** | **table** | 1 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |

## 5. Failure Attribution Breakdown

| Failure Category | Count | Proportion |
| :--- | :---: | :---: |
| `SEMANTIC_KINEMATIC_SIMILARITY` | 36 | 78.3% |
| `SEMANTIC_OVERCONFIDENT_CONFUSION` | 10 | 21.7% |
