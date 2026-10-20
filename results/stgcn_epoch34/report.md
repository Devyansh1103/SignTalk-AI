# SignTalk AI — Evaluation Report: SignSTGCN

- **Evaluated Checkpoint:** `experiments/stgcn/checkpoints/best_checkpoint.pt`
- **Evaluation Split:** `test`
- **Sample Count:** `332`
- **Model Size:** `25.06 MB`
- **Total Parameters:** `2,171,485`

---

## 1. Primary Classification Performance

| Metric | Measured Value |
| :--- | :---: |
| **Top-1 Accuracy** | **0.6536** |
| **Macro Precision** | 0.6751 |
| **Macro Recall** | 0.6655 |
| **Macro F1-Score** | **0.6345** |
| **Weighted F1-Score** | 0.6263 |
| **Top-3 Accuracy** | 0.8765 |
| **Cross-Entropy Loss** | 1.1681 |

## 2. Inference Benchmark & Latency

| Metric | Value | Budget / Target |
| :--- | :---: | :---: |
| **Mean Latency** | `64.26 ms` | $\le 40.0\text{ ms}$ (25 FPS) |
| **Median Latency** | `63.97 ms` | - |
| **P95 Latency** | `67.49 ms` | $\le 60.0\text{ ms}$ |
| **Throughput** | `15.56 FPS` | $\ge 20.0\text{ FPS}$ |
| **Meets Real-Time Budget** | `False` | Required for Edge |

## 3. Confidence & Calibration Diagnostics

- **Expected Calibration Error (ECE):** `0.0463`
- **Brier Score:** `0.4571`
- **Correct Predictions Mean Confidence:** `0.7517`
- **Incorrect Predictions Mean Confidence:** `0.4568`

## 4. Top Confused Sign Pairs

| True Sign | Predicted Sign | Error Count | Probable Contributing Factor |
| :--- | :--- | :---: | :--- |
| **goodnight** | **goodafternoon** | 3 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **ugly** | **beautiful** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **fish** | **time** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **dress** | **clothing** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **red** | **pink** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |

## 5. Failure Attribution Breakdown

| Failure Category | Count | Proportion |
| :--- | :---: | :---: |
| `SEMANTIC_KINEMATIC_SIMILARITY` | 110 | 95.7% |
| `SEMANTIC_OVERCONFIDENT_CONFUSION` | 5 | 4.3% |
