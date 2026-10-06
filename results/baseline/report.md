# SignTalk AI — Evaluation Report: Baseline BiLSTM

- **Evaluated Checkpoint:** `experiments/baseline/checkpoints/best_checkpoint.pt`
- **Evaluation Split:** `test`
- **Sample Count:** `12`
- **Model Size:** `6.88 MB`
- **Total Parameters:** `597,898`

---

## 1. Primary Classification Performance

| Metric | Measured Value |
| :--- | :---: |
| **Top-1 Accuracy** | **0.4167** |
| **Macro Precision** | 0.2917 |
| **Macro Recall** | 0.4500 |
| **Macro F1-Score** | **0.3067** |
| **Weighted F1-Score** | 0.3111 |
| **Top-3 Accuracy** | 0.7500 |
| **Cross-Entropy Loss** | 1.8946 |

## 2. Inference Benchmark & Latency

| Metric | Value | Budget / Target |
| :--- | :---: | :---: |
| **Mean Latency** | `14.07 ms` | $\le 40.0\text{ ms}$ (25 FPS) |
| **Median Latency** | `12.16 ms` | - |
| **P95 Latency** | `30.63 ms` | $\le 60.0\text{ ms}$ |
| **Throughput** | `71.1 FPS` | $\ge 20.0\text{ FPS}$ |
| **Meets Real-Time Budget** | `True` | Required for Edge |

## 3. Confidence & Calibration Diagnostics

- **Expected Calibration Error (ECE):** `0.2108`
- **Brier Score:** `0.7712`
- **Correct Predictions Mean Confidence:** `0.4003`
- **Incorrect Predictions Mean Confidence:** `0.4147`

## 4. Top Confused Sign Pairs

| True Sign | Predicted Sign | Error Count | Probable Contributing Factor |
| :--- | :--- | :---: | :--- |
| **thankyou** | **happy** | 1 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **good** | **hello** | 1 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **monday** | **happy** | 1 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **bird** | **car** | 1 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **time** | **car** | 1 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |

## 5. Failure Attribution Breakdown

| Failure Category | Count | Proportion |
| :--- | :---: | :---: |
| `SEMANTIC_KINEMATIC_SIMILARITY` | 5 | 71.4% |
| `VISUAL_DETECTION_FAILURE` | 2 | 28.6% |
