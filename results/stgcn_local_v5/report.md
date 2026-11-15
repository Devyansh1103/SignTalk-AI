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
| **Top-1 Accuracy** | **0.8744** |
| **Macro Precision** | 0.8958 |
| **Macro Recall** | 0.8795 |
| **Macro F1-Score** | **0.8734** |
| **Weighted F1-Score** | 0.8681 |
| **Top-3 Accuracy** | 0.9659 |
| **Cross-Entropy Loss** | 0.5544 |

## 2. Inference Benchmark & Latency

| Metric | Value | Budget / Target |
| :--- | :---: | :---: |
| **Mean Latency** | `1.58 ms` | $\le 40.0\text{ ms}$ (25 FPS) |
| **Median Latency** | `1.56 ms` | - |
| **P95 Latency** | `1.78 ms` | $\le 60.0\text{ ms}$ |
| **Throughput** | `630.92 FPS` | $\ge 20.0\text{ FPS}$ |
| **Meets Real-Time Budget** | `True` | Required for Edge |

## 3. Confidence & Calibration Diagnostics

- **Expected Calibration Error (ECE):** `0.0432`
- **Brier Score:** `0.1818`
- **Correct Predictions Mean Confidence:** `0.948`
- **Incorrect Predictions Mean Confidence:** `0.6842`

## 4. Top Confused Sign Pairs

| True Sign | Predicted Sign | Error Count | Probable Contributing Factor |
| :--- | :--- | :---: | :--- |
| **dress** | **clothing** | 3 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **husband** | **wife** | 3 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **happy** | **market** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **male** | **man** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |
| **female** | **woman** | 2 | Shared spatial coordinate trajectory or MediaPipe landmark tracking variance. |

## 5. Failure Attribution Breakdown

| Failure Category | Count | Proportion |
| :--- | :---: | :---: |
| `SEMANTIC_KINEMATIC_SIMILARITY` | 59 | 72.8% |
| `SEMANTIC_OVERCONFIDENT_CONFUSION` | 22 | 27.2% |
