# Confidence Filtering and Temporal Smoothing

**Project:** SignTalk AI: Real-Time Indian Sign Language Translation Platform  
**Component:** Prediction History, Confidence Filtering, and Temporal Smoothing Architecture  
**Phase:** Phase 4 — Part 3

---

## 1. Architectural Overview

Real-time sliding window inference updates predictions at high cadence (e.g., 5 frames stride = 200 ms update interval). Even when a signer performs a single steady gesture, individual window inferences can fluctuate due to landmark jitter, brief self-occlusions, or transition boundaries:

$$\text{Raw Window Predictions} \to \text{Confidence Gate} \to \text{Prediction History} \to \text{Smoothing Strategy} \to \text{Stable Candidate}$$

```text
               ST-GCN Window Output
                         │
                         ▼
             Raw Prediction + Confidence
                         │
                         ▼
             [Confidence Filter Gate]
             (conf >= 0.65 & quality >= 0.40)
                   ┌─────┴─────┐
                   │           │
                Valid      Low Conf / Poor Quality
                   │           │
                   ▼           ▼
            Active Record   "UNCERTAIN" Record
                   │           │
                   └─────┬─────┘
                         │
                         ▼
             [Prediction History (FIFO)]
                         │
                         ▼
            [Modular Temporal Smoother]
            - Majority Voting (Baseline)
            - Confidence-Weighted Evidence
            - Temporal Stability (K-consecutive)
                         │
                         ▼
               Smoothed Candidate Sign
```

---

## 2. Confidence Calibration & Terminology

Phase 3 model evaluation determined that ST-GCN has an Expected Calibration Error of $0.2713$. Consequently, raw softmax output $0.90$ does not mathematically represent a 90% objective probability of correctness.

Throughout the codebase and user interfaces, we strictly enforce calibrated terminology:
- **`model confidence score`**: The raw maximum softmax activation emitted by ST-GCN.
- **`input quality score`**: The normalized tracking confidence and presence score emitted by MediaPipe.
- **`stability score`**: The ratio of consensus evidence over the temporal window.

These values are never multiplicatively fused without calibration; they operate as orthogonal filtering gates.

---

## 3. Prediction History Buffer

Implemented in [`src/realtime/prediction_history.py`](file:///d:/SignAI/src/realtime/prediction_history.py):
- **Bounded Rolling Queue (`PredictionHistory`)**: Default capacity of $20$ items (4 seconds of inference history @ 5 strides/sec).
- **Structure (`PredictionRecord`)**:
  ```python
  @dataclass
  class PredictionRecord:
      timestamp: float
      window_start: float
      window_end: float
      class_id: int
      label: str
      confidence: float
      input_quality: float
      gloss: str = ""
      inference_latency_ms: float = 0.0
      is_valid_quality: bool = True
      probabilities: Optional[np.ndarray] = None
  ```

---

## 4. Modular Smoothing Strategies

Implemented under [`src/realtime/smoothing/`](file:///d:/SignAI/src/realtime/smoothing/):

### 4.1 Majority Vote Smoother (`MajorityVoteSmoother`)
- Computes mode of class labels across the trailing $N$ window records ($N=5$).
- Requires minimum consensus quorum $K$ ($K=3$).
- Detects exact ties between top classes and immediately tags output as `UNSTABLE` rather than making arbitrary guesses.
- Filters out low-confidence records before voting.

### 4.2 Confidence-Weighted Smoother (`ConfidenceWeightedSmoother`)
- Computes accumulated evidence for each class $c$:
  $$\text{Score}(c) = \sum_{t=0}^{N-1} \text{conf}_t(c) \cdot \gamma^{N-1-t}$$
  where $\gamma \in (0, 1]$ is the recency decay factor (default $0.85$).
- Allows sustained moderate-confidence gestures to override isolated high-confidence noise spikes.

### 4.3 Temporal Stability Smoother (`TemporalStabilitySmoother`)
- Requires strictly $K$ consecutive identical predictions ($K=2$) before declaring stability.
- Rapidly flags any alternation as `UNSTABLE`.

---

## 5. Comparative Evaluation on Validation Sequences

Empirical comparison evaluated on validation sequences ([`results/smoothing/smoothing_comparison.csv`](file:///d:/SignAI/results/smoothing/smoothing_comparison.csv)):

| Method | Prediction Accuracy | Flip Rate | Mean Stability Duration | Processing Latency (Mean) | Processing Latency (p95) | Emitted Events | False Events |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Predictions** | 75.0% | 0.00 | 0.873 s | 0.001 ms | 0.001 ms | - | - |
| **Majority Vote ($N=5, K=3$)** | 54.2% | 0.00 | 1.067 s | 0.134 ms | 0.377 ms | - | - |
| **Confidence Weighted ($N=5$)** | 60.4% | 0.00 | 1.067 s | 0.085 ms | 0.170 ms | - | - |
| **Temporal Stability ($K=2$)** | 58.3% | 0.00 | 0.505 s | 0.051 ms | 0.085 ms | - | - |
| **Full Pipeline (MV + FSM + Dedup)** | 54.2% | 0.00 | 1.067 s | 0.114 ms | 0.196 ms | 9 | 0 |

### Latency Overhead Analysis
From [`results/temporal_stability/benchmark_latency.json`](file:///d:/SignAI/results/temporal_stability/benchmark_latency.json):
- **ST-GCN Inference:** $143.86$ ms
- **Confidence Filter:** $0.027$ ms ($27$ µs)
- **Prediction History:** $0.006$ ms ($6$ µs)
- **Majority Vote Smoothing:** $0.152$ ms ($152$ µs)
- **Total Post-STGCN Overhead:** $\mathbf{0.201}$ **ms** ($< 0.25$ ms)

The temporal smoothing and filtering layer adds virtually **zero computational overhead** (accounting for only $0.14\%$ of total inference cycle time).
