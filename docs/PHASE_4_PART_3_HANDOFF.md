# Phase 4 — Part 3 Handoff Specification

**SignTalk AI: Real-Time Indian Sign Language Translation Platform**  
*Transition from Sliding-Window Temporal Inference to Continuous Sign Detection & Temporal Smoothing*

---

## 1. System Status & Verification

Phase 4 Part 2 has successfully established a certified, real-time sliding-window inference pipeline connecting the MediaPipe landmark stream to the trained `SignTalk_STGCN_v1` model. The pipeline runs deterministically, satisfies strict shape and numerical invariants, and produces validated predictions at 5.0 updates per second.

---

## 2. Input Specifications (Provided to Part 3)

| Property | Value / Contract | Reference |
| :--- | :--- | :--- |
| **Landmark Frame Type** | [`LandmarkFrame`](file:///d:/SignAI/src/realtime/types.py#L75) dataclass | `src/realtime/types.py` |
| **Landmark Format** | 93 nodes $\times$ 3 channels $(X, Y, Z)$ | `src/data/node_schema.py` |
| **Coordinate Space** | Torso-scale normalized (mid-shoulder center, shoulder-width scale) | `src/preprocessing/normalizer.py` |
| **Sequence Length ($T$)** | Fixed $45\text{ frames}$ ($1.80\text{ s}$ duration) | `TARGET_SEQUENCE_LENGTH = 45` |
| **Ingestion Frame Rate** | $25.0\text{ FPS}$ ($\Delta t = 40.0\text{ ms}$) | Configured in `configs/realtime.yaml` |
| **Temporal Sampling** | Native $25\text{ FPS}$ capture + 1..5 dropped-frame linear interpolation | `docs/REALTIME_TEMPORAL_SAMPLING.md` |
| **Buffer Capacity** | 45 frames FIFO rolling buffer | `src/realtime/temporal_buffer.py` |
| **Inference Stride ($s$)** | 5 frames ($200\text{ ms}$ cadence, $88.9\%$ temporal overlap) | `src/realtime/inference_scheduler.py` |

---

## 3. Model Specifications

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Model Checkpoint** | `experiments/stgcn/checkpoints/best_checkpoint.pt` | Certified best validation checkpoint (Epoch 34) |
| **Architecture** | Spatial-Temporal Graph Convolutional Network (`SignSTGCN`) | 6 ST-GCN blocks, residual connections, edge weights |
| **Parameters** | 2,137,818 trainable weights | Evaluation mode (`eval()`, `requires_grad=False`) |
| **Input Shape** | $[B, C, T, V] = [1, 3, 45, 93]$ | Batch 1, Channels 3, Frames 45, Nodes 93 |
| **Mask Shape** | $[B, 1, T, V] = [1, 1, 45, 93]$ | Visibility and modality mask |
| **Execution Device** | Auto-resolved (`cpu` or `cuda`) | GPU preferred, CPU verified |
| **Raw Output** | Logits tensor $[1, 10]$ | Softmax converted to probabilities |

---

## 4. Prediction Output Contract

Phase 4 Part 3 receives structured [`PredictionResult`](file:///d:/SignAI/src/realtime/types.py#L150) instances:

```python
PredictionResult(
    class_id: int,                      # 0..9 canonical integer class ID
    label: str,                         # e.g. "hello", "thankyou", "house"
    gloss: str,                         # e.g. "HELLO", "THANK_YOU", "HOUSE"
    translation: str,                   # e.g. "Hello", "Thank you", "House"
    confidence: float,                  # Argmax softmax probability [0.0..1.0]
    probabilities: np.ndarray,          # Full 10-class probability distribution
    logits: np.ndarray,                 # Raw unnormalized model logits
    top_k: List[Tuple[int, str, float]],# Top-3 ranked alternatives [(cid, label, prob)]
    timestamp: float,                   # Epoch timestamp of prediction emission
    window_start_time: float,           # Monotonic timestamp of earliest frame in window
    window_end_time: float,             # Monotonic timestamp of latest frame in window
    window_frame_count: int,            # Fixed at 45
    inference_start_time: float,        # Performance profiling timestamps
    inference_end_time: float,
    inference_latency_ms: float,        # Median 120.04 ms on CPU
    input_quality: float,               # Mean landmark quality score across window
    is_valid_quality: bool,             # True if valid frames >= 15 and active hands >= 15%
    buffer_length: int,                 # Current buffer depth (45)
    device: str                         # "cpu" or "cuda"
)
```

### Canonical Class Mapping (Invariant)
- `0`: `hello`
- `1`: `thankyou`
- `2`: `good`
- `3`: `happy`
- `4`: `monday`
- `5`: `car`
- `6`: `bird`
- `7`: `house`
- `8`: `time`
- `9`: `teacher`

---

## 5. Timing Profile Summary

- **Camera Capture FPS**: $25.0\text{ FPS}$
- **Landmark Pipeline FPS**: $25.0\text{ FPS}$ ($12.4\text{ ms}$ extraction per frame)
- **ST-GCN Inference Latency**: $120.0\text{ ms}$ (median CPU) / $149.2\text{ ms}$ (P95)
- **Prediction Emission Cadence**: $5.0\text{ predictions/second}$ (every $200\text{ ms}$ at Stride=5)
- **Temporal Observation Delay**: $1,800\text{ ms}$ ($45\text{ frames}$)
- **End-to-End Latency**: $\approx 320\text{ ms}$ (Observation step delay $+$ inference latency)

---

## 6. Known Limitations & Edge Cases Discovered

1. **Jitter in Transitional Windows**:
   Because windows slide continuously with an 88.9% overlap, transitional frames between gestures can produce flickering or low-confidence alternating predictions (e.g. oscillating between `hello` and `thankyou` during hand descent).
2. **Repeated Emission of Static Signs**:
   A signer holding a static sign will emit repeated identical predictions every $200\text{ ms}$. Duplicate prediction suppression is required in Part 3.
3. **Absence of Silence / Rest Class**:
   When hands rest at the sides without signing, the closed 10-class model will assign non-zero probability mass to the closest phonological match. Threshold-based rejection and rest-state detection are necessary.
4. **Fast Signing Displacements**:
   Rapid hand acceleration across 2..3 frames can cause temporary MediaPipe tracking dropouts. While linear interpolation maintains continuity, sudden jump filtering must be accounted for in continuous segmentation.

---

## 7. Direct Requirements for Phase 4 — Part 3

Phase 4 Part 3 will ingest the continuous prediction stream emitted by [`RealTimeSignPredictor`](file:///d:/SignAI/src/realtime/realtime_predictor.py) and implement:

```
Real-Time ST-GCN Predictions (5.0 Hz)
                ↓
    Temporal Confidence Filter
                ↓
    Sliding Prediction History
                ↓
  Temporal Voting / Exponential Smoothing
                ↓
Duplicate Suppression (Debounce / Cooldown)
                ↓
Continuous Sign Segmentation (Transition vs Hold)
                ↓
    Discrete Stable Sign Events
```

### Components to Implement in Part 3:
1. `PredictionHistoryBuffer`: Sliding FIFO of recent `PredictionResult` objects.
2. `TemporalPredictionSmoother`: Exponential moving average (EMA) or majority voting over recent windows.
3. `SignEventDetector`: Peak confidence detector converting rolling predictions into discrete sign onset, hold, and release events.
4. `DuplicateSuppressor`: Debounce logic preventing repetitive emissions of sustained gestures.
5. `RestStateDetector`: Quality and velocity-aware rejection gate filtering idle postures.
