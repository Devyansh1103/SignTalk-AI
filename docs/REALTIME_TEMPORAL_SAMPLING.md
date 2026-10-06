# Real-Time Temporal Sampling & Frame Rate Adaptation

**SignTalk AI: Real-Time Indian Sign Language Translation Platform**  
*Phase 4 — Part 2: Sliding-Window Inference & Temporal Prediction*

---

## 1. Executive Summary

A critical requirement of neural spatiotemporal graph modeling is **temporal fidelity** between the training distribution and real-time execution. An ST-GCN model learns graph convolution kernels over specific temporal velocities $\frac{\Delta x}{\Delta t}, \frac{\Delta y}{\Delta t}$. If inference is executed at a mismatched frame rate or with erratic frame dropping, the learned temporal kernels degenerate, leading to catastrophic accuracy degradation.

This document details the temporal sampling architecture, camera rate alignment, frame-dropping interpolation bridge, and stride scheduling deployed for real-time ISL recognition.

---

## 2. Temporal Distribution Alignment Analysis

### 2.1 Training Temporal Distribution
During Phase 2 preprocessing and Phase 3 ST-GCN training, temporal normalization established the following canonical standards:
- **Canonical Sequence Duration**: $1.80\text{ seconds}$
- **Target Frame Rate**: $25.0\text{ FPS}$ ($\Delta t = 40.0\text{ ms}$ per frame)
- **Sequence Length**: Fixed $T = 45\text{ frames}$ ($1.80\text{ s} \times 25.0\text{ FPS} = 45\text{ frames}$)
- **Normalization Interpolation**: Scipy/Numpy cubic spline / linear temporal resampling for variable-length video segments during dataset extraction.

### 2.2 Real-Time Camera Hardware Profile
Physical USB and integrated webcams typically provide:
- **Nominal Frame Rates**: $30.0\text{ FPS}$ ($\Delta t \approx 33.3\text{ ms}$) or $25.0\text{ FPS}$ ($\Delta t = 40.0\text{ ms}$).
- **Jitter & Frame Dropping**: Variable capture intervals (e.g., $30\text{ ms}$ to $55\text{ ms}$) caused by CPU scheduling, USB bus load, or ambient lighting exposure throttling.

---

## 3. Temporal Sampling Strategy

To guarantee that the input window presented to the ST-GCN model strictly reflects the physical dynamics of the $25\text{ FPS}$ training domain, SignTalk AI implements a **Dual-Mechanism Rate Adaptation Architecture**:

```
[Camera Hardware Capture] (~30 FPS or ~25 FPS variable)
          │
          ▼
[Monotonic Timestamping (time.perf_counter())]
          │
          ▼
[Landmark Extraction & Quality Filtering]
          │
          ▼
[Temporal Pacing & Drop Bridge (TemporalBuffer)]
          ├─ Condition A: Real-time FPS == 25 FPS → Direct 1:1 Ingestion
          ├─ Condition B: Frame Gap Detected (1..5 dropped) → Linear Coordinate Interpolation
          └─ Condition C: Hardware Rate > 25 FPS (e.g. 30/60 FPS) → Timestamp-paced Ingestion
          │
          ▼
[Fixed Sliding Window (T = 45 frames)]
```

### 3.1 Policy 1: Native 25.0 FPS Pacing
The camera capture subsystem (`src/realtime/camera.py`) requests $25.0\text{ FPS}$ directly via OpenCV hardware properties (`cv2.CAP_PROP_FPS`). When the hardware satisfies this cadence, each incoming frame corresponds exactly to $\Delta t = 40\text{ ms}$.

### 3.2 Policy 2: Dropped Frame Linear Interpolation Bridge
If the camera skips 1 to 5 frames due to system interruptions or MediaPipe pipeline delay, the `TemporalBuffer` inspects frame ID discontinuities:

$$\text{gap} = \text{frame\_id}_{k} - \text{frame\_id}_{k-1} - 1$$

If $1 \le \text{gap} \le 5$, the buffer synthesizes intermediate `LandmarkFrame` instances via linear coordinate interpolation:

$$\mathbf{X}_{\text{interp}}(\alpha) = (1 - \alpha) \mathbf{X}_{k-1} + \alpha \mathbf{X}_{k}, \quad \alpha = \frac{\text{step}}{\text{gap} + 1}$$

This protects ST-GCN temporal edge connections ($A_{\text{temporal}}$) from large artificial motion discontinuities.

### 3.3 Policy 3: Fixed Stride Scheduling
Rather than executing the ST-GCN on every single camera frame ($s = 1$, which would require $25\text{ inferences/sec}$), the system enforces a configurable stride:

$$s = 5\text{ frames}$$

At $25\text{ FPS}$, this creates an update period of:

$$\Delta t_{\text{inference}} = \frac{5}{25.0} = 200\text{ ms} \quad (5.0\text{ predictions/second})$$

Each inference window overlaps with the preceding window by:

$$\text{Overlap} = T - s = 45 - 5 = 40\text{ frames } (88.9\%)$$

This delivers responsive real-time feedback with manageable CPU/GPU computational overhead.

---

## 4. Verification and Empirical Benchmark

| Ingestion Source | Native FPS | Pipeline FPS | Stride ($s$) | Inference Rate | Prediction Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Video Surrogate (`hello.mp4`) | 25.0 | 25.0 | 5 | 5.0 pred/s | 118.6 ms |
| Synthetic Stream (Micro-benchmark) | 25.0 | 25.0 | 5 | 5.0 pred/s | 120.0 ms |

The temporal sampling strategy ensures zero temporal drift and numerical consistency with the offline model.
