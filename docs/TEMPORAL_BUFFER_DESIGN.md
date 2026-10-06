# Temporal Buffer Design Specification

**SignTalk AI: Real-Time Indian Sign Language Translation Platform**  
*Phase 4 — Part 2: Sliding-Window Inference & Temporal Prediction*

---

## 1. Overview and Objectives

The [`TemporalBuffer`](file:///d:/SignAI/src/realtime/temporal_buffer.py) acts as the bridge between the frame-by-frame landmark stream and the temporal sequence requirements of the ST-GCN graph neural network.

Primary design requirements:
1. **Fixed Temporal Capacity**: Strictly maintain $T = 45\text{ frames}$ ($1.80\text{ s}$ @ $25\text{ FPS}$).
2. **Zero Overhead FIFO**: Fast $O(1)$ amortized append and eviction without reallocating arrays.
3. **Dropped-Frame Recovery**: Intelligently bridge temporal gaps of 1..5 dropped frames using linear coordinate interpolation.
4. **Rich Diagnostic Retention**: Maintain per-frame timestamps, raw and normalized coordinates, modality detection statistics, and quality scores.
5. **Quality Gating**: Validate window integrity prior to forward evaluation.

---

## 2. Data Structure and Memory Management

### 2.1 Double-Ended Queue (Deque)
The internal buffer is implemented using Python's `collections.deque(maxlen=45)`.
- When the buffer reaches 45 frames, each subsequent `append(frame)` automatically evicts the oldest frame from the left in $O(1)$ time.
- Memory consumption is strictly bounded: 45 frames $\times \approx 3.2\text{ KB}$ per frame packet $\approx 144\text{ KB}$ in memory.
- Extraction of the model window via `get_window()` produces a shallow list reference of length 45, avoiding unnecessary data copies.

---

## 3. Frame Dropping & Interpolation Bridging

When webcam frame delivery experiences jitter, frame indices become non-contiguous:
$$\text{gap} = \text{frame\_id}_{k} - \text{frame\_id}_{k-1} - 1$$

If $1 \le \text{gap} \le 10$:
For each intermediate step $s \in [1, \text{gap}]$:
$$\alpha = \frac{s}{\text{gap} + 1}$$
$$\mathbf{X}_{\text{interp}} = (1 - \alpha) \mathbf{X}_{k-1} + \alpha \mathbf{X}_{k}$$
$$\mathbf{t}_{\text{interp}} = \mathbf{t}_{k-1} + s \cdot \frac{\mathbf{t}_k - \mathbf{t}_{k-1}}{\text{gap} + 1}$$

Interpolated frames receive combined boolean masks and inherit the parent quality report with an interpolation tag. If $\text{gap} > 10$, interpolation is aborted to prevent synthetic hallucination across extended occlusions.

---

## 4. Sliding Window & Stride Mechanics

Let $t$ denote the camera ingestion frame index.
- **Buffer Warm-up Period**: Frames $0 \le t < 44$. `is_ready()` returns `False`. No inference is triggered.
- **Initial Inference**: Frame $t = 44$. Buffer contains $[0, 1, \dots, 44]$ ($45\text{ frames}$). ST-GCN forward evaluation runs.
- **Stride Progression ($s = 5$)**:
  - Frame $t = 45..48$: Frames appended, no inference triggered.
  - Frame $t = 49$: Buffer contains $[5, 6, \dots, 49]$. Inference triggered.
  - Frame $t = 54$: Buffer contains $[10, 11, \dots, 54]$. Inference triggered.

At each step, the sliding window retains a $40$-frame ($88.9\%$) overlap with the preceding window, delivering smooth spatiotemporal continuity.

---

## 5. Upstream Quality Gate

Before dispatching the window to the model runner, `is_window_valid()` validates:
1. **Valid Frame Ratio**: At least 15 of the 45 frames ($\ge 33.3\%$) must be classified as valid by the `QualityChecker`.
2. **Active Hand Requirement**: At least $15\%$ of frames in the window must exhibit an active left or right hand detection.

If a window fails these gates, inference executes with `is_valid_quality=False`, allowing downstream consumers to reject ungrounded predictions while maintaining system liveness.
