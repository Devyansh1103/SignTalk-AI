# SignTalk AI — Real-Time Landmark Ingestion Data Contract

**Document ID:** `DOC-P4P1-CONTRACT-001`  
**Phase:** Phase 4 — Part 1 (Real-Time Camera & Landmark Streaming)  
**System Title:** SignTalk AI: Real-Time Indian Sign Language Translation Platform  
**Target Architecture:** Spatial-Temporal Graph Convolutional Network (`SignTalk_STGCN_v1`)  
**Status:** CERTIFIED & ACTIVE  

---

## 1. Input Specification (Camera Frame Layer)

| Parameter | Specification | Notes / Validation Rules |
| :--- | :--- | :--- |
| **Input Source** | Live Hardware Webcam / Virtual Device / Replay Video | Configurable via `device_index` (integer or file path) |
| **Native Resolution** | $640 \times 480$ standard ($426 \times 240$ supported) | Target resolution configurable in `configs/realtime.yaml` |
| **Color Space** | BGR (Hardware Capture) $\to$ RGB (MediaPipe Ingestion) | Converted deterministically by `FrameProcessor` |
| **Pixel Data Type** | `uint8` with value range $[0, 255]$ | Strict type check enforced in `FrameProcessor.process()` |
| **Horizontal Mirroring** | Optional horizontal flip (`flip_horizontal: true`) | Mirror preview enabled by default for natural signing |
| **Capture Clock** | Monotonic timing source via `time.perf_counter()` | Used for drift-free latency and jitter calculations |

---

## 2. Landmark Detection Specification

Landmark extraction is executed using MediaPipe Tasks Vision API with models situated in `models/mediapipe/`:

| Modality | Nodes Allocated | MediaPipe Model Asset | Confidence Gates |
| :--- | :---: | :--- | :--- |
| **Left Hand Articulators** | Nodes 0 – 20 (21 joints) | `models/mediapipe/hand_landmarker.task` | Detection: 0.35, Presence: 0.35, Tracking: 0.35 |
| **Right Hand Articulators** | Nodes 21 – 41 (21 joints) | `models/mediapipe/hand_landmarker.task` | Detection: 0.35, Presence: 0.35, Tracking: 0.35 |
| **Upper Pose Anchors** | Nodes 42 – 52 (11 anchors) | `models/mediapipe/pose_landmarker_full.task` | Detection: 0.50, Presence: 0.50, Tracking: 0.50 |
| **Facial Contours** | Nodes 53 – 92 (40 contours) | `models/mediapipe/face_landmarker.task` | Detection: 0.40 (Optional toggle; defaults to upper pose cranial anchors) |

---

## 3. Real-Time Output Tensor Contract

Each captured frame emitted by `RealTimeLandmarkStream` satisfies the canonical representation:

```text
Individual Frame Landmark Object:
  Class:                 LandmarkFrame (src.realtime.types)
  Node Cardinality:      V = 93 nodes
  Feature Channels:      C = 3 channels [X, Y, Z]
  Coordinate Dtype:      numpy.float32 / torch.float32
  Mask Dtype:            numpy.bool_ / torch.float32

Tensor Geometry:
  Unbatched Frame:       [C, 1, V] = [3, 1, 93]
  Batched Frame:         [1, C, 1, V] = [1, 3, 1, 93]
  Accumulated Window:    [B, C, T, V] = [1, 3, 45, 93] (T=45 frames @ 25 FPS)

Channels:
  Channel 0:             X coordinate (torso-centered, scaled by shoulder baseline)
  Channel 1:             Y coordinate (torso-centered, scaled by shoulder baseline)
  Channel 2:             Z relative depth coordinate (scaled by shoulder baseline)
```

---

## 4. Normalization Procedure

Coordinate normalization is strictly synchronized with Phase 2 / Phase 3 training preprocessing:

1. **Torso Reference Anchors:**
   - Left Shoulder = Node 47
   - Right Shoulder = Node 48
2. **Torso Center ($C_{\text{torso}}$):**
   $$C_{\text{torso}} = \frac{P_{47} + P_{48}}{2}$$
   *Fallback:* If only one shoulder is visible, that shoulder serves as center. If neither shoulder is visible, upper pose centroid is used; otherwise default to $(0.5, 0.5, 0.0)$.
3. **Scale Reference ($S$):**
   $$S = \max\left(\|P_{47} - P_{48}\|_2, \; 1.0\times 10^{-4}\right)$$
   *Fallback:* $S = 0.25$ if shoulder distance is unobtainable.
4. **Temporal Anchor Smoothing (Real-Time Mode):**
   To prevent high-frequency breathing artifacts across streaming frames:
   $$C_{\text{torso}}^{(t)} = (1 - \alpha) C_{\text{torso}}^{(t-1)} + \alpha C_{\text{torso, raw}}^{(t)} \quad (\alpha = 0.25)$$
   $$S^{(t)} = (1 - \alpha) S^{(t-1)} + \alpha S_{\text{raw}}^{(t)} \quad (\alpha = 0.25)$$
5. **Coordinate Centering:**
   $$x'_i = \frac{x_i - C_{x}}{S}, \quad y'_i = \frac{y_i - C_{y}}{S}, \quad z'_i = \frac{z_i - C_{z}}{S}$$
6. **Missing-Value Policy:**
   Unobserved nodes ($\text{mask}[i] == \text{False}$) are zeroed out:
   $$P'_i = (0.0, 0.0, 0.0)$$

---

## 5. Quality Validation Gate

Each incoming frame is assessed in real-time:

$$\text{Quality Score} = 0.40 \cdot C_{\text{pose}} + 0.40 \cdot C_{\text{dom\_hand}} + 0.15 \cdot C_{\text{non\_dom\_hand}} + 0.05 \cdot C_{\text{face}}$$

### Status Classification:
* **`GOOD`** ($\text{Score} \ge 0.75$): Full upper-body pose and active signing hand tracked with high confidence.
* **`ACCEPTABLE`** ($\text{Score} \ge 0.50$): Robust pose and at least one hand tracked.
* **`REVIEW`** ($\text{Score} \ge 0.35$): Marginal tracking (e.g. signer present but hands outside camera frame).
* **`REJECT`** ($\text{Score} < 0.35$ or $\text{NaN}$ detected or pose lost): Unusable frame. Downstream temporal window buffer drops or rejects low-quality windows.

### Anomaly Flags:
* `has_nan_or_inf`: Boolean flag checking for numeric divergence.
* `sudden_jumps_detected`: Set if joint Euclidean displacement between consecutive frames exceeds $0.35$ normalized units.
* `rejection_reason`: Explicit diagnostic string for logging and UI feedback.

---

## 6. Error Handling & Recovery Protocols

1. **No Camera Found:** Emits `CameraInitializationError` with diagnostic message and instructions to verify permissions and device index.
2. **Device Disconnection:** Camera background thread attempts automatic reconnection up to 3 times before graceful shutdown.
3. **Queue Overflow:** Fixed ring buffer drops stale frames (`drop_oldest` policy) and records metric in `total_dropped_frames`.
4. **Dimension Mismatch:** `LandmarkFrame` rejects invalid shape inputs with hard `ValueError` before data can propagate downstream.
