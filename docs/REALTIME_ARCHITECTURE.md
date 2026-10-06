# SignTalk AI — Real-Time Streaming Architecture

**Document ID:** `DOC-P4P1-ARCH-001`  
**Phase:** Phase 4 — Part 1 (Real-Time Camera & Landmark Streaming)  
**System Title:** SignTalk AI: Real-Time Indian Sign Language Translation Platform  
**Status:** IMPLEMENTED & BENCHMARKED  

---

## 1. High-Level Architectural Flow

```text
                     PHYSICAL CAMERA / VIDEO STREAM
                                   │
                                   ▼
                       [ Camera Layer (camera.py) ]
                        - DirectShow / V4L2 Device
                        - Monotonic Timestamping (time.perf_counter)
                        - Thread-Safe Ring Buffer (drop_oldest)
                                   │
                              FramePacket
                                   │
                                   ▼
                   [ Frame Preprocessor (frame_processor.py) ]
                        - Dimension & Dtype Validation
                        - Horizontal Flip (Webcam Mirroring)
                        - Color Space Conversion (BGR → RGB)
                                   │
                           Processed FramePacket
                                   │
                                   ▼
                  [ MediaPipe Multimodal Extractor ]
               ┌───────────────────┼───────────────────┐
               ▼                   ▼                   ▼
         Hand Landmarker    Pose Landmarker    Face Landmarker
         (Left + Right)     (11 Upper Anchors) (Salient Contours)
               └───────────────────┬───────────────────┘
                                   │
                                   ▼
                       [ Deterministic Landmark Fusion ]
                        - 93-Node Canonical Schema (0..92)
                        - Strict 3-Channel Array (93, 3)
                        - Modality Detection Attribution
                                   │
                                   ▼
                       [ Torso-Scale Normalizer ]
                        - Mid-Shoulder Centering (Nodes 47 & 48)
                        - Shoulder Distance Scaling
                        - EMA Temporal Anchor Smoothing (α = 0.25)
                                   │
                                   ▼
                       [ Quality & Anomaly Gate ]
                        - NaN / Inf Detection
                        - Coordinate Bounds Checking
                        - Temporal Jump Anomaly Detection
                        - Multi-Tier Quality Scoring (GOOD..REJECT)
                                   │
                                   ▼
                       Canonical LandmarkFrame Stream
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
       Real-Time Debug Visualizer       Downstream Phase 4 Part 2
       (Skeletal Overlay + HUD)         (Temporal Sequence Buffer)
```

---

## 2. Component Design & Responsibilities

### 2.1 Camera Capture Layer (`src/realtime/camera.py`)
* **Decoupled Threading:** Camera capture can run in a dedicated background worker (`threaded=True`) to decouple physical camera I/O shutter timing from landmark inference computation.
* **Bounded Ring Buffer:** Uses a thread-safe `queue.Queue(maxsize=2)`. When camera acquisition runs faster than landmark extraction, oldest unconsumed frames are dropped immediately to ensure downstream inference always receives the freshest current frame.
* **Monotonic Clocks:** Captures frame timestamps using `time.perf_counter()`. Avoids wall-clock system time drift, NTP step adjustments, or leap seconds.
* **Auto-Reconnection:** Detects camera disconnection, releases stale hardware handles, and attempts reconnecting up to 3 times with configurable backoff.

### 2.2 Vision Preprocessing Layer (`src/realtime/frame_processor.py`)
* Converts OpenCV default BGR frames to MediaPipe-required RGB color space.
* Implements horizontal mirroring so the camera feed functions as an intuitive mirror for signers.
* Validates array shapes and `uint8` data types.

### 2.3 Multimodal Extractor & Deterministic Fusion
* Reuses `LandmarkExtractor` from Phase 2 without code duplication.
* MediaPipe Tasks Vision API is executed in `IMAGE` mode.
* Canonical node mapping:
  * `00 - 20`: Left Hand (21 phalanx points)
  * `21 - 41`: Right Hand (21 phalanx points)
  * `42 - 52`: Upper Pose (11 anchors)
  * `53 - 92`: Facial Contours (40 markers)
* Missing-value convention: Unobserved joints are zeroed out with `mask[i] = False`.

### 2.4 Torso-Scale Normalizer with Temporal Anchor Smoothing
* Centering is computed at the anatomical sternum/mid-shoulder:
  $$C_{\text{torso}} = \frac{P_{47} + P_{48}}{2}$$
* Scaling is computed using the inter-shoulder baseline:
  $$S = \|P_{47} - P_{48}\|_2$$
* **Temporal Anchor Smoothing:** In video streaming, tiny frame-to-frame shoulder detections can cause slight "breathing" or jitter across normalized hand joints. Real-time streaming applies Exponential Moving Average (EMA) smoothing to the anchor center and scale ($\alpha = 0.25$):
  $$C^{(t)} = 0.75 C^{(t-1)} + 0.25 C_{\text{raw}}^{(t)}$$
  $$S^{(t)} = 0.75 S^{(t-1)} + 0.25 S_{\text{raw}}^{(t)}$$
  This stabilizes normalized coordinates without altering the static coordinate distribution.

### 2.5 Quality & Diagnostic Anomaly Gate
* Computes weighted frame quality:
  $$Q = 0.40 C_{\text{pose}} + 0.40 C_{\text{dom\_hand}} + 0.15 C_{\text{non\_dom\_hand}} + 0.05 C_{\text{face}}$$
* Inspects Euclidean jump distance between consecutive frames:
  $$\Delta_i = \|P_i^{(t)} - P_i^{(t-1)}\|_2$$
  If $\max(\Delta_i) > 0.35$ normalized units, flags sudden tracking jump warning.

---

## 3. Privacy-First Architectural Design

* **No Default Persistence:** Camera frames are strictly processed in volatile memory. No video or images are written to disk by default.
* **Isolated Debug Mode:** Optional debug recording (`record_landmarks: true` or `record_video: true`) requires explicit opt-in via config or CLI flags.
* **Localized Processing:** 100% of video decoding, landmark extraction, and normalization executes entirely locally on device. No network requests or telemetry transmissions occur.

---

## 4. Downstream Interface for Phase 4 Part 2

The stream emits `LandmarkFrame` instances exposing:
* `normalized_coords`: `(93, 3)` numpy array
* `mask`: `(93,)` boolean mask
* `to_stgcn_frame_tensor()`: returns `[3, 1, 93]` PyTorch tensor
* `quality`: structured quality report (`is_valid`, `classification`, `quality_score`)

Phase 4 Part 2 will feed this stream directly into a 45-frame rolling ring buffer for ST-GCN inference.
