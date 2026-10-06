# SignTalk AI — Real-Time Camera & Streaming Setup Guide

**Document ID:** `DOC-P4P1-SETUP-001`  
**Phase:** Phase 4 — Part 1 (Real-Time Ingestion Layer)  
**System Title:** SignTalk AI: Real-Time Indian Sign Language Translation Platform  

---

## 1. Prerequisites & Dependencies

Ensure your environment satisfies the dependencies established in Phase 1–3:

```bash
# Python 3.10+ (Tested on Python 3.13)
pip install opencv-python mediapipe numpy pyyaml torch
```

Verify that MediaPipe task models are present in the models directory:
```text
models/mediapipe/
├── hand_landmarker.task
├── pose_landmarker_full.task
└── face_landmarker.task
```

---

## 2. Connect & Identify Camera

### Physical USB / Integrated Webcam
1. Plug in your webcam.
2. The default camera index is `0`. If you have multiple webcams (e.g. integrated laptop camera + external USB camera), external webcams are usually index `1` or `2`.

### Verify Camera Access
To verify that OpenCV can access your camera device:
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera 0 Opened:', cap.isOpened()); cap.release()"
```

---

## 3. Configuration (`configs/realtime.yaml`)

Edit `configs/realtime.yaml` to adjust device index, resolution, or frame rates:

```yaml
camera:
  device_index: 0          # 0 for default webcam, or path to an MP4 video file
  width: 640               # Horizontal capture resolution
  height: 480              # Vertical capture resolution
  target_fps: 25.0         # Aligned with trained model target FPS (25 FPS)

processing:
  flip_horizontal: true    # Mirrors preview for natural signing feedback
  color_format: "RGB"      # MediaPipe color space

debug:
  record_landmarks: false  # Privacy-first default: never store frames
  record_video: false
```

---

## 4. Run Real-Time Landmark Stream

### Basic Execution (Live Webcam)
```bash
python scripts/run_realtime_landmarks.py
```

### Specify Camera Device Index
```bash
python scripts/run_realtime_landmarks.py --camera-id 1 --fps 25
```

### Replay from Video File (No Camera Required)
If you do not have a camera attached or want to test reproducibility against a recorded signing sample:
```bash
python scripts/run_realtime_landmarks.py --video-file data/raw/videos/hello_signer_01_rep1.mp4
```

### Run in Headless / Benchmark Mode
```bash
python scripts/run_realtime_landmarks.py --video-file data/interim/landmark_pilot/pilot_sample_01.mp4 --benchmark --headless --max-frames 45
```

---

## 5. Visual Debugging Overlay (HUD)

When running with GUI preview enabled (the default), an OpenCV window displays:
* **Green Skeletal Lines:** Left hand phalanx articulators (Nodes 0–20).
* **Magenta Skeletal Lines:** Right hand phalanx articulators (Nodes 21–41).
* **Cyan/Orange Bones:** Upper-body pose anchors (Shoulders, elbows, wrists, nose, eyes, ears).
* **Yellow Dots:** Facial contour landmarks (Nodes 53–92).
* **Heads-Up Display (HUD) Box (Top Left):**
  * Current Capture & Processing FPS
  * Total pipeline latency & MediaPipe extraction latency
  * Hand detection status (`LH=Y/N`, `RH=Y/N`)
  * Pose tracking status (`DETECTED` / `LOST`)
  * Real-time quality rating (`GOOD`, `ACCEPTABLE`, `REVIEW`, `REJECT`)
  * Frame counter and detected node count (e.g. `42/93` or `93/93`)

**Exiting the Stream:**
Press `q` or `ESC` in the preview window, or press `Ctrl + C` in the terminal.

---

## 6. Privacy & Recording Controls

* **Default State:** Privacy-first design is active. Frames are never written to disk.
* **To Record Debug Landmarks:**
  ```bash
  python scripts/run_realtime_landmarks.py --record
  ```
  Artifacts are saved to `experiments/realtime_debug/landmark_stream_debug_<timestamp>.npz`.
* **To Disable All Recording:**
  Ensure `record_landmarks: false` and `record_video: false` in `configs/realtime.yaml`.

---

## 7. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `Camera unavailable (device=0)` | Device permissions or busy handle | Close other applications using the webcam (Zoom, Teams, Chrome). Check Windows Camera privacy settings (`Settings > Privacy > Camera`). |
| `OpenCV: CAP_DSHOW error` | Backend driver negotiation | Pass `--camera-id 1` to target alternate webcam device. |
| Low Processing FPS on CPU | MediaPipe CPU extraction overhead | Disable facial mesh mesh contours (`enable_face: false` in `configs/realtime.yaml`). Upper facial pose anchors provide cranial stability at higher framerates. |
| Inverted Hands / Chirality | Mirroring toggle mismatched | Use `--no-flip` flag or toggle `flip_horizontal: false`. |
