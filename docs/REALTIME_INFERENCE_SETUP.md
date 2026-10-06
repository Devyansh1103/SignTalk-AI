# Real-Time Inference Setup & Operation Guide

**SignTalk AI: Real-Time Indian Sign Language Translation Platform**  
*Phase 4 — Part 2: Sliding-Window Inference & Temporal Prediction*

---

## 1. Prerequisites and Assets

Ensure the following components and model checkpoints are available in the workspace:

1. **MediaPipe Task Models**:
   - `models/mediapipe/pose_landmarker_full.task`
   - `models/mediapipe/hand_landmarker.task`
2. **ST-GCN Trained Checkpoint**:
   - `experiments/stgcn/checkpoints/best_checkpoint.pt` (Epoch 34, 10 ISL classes)
3. **Vocabulary Registry**:
   - `assets/vocabularies/mvp_10.json`
   - `data/processed/label_mapping.csv`

---

## 2. Configuration Setup

The primary configuration is defined in [`configs/realtime.yaml`](file:///d:/SignAI/configs/realtime.yaml):

```yaml
camera:
  device_index: 0
  width: 640
  height: 480
  target_fps: 25.0
  auto_reconnect: true

buffer:
  max_length: 45           # 45 frames @ 25 FPS = 1.80s
  min_valid_frames: 15
  interpolate_missing_frames: true
  max_consecutive_interpolated: 10

scheduler:
  stride: 5                # Update every 5 frames (200 ms cadence)
  drop_stale_windows: true # Drop queue lag when inference is running

inference:
  enabled: true
  checkpoint_path: "experiments/stgcn/checkpoints/best_checkpoint.pt"
  device: "auto"           # "auto", "cpu", or "cuda"
  top_k: 3
  warmup_iterations: 5
  record_predictions: false
  predictions_csv: "results/realtime/predictions.csv"
```

---

## 3. Running Real-Time Inference

### 3.1 Live Camera Prediction with Visual HUD
```bash
python scripts/run_realtime_prediction.py
```
- Opens live webcam feed (`device_index=0`).
- Displays skeletal overlay, rolling temporal buffer gauge, and top-3 sign predictions in real time.
- Press **'q'** or **ESC** in the preview window to stop.

### 3.2 Video File Surrogate
To validate inference on pre-recorded video samples:
```bash
python scripts/run_realtime_prediction.py \
    --video-file data/raw/videos/hello_signer_01_rep1.mp4 \
    --record-predictions \
    --stride 5
```

### 3.3 Headless Benchmarking Mode
For automated profiling without GUI display overhead:
```bash
python scripts/run_realtime_prediction.py \
    --video-file data/raw/videos/hello_signer_01_rep1.mp4 \
    --headless \
    --benchmark \
    --max-frames 90
```

### 3.4 Micro-Benchmark Pipeline Execution
To execute isolated stage-by-stage latency benchmarking across 100 rolling windows:
```bash
python scripts/benchmark_realtime_inference.py
```

---

## 4. CLI Argument Reference

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--config` | `str` | `configs/realtime.yaml` | Path to YAML configuration. |
| `--camera-id` | `int` | `None` | Camera hardware device index (overrides config). |
| `--video-file` | `str` | `None` | Stream video file as surrogate camera input. |
| `--device` | `str` | `auto` | Execution device (`auto`, `cpu`, `cuda`). |
| `--stride` | `int` | `5` | Sliding window stride in frames. |
| `--top-k` | `int` | `3` | Number of ranked prediction classes reported. |
| `--window-size` | `int` | `45` | Temporal window length (must match model sequence length: 45). |
| `--headless` | `flag` | `False` | Disables OpenCV GUI preview display. |
| `--benchmark` | `flag` | `False` | Prints performance breakdown summary on exit. |
| `--max-frames` | `int` | `None` | Automatically halts after processing N frames. |
| `--record-predictions` | `flag` | `False` | Appends prediction records to CSV log. |
| `--predictions-csv` | `str` | `results/realtime/predictions.csv` | Output path for prediction CSV log. |

---

## 5. Diagnostic Prediction Log Format

When `--record-predictions` is enabled, prediction events are logged to `results/realtime/predictions.csv` with the following columns:

```csv
timestamp,window_start,window_end,prediction,class_id,confidence,input_quality,inference_latency_ms,buffer_length,device
1791296758.1134,305736.4033,305741.9662,thankyou,1,0.4007,0.6775,118.60,45,cpu
```

---

## 6. Interpreting the Visual HUD

The real-time diagnostic overlay displays:
- **Top-Left**: Ingestion FPS, extraction latency, hand/pose visibility, and quality status.
- **Bottom-Left**: Temporal buffer status bar `BUFFER: 45/45 (100%)` and prediction update frequency (`5.0 pred/s`).
- **Top-Right**: ST-GCN prediction card showing:
  - Top-1 Sign Name (`HELLO`) with confidence percentage (`99.8%`)
  - Inference computation latency (`118.6 ms`)
  - Top-3 alternative classes with relative probabilities
