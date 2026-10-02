# Landmark Pilot Report: Feasibility Validation

**Document ID:** STAI-P2P1-005  
**Phase:** Phase 2 — Part 1  
**Status:** Approved  
**Tooling:** MediaPipe Tasks Vision (`PoseLandmarker`, `HandLandmarker`, `FaceLandmarker`)  

---

## 1. Pilot Objective & Setup

Before running the full preprocessing pipeline, a landmark feasibility pilot was conducted on representative ISL signing video clips across 6 environmental conditions:
1. `pilot_sample_01.mp4`: Standard baseline signing video (169 frames, 426×240, 25 FPS).
2. `pilot_sample_02_lowlight.mp4`: Underexposed condition (gamma 0.4).
3. `pilot_sample_03_overexposed.mp4`: Saturated condition (brightness +30, scale 1.6).
4. `pilot_sample_04_motionblur.mp4`: Rapid movement simulation (Gaussian blur 9×9).
5. `pilot_sample_05_fast.mp4`: High-speed signing with skipped frames (85 frames).
6. `pilot_sample_06_cropped.mp4`: Narrow field-of-view / boundary edge condition.

---

## 2. Measured Pilot Results

| Sample ID | Condition | Total Frames | Pose Detection Rate | Right Hand Rate | Left Hand Rate | Face Detection Rate | Mean Frame Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `pilot_01` | Standard Baseline | 169 | 100.0% | 76.3% | 45.6% | 12.4% | 22.4 ms |
| `pilot_02` | Low Light (0.4x) | 169 | 94.7% | 60.9% | 31.4% | 3.5% | 24.1 ms |
| `pilot_03` | Overexposed (1.6x) | 169 | 98.2% | 68.6% | 38.5% | 7.1% | 23.8 ms |
| `pilot_04` | Motion Blur | 169 | 91.1% | 42.0% | 20.7% | 1.8% | 25.2 ms |
| `pilot_05` | Fast Signing | 85 | 98.8% | 71.8% | 42.4% | 11.8% | 22.6 ms |
| `pilot_06` | Cropped Margin | 169 | 97.6% | 73.4% | 43.8% | 8.3% | 23.1 ms |

---

## 3. Key Engineering Findings

1. **Pose Reliability:** Upper-body pose anchors (shoulders, elbows, wrist anchors) achieved $\ge 91.1\%$ detection across all conditions and $100\%$ on standard footage. Shoulders provide an extremely stable anchor for torso centering.
2. **Hand Asymmetry:** In isolated signs, the dominant right hand was actively detected in $76.3\%$ of frames, while the non-dominant hand was active only during two-handed phases ($45.6\%$). Hand presence fluctuates naturally as hands drop out of the signing zone.
3. **Face Landmarker Limitations:** Due to low resolution (426×240) and hand-face occlusion, raw FaceLandmarker detection degraded severely (12.4% standard down to 1.8% in blur).
   - **Approved Architectural Decision:** Pose landmarker upper face landmarks (nose 42, eyes 43-44, ears 45-46) are utilized as robust geometric anchors. When full face mesh is unavailable, pose face points guarantee uninterrupted upper facial structure without crashing.
4. **Latency & Real-Time Headroom:** Average frame extraction latency measured $22.4\text{ ms}$ on CPU ($44.6\text{ FPS}$ equivalent), comfortably exceeding the real-time operational threshold of $25\text{ FPS}$ ($40\text{ ms}$).
