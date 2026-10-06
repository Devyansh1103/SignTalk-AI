# SignTalk AI — Real-Time Troubleshooting & Operations Guide

**Project:** SignTalk AI: Real-Time Indian Sign Language Translation Platform  
**Document:** Operational Troubleshooting & Diagnostics  
**Milestone:** Phase 4 — Part 4  
**Date:** October 6, 2026  

---

## 1. Common Issues & Solutions

### A. Camera Acquisition Issues
* **Problem:** `CameraInitializationError: Failed to open camera device index 0`
  * **Solution:** Verify webcam is plugged in and not in use by other applications (Teams, Zoom, browser). On Windows, check Windows Settings $\to$ Privacy & Security $\to$ Camera $\to$ "Let apps access your camera". Alternatively, try `--camera-id 1`.
* **Problem:** Video preview is frozen or laggy.
  * **Solution:** Enable threaded capture (`threaded: true` in `configs/realtime.yaml`). Ensure USB bandwidth is sufficient if multiple high-resolution webcams share a root hub.

### B. Landmark Detection & Tracking Failures
* **Problem:** Quality score remains $< 0.40$ or status is `UNCERTAIN`.
  * **Solution:** Ensure the signer's entire upper torso, shoulders, and both hands are clearly within the camera frame. Increase ambient lighting to prevent motion blur on fast finger gestures.
* **Problem:** MediaPipe model files missing.
  * **Solution:** Verify `models/mediapipe/pose_landmarker_full.task` and `models/mediapipe/hand_landmarker.task` exist.

### C. Model Inference & Device Issues
* **Problem:** `ModelRunnerError: CUDA requested, but CUDA is not available on this machine.`
  * **Solution:** Change `device: "cpu"` in `configs/realtime.yaml` or run with `--device cpu`.
* **Problem:** `ShapeValidationError: Window length mismatch: got T=... expected 45`
  * **Solution:** Ensure `buffer.max_length: 45` in configuration matches the ST-GCN checkpoint sequence length.

### D. Prediction Instability & Noise
* **Problem:** Predictions rapidly flip between classes without settling.
  * **Solution:** Verify `smoothing.method: "majority_vote"`, `smoothing.history_size: 5`, and `smoothing.min_votes: 3`. For higher stability, increase `stability.min_consecutive_predictions: 3`.

### E. Translation / Transcript Commit Issues
* **Problem:** Finished phrases are not committing to the finalized transcript.
  * **Solution:** Ensure the signer pauses naturally for $\ge 1.5\text{ s}$ after completing a gesture sequence. Alternatively, trigger manual finalization via `pipeline.translator.finalize_phrase()`.
