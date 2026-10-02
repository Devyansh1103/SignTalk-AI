# Dataset Requirements Specification: SignTalk AI

**Document ID:** STAI-P2P1-002  
**Phase:** Phase 2 — Part 1  
**Status:** Approved  

---

## 1. Visual & Spatial Video Requirements

To ensure high-fidelity landmark extraction with Google MediaPipe Tasks API, input video data must satisfy:

1. **Framing & Field of View:**
   - Upper-body frontal composition (torso, arms, hands, head fully in frame).
   - Signer must remain within the central 80% horizontal frame bounds during standard gestures.
   - Distance from camera: 0.5m to 2.0m.

2. **Temporal Resolution:**
   - Native FPS: 25.0 – 30.0 FPS.
   - Minimum acceptable FPS: 20.0 FPS (below 20 FPS, rapid finger transitions exhibit severe temporal aliasing).
   - Target sampling rate for pipeline: 25.0 FPS with uniform temporal resampling.

3. **Spatial Resolution:**
   - Native resolution: 1280×720 or 1920×1080.
   - Processing resolution: Minimum 426×240 for real-time mobile/edge, standard 640×480 for server pipeline.

4. **Illumination & Background:**
   - Minimum illuminance: 150 lux (avoid heavy underexposure).
   - High contrast between hands/skin tone and clothing/background.
   - Avoid strong backlighting that causes silhouette effects.

---

## 2. Skeletal & Landmark Quality Thresholds

- **Pose Joint Visibility:** Shoulders (11, 12) must have visibility confidence $\ge 0.5$ in at least 90% of frames to establish stable torso centering.
- **Hand Presence:** At least one hand (dominant articulator) must be present and detected with confidence $\ge 0.4$ across $\ge 50\%$ of active signing frames.
- **Sequence Continuity:** Maximum tolerable consecutive missing frames before sequence rejection or review is 15 frames (0.6 seconds at 25 FPS).
