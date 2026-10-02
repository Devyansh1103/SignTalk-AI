# 33. Extended Requirements Traceability Matrix (Part 2): SignTalk AI

**Document ID:** STAI-P1P2-033  
**Project Name:** SignTalk AI  
**Document Status:** Approved Engineering Blueprint (Phase 1 — Part 2)  

---

## 1. End-to-End Architectural Traceability

The extended Requirements Traceability Matrix guarantees that every functional and non-functional requirement from Phase 1 Part 1 maps directly to an architectural subsystem, a concrete software module, a measurable metric, and an automated verification test.

$$\text{Requirement} \longrightarrow \text{Architecture Layer} \longrightarrow \text{Implementation Module} \longrightarrow \text{Success Metric} \longrightarrow \text{Automated Test}$$

---

## 2. Master Implementation Traceability Matrix

| Requirement ID | Requirement Statement | Architectural Layer | Concrete Software Module | Evaluated Metric | Verification Test File |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **FR-001** | Live monocular webcam frame capture | Layer 1: Frontend | `frontend/src/components/CameraPreview.tsx` | FPS ($\ge 25\text{ FPS}$) | `tests/e2e/test_camera_stream.py` |
| **FR-002** | 21 3D hand landmarks per hand | Layer 4: Vision | `src/landmarks/mediapipe_extractor.py` | Tracking visibility $> 0.5$ | `tests/unit/test_landmarks.py` |
| **FR-003** | 11 upper-body pose keypoints | Layer 4: Vision | `src/landmarks/mediapipe_extractor.py` | Joint extraction latency | `tests/unit/test_landmarks.py` |
| **FR-004** | 40 salient facial non-manual markers | Layer 4: Vision | `src/landmarks/face_filter.py` | CPU latency $\le 15\text{ ms}$ | `tests/unit/test_face_filter.py` |
| **FR-005** | Torso-centric coordinate normalization| Layer 3: Preprocessing | `src/preprocessing/normalizer.py` | Scale/Distance Invariance | `tests/unit/test_normalizer.py` |
| **FR-006** | Temporal sliding window buffering ($T=45$)| Layer 3: Preprocessing | `backend/app/preprocessing/sliding_buffer.py`| Window stride $\approx 166\text{ ms}$ | `tests/unit/test_sliding_buffer.py` |
| **FR-007** | ST-GCN spatial-temporal graph encoder | Layer 5: ML Engine | `src/models/stgcn_encoder.py` | Top-1 Acc ($\ge 85\%$) / Latency | `tests/model/test_stgcn.py` |
| **FR-008** | Transformer sequence translation | Layer 5: ML Engine | `src/models/transformer_decoder.py` | BLEU-4 ($\ge 22.0$) / WER | `tests/model/test_transformer.py` |
| **FR-009** | Large high-contrast live text captions | Layer 1: Frontend | `frontend/src/components/CaptionPanel.tsx` | Render latency $< 30\text{ ms}$ | `tests/e2e/test_caption_render.py` |
| **FR-010** | Prediction confidence meter | Layer 3 & 1: UI | `frontend/src/components/ConfidenceBar.tsx` | Calibration error (ECE) | `tests/unit/test_confidence.py` |
| **FR-011** | Low-confidence gating & repetition prompt| Layer 3: Safety | `backend/app/services/safety_checker.py` | Triggered when score $< 0.50$| `tests/unit/test_safety.py` |
| **FR-012** | Session lifecycle controls (Start/Stop)| Layer 1: Frontend | `frontend/src/components/SessionControls.tsx` | Clean socket connect/close | `tests/integration/test_session.py` |
| **FR-013** | Interactive landmark canvas overlay | Layer 1: Frontend | `frontend/src/components/LandmarkCanvas.tsx` | Canvas FPS ($\ge 60\text{ Hz}$) | `tests/e2e/test_canvas.py` |
| **FR-014** | Camera permission & hardware error handling| Layer 1: Frontend | `frontend/src/components/StatusBanner.tsx` | Error catch coverage | `tests/e2e/test_error_states.py` |
| **FR-015** | Zero raw-video storage enforcement | Layer 4 & 6: Privacy | `backend/app/main.py` | Zero video files on disk | `tests/performance/test_privacy.py`|
| **FR-016** | Session transcript history panel | Layer 1: Frontend | `frontend/src/components/TranscriptHistory.tsx`| Text export accuracy | `tests/e2e/test_transcript.py` |
| **FR-017** | Domain vocabulary selection mode | Layer 3: Backend | `backend/app/api/endpoints/config.py` | Vocabulary lookup match | `tests/integration/test_config.py` |
| **FR-018** | Offline standalone execution | Layer 7: Deployment | `docker-compose.yml` | 100% offline network pass | `tests/performance/test_offline.py`|

---

## 3. Coverage Analysis
- **Unmapped Requirements:** $0$ (Every requirement has a concrete software component).
- **Extraneous Technical Components:** $0$ (No unnecessary microservices, databases, or LLMs exist).
