# SignTalk AI — Phase 4 Part 4 Implementation Audit

**Project:** SignTalk AI: Real-Time Indian Sign Language Translation Platform  
**System Architecture:** Spatial-Temporal Graph and Transformer-Based Sign-to-Text System  
**Milestone:** Phase 4 — Part 4: End-to-End Real-Time Pipeline & Performance Optimization  
**Date:** October 6, 2026  
**Auditor:** Senior ML & Systems Engineering Team  

---

## 1. Executive Summary

This audit establishes the baseline status of all system components across Phases 1 through Phase 4 Part 3 before assembling the final end-to-end real-time inference pipeline. Every component has been inspected directly in the codebase to evaluate its reusability, tensor contract consistency, device compatibility, and operational boundaries.

---

## 2. Component Inventory & Reusability Assessment

| Component | Repository Location | Status | Reusable? | Notes & Technical Assessment |
| :--- | :--- | :---: | :---: | :--- |
| **Camera Ingestion** | [`src/realtime/camera.py`](file:///d:/SignAI/src/realtime/camera.py) | Complete | **Yes** | OpenCV `VideoCapture` wrapper supporting webcams and video file surrogates; supports monotonic frame timestamps and threaded ring buffer. |
| **Frame Preprocessor** | [`src/realtime/frame_processor.py`](file:///d:/SignAI/src/realtime/frame_processor.py) | Complete | **Yes** | Horizontal mirroring (selfie view), RGB conversion, resizing, and letterboxing. |
| **MediaPipe Landmarker** | [`src/realtime/landmark_stream.py`](file:///d:/SignAI/src/realtime/landmark_stream.py) | Complete | **Yes** | MediaPipe Tasks API (`PoseLandmarker`, `HandLandmarker`). Assembles 93-node multimodal landmark graph with visibility masks. |
| **Landmark Normalization** | [`src/preprocessing/normalizer.py`](file:///d:/SignAI/src/preprocessing/normalizer.py) | Complete | **Yes** | Torso-referenced centering (`mid_shoulder`) and scale normalization (`shoulder_distance`) with temporal exponential smoothing. |
| **Quality Validator** | [`src/preprocessing/quality_checker.py`](file:///d:/SignAI/src/preprocessing/quality_checker.py) | Complete | **Yes** | Frame-level heuristic scoring based on landmark presence, visibility, and biomechanical displacement limits. |
| **Temporal Buffer** | [`src/realtime/temporal_buffer.py`](file:///d:/SignAI/src/realtime/temporal_buffer.py) | Complete | **Yes** | Rolling FIFO buffer of fixed length $T=45$ frames ($1.8\text{ s}$ @ $25\text{ FPS}$) with linear interpolation for dropped frame bridging. |
| **Inference Scheduler** | [`src/realtime/inference_scheduler.py`](file:///d:/SignAI/src/realtime/inference_scheduler.py) | Complete | **Yes** | Sliding stride pacing ($\text{stride}=5$ frames / $200\text{ ms}$) with backpressure dropping of stale windows. |
| **ST-GCN Model** | [`src/models/stgcn.py`](file:///d:/SignAI/src/models/stgcn.py) | Complete | **Yes** | 6-block spatiotemporal graph network with learnable edge weights; parameter count: 2,137,818; expects $[B, 3, 45, 93]$. |
| **Model Runner** | [`src/realtime/model_runner.py`](file:///d:/SignAI/src/realtime/model_runner.py) | Complete | **Yes** | Production inference harness (`STGCNRunner`) validating $[1, 3, 45, 93]$, handling device placement (`cpu`/`cuda`), warm-up, and top-$k$ output. |
| **Confidence Filtering** | [`src/realtime/confidence_filter.py`](file:///d:/SignAI/src/realtime/confidence_filter.py) | Complete | **Yes** | Decouples model confidence ($\tau \ge 0.65$) from tracking quality ($q \ge 0.40$); routes sub-threshold predictions to `UNCERTAIN`. |
| **Prediction History** | [`src/realtime/prediction_history.py`](file:///d:/SignAI/src/realtime/prediction_history.py) | Complete | **Yes** | Bounded ring buffer recording timestamped `PredictionRecord` instances with metadata and latencies. |
| **Temporal Smoothing** | [`src/realtime/smoothing/`](file:///d:/SignAI/src/realtime/smoothing/) | Complete | **Yes** | Modular suite: `MajorityVoteSmoother` ($N=5, K=3$), `ConfidenceWeightedSmoother`, `TemporalStabilitySmoother`. |
| **Sign State Machine** | [`src/realtime/sign_state_machine.py`](file:///d:/SignAI/src/realtime/sign_state_machine.py) | Complete | **Yes** | Debounced finite state machine (`IDLE` $\to$ `CANDIDATE` $\to$ `ACTIVE` $\to$ `ENDING`). |
| **Event Deduplicator** | [`src/realtime/event_deduplicator.py`](file:///d:/SignAI/src/realtime/event_deduplicator.py) | Complete | **Yes** | Temporal debounce enforcing $\ge 800\text{ ms}$ inter-event boundary for repeated gestures. |
| **Sign Event Schema** | [`src/realtime/sign_event.py`](file:///d:/SignAI/src/realtime/sign_event.py) | Complete | **Yes** | Structured dataclass `SignEvent` storing class ID, gloss, duration, confidence, tracking quality, and stability score. |
| **Sign Sequence Buffer** | [`src/realtime/sign_sequence.py`](file:///d:/SignAI/src/realtime/sign_sequence.py) | Complete | **Yes** | Chronological accumulation of sign events into gloss sequences with timeline formatting. |
| **Visual Transformer Model** | [`src/models/sign_translation_model.py`](file:///d:/SignAI/src/models/sign_translation_model.py) | Complete | **Yes** | Hybrid ST-GCN visual encoder + 2-layer autoregressive Transformer decoder (`experiments/transformer/checkpoints/best_checkpoint.pt`). |
| **Sign Tokenizer & Normalizer** | [`src/nlp/tokenizer.py`](file:///d:/SignAI/src/nlp/tokenizer.py) | Complete | **Yes** | Vocabulary mapping (`assets/vocabularies/mvp_10.json`) with special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`) and text cleaning. |
| **Translation Pipeline** | [`src/inference/translation_pipeline.py`](file:///d:/SignAI/src/inference/translation_pipeline.py) | Complete | **Yes** | Visual sequence-to-text pipeline taking $[1, 3, 45, 93]$ and autoregressively decoding gloss/translation. |
| **Configuration System** | [`configs/realtime.yaml`](file:///d:/SignAI/configs/realtime.yaml) | Complete | **Yes** | Structured YAML configuration parsed by `RealTimeConfig` dataclass hierarchy. |

---

## 3. Interfaces & Tensor Dimensions Verification

### Invariant Dimensions
The system enforces strict dimensionality invariance across all modules:
- **Spatial Nodes ($V$):** Exactly $93$ nodes (Upper pose: 11, Left hand: 21, Right hand: 21, Face contours: 40).
- **Coordinate Channels ($C$):** Exactly $3$ channels $(x, y, z)$.
- **Temporal Window ($T$):** Exactly $45$ frames ($1.8\text{ s}$ @ $25\text{ FPS}$).
- **Batch Size ($B$):** Strictly $1$ for real-time low-latency sliding-window inference.
- **Model Input Tensor:** $[1, 3, 45, 93]$ (`torch.float32`).
- **Model Mask Tensor:** $[1, 1, 45, 93]$ (`torch.float32`).
- **ST-GCN Logits / Probs:** $[1, 10]$ (`torch.float32`).

---

## 4. NLP / Translation Layer Status Audit

### A. What the Trained Models Actually Support
1. **Dataset Scope:**
   - The primary dataset (`data/manifests/train.csv`, `val.csv`, `test.csv`) comprises isolated video instances of 10 Indian Sign Language gestures:
     `0: hello`, `1: thankyou`, `2: good`, `3: happy`, `4: monday`, `5: car`, `6: bird`, `7: house`, `8: time`, `9: teacher`.
2. **Translation Transformer Capabilities:**
   - In Phase 3, `SignTranslationModel` was trained on these sequences to autoregressively decode sequence glosses and translations.
   - On the test set, sequence exact match accuracy is $83.33\%$, BLEU-1 is $0.8485$, and ROUGE-L is $0.8333$.
   - **Crucial Limitation:** The model was trained on isolated sign samples and sequence permutations from MVP-10. It is **not** an open-domain machine translation engine for unrestricted conversational Indian Sign Language.
3. **Translation Integration Strategy for Phase 4 Part 4:**
   - Implement a modular `RealTimeTranslator` interface in `src/realtime/translator.py`:
     - **Mode 1 (Sign-to-Text / Gloss-to-Text):** Translates accumulated `SignEvent` streams from `SignSequenceBuffer` into fluent English and Hindi sentences using phrase syntax and dictionary mappings, supporting multi-sign concatenation (e.g. `["hello", "good", "monday"]` $\to$ `"Hello, good Monday!"`, Hindi: `"नमस्ते, अच्छा सोमवार!"`).
     - **Mode 2 (Direct Visual Transformer):** Optionally passes the raw $[1, 3, 45, 93]$ sliding window through the pretrained `SignTranslationModel` via `SignTranslationPipeline` when configured.
     - **Sentence Finalization:** Debounces inter-sign pauses ($\ge 1.5\text{ s}$) or explicit timeout to commit the current phrase to the finalized transcript history.

---

## 5. Technical Debt & Gaps to Resolve in Part 4

1. **Orchestrator Composition:**
   - Currently, `RealTimeSignPredictor` handles frame ingestion through event deduplication, but lacks sentence finalization, multi-lingual translation dispatch, unified lifecycle state tracking (`WARMING_UP`, `RUNNING`, `PAUSED`, `STOPPED`, `ERROR`), and structured transcript state.
   - **Resolution:** Implement `RealtimePipeline` in `src/realtime/realtime_pipeline.py` which composes all lower-level primitives into a single coordinated engine.
2. **Performance Instrumentation:**
   - Existing latency metrics only cover ST-GCN inference and landmark extraction, without microsecond-precision breakdown of every intermediate stage (`Capture`, `MediaPipe`, `Buffer`, `TensorPrep`, `STGCN`, `Filter`, `Smoother`, `StateMachine`, `Deduplicator`, `Translation`, `Display`) or RAM/GPU memory tracking.
   - **Resolution:** Build `PerformanceProfiler` in `src/realtime/performance.py`.
3. **Smoke Test & Deterministic Replay CLI:**
   - Need standard entry points `scripts/run_realtime.py` (live/video runner) and `scripts/replay_realtime_pipeline.py` (offline deterministic replay).
4. **Configuration Centralization:**
   - Integrate translation, performance profiling, and pipeline parameters into `configs/realtime.yaml` and `RealTimeConfig`.
