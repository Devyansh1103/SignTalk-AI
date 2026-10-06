# SignTalk AI — End-to-End Real-Time Pipeline System Guide

**Project:** SignTalk AI: Real-Time Indian Sign Language Translation Platform  
**Document:** End-to-End Real-Time Inference System Architecture & Operating Guide  
**Milestone:** Phase 4 — Part 4  
**Date:** October 6, 2026  

---

## 1. System Overview

The SignTalk AI real-time engine connects optical camera input to stabilized Indian Sign Language recognition and multilingual sentence translation (English and Hindi) through a pipelined architecture:

```text
Camera / Optical Ingestion (640x480 @ 25 FPS)
         │
         ▼
Frame Preprocessor (RGB conversion, letterboxing, horizontal selfie flip)
         │
         ▼
MediaPipe Landmark Stream (93 Multimodal Graph Nodes: Hands, Pose, Face)
         │
         ▼
Torso-Referenced Normalization (Mid-shoulder center, shoulder-distance scale)
         │
         ▼
Quality Gating (Tracking confidence, limb visibility, jump filtering)
         │
         ▼
Temporal Rolling Buffer (FIFO queue of T=45 frames / 1.8s duration)
         │
         ▼
Inference Scheduler (Stride-5 pacing, stale window backpressure dropping)
         │
         ▼
ST-GCN Visual Model Runner (Spatial-Temporal Graph Convolution [1, 3, 45, 93])
         │
         ▼
Confidence Filter (Model confidence τ=0.65 vs Quality q=0.40 decoupling)
         │
         ▼
Prediction History (Rolling 32-record queue with top-k telemetry)
         │
         ▼
Temporal Smoother (Majority Voting N=5, K=3 quorum consensus)
         │
         ▼
Sign State Machine (Debounced transitions: IDLE ──► CANDIDATE ──► ACTIVE ──► ENDING)
         │
         ▼
Event Deduplicator (800ms minimum inter-event boundary gap)
         │
         ▼
Sign Sequence Buffer (Chronological sign event timeline)
         │
         ▼
Linguistic Translation Layer (Phrase templates, bilingual lexicon, pause timeout)
         │
         ▼
Live Transcript & Diagnostic HUD Canvas (OpenCV overlay / WebSocket streaming)
```

---

## 2. Component Responsibilities

1. **`Camera` ([`src/realtime/camera.py`](file:///d:/SignAI/src/realtime/camera.py)):**
   Provides threaded frame acquisition from USB webcams or video files, ensuring monotonic timestamps and ring-buffer protection against camera stalls.
2. **`RealTimeLandmarkStream` ([`src/realtime/landmark_stream.py`](file:///d:/SignAI/src/realtime/landmark_stream.py)):**
   Extracts 93 multimodal landmarks using MediaPipe Tasks API, normalizes coordinates to the signer's torso, and produces a `LandmarkFrame` per frame.
3. **`TemporalBuffer` ([`src/realtime/temporal_buffer.py`](file:///d:/SignAI/src/realtime/temporal_buffer.py)):**
   Maintains a rolling FIFO window of exactly 45 frames ($1.8\text{ s}$ @ $25\text{ FPS}$). Bridges isolated dropped frames via linear coordinate interpolation.
4. **`STGCNRunner` ([`src/realtime/model_runner.py`](file:///d:/SignAI/src/realtime/model_runner.py)):**
   Converts window frames to $[1, 3, 45, 93]$ tensors, validates shapes, and executes optimized PyTorch forward passes producing top-$k$ probabilities and logits.
5. **`ConfidenceFilter` ([`src/realtime/confidence_filter.py`](file:///d:/SignAI/src/realtime/confidence_filter.py)):**
   Gates predictions against validation-selected threshold $\tau=0.65$ and tracking quality $q=0.40$, routing low-confidence outputs to `UNCERTAIN`.
6. **`MajorityVoteSmoother` ([`src/realtime/smoothing/majority_vote.py`](file:///d:/SignAI/src/realtime/smoothing/majority_vote.py)):**
   Filters transient 1-frame prediction flips by enforcing a quorum of 3 out of 5 recent predictions.
7. **`SignStateMachine` ([`src/realtime/sign_state_machine.py`](file:///d:/SignAI/src/realtime/sign_state_machine.py)):**
   Debounces gesture onset and conclusion using a finite state machine (`IDLE` $\to$ `CANDIDATE` $\to$ `ACTIVE` $\to$ `ENDING`).
8. **`EventDeduplicator` ([`src/realtime/event_deduplicator.py`](file:///d:/SignAI/src/realtime/event_deduplicator.py)):**
   Suppresses repeated events for ongoing continuous signs while allowing valid repetitions separated by $\ge 800\text{ ms}$.
9. **`RealTimeTranslator` ([`src/realtime/translator.py`](file:///d:/SignAI/src/realtime/translator.py)):**
   Translates recognized sign sequences into natural English and Hindi phrases, managing pause-based sentence finalization ($1.5\text{ s}$ silence).
10. **`PerformanceProfiler` ([`src/realtime/performance.py`](file:///d:/SignAI/src/realtime/performance.py)):**
    Instruments every pipeline stage with microsecond-resolution timing and evaluates latency percentiles and memory usage.

---

## 3. Configuration System

All real-time parameters are configured via [`configs/realtime.yaml`](file:///d:/SignAI/configs/realtime.yaml) and parsed into type-safe dataclasses in [`src/realtime/realtime_config.py`](file:///d:/SignAI/src/realtime/realtime_config.py).

Key settings:
* `camera.target_fps`: $25.0$
* `buffer.max_length`: $45$
* `scheduler.stride`: $5$
* `confidence.threshold`: $0.65$
* `smoothing.history_size`: $5$
* `smoothing.min_votes`: $3$
* `events.minimum_gap_ms`: $800.0$
* `translation.pause_threshold_sec`: $1.5$

---

## 4. Execution Commands

### Live Camera Recognition
```bash
python scripts/run_realtime.py
```

### Video File Streaming Surrogate
```bash
python scripts/run_realtime.py --video-file data/raw/videos/hello_signer_01_rep1.mp4
```

### Offline Deterministic Replay
```bash
python scripts/replay_realtime_pipeline.py --input data/processed/sequences/val/seq_0037.npz
```

### Official Performance Benchmark
```bash
python scripts/benchmark_realtime.py
```
