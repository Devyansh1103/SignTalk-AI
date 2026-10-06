# Phase 4 — Part 4 Handoff Specification

**Project:** SignTalk AI: Real-Time Indian Sign Language Translation Platform  
**Architecture:** Spatial-Temporal Graph and Transformer-Based Sign-to-Text System  
**From:** Phase 4 — Part 3 (Confidence, Smoothing & Continuous Sign Detection)  
**To:** Phase 4 — Part 4 (Transformer / NLP Layer Integration)  
**Status:** Certified & Ready for Handoff  
**Date:** October 6, 2026

---

## 1. Input Specification for Phase 4 Part 4

Phase 4 Part 4 will consume the structured output emitted by [`SignSequenceBuffer`](file:///d:/SignAI/src/realtime/sign_sequence.py) and [`SignEvent`](file:///d:/SignAI/src/realtime/sign_event.py).

Each emitted sign event provides:
- **`event_id`**: Unique string identifier (`evt_[a-f0-9]{8}`).
- **`class_id`**: Integer class index (`0 .. 9` for MVP-10 vocabulary).
- **`label`**: Canonical gloss string (e.g., `"hello"`, `"thankyou"`, `"good"`, `"happy"`).
- **`start_time`**: Floating-point timestamp (seconds) marking gesture onset.
- **`end_time`**: Floating-point timestamp (seconds) marking gesture conclusion.
- **`duration_ms`**: Elapsed duration of expression in milliseconds.
- **`confidence`**: Mean ST-GCN confidence score during active state ($[0.0, 1.0]$).
- **`input_quality`**: Average MediaPipe landmark quality score ($[0.0, 1.0]$).
- **`stability_score`**: Evidence ratio / voting consensus ($[0.0, 1.0]$).

---

## 2. Sign Sequence Interface

The sequence buffer produces ordered lists of discrete sign glosses:

```text
HELP ──► HOSPITAL ──► TOMORROW
(start: 12.2s)      (start: 14.7s)        (start: 17.1s)
(end:   13.1s)      (end:   16.0s)        (end:   18.3s)
```

Downstream consumption methods on [`SignSequenceBuffer`](file:///d:/SignAI/src/realtime/sign_sequence.py):
- `get_sequence() -> List[SignEvent]`: Chronological event history.
- `get_labels() -> List[str]`: Sequence of gloss strings (e.g. `["help", "hospital", "tomorrow"]`).
- `format_gloss_string() -> str`: Formatted string (`"HELP -> HOSPITAL -> TOMORROW"`).
- `format_timeline() -> str`: Timestamped multi-line timeline representation.
- `last_event() -> Optional[SignEvent]`: Most recently finalized event.

---

## 3. Validated Event Policy

The Phase 4 Part 3 pipeline enforces the following verified decision boundaries:

| Policy | Value | Rationale |
| :--- | :---: | :--- |
| **Model Confidence Threshold ($\tau$)** | `0.65` | Empirically selected from validation dataset; achieves 90.0% precision with only 16.7% rejection. |
| **Landmark Quality Gate ($q_{\text{min}}$)** | `0.40` | Rejects frames with severe hand occlusion or body absence before state evaluation. |
| **Stability Requirement ($K$)** | `2` consecutive | Gestures must sustain consistent identity for at least 2 consecutive sliding inferences ($400$ ms) to enter `ACTIVE`. |
| **Duplicate Debounce Gap** | `800.0` ms | Suppresses repeated events for ongoing continuous signing while permitting distinct repetitions after an $800$ ms pause. |
| **Uncertain Sign Handling** | Tagged `UNCERTAIN` | Sub-threshold predictions do not trigger events; if uncertainty persists $> 1000$ ms, system resets to `IDLE`. |

---

## 4. Measured Real-Time Performance & Overhead

Benchmarked over 100 iterations on CPU ([`results/temporal_stability/benchmark_latency.json`](file:///d:/SignAI/results/temporal_stability/benchmark_latency.json)):

| Pipeline Stage | Mean Latency (ms) | Median Latency (ms) | p95 Latency (ms) | Max Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **ST-GCN Inference** | 143.86 ms | 137.16 ms | 196.58 ms | 288.32 ms |
| **Confidence Filtering** | 0.027 ms | 0.024 ms | 0.042 ms | 0.096 ms |
| **Prediction History Append** | 0.007 ms | 0.006 ms | 0.012 ms | 0.017 ms |
| **Majority Vote Smoothing** | 0.152 ms | 0.139 ms | 0.236 ms | 0.702 ms |
| **Sign State Machine** | 0.012 ms | 0.011 ms | 0.024 ms | 0.031 ms |
| **Event Deduplication** | 0.003 ms | 0.002 ms | 0.004 ms | 0.052 ms |
| **Total Added Post-STGCN Overhead** | **0.201 ms** | **0.184 ms** | **0.309 ms** | **0.745 ms** |
| **Total End-to-End Prediction Cycle** | **144.06 ms** | **137.35 ms** | **196.89 ms** | **289.07 ms** |

---

## 5. Selected Active Configuration

Recorded in [`configs/realtime.yaml`](file:///d:/SignAI/configs/realtime.yaml):

```yaml
confidence:
  threshold: 0.65
  source: "validation"
  uncertain_label: "UNCERTAIN"

smoothing:
  method: "majority_vote"
  history_size: 5
  min_votes: 3
  decay_factor: 0.85
  min_consecutive: 2

stability:
  min_consecutive_predictions: 2
  min_confidence: 0.65
  min_input_quality: 0.40

events:
  minimum_gap_ms: 800.0
  unknown_timeout_ms: 1000.0
  log_events: false
  events_csv: "results/realtime/events.csv"

sequence:
  max_events: 50
```

---

## 6. Known Boundaries & Dataset Limitations

1. **Isolated Sign Limitation:**
   The trained ST-GCN model operates on isolated sign sequences. The continuous-sign pipeline provides an **engineering framework** for debounced temporal segmentation, but does **not** perform true continuous Indian Sign Language discourse parsing.
2. **Vocabulary Bound:**
   Supported signs are strictly bounded by the MVP-10 vocabulary (`hello`, `thankyou`, `good`, `happy`, `monday`, `car`, `bird`, `house`, `time`, `teacher`).
3. **Event-Level Ground Truth:**
   Formal event boundary annotations for continuous sequences do not exist in the current isolated dataset; performance has been certified via validation sequence sweeps and behavioral simulation.

---

## 7. Next Phase Objective (Phase 4 Part 4)

Phase 4 Part 4 will implement:

$$\text{Stable Sign Events} \to \text{Sign Sequence Buffer} \to \text{Transformer / NLP Layer} \to \text{Natural Language Translation}$$

Tasks for Phase 4 Part 4:
1. Ingest `SignEvent` streams from `SignSequenceBuffer`.
2. Convert gloss tokens into grammatically fluent natural language sentences (English and Hindi).
3. Handle inter-sign pauses to detect sentence / clause boundaries.
4. Support real-time streaming translation display.

**Phase 4 Part 3 is certified complete and ready for Part 4 handoff.**
