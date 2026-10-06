# Sign Event Schema Specification

**Project:** SignTalk AI: Real-Time Indian Sign Language Translation Platform  
**Module:** `src/realtime/sign_event.py`  
**Phase:** Phase 4 — Part 3: Confidence, Smoothing & Continuous Sign Detection  
**Consumer:** Phase 4 — Part 4 (Transformer / NLP Layer)

---

## 1. Overview

A `SignEvent` represents the finalized, validated expression of a discrete sign gesture detected from the real-time sliding window stream. It bridges low-level spatiotemporal graph predictions and high-level language translation layers.

---

## 2. Data Structure Specification

Defined in [`src/realtime/sign_event.py`](file:///d:/SignAI/src/realtime/sign_event.py):

```python
@dataclass
class SignEvent:
    event_id: str               # Unique identifier (e.g. 'evt_40abd976')
    class_id: int               # Numerical class index (0..9 in MVP-10)
    label: str                  # Canonical class name (e.g. 'hello', 'thankyou')
    start_time: float           # Wall-clock timestamp in seconds of gesture onset
    end_time: float             # Wall-clock timestamp in seconds of gesture offset
    duration_ms: float          # Gesture duration in milliseconds (end - start) * 1000
    confidence: float           # Mean model confidence score over active duration [0.0..1.0]
    input_quality: float        # Mean landmark tracking quality score [0.0..1.0]
    stability_score: float      # Temporal consensus ratio supporting the event [0.0..1.0]
    source_window_start: int    # Starting frame ID in temporal buffer
    source_window_end: int      # Ending frame ID in temporal buffer
    metadata: Optional[Dict[str, Any]] = None
```

---

## 3. Field Descriptions

| Field | Type | Range / Format | Description |
| :--- | :--- | :--- | :--- |
| `event_id` | `str` | `evt_[a-f0-9]{8}` | Unique 8-character hex UUID identifying this discrete sign occurrence. |
| `class_id` | `int` | `0 .. 9` | Numerical class label from certified vocabulary. |
| `label` | `str` | Lowercase string | Human-readable sign gloss name (e.g., `"hello"`). |
| `start_time` | `float` | $\ge 0.0$ s | Wall-clock timestamp when candidate sign first stabilized. |
| `end_time` | `float` | $\ge \text{start\_time}$ | Wall-clock timestamp when active gesture concluded. |
| `duration_ms` | `float` | $\ge 0.0$ ms | Elapsed duration of the gesture in milliseconds. |
| `confidence` | `float` | $0.0000 .. 1.0000$ | Arithmetic mean of model confidence scores across active frames. |
| `input_quality` | `float` | $0.0000 .. 1.0000$ | Average MediaPipe landmark tracking quality during the gesture. |
| `stability_score` | `float` | $0.0000 .. 1.0000$ | Voting consensus ratio or stability evidence during active state. |
| `source_window_start` | `int` | $\ge 0$ | Buffer frame index corresponding to gesture onset. |
| `source_window_end` | `int` | $\ge \text{start}$ | Buffer frame index corresponding to gesture conclusion. |

---

## 4. CSV Logging Format

When exported to disk ([`results/realtime/events.csv`](file:///d:/SignAI/results/realtime/events.csv)), rows adhere to the following schema:

### Header
```csv
event_id,label,class_id,start_time,end_time,duration_ms,confidence,input_quality,stability_score,source_window_start,source_window_end
```

### Sample Records
```csv
evt_40abd976,hello,0,0.0000,2.8000,2800.00,0.9897,0.7878,0.8800,0,0
evt_bedbf93f,thankyou,1,0.4000,4.4000,4000.00,0.8923,0.7020,0.8444,0,0
evt_74fd2d0c,good,2,2.0000,5.2000,3200.00,0.9829,0.4333,0.7200,0,0
evt_ec6813cc,monday,4,3.6000,6.8000,3200.00,0.7590,0.4000,0.7200,0,0
evt_61e716dd,car,5,4.4000,7.6000,3200.00,0.7013,0.4456,0.7200,0,0
evt_f00749f8,bird,6,5.2000,8.4000,3200.00,0.9711,0.5344,0.7200,0,0
evt_925f5cda,time,8,6.0000,9.2000,3200.00,0.7796,0.6622,0.7200,0,0
evt_1c578cf9,house,7,6.8000,10.0000,3200.00,0.7448,0.4089,0.7200,0,0
evt_6c906883,time,8,7.6000,10.8000,3200.00,0.8974,0.7478,0.7200,0,0
```

---

## 5. Downstream Contract (Phase 4 Part 4)

The Transformer/NLP layer in Phase 4 Part 4 consumes an ordered stream of `SignEvent` instances:
1. **Input Sequence:** `List[SignEvent]` ordered by `start_time`.
2. **Gloss Sequence:** `[event.label for event in events]` (e.g. `["help", "water", "hospital"]`).
3. **Temporal Gaps:** Intersign pause $\Delta t = \text{event}_{i+1}.\text{start\_time} - \text{event}_{i}.\text{end\_time}$ signals clause and sentence boundaries.
4. **Confidence Weighting:** Downstream token decoding can utilize `event.confidence` and `event.stability_score` to weight beam search hypotheses or request clarification on ambiguous gestures.
