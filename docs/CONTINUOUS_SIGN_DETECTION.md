# Continuous Sign Detection & Event Segmentation

**Project:** SignTalk AI: Real-Time Indian Sign Language Translation Platform  
**Component:** Temporal State Machine, Duplicate Suppression, and Continuous Sign Sequence Buffer  
**Phase:** Phase 4 — Part 3

---

## 1. Architectural Objective & Engineering Framework

In real-time sign language interaction, a signer naturally communicates multiple signs sequentially (e.g. `HELP` $\to$ `WATER` $\to$ `HOSPITAL`) without manually pressing buttons between gestures.

Because raw sliding-window models produce repetitive predictions for sustained movements, this layer serves as an **engineering framework** that translates a stream of smoothed predictions into discrete, ordered **Sign Events**:

$$\text{Smoothed Predictions} \to \text{Finite State Machine} \to \text{Duplicate Suppression} \to \text{Continuous Sign Buffer}$$

---

## 2. Stability State Machine

Implemented in [`src/realtime/sign_state_machine.py`](file:///d:/SignAI/src/realtime/sign_state_machine.py), the Finite State Machine (FSM) tracks the lifecycle of every gesture across 4 discrete states:

```text
                  ┌──────────────┐
                  │     IDLE     │◄─────────────────┐
                  └──────┬───────┘                  │
                         │ valid candidate          │
                         ▼                          │
                  ┌──────────────┐                  │ drop / timeout
                  │  CANDIDATE   │──────────────────┤
                  └──────┬───────┘                  │
                         │ repeated consistency     │
                         ▼                          │
                  ┌──────────────┐                  │
                  │    ACTIVE    │                  │
                  └──────┬───────┘                  │
                         │ class change / conf drop │
                         ▼                          │
                  ┌──────────────┐                  │
                  │    ENDING    │──────────────────┘
                  └──────────────┘
                         │
                         ▼
                   [Emit SignEvent]
```

### State Definitions & Trigger Policies
1. **`IDLE`**:
   - Baseline resting state.
   - Transitions to `CANDIDATE` when a smoothed prediction satisfies `is_valid` (`confidence >= 0.65`, `input_quality >= 0.40`, class not in `UNCERTAIN`/`IDLE`).
2. **`CANDIDATE`**:
   - A promising sign gesture has appeared.
   - If the next consecutive prediction matches the same sign class, increment `consecutive_count`.
   - Once `consecutive_count >= min_consecutive_predictions` ($K=2$), transitions to `ACTIVE`.
   - If class identity changes before reaching quorum, resets candidate tracking to the new class.
   - If confidence falls below threshold, resets to `IDLE`.
3. **`ACTIVE`**:
   - The sign gesture is confirmed as actively and stably expressed.
   - Continues accumulating duration, confidence, and quality metrics as long as predictions remain consistent.
   - Exits to `ENDING` when prediction class shifts or confidence drops below threshold.
4. **`ENDING`**:
   - The sustained sign has concluded.
   - Finalizes and packages a completed `SignEvent` record.
   - If the new prediction is valid, immediately transitions into `CANDIDATE` for the new sign; otherwise returns to `IDLE`.

---

## 3. Duplicate Suppression & Inter-Event Timing

Without duplicate suppression, a single held sign would generate repeated events.

Implemented in [`src/realtime/event_deduplicator.py`](file:///d:/SignAI/src/realtime/event_deduplicator.py):
- **Rule 1 (Different Class):** If $\text{class\_id}_{t} \neq \text{class\_id}_{t-1}$, immediately emit as a new sign event.
- **Rule 2 (Same Class, Sub-Gap Continuation):** If $\Delta t = (t_{\text{start}} - t_{\text{end, prev}}) < \text{minimum\_gap\_ms}$ ($800$ ms), suppress candidate event and extend the trailing event boundary:
  $$t_{\text{end, prev}} \leftarrow \max(t_{\text{end, prev}}, t_{\text{end}})$$
- **Rule 3 (Same Class, Valid Gap Repetition):** If $\Delta t \ge \text{minimum\_gap\_ms}$, permit emission as a distinct repeated occurrence of the same sign.

---

## 4. Continuous Sign Sequence Buffer

Implemented in [`src/realtime/sign_sequence.py`](file:///d:/SignAI/src/realtime/sign_sequence.py):
- **FIFO Buffer (`SignSequenceBuffer`)**: Bounded capacity (default $50$ events).
- **Intermediate Representation**: Stores structured glosses with start and end timestamps:
  ```text
  HELP
  start: 12.2s
  end: 13.1s

  WATER
  start: 13.5s
  end: 14.3s

  HOSPITAL
  start: 14.7s
  end: 16.0s
  ```
- Exposes gloss sequence (`HELP -> WATER -> HOSPITAL`) for consumption by downstream language models.

---

## 5. Critical Limitation & Boundary Constraints

> [!WARNING]
> **Dataset Limitation Notice:**  
> The ST-GCN model was trained on **isolated sign recordings** (single sign per clip).  
> The continuous sign detection pipeline implemented here is an **engineering framework** for real-time temporal segmentation and debounce filtering.  
> It does **NOT** represent unrestricted continuous ISL sentence recognition or grammatical translation. Continuous sign language features co-articulation, movement epenthesis, and non-manual markers that require continuous multi-signer annotated corpora for formal training and benchmarking.
