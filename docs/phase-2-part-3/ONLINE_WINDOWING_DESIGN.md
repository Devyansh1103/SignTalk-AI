# Online Sliding Window Design: SignTalk AI

**Document ID:** STAI-P2P3-035  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead System Architect & Real-Time Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Streaming Buffer Architecture

For real-time webcam streaming, the sequence processing pipeline operates on a continuous First-In-First-Out (FIFO) circular buffer:

```
Webcam Frame Ingestion (25 FPS)
               │
               ▼
[MediaPipe Tasks Landmark Extraction] ──► Produces Joint Vector p_t ∈ R^{3 × 93}
               │
               ▼
[FIFO Ring Buffer of Capacity W = 45 Frames]
               │
  ┌────────────┴───────────────────────────┐
  │ Check Buffer Size:                     │
  │ If count < 45: Emit WARMUP State       │
  │ If count == 45 and (t % S == 0):       │
  └────────────┬───────────────────────────┘
               │
               ▼
[Torso Centering & Running Scale Normalization]
               │
               ▼
[Model Forward Pass: Tensor shape [1, 3, 45, 93]]
               │
               ▼
[Temporal Logit Smoothing & Debounce Filter] ──► Emits Stable Translation
```

---

## 2. Operational Parameters

- **Buffer Capacity ($W$):** $45$ frames ($1.80\text{ s}$ temporal history).
- **Inference Evaluation Stride ($S$):** $5$ frames ($200\text{ ms}$ evaluation frequency = $5.0\text{ Hz}$).
- **Warm-Up Period:** First $44$ frames following stream initialization. During warm-up, the system emits an `ACQUIRING_BASELINE` status without dispatching incomplete windows to the neural network.
- **Running Normalization Anchor:** The torso midpoint $\mathbf{c}_{torso}$ and shoulder scale $s$ are smoothed across the 45-frame buffer using an exponential moving average ($\alpha = 0.1$) to prevent rapid camera shake or micro-posture shifts from distorting coordinate geometry.

---

## 3. Temporal Debounce and Prediction Stabilization

To prevent high-frequency prediction flickering during transitional hand movements, the online engine employs a 3-stage stabilization pipeline:
1. **Softmax Logit Windowing:** Predicted probability vectors $\mathbf{p}_t$ are averaged over a rolling window of $K = 3$ consecutive evaluations ($600\text{ ms}$):
   $$\bar{\mathbf{p}}_t = \frac{1}{3}\sum_{k=0}^2 \mathbf{p}_{t - k \cdot S}$$
2. **Confidence Thresholding:** A prediction is committed to the user interface only if:
   $$\max_c \bar{p}_t(c) \ge \tau_{commit} \quad (\tau_{commit} = 0.65)$$
3. **Transition Debounce:** Once a sign is committed, identical consecutive predictions are suppressed to avoid repeating identical words, while a refractory period of $2$ evaluation steps ($400\text{ ms}$) prevents immediate premature transitions.
