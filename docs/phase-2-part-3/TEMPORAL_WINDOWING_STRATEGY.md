# Temporal Windowing Strategy: SignTalk AI

**Document ID:** STAI-P2P3-006  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead ML Data Engineer & Computer Vision Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Overview & Strategy Taxonomy

Temporal windowing governs how continuous or segmented video recordings are partitioned into discrete input sequences for neural sequence models. In SignTalk AI, we implement a dual-mode windowing architecture:
1. **Mode 1: Canonical Full-Utterance Windowing (Primary Training Mode)**  
   Transforms each isolated sign recording into exactly one canonical, balanced sequence ($T = 45$ frames).
2. **Mode 2: Overlapping Sliding-Window Generation (Inference Simulation Mode)**  
   Extracts overlapping sub-windows ($W = 45$ frames, stride $S = 5$ frames) to simulate real-time sliding-window buffering and evaluate temporal shift invariance.

---

## 2. Mode 1: Canonical Full-Utterance Windowing (Primary)

### 2.1 Specification
- **Window Size ($W$):** $45$ frames ($1.80\text{ seconds}$ at $25.0\text{ FPS}$).
- **Stride ($S$):** Equal to recording length (non-overlapping, 1:1 mapping).
- **Target Temporal Resolution:** Uniformly resampled across the duration $[t_{start}, t_{end}]$.
- **Class & Signer Balance:** Strictly maintains $100\%$ class balance ($6$ sequences per class) and signer balance ($20$ sequences per signer).
- **Leakage Immunity:** Guarantees zero adjacent-window correlation across evaluation splits.

### 2.2 Boundary Handling
- Frame indices: $0 \le t < N$ where $N \in [45, 55]$.
- Active stroke identification: The sign nucleus occurs between $20\%$ and $80\%$ of the recording interval. Uniform temporal resampling naturally centers the stroke within the $45$-frame window.

---

## 3. Mode 2: Overlapping Sliding-Window Generation (Inference Simulation)

### 3.1 Parameterization
```yaml
sliding_window:
  window_size: 45        # Fixed temporal window length (T=45 ~ 1.8s)
  stride: 5              # Temporal hop size (S=5 ~ 200ms)
  overlap_ratio: 0.889   # (45 - 5) / 45 = 88.89%
  min_frames: 40         # Minimum valid frames required to accept a window
  boundary_padding: "edge" # Edge frame replication for boundary alignment
```

### 3.2 Mathematical Formulation
For a recording containing $N$ frames, the number of generated sliding windows $K$ is given by:
$$K = \left\lfloor \frac{N - W}{S} \right\rfloor + 1$$
- For $N = 45$: $K = \lfloor(45 - 45)/5\rfloor + 1 = 1\text{ window}$ ($t \in [0, 44]$).
- For $N = 53$: $K = \lfloor(53 - 45)/5\rfloor + 1 = 2\text{ windows}$ ($t \in [0, 44]$ and $[5, 49]$).
- For $N = 55$: $K = \lfloor(55 - 45)/5\rfloor + 1 = 3\text{ windows}$ ($t \in [0, 44]$, $[5, 49]$, and $[10, 54]$).

### 3.3 Overlap Analysis & Duplication Safeguard
While overlapping sliding windows generate additional training frames, adjacent windows with $88.9\%$ overlap share $40$ out of $45$ frames.
- **Rule 1 (Strict Partition Isolation):** All sliding windows originating from a given source video MUST remain in the same partition (`train`, `val`, or `test`). Moving overlapping windows across splits constitutes severe temporal leakage.
- **Rule 2 (Primary Benchmark Ground Truth):** Model evaluation and validation benchmarks MUST evaluate on Canonical Mode 1 sequences to prevent inflated evaluation accuracy caused by test-time window autocorrelation.
