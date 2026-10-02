# Transformer Data Compatibility Analysis

**Document ID:** STAI-P2P1-007  
**Phase:** Phase 2 — Part 1  
**Status:** Approved  

---

## 1. Sequence Representation & Temporal Stride

The sequence modeling stage (Transformer) operates on the spatial feature embeddings produced by ST-GCN or directly on normalized temporal landmark sequences:

1. **Input Sequence Length ($T$):**
   - Fixed temporal window: $T = 45$ frames (1.8 seconds) for isolated signs in INCLUDE-50.
   - For continuous sentences (ISLTranslate / ISL-CSLTR), sliding temporal windows of length $T_{win} = 45$ with stride $S = 15$ frames (0.6 seconds overlap) enable stream processing.

2. **Feature Dimension ($D_{model}$):**
   - Direct landmark input: $D = V \times C = 93 \times 3 = 279$ continuous coordinates.
   - ST-GCN spatial feature embedding: $D_{model} = 256$ or $512$.

3. **Temporal Invariance:**
   - Normalization and temporal resampling ensure sequences from 25 FPS or 30 FPS cameras are interpolated onto a standard time-base without distorting signing velocity.
