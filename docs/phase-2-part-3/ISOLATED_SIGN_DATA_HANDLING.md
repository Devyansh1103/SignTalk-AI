# Isolated Sign Data Handling: SignTalk AI

**Document ID:** STAI-P2P3-008  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead ML Data Engineer & Computer Vision Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Principles of Isolated Sign Representation

Isolated sign language recognition (ISLR) forms the empirical foundation of the SignTalk AI spatial-temporal modeling pipeline. In this phase, each valid recording is treated as a self-contained lexical gesture sequence.

### 1.1 Integrity Constraints
1. **No Artificial Concatenation:** We strictly prohibit stitching together isolated sign clips (e.g. `hello` + `good` + `time`) to manufacture synthetic sentences. Such synthetic sequences lack natural coarticulation, introduce boundary artifacts, and yield deceptive evaluation metrics.
2. **No Fabricated Continuous Targets:** Lexical labels are maintained as isolated categorical targets $[0..K-1]$ with corresponding linguistic glosses (e.g. `HELLO`, `CAR`).
3. **Temporal Self-Sufficiency:** Each recording encompasses the complete gestural stroke lifecycle:
   - **Preparation Phase:** Movement of hands from a resting pose to the designated signing space.
   - **Stroke Nucleus:** The primary morphological gesture carrying semantic meaning (handshape, orientation, location, movement).
   - **Retraction / Post-Stroke Phase:** Return of hands toward rest or transition.

---

## 2. Temporal Processing Workflow

```
Raw Isolated Recording [45..55 frames]
       │
       ▼
Uniform Temporal Normalization [T = 45 frames]
       │
       ▼
Torso-Centered Spatial Normalization
       │
       ▼
Anatomical Graph Formatting [C=3, T=45, V=93]
       │
       ▼
Canonical Isolated Sign Sequence Object
```

### 2.1 Uniform Temporal Resampling
Because signers execute signs at natural human tempo variations ($1.8\text{ s}$ to $2.2\text{ s}$), uniform temporal resampling aligns all gestural trajectories to a standard $T = 45$ frame canvas ($1.80\text{ s}$ at $25.0\text{ FPS}$). This preserves:
- The relative velocity profile of the hands and fingers.
- The inflection points of joint angle trajectories.
- The spatial geometry of handshapes at the apex of the sign stroke.

---

## 3. Bridge to Future Continuous Translation

Treating isolated signs as standardized temporal sequences establishes a modular architecture:
1. **Isolated Sign Classifier:** $\text{ST-GCN}(\mathbf{X}_{1:45}) \to \mathbf{z} \in \mathbb{R}^K$.
2. **Sliding Window Buffer:** In real-time webcam streaming, a FIFO buffer of length $45$ frames collects landmarks and evaluates predictions every $S = 5$ frames.
3. **Continuous Sentence Decoder:** In Phase 3 and Phase 4, the spatial-temporal embeddings $\mathbf{H} = \text{ST-GCN}(\mathbf{X})$ serve as token feature vectors that feed into a Transformer autoregressive language model or CTC beam search decoder.
