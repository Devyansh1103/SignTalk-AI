# SignTalk AI — ST-GCN Objective & Research Rationale

**Document ID:** `DOC-P3P2-OBJECTIVE-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Author:** Lead Computer Vision & Machine Learning Engineering Team  
**Date:** October 2, 2026  

---

## 1. Context and Problem Statement

Indian Sign Language (ISL) communication combines:
1. **Dynamic Manual Articulation:** Fine-grained finger motions, hand configurations, orientations, and trajectories of dominant and non-dominant hands.
2. **Gross Skeletal Kinematics:** Arm raises, shoulder shifts, elbow extensions, and torso orientation.
3. **Non-Manual Grammatical Cues:** Facial expressions, eyebrow furrowing/raising, mouth shapes, and head orientation.

In Phase 3 Part 1, the baseline model (`SignBaselineModel`) achieved $41.67\%$ Top-1 accuracy and $75.00\%$ Top-3 accuracy on the isolated test set using a 2-layer Bidirectional GRU. However, empirical error analysis ([`BASELINE_ERROR_ANALYSIS.md`](file:///d:/SignAI/docs/phase-3-part-1/BASELINE_ERROR_ANALYSIS.md)) revealed a fundamental architectural bottleneck:

> **The Flattened Projection Limitation:** The baseline model vectorized the 93 skeletal landmarks into an unstructured 279-dimensional array ($3 \times 93$). By discarding the underlying physical connectivity of the human skeleton, the model failed to distinguish signs that share similar macroscopic hand trajectories but differ in subtle finger-joint articulations (e.g. `BIRD` pinching vs. `GOOD` thumbs-up vs. `TIME` index tap).

---

## 2. Why Spatial-Temporal Graph Convolutional Networks (ST-GCN)?

Human signing gestures are naturally structured as a **spatiotemporal graph**:
- **Spatial Dimension ($V = 93$ Nodes):** Physical bones, phalanges, facial contours, and joints naturally form a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$. Joints connected by rigid bones (e.g. thumb MCP to thumb IP, shoulder to elbow) constrain motion, while neighboring fingers interact during shape formation.
- **Temporal Dimension ($T = 45$ Frames):** Joint positions evolve continuously over time, capturing velocity, acceleration, and rhythmic repetitions.

```
Spatiotemporal Graph Architecture:
      Frame t-1                    Frame t                    Frame t+1
   (Spatial Graph)             (Spatial Graph)             (Spatial Graph)
       (Node)                      (Node)                      (Node)
       /    \                      /    \                      /    \
    (Node)-(Node) ------------> (Node)-(Node) ------------> (Node)-(Node)
       \    /   (Temporal Edge)    \    /   (Temporal Edge)    \    /
       (Node)                      (Node)                      (Node)
```

The Spatial-Temporal Graph Convolutional Network (Yan et al., AAAI 2018) is designed to exploit this dual inductive bias:
1. **Spatial Graph Convolutions:** Aggregate features along localized anatomical bone edges defined by the adjacency matrix $\mathbf{A}$, extracting joint configuration patterns invariant to global translation.
2. **Temporal 1D Convolutions:** Model feature evolution across consecutive frames along temporal edges, extracting velocity and rhythm without recurrent vanishing gradient constraints.

---

## 3. Concrete Phase 3 Part 2 Objective

Given a preprocessed, torso-normalized landmark sequence:
$$\mathbf{X} \in \mathbb{R}^{B \times C \times T \times V}$$
where $B$ is batch size, $C=3$ spatial channels $(x, y, z)$, $T=45$ temporal frames, and $V=93$ anatomical nodes, the objective is to:

1. **Construct a rigorous, anatomical graph topology** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with normalized adjacency matrices $\mathbf{A} \in \mathbb{R}^{K \times V \times V}$.
2. **Implement modular ST-GCN building blocks** consisting of spatial graph convolution, batch normalization, ReLU activation, temporal 1D convolution, dropout, and residual connections.
3. **Train and validate the ST-GCN network** on the canonical sequence dataset (`signTalk-seq-v1.0.0`) using the verified training coordinator.
4. **Evaluate on the isolated test partition** and measure the exact empirical delta against the BiGRU baseline ($41.67\%$ Top-1 Accuracy, $0.3067$ Macro-F1).
5. **Prepare clean, standardized spatiotemporal feature embeddings** for subsequent Transformer handoff without implementing sentence-level translation prematurely.

---

## 4. Scope and Guardrails

- **Strict Task Boundary:** This phase focuses exclusively on **Isolated Sign/Class Recognition** ($10$ classes).
- **No Transformer/NLP:** The autoregressive Transformer decoder will **NOT** be implemented in this phase.
- **No Fabricated Labels:** All evaluations are conducted against real ground-truth gloss labels. No continuous sentence annotations will be simulated.
