# ST-GCN Sequence Format Specification: SignTalk AI

**Document ID:** STAI-P2P3-014  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead Architect & ML Data Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Input Tensor Specification

Spatial-Temporal Graph Convolutional Networks (ST-GCN) ingest skeleton sequences formatted as 4-dimensional tensors:

$$\mathbf{X} \in \mathbb{R}^{B \times C \times T \times V}$$

| Axis | Symbol | Standard Dimension | Semantics & Content | Memory Layout |
| :---: | :---: | :---: | :--- | :--- |
| **0** | $B$ | Configurable ($8, 16, 32$) | Batch size of independent sign sequences | Contiguous |
| **1** | $C$ | $3$ (or $6$ with velocity) | Feature channels: $(x, y, z)$ or $(x, y, z, \Delta x, \Delta y, \Delta z)$ | Contiguous |
| **2** | $T$ | $45$ frames | Temporal sequence duration ($1.80\text{ s}$ at $25.0\text{ FPS}$) | Contiguous |
| **3** | $V$ | $93$ vertices | Curated skeletal graph nodes | Contiguous |

---

## 2. Graph Node Partitioning ($V = 93$)

The 93 nodes represent anatomical landmarks organized into four structurally contiguous subgraphs:

```
Nodes 00..20: Left Hand Articulators   [21 joints]
Nodes 21..41: Right Hand Articulators  [21 joints]
Nodes 42..52: Upper-Body Pose Anchors  [11 joints]
Nodes 53..92: Facial Non-Manual Markers [40 joints]
```

### 2.1 Hand Node Indices ($0 \le v \le 20$ for Left, $21 \le v \le 41$ for Right)
- Wrist: $0$ (Left), $21$ (Right)
- Thumb: $1..4$ (Left), $22..25$ (Right)
- Index: $5..8$ (Left), $26..29$ (Right)
- Middle: $9..12$ (Left), $30..33$ (Right)
- Ring: $13..16$ (Left), $34..37$ (Right)
- Pinky: $17..20$ (Left), $38..41$ (Right)

### 2.2 Upper-Body Pose Node Indices ($42 \le v \le 52$)
- $42$: Nose Anchor
- $43, 44$: Left Eye, Right Eye
- $45, 46$: Left Ear, Right Ear
- $47, 48$: Left Shoulder, Right Shoulder (Torso reference midpoint)
- $49, 50$: Left Elbow, Right Elbow
- $51, 52$: Left Pose Wrist, Right Pose Wrist (Cross-modality kinematic bridge nodes)

---

## 3. Spatial Adjacency Matrix Layout ($\mathbf{\Lambda} \in \mathbb{R}^{3 \times 93 \times 93}$)

Directional spatial graph convolutions employ a 3-partition adjacency tensor saved at [`assets/graphs/kinematic_adjacency_93.npy`](file:///d:/SignAI/assets/graphs/kinematic_adjacency_93.npy):
$$\mathbf{\Lambda}[0, :, :] = \mathbf{\Lambda}_{root} = \mathbf{I}_{93} \quad (\text{Self-loops})$$
$$\mathbf{\Lambda}[1, :, :] = \mathbf{\Lambda}_{centripetal} \quad (\text{Inward flow toward torso midpoint})$$
$$\mathbf{\Lambda}[2, :, :] = \mathbf{\Lambda}_{centrifugal} \quad (\text{Outward flow toward fingertips and facial contours})$$

Normalized symmetrically:
$$\mathbf{\Lambda}_k = \mathbf{D}_k^{-\frac{1}{2}} \mathbf{A}_k \mathbf{D}_k^{-\frac{1}{2}}$$

---

## 4. First-Order Temporal Velocity Features ($C = 6$)

To capture articulation speed and acceleration explicitly, optional first-order backward finite differences are computed:
$$\Delta \mathbf{p}_t = \mathbf{p}_t - \mathbf{p}_{t-1}, \quad \text{with } \Delta \mathbf{p}_0 = \mathbf{0}$$
Concatenating spatial coordinates with velocity yields:
$$\mathbf{X}_{aug} = [\mathbf{p}_t \,\|\, \Delta \mathbf{p}_t] \in \mathbb{R}^{B \times 6 \times T \times V}$$
Implemented via `compute_temporal_velocity()` in [`src/data/stgcn_tensor.py`](file:///d:/SignAI/src/data/stgcn_tensor.py).
