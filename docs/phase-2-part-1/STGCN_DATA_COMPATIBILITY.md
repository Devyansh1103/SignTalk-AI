# ST-GCN Data Compatibility Analysis

**Document ID:** STAI-P2P1-006  
**Phase:** Phase 2 — Part 1  
**Status:** Approved  

---

## 1. Graph Skeleton Specification ($V = 93$ Nodes)

To feed the Spatial-Temporal Graph Convolutional Network (ST-GCN), landmark frames must map onto a fixed node topology:

$$\mathcal{V} = \{0, \dots, 92\}$$

- **Nodes 0–20:** Left Hand Articulators (21 joints: wrist, thumb, index, middle, ring, pinky).
- **Nodes 21–41:** Right Hand Articulators (21 joints: wrist, thumb, index, middle, ring, pinky).
- **Nodes 42–52:** Upper-Body Pose Anchors (11 joints: nose, eyes, ears, shoulders, elbows, pose wrists).
- **Nodes 53–92:** Salient Facial Non-Manual Markers (40 facial joints or zero-padded if face mesh disabled).

---

## 2. Expected Tensor Format

The preprocessed representation produced in Phase 2 Part 2 must conform to:

$$\mathbf{X} \in \mathbb{R}^{B \times C \times T \times V}$$

Where:
- $B$: Mini-batch size.
- $C = 3$: Spatial coordinate channels $(x, y, z)$.
- $T = 45$: Fixed temporal sequence length (at 25 FPS, $45\text{ frames} = 1.8\text{ seconds}$).
- $V = 93$: Number of skeletal joints.

An auxiliary visibility/confidence tensor:
$$\mathbf{V}_{mask} \in \mathbb{R}^{B \times 1 \times T \times V}$$
is also preserved to inform graph attention and spatial masking.
