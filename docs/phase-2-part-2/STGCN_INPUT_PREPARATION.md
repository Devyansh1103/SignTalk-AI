# ST-GCN Input Preparation Specification: SignTalk AI

**Document ID:** STAI-P2P2-005  
**Project:** SignTalk AI  
**Phase:** Phase 2 — Part 2  
**Status:** Approved  

---

## 1. Input Tensor Dimension & Semantic Layout

Spatial-Temporal Graph Convolutional Networks (ST-GCN) operate on skeletal joint graphs evolved across time. The preprocessed data produced by this pipeline maps directly into:

$$\mathbf{X} \in \mathbb{R}^{B \times C \times T \times V}$$

| Axis | Dimension | Parameter Name | Description |
| :---: | :---: | :--- | :--- |
| **0** | $B$ | `batch_size` | Number of sign sequences processed in parallel (e.g. 8, 16, 32). |
| **1** | $C = 3$ | `channels` | Normalized spatial joint coordinates $(x, y, z)$ in body-centered space. |
| **2** | $T = 45$ | `sequence_length` | Temporal duration (45 frames at $25\text{ FPS} = 1.8\text{ seconds}$). |
| **3** | $V = 93$ | `num_nodes` | Curated skeletal joints (21 LH + 21 RH + 11 Pose + 40 Face). |

An auxiliary confidence mask tensor:
$$\mathbf{V}_{mask} \in \mathbb{R}^{B \times 1 \times T \times V}$$
is fed to graph attention modules to modulate edge weights when joints are occluded.

---

## 2. Anatomical Graph Edge Topology ($102$ Edges)

The physical graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ connects $V = 93$ joints through $102$ anatomical and kinematic edges:

### 2.1 Hand Articulators Subgraphs (40 Edges)
- **Left Hand (20 edges):**
  - Thumb: $0-1-2-3-4$
  - Index: $0-5-6-7-8$
  - Middle: $0-9-10-11-12$
  - Ring: $0-13-14-15-16$
  - Pinky: $0-17-18-19-20$
- **Right Hand (20 edges):**
  - Symmetrically connects nodes $21$ to $41$.

### 2.2 Upper-Body Pose Subgraph (12 Edges)
- Shoulder baseline: $(47, 48)$
- Left arm chain: $(47, 49), (49, 51)$
- Right arm chain: $(48, 50), (50, 52)$
- Facial anchor bridges: $(47, 42), (48, 42), (42, 43), (42, 44), (43, 45), (44, 46)$

### 2.3 Kinematic Wrist Cross-Bridges (2 Edges)
- Left Hand Wrist (Node 0) $\longleftrightarrow$ Left Wrist Pose (Node 51)
- Right Hand Wrist (Node 21) $\longleftrightarrow$ Right Wrist Pose (Node 52)

---

## 3. Spatial Configuration Adjacency Partitioning ($K = 3$)

To enable directional spatial graph convolutions, the raw binary adjacency matrix $\mathbf{A} \in \{0, 1\}^{93 \times 93}$ is partitioned into 3 sub-matrices based on distance from the body torso anchor:

$$\mathbf{A} + \mathbf{I} = \mathbf{A}_{root} + \mathbf{A}_{centripetal} + \mathbf{A}_{centrifugal}$$

1. **Root Partition ($\mathbf{A}_{root} = \mathbf{I}_{93}$):** Self-loops connecting each joint to itself.
2. **Centripetal Partition ($\mathbf{A}_{centripetal}$):** Edges connecting a joint $j$ closer to the torso center than joint $i$ (inward energy flow).
3. **Centrifugal Partition ($\mathbf{A}_{centrifugal}$):** Edges connecting a joint $j$ further from the torso center than joint $i$ (outward flow to fingertips).

Each partitioned adjacency matrix is normalized symmetrically:
$$\mathbf{\Lambda}_k = \mathbf{D}_k^{-\frac{1}{2}} \mathbf{A}_k \mathbf{D}_k^{-\frac{1}{2}}, \quad k \in \{1, 2, 3\}$$

The static normalized adjacency tensor:
$$\mathbf{\Lambda} \in \mathbb{R}^{3 \times 93 \times 93}$$
is precomputed and cached for zero-latency graph convolution forward passes.

---

## 4. PyTorch DataLoader Integration

The preprocessed data format is loadable by standard PyTorch `DataLoader` pipelines:

```python
from src.data.landmark_dataset import create_landmark_dataloader

train_loader = create_landmark_dataloader(
    data_dir="data/processed/landmarks",
    split="train",
    batch_size=16,
    shuffle=True
)

for batch_data, batch_mask, batch_labels, meta in train_loader:
    # batch_data: [16, 3, 45, 93]
    # batch_mask: [16, 1, 45, 93]
    # batch_labels: [16]
    output = stgcn_model(batch_data, batch_mask)
```
