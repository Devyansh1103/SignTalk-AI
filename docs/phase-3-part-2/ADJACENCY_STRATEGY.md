# SignTalk AI — Graph Adjacency & Normalization Strategy

**Document ID:** `DOC-P3P2-ADJACENCY-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Schema Identifier:** `93-node-v1`  
**Code Reference:** [`src/models/graph.py`](file:///d:/SignAI/src/models/graph.py)  

---

## 1. Mathematical Formulation

In standard Graph Convolutional Networks (Kipf & Welling, ICLR 2017), spatial feature propagation on an undirected graph is expressed as:
$$\mathbf{Z} = \mathbf{\hat{A}} \mathbf{X} \mathbf{W}$$
where $\mathbf{\hat{A}} = \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}}$ is the symmetrically normalized adjacency matrix with added self-loops $\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{I}$, and $\mathbf{\tilde{D}}_{ii} = \sum_j \mathbf{\tilde{A}}_{ij}$ is the degree diagonal matrix.

While symmetric normalization works effectively for homogenous citation networks, human skeletal motion is fundamentally **asymmetric and directional**:
1. Proximal joints (e.g. shoulders, elbows, wrists) drive distal joints (e.g. fingers, phalanges).
2. Movement towards the kinematic root (centripetal motion) has distinct physical semantics from movement away from the root (centrifugal extension).
3. Self-dynamics (the joint's own coordinate trajectory) must not be dominated by adjacent high-degree nodes (e.g. wrist connected to 5 metacarpals).

---

## 2. Partitioning Strategies Supported

[`src/models/graph.py`](file:///d:/SignAI/src/models/graph.py) implements three distinct partitioning formulations:

### Strategy 1: Spatial Configuration Partitioning ($K=3$, Default ST-GCN)
Following Yan et al. (AAAI 2018), the 1-hop spatial neighborhood $B(v_i)$ of node $v_i$ is partitioned into $K=3$ mutually exclusive subsets based on radial graph distance from the kinematic root $r_{\text{root}}$ (Node 42: Nose / cranial reference):

$$B(v_i) = \{v_j \mid d(v_i, v_j) \le 1\}$$

1. **Root / Self-Connection Subset ($k=0$):**
   $$B_0(v_i) = \{v_i\}$$
   *Models the joint's intrinsic coordinate position and temporal velocity.*
2. **Centripetal Subset ($k=1$):**
   $$B_1(v_i) = \{v_j \mid d(v_i, v_j) = 1 \text{ and } d(v_j, r_{\text{root}}) < d(v_i, r_{\text{root}})\}$$
   *Aggregates message-passing signals flowing inward towards the core body frame.*
3. **Centrifugal Subset ($k=2$):**
   $$B_2(v_i) = \{v_j \mid d(v_i, v_j) = 1 \text{ and } d(v_j, r_{\text{root}}) \ge d(v_i, r_{\text{root}})\}$$
   *Aggregates message-passing signals flowing outward towards finger tips and distal extremities.*

#### Row-Degree Normalization per Subset:
For each partition subset $k \in \{0, 1, 2\}$, the unweighted adjacency matrix is defined by:
$$\mathbf{A}_k(i, j) = \begin{cases} 1 & \text{if } v_j \in B_k(v_i) \\ 0 & \text{otherwise} \end{cases}$$

To prevent numerical gradient explosion when summing varying neighbor degrees, each matrix is normalized by its out-degree:
$$\mathbf{\hat{A}}_k = \mathbf{D}_k^{-1} \mathbf{A}_k$$
where $\mathbf{D}_k(i, i) = \sum_j \mathbf{A}_k(i, j) + \epsilon$.

The stacked adjacency tensor has shape:
$$\mathbf{\hat{A}} \in \mathbb{R}^{3 \times 93 \times 93}$$

---

### Strategy 2: Distance Partitioning ($K=2$)
Partitions the neighborhood strictly by graph distance:
- **Subset 0:** Self-loops ($d = 0$, $\mathbf{I}_{93 \times 93}$)
- **Subset 1:** 1-hop physical skeletal neighbors ($d = 1$, $\mathbf{A}_{\text{physical}}$)
Each subset is row-normalized by $\mathbf{D}_k^{-1}$. Resulting tensor: $\mathbf{\hat{A}} \in \mathbb{R}^{2 \times 93 \times 93}$.

---

### Strategy 3: Uniform Partitioning ($K=1$)
All neighbors and self-connections share the same kernel weight:
$$\mathbf{\hat{A}} = \mathbf{\tilde{D}}^{-\frac{1}{2}} (\mathbf{A} + \mathbf{I}) \mathbf{\tilde{D}}^{-\frac{1}{2}} \in \mathbb{R}^{1 \times 93 \times 93}$$

---

## 3. Spatial Graph Convolution with Partitioned Adjacency

Given input features $\mathbf{X} \in \mathbb{R}^{B \times C_{\text{in}} \times T \times V}$ and partitioned normalized adjacency $\mathbf{\hat{A}} \in \mathbb{R}^{K \times V \times V}$, the spatial graph convolution computes:

$$\mathbf{Y}_{b, :, t, :} = \sum_{k=0}^{K-1} \mathbf{W}_k \mathbf{X}_{b, :, t, :} \mathbf{\hat{A}}_k^{\top}$$

where $\mathbf{W}_k \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}}}$ is a learnable parameter matrix for each spatial partition. This provides independent learnable weight matrices for self-dynamics, inward body flow, and outward finger articulation.

---

## 4. Learnable Edge Importance Weighting (Mask Matrix $\mathbf{M}$)

In addition to fixed topological adjacency $\mathbf{\hat{A}}$, each ST-GCN block includes a learnable edge importance weight tensor:
$$\mathbf{M} \in \mathbb{R}^{K \times V \times V}$$
initialized to all ones ($\mathbf{M} = \mathbf{1}$). The effective spatial adjacency becomes:
$$\mathbf{A}_{\text{eff}} = \mathbf{\hat{A}} \odot \mathbf{M}$$
where $\odot$ denotes element-wise Hadamard product. This allows the model to dynamically strengthen informative anatomical edges (e.g. index-to-thumb during `BIRD`) and suppress uninformative edges during training.
