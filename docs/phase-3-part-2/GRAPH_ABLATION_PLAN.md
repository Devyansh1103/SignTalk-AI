# SignTalk AI — Graph Topology & Partitioning Ablation Plan

**Document ID:** `DOC-P3P2-ABLATION-GRAPH-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Code Reference:** [`src/models/graph.py`](file:///d:/SignAI/src/models/graph.py)  

---

## 1. Motivation

A core hypothesis of ST-GCN is that physical anatomical bones and directional spatial partitioning provide inductive biases superior to unconstrained or uniformly connected graphs.

This ablation plan defines a structured comparison to test:
> *"Does directional spatial partitioning ($K=3$) and explicit anatomical bone connectivity outperform uniform or random graph representations?"*

---

## 2. Compared Graph Topologies

1. **Topology 1: Spatial Configuration Partitioning ($K=3$, Canonical ST-GCN):**
   - Natural bone connectivity (105 undirected edges).
   - Partitioned into 3 subsets: root/self ($k=0$), centripetal inward flow ($k=1$), centrifugal outward flow ($k=2$).
   - Learnable edge importance mask $\mathbf{M} \in \mathbb{R}^{3 \times 93 \times 93}$.
2. **Topology 2: Distance Partitioning ($K=2$):**
   - Subset 0: Self-loops ($I$).
   - Subset 1: 1-hop physical skeletal neighbors.
3. **Topology 3: Uniform Partitioning ($K=1$):**
   - Single symmetrically normalized adjacency $\mathbf{\hat{A}} = \mathbf{\tilde{D}}^{-\frac{1}{2}} (\mathbf{A} + \mathbf{I}) \mathbf{\tilde{D}}^{-\frac{1}{2}}$.
   - Treats all neighboring joint connections identically without directional bias.
4. **Topology 4: Fully Connected Graph / Data-Driven (Dense Topology):**
   - Adjacency initialized as dense uniform or fully learned $93 \times 93$ matrix.
   - Evaluates whether unconstrained spatial attention can discover skeletal anatomy from scratch on small datasets.

---

## 3. Evaluation Protocol

- Identical model depth (6 blocks), temporal kernel ($K_t=9$), and training protocol across all topology variants.
- Test metrics compared: Top-1 Accuracy, Macro-F1, convergence speed, and parameter footprint.
