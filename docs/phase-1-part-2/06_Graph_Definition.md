# 06. Graph Definition & Adjacency Mathematics: SignTalk AI

**Document ID:** STAI-P1P2-006  
**Project Name:** SignTalk AI  
**Document Status:** Approved Engineering Blueprint (Phase 1 — Part 2)  

---

## 1. Mathematical Graph Formulation

The signing human skeleton across a sliding temporal window of $T$ frames is modeled as a dynamic spatial-temporal graph:

$$\mathcal{G} = (\mathcal{V}, \mathcal{E})$$

Where:
- $\mathcal{V} = \{ v_{t, i} \mid t \in \{1, \dots, T\}, i \in \{1, \dots, N\} \}$ is the node set containing $N = 93$ joints per frame across $T = 45$ frames ($4,185$ total space-time nodes).
- $\mathcal{E} = \mathcal{E}_{spatial} \cup \mathcal{E}_{temporal}$ is the complete edge set:
  - $\mathcal{E}_{spatial} = \{ (v_{t, i}, v_{t, j}) \mid (i, j) \in \mathcal{E}_{skeleton}, t \in \{1, \dots, T\} \}$ represents intra-frame anatomical connectivity.
  - $\mathcal{E}_{temporal} = \{ (v_{t, i}, v_{t+1, i}) \mid t \in \{1, \dots, T-1\}, i \in \{1, \dots, N\} \}$ represents inter-frame joint trajectory persistence.

---

## 2. Anatomical Edge Connectivity Specification

The spatial graph contains **102 physical and kinematic bone edges** across four interconnected subsystems:

```mermaid
graph TD
    subgraph LeftHandEdges [Left Hand Kinematics: 20 Edges]
        L0[0: Wrist] --- L1[1: Thumb CMC] --- L2[2: Thumb MCP] --- L3[3: Thumb IP] --- L4[4: Thumb Tip]
        L0 --- L5[5: Index MCP] --- L6[6: Index PIP] --- L7[7: Index DIP] --- L8[8: Index Tip]
        L0 --- L9[9: Middle MCP] --- L10[10: Middle PIP] --- L11[11: Middle DIP] --- L12[12: Middle Tip]
        L0 --- L13[13: Ring MCP] --- L14[14: Ring PIP] --- L15[15: Ring DIP] --- L16[16: Ring Tip]
        L0 --- L17[17: Pinky MCP] --- L18[18: Pinky PIP] --- L19[19: Pinky DIP] --- L20[20: Pinky Tip]
    end

    subgraph RightHandEdges [Right Hand Kinematics: 20 Edges]
        R21[21: Wrist] --- R22[22: Thumb CMC] --- R23[23: Thumb MCP] --- R24[24: Thumb IP] --- R25[25: Thumb Tip]
        R21 --- R26[26: Index MCP] --- R27[27: Index PIP] --- R28[28: Index DIP] --- R29[29: Index Tip]
        R21 --- R30[30: Middle MCP] --- R31[31: Middle PIP] --- R32[32: Middle DIP] --- R33[33: Middle Tip]
        R21 --- R34[34: Ring MCP] --- R35[35: Ring PIP] --- R36[36: Ring DIP] --- R37[37: Ring Tip]
        R21 --- R38[38: Pinky MCP] --- R39[39: Pinky PIP] --- R40[40: Pinky DIP] --- R41[41: Pinky Tip]
    end

    subgraph UpperPoseEdges [Upper Pose Chains: 12 Edges]
        P47[47: L Shoulder] --- P48[48: R Shoulder]
        P47 --- P49[49: L Elbow] --- P51[51: L Wrist Pose]
        P48 --- P50[50: R Elbow] --- P52[52: R Wrist Pose]
        P47 --- P42[42: Nose] --- P48
        P42 --- P43[43: L Eye] --- P45[45: L Ear]
        P42 --- P44[44: R Eye] --- P46[46: R Ear]
    end

    subgraph CrossBridges [Kinematic Bridges: 2 Edges]
        L0 -.-|Left Wrist Bridge| P51
        R21 -.-|Right Wrist Bridge| P52
    end
```

---

## 3. Adjacency Matrix Normalization & Partitioning

To compute spatial graph convolutions, the raw binary adjacency matrix $\mathbf{A} \in \{0, 1\}^{N \times N}$ is partitioned according to **Spatial Configuration Partitioning** into $K = 3$ sub-matrices:

$$\mathbf{A} + \mathbf{I} = \mathbf{A}_{root} + \mathbf{A}_{centripetal} + \mathbf{A}_{centrifugal}$$

1. **Root Partition ($\mathbf{A}_{root} = \mathbf{I}$):** Represents self-loops connecting each node to itself.
2. **Centripetal Partition ($\mathbf{A}_{centripetal}$):** Contains edges $(i, j)$ where joint $j$ is closer to the body torso anchor than joint $i$ (inward kinematic flow).
3. **Centrifugal Partition ($\mathbf{A}_{centrifugal}$):** Contains edges $(i, j)$ where joint $j$ is further from the body torso anchor than joint $i$ (outward kinematic flow to fingertips).

Each partitioned adjacency matrix is symmetrically normalized:
$$\mathbf{\Lambda}_k = \mathbf{D}_k^{-\frac{1}{2}} \mathbf{A}_k \mathbf{D}_k^{-\frac{1}{2}}, \quad k \in \{root, centripetal, centrifugal\}$$
where $\mathbf{D}_k^{i, i} = \sum_j \mathbf{A}_k^{i, j} + \alpha$ is the diagonal degree matrix ($\alpha = 0.001$ avoids division by zero).

The resulting static tensor $\mathbf{\Lambda} \in \mathbb{R}^{3 \times 93 \times 93}$ is precomputed once during model initialization and stored directly in GPU/CPU memory for zero-latency indexing.
