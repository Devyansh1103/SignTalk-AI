# Baseline Input Format Specification: SignTalk AI

**Document ID:** STAI-P3P1-004  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 3 — Part 1 (Baseline Model & Training Infrastructure)  
**Author:** Lead ML Data Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Input Tensor Invariance

To maintain data pipeline integrity and eliminate unnecessary serialization duplication, the baseline model ingests the **exact same canonical sequence format** finalized in Phase 2 Part 3:

$$\mathbf{X} \in \mathbb{R}^{B \times C \times T \times V}$$

| Axis | Symbol | Standard Dimension | Semantics & Content |
| :---: | :---: | :---: | :--- |
| **0** | $B$ | Configurable ($8, 16, 32$) | Batch size of independent sequence samples |
| **1** | $C$ | $3$ (or $6$ with velocity) | Spatial Cartesian coordinates $(x, y, z)$ in torso units |
| **2** | $T$ | $45$ frames | Temporal duration ($1.80\text{ s}$ at $25.0\text{ FPS}$) |
| **3** | $V$ | $93$ nodes | Curated skeletal joints (21 LH, 21 RH, 11 Pose, 40 Face) |

---

## 2. In-Model Deterministic Spatial Flattening

Because recurrent neural networks (such as BiGRU) operate on feature vectors across sequential timesteps, the 4-dimensional spatial-temporal tensor is reshaped internally inside `SignBaselineModel.forward()`:

$$\mathbf{X} \in \mathbb{R}^{B \times C \times T \times V} \xrightarrow{\text{Permute } (0, 2, 1, 3)} \mathbf{X}' \in \mathbb{R}^{B \times T \times C \times V} \xrightarrow{\text{Flatten } (2, 3)} \mathbf{X}_{seq} \in \mathbb{R}^{B \times T \times D_{in}}$$

where:
$$D_{in} = C \times V = 3 \times 93 = 279\text{ features per frame}$$

### 2.1 Properties of the Transformation
1. **Zero Data Duplication:** No duplicate datasets or flattened files are saved to disk.
2. **Temporal Order Preservation:** The temporal dimension $T = 45$ is preserved as the sequence axis for the recurrent layers.
3. **Deterministic & Reversible:** Frame $t$ contains precisely the ordered coordinates:
   $$\mathbf{x}_t = [x_{t, 0}, y_{t, 0}, z_{t, 0}, x_{t, 1}, y_{t, 1}, z_{t, 1}, \dots, x_{t, 92}, y_{t, 92}, z_{t, 92}]^T \in \mathbb{R}^{279}$$

---

## 3. Auxiliary Mask Handling

The auxiliary binary validity mask tensor $\mathbf{M} \in \mathbb{R}^{B \times 1 \times T \times V}$ can optionally be compressed along the node axis:
$$\mathbf{m}_{frame} = \frac{1}{V}\sum_{v=0}^{V-1} \mathbf{M}[:, 0, :, v] \in \mathbb{R}^{B \times T}$$
indicating the fraction of validly detected joints per frame. In the standard baseline, frames with zero detected hands contribute zero coordinates naturally after torso centering.
