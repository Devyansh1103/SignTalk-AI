# Padding and Masking Strategy: SignTalk AI

**Document ID:** STAI-P2P3-013  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead ML Data Engineer & Research Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Architectural Challenge

Neural sequence architectures (such as ST-GCN and Transformers) require batch tensors of uniform dimensions. However, sign language recordings exhibit natural duration variability ($45$ to $55$ frames in the active benchmark). When temporal sequences are aligned, the choice of padding, resampling, or masking fundamentally impacts model gradients and temporal convolution kernels.

---

## 2. Evaluation of Alignment Paradigms

| Alignment Method | Mathematical Behavior | Impact on ST-GCN Temporal Conv | Impact on Transformer Attention | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Zero Padding** | Frames $t > N$ set to $(0, 0, 0)$ | Destructive: In torso-centered space, $(0,0,0)$ represents the body chest center, falsely indicating hand collapse. | Harmless if attention mask $\mathbf{M}_{attn} = 0$ is applied. | **Avoid for Coordinates**; use only for mask tensors. |
| **Edge / Repeat Padding** | Frame $N$ replicated for $t > N$ | Generates zero-velocity flatlines ($\Delta \mathbf{p} = \mathbf{0}$); may cause false sign "holds". | Supported via causal masking. | **Secondary / Fallback** for sliding window boundaries. |
| **Uniform Temporal Resampling** | Continuous trajectory $\mathbf{p}(t)$ interpolated to fixed $T = 45$ | Preserves continuous kinematic velocity profile without artificial static holds. | Fully compatible with fixed positional encodings. | **Selected Primary Strategy** for full-utterance sequences. |

---

## 3. Selected Strategy: Uniform Trajectory Resampling with Dual-Level Masking

### 3.1 Continuous Trajectory Resampling
For all primary canonical sequences, full recordings ($N \in [45, 55]$) are uniformly resampled onto the standardized temporal grid $T = 45$ frames ($1.80\text{ s}$ at $25.0\text{ FPS}$) using linear temporal interpolation:
$$t'_k = \frac{k}{T - 1} \times (N - 1), \quad k \in \{0, 1, \dots, T - 1\}$$
This guarantees:
- Zero padded dummy frames during standard training.
- Natural preservation of peak articulation velocities.
- Elimination of edge discontinuities at sequence boundaries.

### 3.2 Auxiliary Binary Mask Tensor ($\mathbf{M} \in \{0, 1\}^{1 \times T \times V}$)
To account for detector occlusions, dropped hands, or out-of-frame articulators, an explicit binary mask tensor is serialized alongside the coordinates:

$$M[0, t, v] = \begin{cases} 
1.0 & \text{if joint } v \text{ at frame } t \text{ is genuine or validly interpolated} \\ 
0.0 & \text{if joint } v \text{ at frame } t \text{ is missing, dormant, or invalid}
\end{cases}$$

### 3.3 Transformer Attention Masking
For batch collation of variable-length sequences or token sequences:
$$\mathbf{M}_{attn}[i, j] = \begin{cases} 1 & \text{if frame } j \le \text{valid length} \\ 0 & \text{if frame } j \text{ is padding} \end{cases}$$
The Transformer multi-head attention computes:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + (1 - \mathbf{M}_{attn}) \cdot (-10^9)\right)\mathbf{V}$$
ensuring zero gradient leakage into unobserved frames.
