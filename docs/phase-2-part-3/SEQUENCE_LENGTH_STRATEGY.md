# Sequence Length Strategy: SignTalk AI

**Document ID:** STAI-P2P3-005  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead AI/ML Data Engineer & Lead Architect  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Architectural Trade-Off Analysis

Sequence length design determines memory consumption, computational graph staticness, batch parallelization efficiency, and compatibility with spatial-temporal graph convolutions (ST-GCN). We evaluate three potential sequence length paradigms:

| Architectural Dimension | Option A: Strictly Fixed ($T = 45$) | Option B: Purely Variable ($T \in [45, 55]$) | Option C: Standardized Window ($T = 45$) + Native Metadata (Selected) |
| :--- | :--- | :--- | :--- |
| **ST-GCN Batching** | Seamless static tensor $[B, C, T, V]$ without runtime dynamic graph rebuilding. | Requires custom collate function with batch-level zero padding to $\max(T_{batch})$. | **Optimal:** Native $[B, C, 45, V]$ static tensor ensures $100\%$ GPU memory predictability. |
| **Transformer Processing** | Compatible with fixed-length positional encodings. | Supports dynamic sequence lengths via attention masks $\mathbf{M}_{attn}$. | **Supported:** Standardized $T=45$ supports both fixed and masked attention layers. |
| **Edge & Real-Time Sync** | Matches real-time FIFO sliding window ($W = 45$ frames $\approx 1.8\text{ s}$). | Mismatch between dynamic training lengths and static edge buffer. | **Exact Match:** Training window perfectly mirrors the inference sliding window ($W=45$). |
| **Information Preservation**| Resamples 55 frames to 45 or crops resting tail frames. | Retains exact original frame indices. | **Preserved:** All native frame counts, durations, and boundary indices stored in metadata. |
| **Padding Overhead** | $0\%$ padding overhead. | Up to $18.2\%$ padding overhead per batch ($10/55$ frames). | **$0\%$ padding overhead** during primary model execution. |

---

## 2. Selected Strategy: Standardized Fixed Window ($T = 45$) with Preserved Native Temporal Metadata

### 2.1 Justification Based on Empirical Statistics
As documented in [`SEQUENCE_LENGTH_STATISTICS.md`](file:///d:/SignAI/docs/phase-2-part-3/SEQUENCE_LENGTH_STATISTICS.md):
- Exactly $50\%$ of dataset instances have a native duration of exactly $45$ frames ($1.80\text{ s}$).
- The remaining instances span $53$ to $55$ frames ($2.12\text{ s}$ to $2.20\text{ s}$), where the additional $8$ to $10$ frames correspond strictly to the signer retracting hands into rest position following sign execution.
- Standardizing to $T = 45$ frames captures the entire active signing stroke (preparation, stroke nucleus, and hold) without truncating meaningful semantic content.

### 2.2 Mathematical Transformation
For a source recording of length $N$ frames, temporal resampling to standardized length $T = 45$ is computed via uniform linear temporal interpolation:
$$t'_k = \frac{k}{T - 1} \times (N - 1), \quad k \in \{0, 1, \dots, T - 1\}$$
$$\mathbf{X}_{std}[c, k, v] = (1 - \alpha) \cdot \mathbf{X}_{raw}[c, \lfloor t'_k \rfloor, v] + \alpha \cdot \mathbf{X}_{raw}[c, \lceil t'_k \rceil, v]$$
where $\alpha = t'_k - \lfloor t'_k \rfloor$.

### 2.3 Auxiliary Mask Representation
Alongside the coordinate tensor $\mathbf{X} \in \mathbb{R}^{3 \times 45 \times 93}$, an explicit binary validity mask $\mathbf{M} \in \{0, 1\}^{1 \times 45 \times 93}$ is stored:
$$M[0, t, v] = \begin{cases} 1 & \text{if joint } v \text{ was directly detected or validly interpolated at frame } t \\ 0 & \text{if joint } v \text{ is occluded, dormant, or missing} \end{cases}$$
This ensures downstream graph attention layers and temporal pooling operators can selectively ignore unobserved articulator nodes.
