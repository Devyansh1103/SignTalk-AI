# Batching Strategy: SignTalk AI

**Document ID:** STAI-P2P3-027  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead ML Data Engineer & System Architect  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Batch Tensor Architecture

Spatial-Temporal Graph Convolutional Networks (ST-GCN) and Transformer architectures require batched multidimensional tensors. SignTalk AI implements two distinct collation strategies:

### 1.1 Mode 1: Static Synchronous Batching (`sign_sequence_collate_fn`)
Used for all primary training and evaluation runs where sequence duration is standardized to $T = 45$ frames.
- **Input Skeletal Tensor:** $\mathbf{X} \in \mathbb{R}^{B \times C \times 45 \times 93}$ (float32, C-contiguous).
- **Auxiliary Mask Tensor:** $\mathbf{M} \in \mathbb{R}^{B \times 1 \times 45 \times 93}$ (float32 binary presence mask).
- **Target Categorical Labels:** $\mathbf{y} \in \mathbb{Z}^{B}$ (int64 scalar class indices $[0..9]$).
- **Gloss Token IDs:** $\mathbf{g} \in \mathbb{Z}^{B}$ (int64 token identifiers $[4..13]$).
- **Decoder Input Tokens:** $\mathbf{Y}_{in} \in \mathbb{Z}^{B \times 2}$ (`[<BOS>, GLOSS_ID]`).
- **Decoder Target Labels:** $\mathbf{Y}_{tgt} \in \mathbb{Z}^{B \times 2}$ (`[GLOSS_ID, <EOS>]`).
- **Causal Attention Mask:** $\mathbf{M}_{attn} \in \mathbb{Z}^{B \times 2}$ (all ones `[1, 1]`).

### 1.2 Mode 2: Dynamic Temporal Padding Collation (`pad_variable_sequence_collate_fn`)
Used when ingesting unnormalized, variable-duration sequence recordings ($N \in [45, 55]$).
- Pads along the time dimension to $T_{max} = \max_{b}(T_b)$.
- Automatically pads the binary mask tensor with zeros for $t > T_b$, preventing spatial-temporal graph kernels and attention layers from computing activations over padded intervals.

---

## 2. Memory & Throughput Characteristics

- **Memory Footprint per Batch ($B = 16$):**
  $$\text{Memory} = 16 \times (3 \times 45 \times 93 \times 4\text{ bytes}) \approx 803.5\text{ KB}$$
  The compact representation allows large batch sizes ($B = 32, 64, 128$) to reside entirely within fast GPU L2 / VRAM cache without memory fragmentation.
- **DataLoader Prefetching:** On Windows local environments, `num_workers=0` eliminates multiprocessing fork overhead; on Linux clusters, `num_workers=4` with `pin_memory=True` achieves over 450 batches/second.
