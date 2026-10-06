# SignTalk AI — ST-GCN to Transformer Interface Specification

**Document ID:** `DOC-P3P3-STGCN-INTERFACE-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** SPECIFIED & IMPLEMENTED  

---

## 1. Visual Feature Tensor Extraction

The spatial-temporal graph convolutional network (`SignSTGCN`) processes 4D coordinate tensors:
$$\mathbf{X} \in \mathbb{R}^{B \times C \times T \times V}$$
where:
- $B$: Batch size
- $C = 3$: Spatial landmark coordinates $(x, y, z)$
- $T = 45$: Fixed-length temporal window
- $V = 93$: Canonical skeletal graph nodes

### Downsampling & Spatial Pooling Pipeline
1. **Block Stacking:**
   - Blocks 1 & 2: Stride 1 $\implies T_1 = 45, C_1 = 64$
   - Block 3: Stride 2 $\implies T_2 = 23, C_2 = 128$
   - Block 4: Stride 1 $\implies T_3 = 23, C_3 = 128$
   - Block 5: Stride 2 $\implies T_4 = 12, C_4 = 256$
   - Block 6: Stride 1 $\implies T_5 = 12, C_5 = 256$
2. **Intermediate Block 6 Representation:**
   $$\mathbf{H}_6 \in \mathbb{R}^{B \times 256 \times 12 \times 93}$$
3. **Spatial Node Reduction:**
   Spatial relationships across the 93 skeletal nodes have been integrated across 6 layers of partitioned graph convolution. The graph nodes are then pooled via spatial average pooling:
   $$\mathbf{X}_{\text{temp}} = \frac{1}{V} \sum_{v=1}^{V} \mathbf{H}_6[:, :, :, v] \in \mathbb{R}^{B \times 256 \times 12}$$
4. **Dimension Permutation to Sequence Format:**
   $$\mathbf{Z}_{\text{visual}} = \mathbf{X}_{\text{temp}}^{\top_{(1, 2)}} \in \mathbb{R}^{B \times 12 \times 256}$$

---

## 2. Temporal Validity Mask Downsampling

When sequences contain missing or zero-padded frames at the input ($M \in \{0, 1\}^{B \times 1 \times T \times V}$), the temporal validity mask must be projected to match the downsampled sequence length $T'=12$:
1. A frame $t \in [1, 45]$ is marked valid if any landmark has visibility $\ge 0.5$.
2. The downsampled mask $M_{\text{down}} \in \{0, 1\}^{B \times 12}$ is computed by max-pooling or sampling the input mask across temporal stride receptive fields.
3. For Transformer attention, this is converted to a PyTorch `src_key_padding_mask` of type `torch.bool`, where `True` indicates padded/ignored frames.

---

## 3. Projection to Transformer Latent Space

To ensure flexibility in Transformer model sizing, visual features are passed through a dedicated `FeatureProjection` layer:
$$\mathbf{H}_{\text{proj}} = \text{Dropout}(\text{LayerNorm}(\mathbf{W}_p \mathbf{Z}_{\text{visual}} + \mathbf{b}_p)) \in \mathbb{R}^{B \times 12 \times D_{\text{model}}}$$
where $\mathbf{W}_p \in \mathbb{R}^{D_{\text{model}} \times 256}$. This allows decoupled tuning of ST-GCN channels ($C_{\text{final}} = 256$) and Transformer hidden dimension ($D_{\text{model}} \in \{128, 256\}$).
