# ADR-004: Selection of Spatial-Temporal Graph Convolutional Networks (ST-GCN)

**Status:** APPROVED  
**Date:** October 2026  
**Context:** Skeletal landmarks possess an intrinsic biological topology (joints connected by bones). Treating coordinates as unstructured vectors discards this spatial hierarchy, while applying 2D convolutions ignores physical connectivity.

## Decision
Adopt **Spatial-Temporal Graph Convolutional Networks (ST-GCN)** (Yan et al., AAAI 2018) as the primary kinematic feature encoder.

## Evaluated Alternatives
1. **Flattened LSTM / Bi-LSTM:** Simple to implement, but discards skeletal graph structure by treating 3D coordinates as a 1D vector, resulting in poor spatial generalization on low-resource datasets.
2. **Dense 3D Convolutions:** Ignores joint connectivity and requires high-dimensional voxel processing.
3. **Temporal Convolutional Networks (TCN):** Strong temporal modeling, but lacks explicit spatial graph convolution operations along biological bone edges.

## Consequences
- **Positive:** Strong inductive bias reflecting human anatomical constraints; reduces parameter count ($\approx 2.5\text{M}$ parameters); prevents catastrophic overfitting on small ISL datasets.
- **Negative:** Requires pre-defining and normalizing spatial adjacency matrices $\mathbf{A} \in \mathbb{R}^{N \times N}$ and managing spatial configuration partitioning.
