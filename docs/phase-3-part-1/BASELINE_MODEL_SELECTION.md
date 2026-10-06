# Baseline Model Selection Specification: SignTalk AI

**Document ID:** STAI-P3P1-003  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 3 — Part 1 (Baseline Model & Training Infrastructure)  
**Author:** Lead AI/ML Data Engineer & Lead Architect  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Candidate Architecture Evaluation

Before selecting a primary baseline, we evaluate six candidate architectures against operational criteria for sign language sequence classification:

| Candidate Model | Temporal Modeling | Parameter Efficiency | Small-Sample Convergence | Spatial Feature Retention | Implementation Complexity | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Standard MLP** | None (Flattens all frames) | Very Poor ($12,555$ input features) | Poor (Severe overfitting) | Poor (Destroys time axis) | Very Low | Rejected |
| **2. 1D Temporal CNN** | Local kernel receptive field | Moderate | Moderate | Moderate | Low | Viable alternative |
| **3. Standard LSTM** | Unidirectional recurrence | High ($4$ gate matrices) | Moderate | High (Step-by-step) | Moderate | Good, but heavier than GRU |
| **4. Bidirectional GRU (BiGRU)** | **Full bidirectional context** | **Optimal ($2$ gate matrices)** | **High (Fast convergence)** | **High (Frame embeddings)** | **Moderate** | **SELECTED PRIMARY** |
| **5. CNN + LSTM Hybrid** | Convolution + recurrence | High parameter count | Prone to overfitting | High | High | Over-engineered for baseline |
| **6. Dilated Temporal CNN (TCN)**| Causal dilated convolutions | Moderate | Requires receptive tuning | High | Moderate | Complex hyperparameter search |

---

## 2. Selected Primary Baseline: Bidirectional Gated Recurrent Unit (BiGRU)

### 2.1 Architectural Rationale
1. **Bidirectional Temporal Context:** Human sign language gestures consist of preparation, stroke nucleus, and retraction phases. A bidirectional model evaluates both past frame trajectory ($\vec{\mathbf{h}}_t$) and future movement intent ($\overleftarrow{\mathbf{h}}_t$) at every timestep.
2. **Parameter Economy & Gradient Stability:** Compared to LSTM (which uses 4 gates: input, forget, output, cell candidate), GRU utilizes only 2 gates (reset and update). On compact datasets ($60$ sequences), GRU exhibits substantially lower variance, converges in fewer epochs, and resists overfitting.
3. **Absence of Spatial Graph Prior:** Unlike ST-GCN (which enforces physical bone connectivity through adjacency message passing), the BiGRU baseline treats the 279 spatial coordinates as an unstructured feature vector. This isolates the exact empirical performance gain attributable to graph convolutions when ST-GCN is evaluated in Phase 3 Part 2.

---

## 3. Structural Architecture Specification

```
Input Tensor: X ∈ R^{B × 3 × 45 × 93}
       │
       ▼ [Reshape & Flatten Spatial Nodes: [B, 45, 279]]
       │
       ▼ [Linear Feature Projection: 279 ──► 128] + LayerNorm + ReLU + Dropout(0.2)
       │
       ▼ [2-Layer Bidirectional GRU: Hidden Size H = 128, Dropout = 0.3]
       │ Produces H_seq ∈ R^{B × 45 × 256}
       │
       ▼ [Dual Temporal Pooling: Mean-Pool(256) ⊕ Max-Pool(256)]
       │ Produces Representation h_pooled ∈ R^{B × 512}
       │
       ▼ [Classification Head: 512 ──► 128 ──► num_classes] + LayerNorm + Dropout(0.3)
       │
       ▼
Output Logits: z ∈ R^{B × 10}
```

### 3.1 Input and Output Formats
- **Input:** Batched tensor $\mathbf{X} \in \mathbb{R}^{B \times 3 \times 45 \times 93}$ or flattened $\mathbf{X}_{flat} \in \mathbb{R}^{B \times 45 \times 279}$.
- **Output:** Categorical logit vector $\mathbf{z} \in \mathbb{R}^{B \times 10}$ mapped to predicted class probabilities via Softmax.
