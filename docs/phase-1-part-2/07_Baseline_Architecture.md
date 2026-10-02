# 07. Baseline Model Architecture & Benchmarking: SignTalk AI

**Document ID:** STAI-P1P2-007  
**Project Name:** SignTalk AI  
**Document Status:** Approved Engineering Blueprint (Phase 1 — Part 2)  

---

## 1. Rationale for Baseline Models in Academic Research

In scholarly AI research, an advanced architecture (such as an ST-GCN or Transformer) cannot be asserted as effective without establishing a rigorous, empirically verified performance baseline on identical data splits.

SignTalk AI establishes two formal baseline models:
1. **Baseline Model 1 (Static Spatial Baseline):** Multi-Layer Perceptron (MLP) operating on flattened frame-averaged landmark vectors.
2. **Baseline Model 2 (Temporal Sequence Baseline):** Bidirectional Long Short-Term Memory (Bi-LSTM) network operating on flattened temporal coordinate sequences.

```mermaid
graph LR
    subgraph BaselineInput [Input Preprocessing]
        A[Sliding Window Tensor: T=45, N=93, C=3] --> B[Flatten Spatial Nodes: D_flat = 93 x 3 = 279]
        B --> C[Temporal Sequence Vector: x_t in R^45 x 279]
    end

    subgraph BiLSTM_Architecture [Baseline Model 2: Bi-LSTM Sequence Classifier]
        C --> D[Linear Projection Layer: 279 -> 256]
        D --> E[Bi-LSTM Layer 1: Hidden Dim 256, Dropout 0.3]
        E --> F[Bi-LSTM Layer 2: Hidden Dim 256, Dropout 0.3]
        F --> G[Global Temporal Average Pooling]
        G --> H[Fully Connected Head: 512 -> Num_Classes]
        H --> I[Softmax Classification Distribution]
    end
```

---

## 2. Baseline Architecture Specifications

### 2.1 Baseline Model 1: Multi-Layer Perceptron (MLP)
- **Input Dimension:** Flattened temporal mean vector $\bar{\mathbf{x}} = \frac{1}{T} \sum_{t=1}^T \mathbf{x}_t \in \mathbb{R}^{279}$.
- **Hidden Layers:**
  - Layer 1: Linear $(279 \rightarrow 512)$, BatchNorm1D, ReLU, Dropout ($p = 0.3$).
  - Layer 2: Linear $(512 \rightarrow 256)$, BatchNorm1D, ReLU, Dropout ($p = 0.3$).
  - Classification Head: Linear $(256 \rightarrow C)$, where $C = 50$ (MVP vocabulary).
- **Total Parameter Count:** $\approx 290,000$ parameters.

### 2.2 Baseline Model 2: Bidirectional LSTM (Bi-LSTM)
- **Input Dimension:** Temporal sequence of vectors $\mathbf{X} \in \mathbb{R}^{B \times 45 \times 279}$.
- **Network Layers:**
  - Linear Input Projection: $(279 \rightarrow 256)$ with LayerNorm and ReLU.
  - Bidirectional LSTM Stack: 2 layers, hidden size $d_h = 256$ ($512$ bidirectional outputs per time step), dropout $p = 0.3$.
  - Temporal Pooling: Mean-pooling over the 45 temporal steps to collapse into a single representation $\mathbf{h}_{seq} \in \mathbb{R}^{512}$.
  - Classification Head: Linear $(512 \rightarrow 256) \rightarrow \text{ReLU} \rightarrow \text{Linear} (256 \rightarrow C)$.
- **Total Parameter Count:** $\approx 2.45\text{M}$ parameters (comparable in parameter scale to the proposed ST-GCN, ensuring a fair architectural comparison).

---

## 3. Training & Evaluation Protocol for Baselines

| Hyperparameter | Baseline 1 (MLP) | Baseline 2 (Bi-LSTM) | Proposed ST-GCN (For Comparison) |
| :--- | :--- | :--- | :--- |
| **Input Representation** | Static Mean Vector $\mathbb{R}^{279}$ | Flattened Sequence $\mathbb{R}^{45 \times 279}$ | Kinematic Graph Tensor $\mathbb{R}^{3 \times 45 \times 93}$ |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW ($\beta_1=0.9, \beta_2=0.999$) |
| **Base Learning Rate** | $1 \times 10^{-3}$ | $5 \times 10^{-4}$ | $1 \times 10^{-3}$ (with Cosine Decay) |
| **Weight Decay** | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| **Batch Size** | 64 | 32 | 32 |
| **Epochs / Early Stop**| 50 (Patience = 10) | 60 (Patience = 12) | 60 (Patience = 12) |
| **Evaluation Metrics**| Top-1 Acc, Top-5 Acc, F1 | Top-1 Acc, Top-5 Acc, F1, Latency | Top-1 Acc, Top-5 Acc, F1, Latency |

### Scholarly Comparison Hypothesis:
The Bi-LSTM baseline will capture high-level temporal velocity but will struggle to distinguish fine-grained finger articulation and minimal lexical sign pairs because physical joint topology is flattened into an unstructured 1D vector. The ST-GCN is hypothesized to achieve statistically superior classification accuracy ($\ge 8-12\%$ gain on Top-1) due to its spatial kinematic inductive bias.
