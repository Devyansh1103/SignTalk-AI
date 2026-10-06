# SignTalk AI — Model Size & Parameter Report

**Document ID:** `DOC-P3P1-MODEL-SIZE-001`  
**Phase:** Phase 3 — Part 1 (Baseline Model & Training Infrastructure)  
**Model Name:** `SignBaselineModel` (Bidirectional GRU with Linear Spatial Projection)  
**Code Reference:** [`src/models/baseline.py`](file:///d:/SignAI/src/models/baseline.py)  
**Configuration Reference:** [`configs/baseline.yaml`](file:///d:/SignAI/configs/baseline.yaml)  

---

## 1. Executive Summary

This report documents the architectural parameters, parameter counts, memory footprints, and disk checkpoint footprints for the Phase 3 Part 1 baseline model (`SignBaselineModel`). The baseline model is designed as a computationally lightweight temporal sequence baseline that operates directly on the standardized sequence representation $[B, 3, 45, 93]$ without spatial graph convolutions.

| Metric | Measured Value | Unit / Format |
| :--- | :--- | :--- |
| **Total Parameters** | **597,898** | Parameters |
| **Trainable Parameters** | **597,898** | Parameters (100.0%) |
| **Non-Trainable Parameters** | **0** | Parameters (0.0%) |
| **Raw Model Weights (fp32)** | **2.28** | MB ($2,391,592$ bytes) |
| **Checkpoint Size (`best_checkpoint.pt`)** | **7.21** | MB ($7,209,887$ bytes) |
| **Checkpoint Size (`latest_checkpoint.pt`)** | **7.21** | MB ($7,210,107$ bytes) |
| **Theoretical FLOPs per Sequence ($T=45$)** | **~53.8** | MFLOPs |

---

## 2. Layer-by-Layer Parameter Breakdown

The baseline model architecture consists of three principal submodules:
1. **Spatial Projection Submodule:** Linear projection mapping flattened spatial coordinate frames ($3 \times 93 = 279$) to a latent feature dimension ($128$).
2. **Temporal Modeling Submodule:** A 2-layer Bidirectional Gated Recurrent Unit (BiGRU) operating across the temporal length ($T=45$).
3. **Classification Head Submodule:** Dual temporal pooling (concatenated temporal mean and max pooling yielding $2 \times 256 = 512$ features), followed by LayerNorm, Dropout, intermediate projection, and logits projection to $C=10$ classes.

```
====================================================================================================
Layer (type:depth-idx)                   Output Shape              Param #     Requires Grad
====================================================================================================
SignBaselineModel                        [B, 10]                   --          --
├─ Linear: 1-1 (spatial_proj)            [B, 45, 128]              35,840      True
├─ LayerNorm: 1-2 (spatial_norm)         [B, 45, 128]              256         True
├─ Dropout: 1-3 (spatial_drop)           [B, 45, 128]              0           --
├─ GRU: 1-4 (gru)                        [B, 45, 256]              --          --
│    ├─ Layer 1 (Bidirectional GRU)      [B, 45, 256]              198,144     True
│    └─ Layer 2 (Bidirectional GRU)      [B, 45, 256]              296,448     True
├─ TemporalPooling: 1-5 (mean_max)       [B, 512]                  0           --
├─ Sequential: 1-6 (classifier)          [B, 10]                   --          --
│    ├─ Linear: 2-1                      [B, 128]                  65,664      True
│    ├─ LayerNorm: 2-2                   [B, 128]                  256         True
│    ├─ ReLU: 2-3                        [B, 128]                  0           --
│    ├─ Dropout: 2-4                     [B, 128]                  0           --
│    └─ Linear: 2-5                      [B, 10]                   1,290       True
====================================================================================================
Total params: 597,898
Trainable params: 597,898
Non-trainable params: 0
Total mult-adds (MFLOPs): ~53.8
====================================================================================================
```

### Detailed Parameter Calculation

1. **Spatial Projection Submodule:**
   - Weight matrix: $W \in \mathbb{R}^{128 \times 279} \implies 128 \times 279 = 35,712$ parameters.
   - Bias vector: $b \in \mathbb{R}^{128} \implies 128$ parameters.
   - LayerNorm: $\gamma \in \mathbb{R}^{128}, \beta \in \mathbb{R}^{128} \implies 256$ parameters.
   - **Subtotal:** $35,840 + 256 = 36,096$ parameters.

2. **Bidirectional GRU Submodule:**
   - **Layer 1** ($D_{\text{in}} = 128, D_{\text{out}} = 128$, Bidirectional):
     - Forward GRU: $3 \times (128 \times 128 + 128 \times 128 + 2 \times 128) = 99,072$ parameters.
     - Reverse GRU: $3 \times (128 \times 128 + 128 \times 128 + 2 \times 128) = 99,072$ parameters.
     - Layer 1 Total: $198,144$ parameters.
   - **Layer 2** ($D_{\text{in}} = 256, D_{\text{out}} = 128$, Bidirectional):
     - Forward GRU: $3 \times (256 \times 128 + 128 \times 128 + 2 \times 128) = 148,224$ parameters.
     - Reverse GRU: $3 \times (256 \times 128 + 128 \times 128 + 2 \times 128) = 148,224$ parameters.
     - Layer 2 Total: $296,448$ parameters.
   - **Subtotal:** $198,144 + 296,448 = 494,592$ parameters.

3. **Classifier Head Submodule:**
   - Temporal Pooling (Mean + Max): Feature dimension $D = 256 \times 2 = 512$. Parameter count: $0$.
   - Intermediate Linear: $W \in \mathbb{R}^{128 \times 512} \implies 65,536$ params; bias $\implies 128$ params. Total: $65,664$.
   - LayerNorm: $\gamma, \beta \in \mathbb{R}^{128} \implies 256$ parameters.
   - Final Logits Linear: $W \in \mathbb{R}^{10 \times 128} \implies 1,280$ params; bias $\implies 10$ params. Total: $1,290$.
   - **Subtotal:** $65,664 + 256 + 1,290 = 67,210$ parameters.

**Grand Total:** $36,096 + 494,592 + 67,210 = 597,898$ parameters.

---

## 3. Checkpoint Footprint Analysis

PyTorch `.pt` checkpoints contain not only model parameters but also complete optimizer states and metadata for resuming training:

```
Checkpoint File: experiments/baseline/checkpoints/best_checkpoint.pt
File Size: 7,209,887 bytes (7.21 MB)
Contents:
├── "epoch": int (Epoch index, e.g. 25)
├── "model_state_dict": 597,898 fp32 weights (~2.28 MB)
├── "optimizer_state_dict":
│   ├── "state": AdamW first moment (m) and second moment (v) for 597,898 weights (~4.56 MB)
│   └── "param_groups": Hyperparameters, learning rate, weight decay
├── "scheduler_state_dict": CosineAnnealingLR step counter
├── "val_loss": 1.7766
├── "val_macro_f1": 0.2667
├── "config": Exact configuration dictionary
└── "history": Full training trajectory across all epochs
```

---

## 4. Memory Footprint During Execution

Memory usage was measured on the reference test system:
- **Model Parameters (fp32):** $2.28$ MB
- **Forward Activation Memory (Batch Size = 1):** $< 1.5$ MB
- **Forward Activation Memory (Batch Size = 8):** $< 8.2$ MB
- **Peak Process RSS Memory (Python + PyTorch Runtime):** $184.2$ MB

---

## 5. Architectural Implications for ST-GCN

1. **Parameter Efficiency:** The baseline uses $597\text{K}$ parameters primarily inside the recurrent transition matrices ($494\text{K}$ params, $82.7\%$). In contrast, a standard 9-layer ST-GCN typically employs between $1.2\text{M}$ and $3.1\text{M}$ parameters distributed across spatial graph convolutions and temporal 1D convolutions.
2. **Computational Locality:** The baseline treats all 93 landmarks as a concatenated 279-element 1D vector at each time step, losing all explicit anatomical connectivity. ST-GCN will constrain spatial message passing to true skeletal graph edges, providing far higher parameter efficiency per edge relationship.
