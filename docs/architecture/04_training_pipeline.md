# 04. Training Pipeline & Evaluation Architecture: SignTalk AI

**Document ID:** STAI-ARCH-004  
**Project Name:** SignTalk AI  
**Document Status:** Approved Engineering Blueprint (Phase 1 — Part 2)  

---

## 1. End-to-End Training Lifecycle

The training architecture is structured to ensure complete reproducibility, prevent data leakage, and maximize generalization across unseen human signers.

```mermaid
graph TD
    subgraph DataPrep [Phase 1: Ingestion, Extraction & Quality Assurance]
        D1[Raw Video Datasets: INCLUDE-50 & ISL-CSLTR] --> D2[Checksum & Metadata Validation]
        D2 --> D3[Offline Batch MediaPipe Holistic Extraction]
        D3 --> D4[Keypoint Quality Gate: Drop frames with confidence < 0.4]
        D4 --> D5[Torso Normalization & Scaling]
        D5 --> D6[Parquet / HDF5 Skeletal Landmark Storage]
    end

    subgraph SplitProtocol [Phase 2: Signer-Independent Partitioning]
        D6 --> SP1[Signer Stratification by signer_id]
        SP1 --> TR_SET[Train Set: 70% ~ Signers S1-S5]
        SP1 --> VAL_SET[Validation Set: 15% ~ Signer S6]
        SP1 --> TE_SET[Test Set: 15% ~ Signers S7-S8]
    end

    subgraph Augment [Phase 3: Spatial-Temporal Coordinate Augmentation]
        TR_SET --> AG1[Temporal Resampling: Speed 0.85x to 1.15x]
        AG1 --> AG2[Spatial Rotation: +/- 10 deg in 2D Plane]
        AG2 --> AG3[Joint Jittering: Gaussian Noise N(0, 0.005)]
        AG3 --> AG4[Hand Mirroring: Synthetic Left/Right Flips]
    end

    subgraph ModelTrain [Phase 4: Two-Stage Model Training]
        AG4 --> STAGE1[Stage 1: Pretrain ST-GCN on Isolated INCLUDE-50]
        STAGE1 --> STAGE1_EVAL[Validate Top-1 Accuracy on Val Set]
        STAGE1_EVAL --> STAGE2[Stage 2: Joint Fine-Tuning with Transformer on ISL-CSLTR]
        VAL_SET -.->|Early Stopping Guidance| STAGE2
        STAGE2 --> LOSS[Loss: Cross-Entropy + Auxiliary CTC Loss]
    end

    subgraph Export [Phase 5: Evaluation & Artifact Packaging]
        STAGE2 --> TE_EVAL[Final Evaluation on Held-Out Test Signers]
        TE_EVAL --> METRICS[Compute BLEU 1-4, WER, Macro F1, Latency]
        TE_EVAL --> EXP_REG[Log Checkpoint & Metrics in Model Registry]
        STAGE2 --> ONNX_EXP[Export to ONNX / TorchScript FP16]
    end
```

---

## 2. Prevention of Signer Leakage: Signer-Independent Partitioning

A critical methodological flaw in amateur sign language machine learning projects is **random frame- or clip-level shuffling**. When multiple video clips of the same signer are randomly dispersed across training and testing partitions, the neural network memorizes individual signer clothing, skin tone, background lighting, and unique personal idiosyncrasies rather than learning generalized sign language kinematics.

To prevent signer leakage:
- All dataset samples are grouped strictly by `signer_id`.
- Partitions are divided along signer boundaries:
  - **Training Split ($70\%$):** Formed exclusively from Signers $S_1$ through $S_5$.
  - **Validation Split ($15\%$):** Formed exclusively from Signer $S_6$ (used for hyperparameter tuning and early stopping).
  - **Testing Split ($15\%$):** Formed exclusively from Signers $S_7$ and $S_8$ (never seen during training or tuning).
- Generalization is confirmed only when high accuracy and BLEU scores are achieved on these unseen test signers.

---

## 3. Spatial-Temporal Coordinate Augmentation Suite

To prevent overfitting on limited ISL training corpora, data augmentation is executed directly on the normalized $(x, y, z)$ skeletal landmark arrays:

| Augmentation Technique | Applied Mathematical Transformation | Linguistic Justification |
| :--- | :--- | :--- |
| **Temporal Resampling** | Linear spline interpolation stretching or compressing sequence length by factor $\alpha \in [0.85, 1.15]$. | Simulates signers executing identical signs at different speeds. |
| **Spatial 2D Rotation** | Coordinate rotation around the z-axis: $\mathbf{p}' = \mathbf{R}_z(\theta) \mathbf{p}$ for $\theta \sim \mathcal{U}(-10^\circ, +10^\circ)$. | Simulates variable camera tilts and non-level laptop placement. |
| **Random Scaling** | Uniform scaling: $\mathbf{p}' = s \cdot \mathbf{p}$ where $s \sim \mathcal{U}(0.9, 1.1)$. | Simulates residual distance variations after torso normalization. |
| **Joint Jittering** | Adding zero-mean Gaussian noise: $\mathbf{p}' = \mathbf{p} + \epsilon, \epsilon \sim \mathcal{N}(0, 0.005)$. | Simulates sensor noise and subtle finger trembling. |
| **Horizontal Mirroring** | Inverting x-coordinates and swapping left-hand and right-hand landmark node indices. | Generates synthetic data representing left-handed signers. |

---

## 4. Multi-Task Loss Formulation

During continuous sentence translation training, the network is optimized using a combined multi-task loss:

$$\mathcal{L}_{total} = \mathcal{L}_{CE}(\mathbf{Y}, \hat{\mathbf{Y}}) + \lambda_{CTC} \mathcal{L}_{CTC}(\mathbf{G}, \hat{\mathbf{G}}) + \lambda_{reg} \|\mathbf{\Theta}\|_2^2$$

Where:
- $\mathcal{L}_{CE}$: Cross-entropy loss with label smoothing ($\alpha = 0.1$) over natural-language English tokens.
- $\mathcal{L}_{CTC}$: Connectionist Temporal Classification loss applied to intermediate ST-GCN frame features to enforce monotonic alignment with word-level sign glosses $\mathbf{G}$.
- $\lambda_{CTC}$: Balancing weight ($\lambda_{CTC} = 0.3$).
- $\lambda_{reg}$: Weight decay factor ($\lambda_{reg} = 1 \times 10^{-4}$) to prevent overfitting.
