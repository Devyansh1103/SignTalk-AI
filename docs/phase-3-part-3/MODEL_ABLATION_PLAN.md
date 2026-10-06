# SignTalk AI — Model Ablation Plan (Phase 3 Part 3)

**Document ID:** `DOC-P3P3-ABLATION-PLAN-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** COMPLETE & DOCUMENTED  

---

## 1. Objective

This ablation plan defines systematic comparative experiments to evaluate the isolated contributions of architectural components across the visual-language boundary.

---

## 2. Planned Ablation Configurations

### Ablation 1: Architecture Comparison (Baseline vs ST-GCN vs ST-GCN + Transformer)
- **Model A (Baseline):** Bidirectional GRU with temporal frame attention (Phase 3 Part 1).
- **Model B (ST-GCN Alone):** 6-block partitioned ST-GCN with spatial-temporal average pooling and linear classification head (Phase 3 Part 2).
- **Model C (ST-GCN + Transformer):** Pretrained ST-GCN visual encoder + FeatureProjection + SignLanguageTransformer autoregressive decoder (This phase).
- **Objective:** Measure whether adding the Transformer cross-attention decoder maintains or improves sign recognition accuracy while establishing autoregressive translation capabilities.

### Ablation 2: Visual Encoder Freezing (Frozen vs Joint Fine-Tuning)
- **Configuration 1 (Frozen):** ST-GCN initialized from `best_checkpoint.pt` with $\nabla_{\theta_{\text{GCN}}} = 0$. Only `FeatureProjection` and `Transformer` are trained.
- **Configuration 2 (Joint Fine-Tuning):** Full end-to-end backpropagation with visual learning rate $\eta_{\text{GCN}} = 10^{-5}$ and language learning rate $\eta_{\text{TF}} = 10^{-3}$.
- **Objective:** Determine if joint gradients refine spatial graph representations or induce catastrophic overfitting on small sequence corpora.

### Ablation 3: Decoder Layer Depth and Positional Encodings
- **Depth Ablation:** Compare 1-layer vs 2-layer vs 4-layer Transformer decoder.
- **Positional Encoding Ablation:** Sinusoidal analytical encoding vs learned embedding table.
- **Objective:** Identify the optimal parameter-efficiency frontier for low-latency CPU and edge deployment.
