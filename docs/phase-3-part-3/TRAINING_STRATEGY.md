# SignTalk AI — Two-Stage Training Strategy

**Document ID:** `DOC-P3P3-TRAINING-STRAT-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** COMPLETE & DOCUMENTED  

---

## 1. Overview of Paradigms

Training an end-to-end visual-to-language translation system can follow two distinct paradigms:

### APPROACH A: Two-Stage Decoupled Training (Frozen Visual Encoder)
1. **Stage 1 (Completed in Phase 3 Part 2):** Train the ST-GCN visual graph convolutional network to convergence on sign recognition tasks.
2. **Stage 2 (Current Phase):** Freeze all ST-GCN weights ($\nabla_{\theta_{\text{STGCN}}} \mathcal{L} = 0$). Train only the `FeatureProjection` and `SignLanguageTransformer` decoder layers.
- **Advantages:**
  - Isolates language layer dynamics and gradient behavior.
  - Prevents catastrophic forgetting of spatial-temporal graph representations.
  - Substantially faster training epochs and lower memory usage since visual encoder backpropagation is skipped.
  - Guarantees that language layer convergence is driven by the quality of the learned spatial-temporal visual embeddings.

### APPROACH B: End-to-End Joint Fine-Tuning
1. Initialize ST-GCN from the best validation checkpoint (`experiments/stgcn/checkpoints/best_checkpoint.pt`).
2. Train all parameters jointly with differential learning rates (e.g., $\text{lr}_{\text{STGCN}} = 10^{-5}$, $\text{lr}_{\text{Transformer}} = 10^{-3}$).
- **Trade-offs:** Can allow the visual encoder to adjust representations for downstream language cross-attention, but risks overfitting on small-scale datasets.

---

## 2. Decision and Protocol

We adopt **Approach A** as the primary training strategy for Phase 3 Part 3:
- ST-GCN weights are initialized from `experiments/stgcn/checkpoints/best_checkpoint.pt`.
- All ST-GCN parameters are frozen (`requires_grad = False`).
- Only `FeatureProjection` and `SignLanguageTransformer` are updated.
- An ablation comparison evaluating Approach B (joint fine-tuning) is outlined in the model ablation plan.
