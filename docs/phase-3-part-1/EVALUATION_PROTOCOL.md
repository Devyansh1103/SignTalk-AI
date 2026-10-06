# Evaluation Protocol Specification: SignTalk AI

**Document ID:** STAI-P3P1-010  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 3 — Part 1 (Baseline Model & Training Infrastructure)  
**Author:** Lead Research Engineer & Safety Auditor  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Partition Roles and Boundary Enforcement

To maintain strict scientific integrity and eliminate subtle data leakage, the roles of the three dataset partitions are immutably defined:

```
[TRAIN PARTITION: 36 Samples]
  └── Compute forward pass, calculate CrossEntropyLoss, compute gradients, update weights.
  └── Optional training-time stochastic data augmentation (jitter, scaling, planar rotation).

[VALIDATION PARTITION: 12 Samples]
  └── Ingested every epoch under torch.no_grad() without data augmentation.
  └── Evaluates loss and macro F1; triggers early stopping and best-checkpoint serialization.
  └── Informs all hyperparameter choices and architectural decisions.

[TEST PARTITION: 12 Samples]
  └── STRICTLY ISOLATED during all training epochs and model selection cycles.
  └── Evaluated EXACTLY ONCE on the finalized 'best_checkpoint.pt'.
  └── NEVER used for model selection, early stopping, or tuning.
```

---

## 2. Test Set Evaluation Trigger Conditions

The test set may be unblinded and evaluated ONLY when ALL the following conditions are met:
1. Model training has concluded cleanly (either through epoch completion or early stopping).
2. The `best_checkpoint.pt` has been restored based strictly on peak validation macro F1.
3. No further hyperparameter changes, learning rate adjustments, or architectural edits are permitted for that experiment run.
4. All test metrics are recorded directly into `data/manifests/` and experimental logs without retrospective cherry-picking.
