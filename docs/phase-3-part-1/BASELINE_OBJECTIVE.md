# Baseline Objective Specification: SignTalk AI

**Document ID:** STAI-P3P1-002  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 3 — Part 1 (Baseline Model & Training Infrastructure)  
**Author:** Research Engineer & Lead Architect  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Why a Baseline is Scientifically Mandatory

In applied machine learning research, advancing directly to complex, specialized architectures (such as Spatial-Temporal Graph Convolutional Networks) without establishing a defensible baseline creates an ungrounded empirical evaluation. 

The baseline answers the fundamental scientific question:
> **"How well can the spatial-temporal landmark sequence representation be classified using standard sequence models without the spatial graph topology modeling of ST-GCN?"**

Without this reference point, any performance claims regarding ST-GCN are uninterpretable: one cannot determine whether accuracy gains arise from the preprocessed landmark representation itself or from the graph convolution operator.

---

## 2. Quantitative Reference Framework

The baseline establishes the empirical floor for:
1. **Classification Accuracy & Macro F1:** Establishing classification performance across the 10 vocabulary classes.
2. **Computational Complexity & Parameter Count:** Measuring baseline model parameter volume and memory overhead.
3. **Inference Latency:** Providing latency benchmarks against which ST-GCN edge inference will be evaluated.
4. **Error Patterns & Confusion Clusters:** Identifying which gesture classes share similar temporal trajectories and require fine-grained spatial finger-joint graph modeling.

---

## 3. Strict Research Constraints

1. **No Performance Conflation:** The baseline performance does NOT represent final SignTalk AI product capability.
2. **Identical Data Pipeline:** The baseline is trained on the exact same dataset (`signTalk-seq-v1.0.0`), partitions (36 train, 12 val, 12 test), and preprocessing normalizations as ST-GCN.
3. **Reusable Infrastructure:** All training, evaluation, logging, and checkpointing infrastructure developed in this phase must directly support ST-GCN in Phase 3 Part 2 without rewriting core training loops.
