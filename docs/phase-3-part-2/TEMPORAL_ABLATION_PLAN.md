# SignTalk AI — Temporal Convolution Ablation Plan

**Document ID:** `DOC-P3P2-ABLATION-TEMPORAL-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Layer Reference:** [`src/models/layers/temporal_conv.py`](file:///d:/SignAI/src/models/layers/temporal_conv.py)  

---

## 1. Motivation

Sign language gestures unfold continuously across time. While spatial graph convolutions capture intra-frame joint configurations, temporal convolutions capture movement dynamics, velocities, and rhythm.

This ablation plan establishes experiments to investigate:
1. The contribution of temporal modeling versus purely static spatial graph pooling.
2. The optimal temporal receptive field ($K_t$) for $25.0\text{ FPS}$ signing sequences.

---

## 2. Compared Configurations

```
Temporal Architecture Variants:
├── Variant 1: Spatial-Only Graph Model (K_t = 1)
│   └── Receptive field: 1 frame (0.04s). Eliminates cross-frame convolution;
│       aggregates temporal information purely via final global temporal pooling.
│
├── Variant 2: Narrow Temporal Window (K_t = 5)
│   └── Receptive field: 5 frames (~0.20s). Captures instantaneous velocity.
│
├── Variant 3: Standard Temporal Window (K_t = 9, Default)
│   └── Receptive field: 9 frames (~0.36s). Captures complete phonic transitions.
│
└── Variant 4: Broad Temporal Window (K_t = 13)
    └── Receptive field: 13 frames (~0.52s). Models extended gesture trajectories.
```

---

## 3. Evaluation Hypotheses

1. **Spatial-Only Collapse:** $K_t=1$ is expected to degrade sharply on dynamic trajectory signs (e.g. `MONDAY`, `HAPPY`, `TEACHER`) that require temporal ordering, while maintaining moderate performance on static shape signs (e.g. `HOUSE`).
2. **Receptive Field Saturation:** Receptive fields larger than $K_t=9$ are expected to offer diminishing accuracy returns while increasing parameter volume and CPU inference latency.
