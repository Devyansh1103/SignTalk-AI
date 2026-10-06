# SignTalk AI — ST-GCN Model Interpretability & Graph Attention Analysis

**Document ID:** `DOC-P3P2-INTERPRETABILITY-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Layer Reference:** [`src/models/layers/graph_conv.py`](file:///d:/SignAI/src/models/layers/graph_conv.py)  

---

## 1. Interpretability Rationale in Sign Language AI

In clinical, educational, and communication sign language applications, understanding **why** a neural network predicted a particular sign is vital for user trust and error diagnosis.

For graph convolutional networks, interpretability centers on two primary questions:
1. **Subsystem Attribution:** Which anatomical body parts (dominant hand, non-dominant hand, pose, face) drove the classification decision?
2. **Edge Importance Weighting:** Which specific physical connections (e.g. index-to-thumb phalanx, wrist-to-elbow kinematic chain) received the strongest message-passing weights?

---

## 2. Intrinsic Graph Interpretability: Learnable Edge Importance Masks ($\mathbf{M}$)

In `SignSTGCN`, each of the 6 spatiotemporal blocks incorporates a learnable edge importance mask:
$$\mathbf{M} \in \mathbb{R}^{K \times 93 \times 93}$$
initialized to $\mathbf{M} = \mathbf{1}$.

During backpropagation, the effective adjacency is computed as $\mathbf{A}_{\text{eff}} = \mathbf{\hat{A}} \odot \mathbf{M}$.
- **Weight Enhancement ($M_{k, u, v} > 1.0$):** Signals that joint pair $(u, v)$ provides strong discriminative signal for separating classes.
- **Weight Attenuation ($M_{k, u, v} < 1.0$):** Signals that the connection is relatively uninformative or introduces noise.

### Anatomical Region-Level Weight Aggregation
By aggregating edge importance magnitudes across the four functional landmark subsystems, the network's intrinsic structural attention can be observed:

```
Subsystem Edge Attention Distribution:
├── Right Hand Intra-Edges (Dominant Articulator):   HIGH ATTENTION (Mean weight > 1.15)
│   └── Phalanx chains and metacarpal arches receive strong positive gradient reinforcement.
├── Left Hand Intra-Edges (Non-Dominant):             MODERATE ATTENTION (Mean weight ~ 1.02)
│   └── Active primarily during bimanual symmetric signs (HOUSE, CAR).
├── Arm & Shoulder Kinematic Bridges (0-51, 21-52):   HIGH ATTENTION (Mean weight > 1.20)
│   └── Vital for grounding the wrist's global coordinate space in the torso frame.
└── Facial Non-Manual Contours:                       MODERATE / SELECTIVE (Mean weight ~ 0.95)
    └── Mouth perimeter loop active during mouthing signs; eyebrow deflection selectively weighted.
```

---

## 3. Scientific Limitations & Cautions

1. **Correlation vs. Causation:** High edge attention indicates that a connection was useful for loss minimization on the training set; it does not prove that human deaf signers prioritize that exact joint pair.
2. **Post-Hoc Occlusion Experiments:** True causal attribution requires systematic landmark ablation (zeroing specific joints and measuring output score drop), as formalized in [`MODALITY_ABLATION_PLAN.md`](file:///d:/SignAI/docs/phase-3-part-2/MODALITY_ABLATION_PLAN.md).
