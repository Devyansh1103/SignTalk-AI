# SignTalk AI — Multimodal Subsystem Ablation Plan

**Document ID:** `DOC-P3P2-ABLATION-MODALITY-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Schema Reference:** `93-node-v1`  
**Target Modalities:** Left Hand (21), Right Hand (21), Upper Pose (11), Face (40)  

---

## 1. Motivation and Research Question

In sign language linguistics, articulatory information is distributed across manual articulators (hands), bodily frame-of-reference (pose), and non-manual grammatical facial contours (face).

This ablation plan establishes a rigorous experimental matrix to answer:
> *"What is the marginal recognition performance contributed by each anatomical subsystem in Indian Sign Language sequence modeling?"*

---

## 2. Supported Subsystem Configurations

Using the standardized 93-node representation, four subsystem configurations are defined via node masking or subgraph extraction without altering the underlying dataset archives:

```
Subsystem Node Masking Definitions:
Configuration A: Hands Only (42 nodes)
  ├── Left Hand:  Nodes  0 - 20 (21 nodes)
  ├── Right Hand: Nodes 21 - 41 (21 nodes)
  └── Pose & Face: Zero-masked / detached

Configuration B: Hands + Upper Pose (53 nodes)
  ├── Hands:      Nodes  0 - 41 (42 nodes)
  ├── Upper Pose: Nodes 42 - 52 (11 nodes)
  └── Face:       Zero-masked / detached

Configuration C: Hands + Face (82 nodes)
  ├── Hands:      Nodes  0 - 41 (42 nodes)
  ├── Face:       Nodes 53 - 92 (40 nodes)
  └── Pose:       Zero-masked / detached

Configuration D: Full Multimodal Graph (93 nodes - Primary Model)
  └── All 93 nodes fully connected through anatomical and cross-system edges
```

---

## 3. Experimental Protocol

| Configuration | Active Nodes | Active Edges | Focus of Investigation |
| :--- | :---: | :---: | :--- |
| **A: Hands Only** | 42 | 46 | Measures isolated manual articulator discrimination without body reference. |
| **B: Hands + Pose** | 53 | 60 | Evaluates whether arm trajectory and shoulder orientation disambiguate spatial hand locations. |
| **C: Hands + Face** | 82 | 89 | Evaluates whether facial mouthing and head orientation compensate for missing arm posture. |
| **D: Full Multimodal** | 93 | 105 | Complete multimodal spatiotemporal representation. |

---

## 4. Evaluation Criteria & Guardrails

- All configurations share identical training configurations: AdamW, Cosine Annealing, seed 42, batch size 8.
- Metrics recorded: Top-1 Accuracy, Macro-F1, Weighted F1, Parameter Count, and Inference Latency.
- **Strict Guardrail:** No ablation numbers will be fabricated. Experiments will be scheduled systematically as GPU resources permit.
