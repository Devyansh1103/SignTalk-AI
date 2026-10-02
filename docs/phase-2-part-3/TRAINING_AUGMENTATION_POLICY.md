# Training Augmentation Policy: SignTalk AI

**Document ID:** STAI-P2P3-024  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead ML Data Engineer & Research Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Policy Principles & Non-Negotiables

Data augmentation prevents overfitting in deep spatial-temporal models. However, in sign language processing, reckless spatial transformations alter lexical semantics (e.g. converting a positive gesture into an entirely different sign).

We establish the following non-negotiable augmentation rules:
1. **Training-Time Only Execution:** Augmentation is strictly applied dynamically inside the training `DataLoader`. No permanently augmented synthetic samples are serialized to disk in this phase.
2. **Zero Augmentation on Val/Test Splits:** Validation and test splits are strictly evaluated in their native, unaugmented form to reflect real-world performance accurately.
3. **Semantic Invariance:** All transformations must preserve the phonological components of Indian Sign Language (handshape, location, orientation, movement, and non-manual markers).

---

## 2. Permitted Landmark Augmentation Transformations

The following stochastic transformations are approved for training-time application:

| Transformation | Mathematical Definition | Parameter Range | Rationale |
| :--- | :--- | :---: | :--- |
| **Gaussian Spatial Jitter** | $\hat{\mathbf{p}}_{i,t} = \mathbf{p}_{i,t} + \mathcal{N}(0, \sigma^2)$ | $\sigma = 0.005$ units | Simulates MediaPipe joint localization noise and sensor jitter. |
| **Global Scale Variation** | $\hat{\mathbf{p}}_{i,t} = \alpha \cdot \mathbf{p}_{i,t}$ | $\alpha \in [0.95, 1.05]$ | Simulates variable camera distances and participant body proportions. |
| **Planar Rotation** | $\hat{\mathbf{p}}_{i,t} = \mathbf{R}_z(\theta) \cdot \mathbf{p}_{i,t}$ | $\theta \in [-5^\circ, +5^\circ]$ | Simulates slight head/body tilts without flipping sign orientation. |
| **Temporal Frame Dropping** | Uniformly drops up to $k$ frames; re-interpolates | $k \in [1, 4]$ frames | Simulates dropped webcam frames and temporal network packet loss. |
| **Temporal Speed Perturbation** | Resamples trajectory with speed factor $\gamma$ | $\gamma \in [0.90, 1.10]$ | Simulates fast vs. deliberate signing cadences. |

---

## 3. Strictly Prohibited Augmentations

1. **Uncontrolled Horizontal Mirroring:** In sign language, left/right hand assignment carries semantic weight. Arbitrary horizontal reflection without label verification converts right-handed signs into left-handed signs and alters directional verbs.
2. **Extreme Rotation ($> 15^\circ$):** Inverts gravity-relative gestures and corrupts vertical orientation cues.
3. **Independent Finger Scrambling:** Applying independent noise to finger joints destroys rigid bone constraints.
