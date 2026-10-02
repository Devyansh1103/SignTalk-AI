# Data Augmentation Strategy: SignTalk AI

**Document ID:** STAI-P2P2-004  
**Project:** SignTalk AI  
**Phase:** Phase 2 — Part 2  
**Status:** Approved  

---

## 1. Augmentation Philosophy & Linguistic Constraints

In sign language recognition and translation, arbitrary visual augmentations (such as aggressive shear, extreme rotation, or indiscriminate temporal reversal) are catastrophic:
- **Spatial Orientation Sensitivity:** Flipping a sign upside down destroys the linguistic meaning of signs (e.g. distinguishing `fall` from `rise` or `sun` from `ground`).
- **Temporal Directionality:** Reversing a temporal sequence inverts the syntactic movement trajectory.
- **Fingerspelling & Articulator Precision:** Finger joint positions define distinct lexical units; adding excessive Gaussian noise disrupts fine motor distinction between letters and numbers.

Therefore, SignTalk AI defines a strictly constrained **Linguistically-Preserving Augmentation Policy**.

---

## 2. Training-Time Augmentation Policy (Phase 3)

> [!IMPORTANT]
> **No Offline Augmentation During Phase 2:**
> In accordance with academic reproducibility and data integrity standards, the processed dataset produced in Phase 2 Part 2 contains **zero synthetic duplicates or offline augmentations**. All augmentation transformations are designated strictly as **Training-Time On-The-Fly Augmentations** inside the PyTorch `Dataset` during model training in Phase 3.
> 
> **Zero Leakage:** Augmentation is strictly prohibited on Validation and Test partitions.

---

## 3. Approved Training-Time Transformations

| Augmentation Technique | Parameter Bounds | Probability ($p$) | Linguistic Rationale |
| :--- | :--- | :---: | :--- |
| **Bilateral Mirroring** | Horizontal flip + node swap | $p = 0.5$ | Enables hand-dominance invariance (left vs right-handed signers). |
| **Spatial Coordinate Jitter** | Gaussian noise $\mathcal{N}(0, \sigma^2)$, $\sigma \le 0.01$ | $p = 0.3$ | Simulates webcam sensor noise and MediaPipe detection micro-jitter. |
| **Subtle 2D In-Plane Rotation** | Angle $\theta \in [-8^\circ, +8^\circ]$ | $p = 0.3$ | Simulates slight head tilt or slight off-axis webcam mounting angles. |
| **Global Scale Variation** | Uniform scale $s \in [0.92, 1.08]$ | $p = 0.4$ | Simulates slight residual body-depth shifts not removed by normalization. |
| **Temporal Speed Perturbation** | Resampling factor $\alpha \in [0.85, 1.15]$ | $p = 0.3$ | Simulates natural signer cadence variations (fast vs deliberate signing). |
| **Random Joint Dropping** | 1 to 3 non-shoulder nodes zero-masked | $p = 0.2$ | Trains ST-GCN graph attention to be robust against momentary finger occlusion. |

---

## 4. Forbidden Transformations

- **Temporal Time Reversal:** Strictly forbidden. Reversing time destroys gesture semantics.
- **Vertical Inversion ($y$-flip):** Strictly forbidden. Changes the physical direction of signs.
- **Excessive Scaling ($> 25\%$):** Strictly forbidden. Distorts relative distances between hands and head anchors.
- **Validation/Test Augmentation:** Strictly forbidden. Evaluates real unaltered human signing.
