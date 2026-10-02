# Handedness Strategy: SignTalk AI

**Document ID:** STAI-P2P2-003  
**Project:** SignTalk AI  
**Phase:** Phase 2 — Part 2  
**Status:** Approved  

---

## 1. Handedness in Indian Sign Language (ISL)

In sign languages, including ISL, signers possess a dominant signing hand (typically the right hand for right-handed signers and left hand for left-handed signers). 
- In **one-handed signs** (e.g. `hello`, `thankyou`, `girl`), only the dominant hand performs lexical articulation, while the non-dominant hand remains resting or lowered outside the camera field.
- In **two-handed asymmetric signs** (e.g. `pen`, `time`), the non-dominant hand acts as a passive base (the paper or watch surface) while the dominant hand performs dynamic action.
- In **two-handed symmetric signs** (e.g. `car`, `house`), both hands perform mirror-symmetric movement trajectories.

---

## 2. Anatomical Subgraph Representation

SignTalk AI explicitly separates left and right hands into distinct graph node partitions:
- **Left Hand Subgraph:** Nodes $0$ through $20$.
- **Right Hand Subgraph:** Nodes $21$ through $41$.

Unlike naive models that blindly collapse or arbitrarily assign whichever hand is visible to a generic "hand" channel, SignTalk AI preserves anatomical handedness:
1. **Kinematic Wrist Bridge:** Node $0$ connects to Left Wrist Pose Node $51$; Node $21$ connects to Right Wrist Pose Node $52$.
2. **Missing Hand Representation:** If the non-dominant hand is lowered ($c_{vis} < 0.35$), nodes $0-20$ are zero-masked and their visibility flags set to `False`. The ST-GCN adjacency and attention layers learn that an inactive node represents a dormant hand, rather than missing information.

---

## 3. Mathematical Mirroring Transformation (Handedness Invariance)

To allow a model trained predominantly on right-handed signers to recognize signs performed by left-handed signers (or mirrored webcam streams), SignTalk AI defines a strict **Bilateral Inversion Transform** $\mathcal{M}(\mathbf{X})$:

### 3.1 Horizontal Reflection
$$x'_{t, i} = -x_{t, i}, \quad \forall t \in \{1, \dots, T\}, \forall i \in \{0, \dots, 92\}$$

### 3.2 Articulator Node Swapping
$$\mathbf{X}'[0:21] \longleftrightarrow \mathbf{X}'[21:42]$$
$$\mathbf{M}_{mask}'[0:21] \longleftrightarrow \mathbf{M}_{mask}'[21:42]$$

### 3.3 Bilateral Pose Anchors Swapping
The bilateral pose pairs must also be transposed across the sagittal symmetry plane:
- Eyes: Node $43 \longleftrightarrow 44$
- Ears: Node $45 \longleftrightarrow 46$
- Shoulders: Node $47 \longleftrightarrow 48$
- Elbows: Node $49 \longleftrightarrow 50$
- Pose Wrists: Node $51 \longleftrightarrow 52$

```python
# Implemented in CoordinateNormalizer.mirror_horizontal
mirrored_coords, mirrored_mask = CoordinateNormalizer.mirror_horizontal(coords, mask)
```

---

## 4. Policy on Transformation Application

- **During Preprocessing:** The dataset is saved in its authentic observed orientation. No destructive or permanent swapping is applied to raw or interim landmark archives.
- **During Model Training (Phase 3):** Bilateral mirroring is applied probabilistically ($p = 0.5$) strictly on **Training Split** samples as an approved spatial augmentation.
- **During Validation & Test:** Mirroring is **strictly prohibited** to preserve genuine real-world test evaluation.
