# SignTalk AI — Phase 4 Part 1 Handoff Audit

**Document ID:** `DOC-P4P1-AUDIT-001`  
**Phase:** Phase 4 — Part 1 (Real-Time Camera & Landmark Streaming)  
**System Title:** SignTalk AI: Real-Time Indian Sign Language Translation Platform  
**Technical Subtitle:** A Spatial-Temporal Graph and Transformer-Based Sign-to-Text System  
**Audit Date:** October 2026  
**Status:** COMPLETED & VERIFIED AGAINST TRAINED ARTIFACTS  

---

## 1. Executive Summary & Objective

This audit establishes the empirical ground truth for **Phase 4 Part 1** by inspecting:
1. `docs/PHASE_4_HANDOFF.md`
2. `docs/PHASE_3_FINAL_EVALUATION_REPORT.md`
3. `docs/MODEL_SELECTION_REPORT.md`
4. `models/model_registry.yaml`
5. `configs/final_model.yaml`
6. `configs/preprocessing.yaml`
7. Checkpoint: `experiments/stgcn/checkpoints/best_checkpoint.pt`

The real-time streaming pipeline constructed in Phase 4 Part 1 must faithfully produce the exact landmark schema, node ordering, coordinate space, normalization convention, and temporal contract expected by the selected final model without discrepancy.

---

## 2. Selected Model Specification

| Parameter | Empirically Audited Specification | Source Artifact |
| :--- | :--- | :--- |
| **Model Name** | `SignTalk_STGCN_v1` (Model ID: `signtalk_stgcn_v1`) | `models/model_registry.yaml` |
| **Version** | `1.0.0` | `models/model_registry.yaml` |
| **Architecture** | Spatial-Temporal Graph Convolutional Network (ST-GCN) | `configs/final_model.yaml` |
| **Checkpoint Path** | `experiments/stgcn/checkpoints/best_checkpoint.pt` | Verified on disk (`epoch=34`, 116 weights) |
| **Checkpoint Disk Size** | `24.68 MB` (25,876,189 bytes) | Verified on disk |
| **Total Parameters** | `2,137,818` | `models/model_registry.yaml` |
| **Vocabulary Size ($K$)** | 10 canonical sign classes | `assets/vocabularies/mvp_10.json` |
| **Classes** | `hello` (0), `thankyou` (1), `good` (2), `happy` (3), `monday` (4), `car` (5), `bird` (6), `house` (7), `time` (8), `teacher` (9) | `configs/final_model.yaml` |
| **Task Definition** | Isolated Indian Sign Language Classification & Gloss Alignment | `docs/EVALUATION_TASK_DEFINITION.md` |

---

## 3. Input Tensor & Data Representation Contract

The downstream ST-GCN model inference requires the following tensor specification:

```text
Input Tensor:
  Name:                  "landmark_sequence"
  Shape:                 [B, C, T, V] = [1, 3, 45, 93]
  Data Type:             torch.float32 (NumPy float32 during extraction)

Channels (C = 3):
  Channel 0 (Index 0):   X coordinate (normalized 0.0 to 1.0, root-centered & scaled)
  Channel 1 (Index 1):   Y coordinate (normalized 0.0 to 1.0, root-centered & scaled)
  Channel 2 (Index 2):   Z coordinate (relative depth, root-centered & scaled)

Temporal Window (T = 45):
  Frames per Window:     45 frames
  Target Frame Rate:     25.0 FPS (corresponds to 1.8 seconds temporal span)
  Streaming Strategy:    Sliding temporal buffer with linear interpolation fallback

Graph Nodes (V = 93):
  Total Nodes:           93 multimodal skeletal nodes
  Partitioning:
    ├── Nodes 00 - 20:   Left Hand (21 phalanx keypoints)
    ├── Nodes 21 - 41:   Right Hand (21 phalanx keypoints)
    ├── Nodes 42 - 52:   Upper Pose (11 torso/head anatomical anchors)
    └── Nodes 53 - 92:   Face Contours (40 salient eyebrow, mouth, and jawline points)

Optional Validity Mask:
  Name:                  "visibility_mask"
  Shape:                 [B, 1, T, V] = [1, 1, 45, 93]
  Data Type:             torch.float32 (1.0 = observed/imputed joint, 0.0 = missing)
```

---

## 4. Graph Node Schema & Modality Mapping

The canonical 93-node index mapping established in Phase 2 Part 2 and maintained throughout Phase 3:

### 4.1 Left Hand (Nodes 0 – 20)
* `0`: Left Wrist
* `1`–`4`: Thumb (CMC, MCP, IP, TIP)
* `5`–`8`: Index (MCP, PIP, DIP, TIP)
* `9`–`12`: Middle (MCP, PIP, DIP, TIP)
* `13`–`16`: Ring (MCP, PIP, DIP, TIP)
* `17`–`20`: Pinky (MCP, PIP, DIP, TIP)

### 4.2 Right Hand (Nodes 21 – 41)
* `21`: Right Wrist
* `22`–`25`: Thumb (CMC, MCP, IP, TIP)
* `26`–`29`: Index (MCP, PIP, DIP, TIP)
* `30`–`33`: Middle (MCP, PIP, DIP, TIP)
* `34`–`37`: Ring (MCP, PIP, DIP, TIP)
* `38`–`41`: Pinky (MCP, PIP, DIP, TIP)

### 4.3 Upper Pose Anchors (Nodes 42 – 52)
Extracted from MediaPipe Pose 33-landmark model using `POSE_KEYPOINT_MAP`:
* `42`: Nose (MediaPipe Pose 0)
* `43`: Left Eye (MediaPipe Pose 2)
* `44`: Right Eye (MediaPipe Pose 5)
* `45`: Left Ear (MediaPipe Pose 7)
* `46`: Right Ear (MediaPipe Pose 8)
* `47`: Left Shoulder (MediaPipe Pose 11) — **Key Torso Anchor**
* `48`: Right Shoulder (MediaPipe Pose 12) — **Key Torso Anchor**
* `49`: Left Elbow (MediaPipe Pose 13)
* `50`: Right Elbow (MediaPipe Pose 14)
* `51`: Left Wrist (MediaPipe Pose 15) — **Pose-Hand Kinematic Bridge**
* `52`: Right Wrist (MediaPipe Pose 16) — **Pose-Hand Kinematic Bridge**

### 4.4 Facial Non-Manual Contours (Nodes 53 – 92)
Extracted from MediaPipe Face Mesh using `FACE_KEYPOINT_MAP`:
* `53`–`56`: Left Eyebrow contour (4 points: [70, 63, 105, 66])
* `57`–`60`: Right Eyebrow contour (4 points: [300, 293, 334, 296])
* `61`–`68`: Outer Lip contour (8 points: [61, 146, 91, 181, 84, 17, 314, 405])
* `69`–`76`: Inner Lip contour (8 points: [78, 95, 88, 178, 87, 14, 317, 402])
* `77`–`92`: Mandibular & Jawline contour (16 points: [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323])

---

## 5. Landmark Normalization Procedure

The model was trained exclusively using the **`torso_scale`** normalization method defined in `src/preprocessing/normalizer.py`:

1. **Torso Center ($C_{\text{torso}}$):**
   $$C_{\text{torso}} = \frac{P_{\text{left\_shoulder}} + P_{\text{right\_shoulder}}}{2} = \frac{\text{coords}[47] + \text{coords}[48]}{2}$$
   *Fallback:* If only one shoulder is visible, that shoulder serves as center. If neither shoulder is visible, the centroid of valid upper pose landmarks is used; if no pose landmarks are visible, default to $(0.5, 0.5, 0.0)$.

2. **Scale Factor ($S$):**
   $$S = \max\left(\|P_{\text{left\_shoulder}} - P_{\text{right\_shoulder}}\|_2, \; \epsilon\right) \quad (\epsilon = 1.0\times 10^{-4})$$
   *Fallback:* If shoulders are not mutually visible, fallback to default scale $S = 0.25$.

3. **Normalized Coordinates ($P'_i$ for $i \in [0, 92]$):**
   $$x'_i = \frac{x_i - C_{\text{torso}, x}}{S}$$
   $$y'_i = \frac{y_i - C_{\text{torso}, y}}{S}$$
   $$z'_i = \frac{z_i - C_{\text{torso}, z}}{S} \times z_{\text{scale\_factor}} \quad (z_{\text{scale\_factor}} = 1.0)$$

4. **Missing Landmarks:**
   Unobserved joints ($\text{mask}[i] = \text{False}$) are zeroed out:
   $$P'_i = (0.0, 0.0, 0.0)$$

---

## 6. MediaPipe Detection & Confidence Thresholds

In accordance with `configs/preprocessing.yaml`:
* **Pose Detection:**
  * Model: `models/mediapipe/pose_landmarker_full.task`
  * Detection confidence: $\ge 0.50$
  * Presence confidence: $\ge 0.50$
  * Tracking confidence: $\ge 0.50$
* **Hand Detection:**
  * Model: `models/mediapipe/hand_landmarker.task`
  * Detection confidence: $\ge 0.35$
  * Presence confidence: $\ge 0.35$
  * Tracking confidence: $\ge 0.35$
  * Max hands: 2
* **Face Detection:**
  * Model: `models/mediapipe/face_landmarker.task`
  * Detection confidence: $\ge 0.40$
  * Presence confidence: $\ge 0.40$
  * Max faces: 1
  * Facial mesh toggle: `enable_face_mesh` configurable; fallback upper-facial pose anchors (eyes, nose, ears) provide robust non-manual tracking if mesh is disabled.

---

## 7. Quality Gate & Acceptance Criteria

Quality criteria established in Phase 3:
1. **Frame-Level Quality ($Q_{\text{frame}} \in [0.0, 1.0]$):**
   $$Q = 0.40 \cdot C_{\text{pose}} + 0.40 \cdot C_{\text{dominant\_hand}} + 0.15 \cdot C_{\text{non\_dominant\_hand}} + 0.05 \cdot C_{\text{face}}$$
2. **Quality Categories:**
   * $\ge 0.75$: `GOOD`
   * $\ge 0.50$: `ACCEPTABLE`
   * $\ge 0.35$: `REVIEW`
   * $< 0.35$: `REJECT`
3. **Sequence Gate:**
   $$\text{valid\_frames} \ge 15 \quad \text{and} \quad (\text{left\_hand\_detected} \lor \text{right\_hand\_detected})$$

---

## 8. Known Model Limitations & Engineering Mitigations

1. **Temporal Discontinuity Sensitivity:**
   * *Ablation Finding:* Dropping 25% of temporal frames degraded ST-GCN accuracy from 83.3% to 0.0%.
   * *Phase 4 Mitigation:* Real-time camera capture must enforce a ring buffer that never emits sequences with temporal gaps. If frames drop, linear interpolation bridges up to 10 missing frames.
2. **Hand Swapping / Chirality Inversion:**
   * *Ablation Finding:* Mirroring hands inverted accuracy to 16.7%.
   * *Phase 4 Mitigation:* Strict left/right hand assignment mapping based on MediaPipe's handedness classification score with identity tracking across frames.
3. **Real-Time Latency Ceiling:**
   * Single frame landmark extraction on CPU takes ~25–45 ms across Pose + Hand models.
   * Frame capture + landmark extraction + normalization + validation must complete within $\le 85\text{ ms}$ to maintain streaming responsiveness.

---

## 9. Conclusion

All specifications are locked. The Phase 4 Part 1 implementation will directly conform to these verified constraints.
