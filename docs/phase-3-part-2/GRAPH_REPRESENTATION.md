# SignTalk AI — 93-Node Multimodal Skeletal Graph Representation

**Document ID:** `DOC-P3P2-GRAPH-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Schema Identifier:** `93-node-v1`  
**Canonical Landmark Reference:** [`docs/phase-2-part-2/LANDMARK_DATA_SCHEMA.md`](file:///d:/SignAI/docs/phase-2-part-2/LANDMARK_DATA_SCHEMA.md)  
**Implementation:** [`src/models/graph.py`](file:///d:/SignAI/src/models/graph.py)  

---

## 1. Graph Definition and Structural Dimensions

The skeletal representation of the signer is modeled as a bidirectional, connected spatiotemporal graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:

- **Node Set ($\mathcal{V}$):** $V = 93$ canonical anatomical landmarks.
- **Edge Set ($\mathcal{E}$):** Intra-frame spatial bones and physical connections connecting neighboring anatomical landmarks.
- **Spatial Adjacency ($\mathbf{A}$):** $\mathbf{A} \in \mathbb{R}^{K \times V \times V}$, where $K$ denotes the partitioning subsets (e.g. $K=1$ uniform, $K=3$ spatial configuration).

```
Landmark Subsystem Distribution (V = 93):
┌────────────────────────────────────────────────────────┐
│ Left Hand Articulators:     Nodes  0 - 20 (21 nodes)   │
│ Right Hand Articulators:    Nodes 21 - 41 (21 nodes)   │
│ Upper Pose Anchors:         Nodes 42 - 52 (11 nodes)   │
│ Facial Contours:            Nodes 53 - 92 (40 nodes)   │
│   ├─ Eyebrows:              Nodes 53 - 60 ( 8 nodes)   │
│   ├─ Lips & Mouth:          Nodes 61 - 76 (16 nodes)   │
│   └─ Lower Jawline:         Nodes 77 - 92 (16 nodes)   │
└────────────────────────────────────────────────────────┘
```

---

## 2. Anatomical Subsystems and Node Registry

The table below catalogs all 93 nodes with exact 0-indexed integer identifiers, anatomical names, and parent kinematic roots:

### A. Left Hand Articulators (Nodes 0 – 20)
| Node ID | Anatomical Joint Name | Parent / Prior Joint | Kinematic Segment |
| :---: | :--- | :---: | :--- |
| **0** | Left Wrist (Carpi) | 51 (Pose Left Wrist) | Hand Root / Carpus |
| **1** | Left Thumb CMC | 0 | First Metacarpal Base |
| **2** | Left Thumb MCP | 1 | Thumb Knuckle |
| **3** | Left Thumb IP | 2 | Thumb Interphalangeal |
| **4** | Left Thumb Tip | 3 | Distal Thumb |
| **5** | Left Index MCP | 0 | Second Metacarpal Knuckle |
| **6** | Left Index PIP | 5 | Proximal Interphalangeal |
| **7** | Left Index DIP | 6 | Distal Interphalangeal |
| **8** | Left Index Tip | 7 | Distal Index |
| **9** | Left Middle MCP | 0 | Third Metacarpal Knuckle |
| **10** | Left Middle PIP | 9 | Proximal Interphalangeal |
| **11** | Left Middle DIP | 10 | Distal Interphalangeal |
| **12** | Left Middle Tip | 11 | Distal Middle |
| **13** | Left Ring MCP | 0 | Fourth Metacarpal Knuckle |
| **14** | Left Ring PIP | 13 | Proximal Interphalangeal |
| **15** | Left Ring DIP | 14 | Distal Interphalangeal |
| **16** | Left Ring Tip | 15 | Distal Ring |
| **17** | Left Pinky MCP | 0 | Fifth Metacarpal Knuckle |
| **18** | Left Pinky PIP | 17 | Proximal Interphalangeal |
| **19** | Left Pinky DIP | 18 | Distal Interphalangeal |
| **20** | Left Pinky Tip | 19 | Distal Pinky |

### B. Right Hand Articulators (Nodes 21 – 41)
| Node ID | Anatomical Joint Name | Parent / Prior Joint | Kinematic Segment |
| :---: | :--- | :---: | :--- |
| **21** | Right Wrist (Carpi) | 52 (Pose Right Wrist) | Hand Root / Carpus |
| **22** | Right Thumb CMC | 21 | First Metacarpal Base |
| **23** | Right Thumb MCP | 22 | Thumb Knuckle |
| **24** | Right Thumb IP | 23 | Thumb Interphalangeal |
| **25** | Right Thumb Tip | 24 | Distal Thumb |
| **26** | Right Index MCP | 21 | Second Metacarpal Knuckle |
| **27** | Right Index PIP | 26 | Proximal Interphalangeal |
| **28** | Right Index DIP | 27 | Distal Interphalangeal |
| **29** | Right Index Tip | 28 | Distal Index |
| **30** | Right Middle MCP | 21 | Third Metacarpal Knuckle |
| **31** | Right Middle PIP | 30 | Proximal Interphalangeal |
| **32** | Right Middle DIP | 31 | Distal Interphalangeal |
| **33** | Right Middle Tip | 32 | Distal Middle |
| **34** | Right Ring MCP | 21 | Fourth Metacarpal Knuckle |
| **35** | Right Ring PIP | 34 | Proximal Interphalangeal |
| **36** | Right Ring DIP | 35 | Distal Interphalangeal |
| **37** | Right Ring Tip | 36 | Distal Ring |
| **38** | Right Pinky MCP | 21 | Fifth Metacarpal Knuckle |
| **39** | Right Pinky PIP | 38 | Proximal Interphalangeal |
| **40** | Right Pinky DIP | 39 | Distal Interphalangeal |
| **41** | Right Pinky Tip | 40 | Distal Pinky |

### C. Upper Pose Anchors (Nodes 42 – 52)
| Node ID | Anatomical Joint Name | Kinematic Function |
| :---: | :--- | :--- |
| **42** | Nose | Facial Center Anchor / Head Apex |
| **43** | Left Eye Inner/Outer | Cranial Reference Anchor |
| **44** | Right Eye Inner/Outer | Cranial Reference Anchor |
| **45** | Left Ear | Cranial Lateral Reference |
| **46** | Right Ear | Cranial Lateral Reference |
| **47** | Left Shoulder | Clavicular Left Pivot (Torso Base) |
| **48** | Right Shoulder | Clavicular Right Pivot (Torso Base) |
| **49** | Left Elbow | Left Arm Articulator |
| **50** | Right Elbow | Right Arm Articulator |
| **51** | Left Wrist Pose Anchor | Arm-to-Hand Left Interface |
| **52** | Right Wrist Pose Anchor | Arm-to-Hand Right Interface |

### D. Facial Non-Manual Contours (Nodes 53 – 92)
- **Eyebrows (53 – 60):** 4 nodes per brow tracking vertical deflection during questions and emphasis.
  - Left Eyebrow: `53, 54, 55, 56`
  - Right Eyebrow: `57, 58, 59, 60`
- **Lips & Mouth Perimeter (61 – 76):** 16 nodes forming the outer elliptical contour of the mouth for mouthing patterns.
  - Contiguous perimeter loop: `61 -> 62 -> ... -> 76 -> 61`
- **Lower Jawline (77 – 92):** 16 nodes tracking mandibular drop and lateral rotation.
  - Contiguous jaw perimeter: `77 -> 78 -> ... -> 92`

---

## 3. Physical Edge Connectivity

Edges represent physical kinematic constraints and biological connectivity:

1. **Left Hand Bones (23 edges):**
   - Phalanx chains: `(0, 1), (1, 2), (2, 3), (3, 4)` (Thumb)
   - `(0, 5), (5, 6), (6, 7), (7, 8)` (Index)
   - `(0, 9), (9, 10), (10, 11), (11, 12)` (Middle)
   - `(0, 13), (13, 14), (14, 15), (15, 16)` (Ring)
   - `(0, 17), (17, 18), (18, 19), (19, 20)` (Pinky)
   - Metacarpal transverse palm arches: `(5, 9), (9, 13), (13, 17)`
2. **Right Hand Bones (23 edges):**
   - Identical phalanx chains and transverse palm arches offset by $+21$.
3. **Upper Body Skeleton (12 edges):**
   - Shoulder girdle: `(47, 48)`
   - Left Arm: `(47, 49), (49, 51)`
   - Right Arm: `(48, 50), (50, 52)`
   - Cranial triangle: `(42, 43), (42, 44), (43, 45), (44, 46)`
   - Torso-neck anchors: `(42, 47), (42, 48)`
4. **Kinematic Cross-System Bridges (2 critical edges):**
   - Left Wrist Hand-to-Arm Bridge: `(0, 51)`
   - Right Wrist Hand-to-Arm Bridge: `(21, 52)`
   *These edges unite isolated hand subgraphs with the body kinematic chain, allowing arm trajectory to contextualize hand location.*
5. **Facial Structural Edges (43 edges):**
   - Brow arches: `(53, 54), (54, 55), (55, 56)` and `(57, 58), (58, 59), (59, 60)`
   - Brow-to-Eye anchors: `(53, 43), (56, 42), (57, 42), (60, 44)`
   - Lip perimeter loop: 16 contiguous edges `(61, 62) ... (76, 61)`
   - Mouth-to-Nose anchors: `(61, 42), (69, 42)`
   - Jaw arc: 15 contiguous edges `(77, 78) ... (91, 92)`
   - Jaw-to-Ear/Cranial anchors: `(77, 45), (92, 46), (84, 42)`

**Total Undirected Physical Edges:** **103 unique spatial edges**.

---

## 4. Symmetry and Self-Connections

- **Undirected Property:** For every spatial edge $(u, v) \in \mathcal{E}$, $(v, u) \in \mathcal{E}$.
- **Self-Loops:** Every node $i \in \mathcal{V}$ possesses a self-connection $(i, i)$ to preserve node identity representations during spatial graph message passing.
