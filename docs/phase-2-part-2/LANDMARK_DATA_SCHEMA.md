# Landmark Data Schema: SignTalk AI

**Document ID:** STAI-P2P2-001  
**Project:** SignTalk AI  
**Phase:** Phase 2 — Part 2 (Landmark Extraction & Preprocessing)  
**Status:** Approved Engineering Specification  

---

## 1. Landmark Node Topology ($V = 93$ Nodes)

SignTalk AI operates on an anatomically curated 93-node skeletal graph derived from Google MediaPipe Tasks Vision models (`PoseLandmarker`, `HandLandmarker`, `FaceLandmarker`).

```mermaid
graph TD
    subgraph LeftHand [Left Hand Articulators: Nodes 0 - 20]
        L0[0: Wrist] --> L1[1: Thumb CMC] --> L2[2: Thumb MCP] --> L3[3: Thumb IP] --> L4[4: Thumb Tip]
        L0 --> L5[5: Index MCP] --> L6[6: Index PIP] --> L7[7: Index DIP] --> L8[8: Index Tip]
        L0 --> L9[9: Middle MCP] --> L10[10: Middle PIP] --> L11[11: Middle DIP] --> L12[12: Middle Tip]
        L0 --> L13[13: Ring MCP] --> L14[14: Ring PIP] --> L15[15: Ring DIP] --> L16[16: Ring Tip]
        L0 --> L17[17: Pinky MCP] --> L18[18: Pinky PIP] --> L19[19: Pinky DIP] --> L20[20: Pinky Tip]
    end

    subgraph RightHand [Right Hand Articulators: Nodes 21 - 41]
        R21[21: Wrist] --> R22[22: Thumb CMC] --> R23[23: Thumb MCP] --> R24[24: Thumb IP] --> R25[25: Thumb Tip]
        R21 --> R26[26: Index MCP] --> R27[27: Index PIP] --> R28[28: Index DIP] --> R29[29: Index Tip]
        R21 --> R30[30: Middle MCP] --> R31[31: Middle PIP] --> R32[32: Middle DIP] --> R33[33: Middle Tip]
        R21 --> R34[34: Ring MCP] --> R35[35: Ring PIP] --> R36[36: Ring DIP] --> R37[37: Ring Tip]
        R21 --> R38[38: Pinky MCP] --> R39[39: Pinky PIP] --> R40[40: Pinky DIP] --> R41[41: Pinky Tip]
    end

    subgraph PoseAnchors [Upper Pose Anchors: Nodes 42 - 52]
        P42[42: Nose]
        P43[43: Left Eye]
        P44[44: Right Eye]
        P45[45: Left Ear]
        P46[46: Right Ear]
        P47[47: Left Shoulder]
        P48[48: Right Shoulder]
        P49[49: Left Elbow]
        P50[50: Right Elbow]
        P51[51: Left Wrist Pose]
        P52[52: Right Wrist Pose]
    end

    subgraph FaceNonManuals [Salient Facial Markers: Nodes 53 - 92]
        F53_60[53 - 60: Eyebrows Contour - 8 Nodes]
        F61_76[61 - 76: Lips & Mouth Contour - 16 Nodes]
        F77_92[77 - 92: Lower Jawline Contour - 16 Nodes]
    end
```

---

## 2. Anatomical Subsystems Breakdown

| Node Range | Subsystem | Count | Source Detector | Role in Indian Sign Language |
| :---: | :--- | :---: | :--- | :--- |
| **0 – 20** | Left Hand Articulators | 21 | `HandLandmarker` | Secondary or symmetric sign articulator, fingerspelling base |
| **21 – 41** | Right Hand Articulators | 21 | `HandLandmarker` | Primary/dominant sign articulator, trajectory carrier |
| **42 – 52** | Upper Pose Anchors | 11 | `PoseLandmarker` | Anatomical spatial frame-of-reference, torso center, shoulder scale |
| **53 – 60** | Eyebrow Contours | 8 | `FaceLandmarker` | Non-manual question markers (wh-furrow, polar-raise) |
| **61 – 76** | Lip & Mouth Contours | 16 | `FaceLandmarker` | Mouthing morphemes, grammatical facial adjectives |
| **77 – 92** | Jawline Contours | 16 | `FaceLandmarker` | Head nods/shakes, spatial perspective shifts |

---

## 3. Storage Container Schema: NumPy Compressed (`.npz`)

All processed samples are serialized as standalone `.npz` archive files located at `data/processed/landmarks/{split}/{sample_id}.npz`:

| Field Key | Data Type | Array Shape | Description |
| :--- | :--- | :---: | :--- |
| `data` | `float32` | `(3, 45, 93)` | Normalized spatial coordinates $[C, T, V]$ ($C=3$ channels: $x, y, z$). |
| `mask` | `bool` | `(1, 45, 93)` | Binary visibility mask indicating valid/imputed vs missing nodes. |
| `raw_coords` | `float32` | `(T_orig, 93, 3)` | Original unnormalized coordinates prior to temporal resampling. |
| `raw_mask` | `bool` | `(T_orig, 93)` | Original binary detection mask from MediaPipe detectors. |
| `class_id` | `int64` | Scalar | Numerical class label index ($0$ to $49$). |
| `class_label`| `str` | Scalar | English gloss text string (e.g. `'hello'`, `'thankyou'`). |
| `signer_id` | `str` | Scalar | Identifier of the human subject (e.g. `'signer_01'`). |
| `sample_id` | `str` | Scalar | Unique sequence identifier (e.g. `'raw_0001'`). |
| `split` | `str` | Scalar | Dataset partition (`'train'`, `'val'`, `'test'`). |
| `fps` | `float64` | Scalar | Effective video frame rate (standardized to $25.0\text{ FPS}$). |
| `quality_score`| `float64` | Scalar | Sequence-level quality evaluation score $[0.0, 1.0]$. |
| `classification`| `str` | Scalar | Quality classification (`'GOOD'`, `'ACCEPTABLE'`, `'REVIEW'`, `'REJECT'`). |

---

## 4. Metadata Manifest Schema (`processed_dataset_manifest.csv`)

The accompanying dataset manifest records:
- `sample_id`: Unique identifier.
- `split`: Partition assignment (`train`, `val`, `test`).
- `class_label`: Vocabulary gloss name.
- `class_id`: Integer label.
- `signer_id`: Signer code.
- `processed_file`: Relative path to the `.npz` file.
- `original_frames`: Frame count prior to resampling.
- `target_frames`: Standardized sequence length ($T = 45$).
- `average_quality`: Weighted sequence quality score.
- `classification`: Quality status.
- `classification_reason`: Documented justification.
- `pose_detection_rate`: Percentage of frames with valid pose.
- `left_hand_detection_rate`: Percentage of frames with valid left hand.
- `right_hand_detection_rate`: Percentage of frames with valid right hand.
- `face_detection_rate`: Percentage of frames with valid face.
- `elapsed_seconds`: Extraction latency.
- `status`: Execution checkpoint (`COMPLETED` or `FAILED`).
