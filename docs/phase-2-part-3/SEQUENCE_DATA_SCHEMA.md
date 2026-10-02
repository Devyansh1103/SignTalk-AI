# Canonical Sequence Data Schema: SignTalk AI

**Document ID:** STAI-P2P3-003  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** ML Data Engineer & Lead Architect  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Overview

This document specifies the canonical sequence schema for all processed sequence objects in SignTalk AI. Every sequence represents an ordered temporal segment of skeletal landmark coordinates with corresponding labels, structural metadata, quality metrics, and alignment indicators.

---

## 2. Canonical Sequence Metadata Schema

Every generated sequence record contains the following standardized attributes stored in the master manifest (`sequence_manifest.csv`) and embedded within the binary sequence archive:

| Field Name | Data Type | Nullable | Description & Reference | Example Value |
| :--- | :--- | :---: | :--- | :--- |
| `sequence_id` | String | No | Unique canonical sequence identifier | `seq_0001` |
| `source_dataset` | String | No | Originating dataset benchmark | `INCLUDE-50` |
| `source_recording_id` | String | No | Original raw recording or sample ID | `raw_0001` |
| `source_video_id` | String | No | Original source video filename | `hello_signer_01_rep1.mp4` |
| `signer_id` | String | No | Anonymized participant identifier | `signer_01` |
| `split` | String | No | Designated evaluation partition (`train`, `val`, `test`) | `train` |
| `class_id` | Integer | No | Discrete categorical index $[0, 9]$ | `0` |
| `label` | String | No | Canonical lowercase vocabulary label | `hello` |
| `gloss` | String | No | Uppercase linguistic sign gloss | `HELLO` |
| `translation` | String | Yes | Natural language English translation equivalent | `Hello` |
| `sequence_start_frame` | Integer | No | Starting frame index in source video (0-indexed) | `0` |
| `sequence_end_frame` | Integer | No | Ending frame index in source video | `44` |
| `sequence_start_time` | Float | No | Start timestamp in seconds | `0.000` |
| `sequence_end_time` | Float | No | End timestamp in seconds | `1.800` |
| `frame_count` | Integer | No | Number of frames in raw extraction window | `45` |
| `fps` | Float | No | Operational frame rate | `25.0` |
| `duration_seconds` | Float | No | Sequence duration in seconds ($\text{frames} / \text{fps}$) | `1.800` |
| `landmark_schema_version` | String | No | Underlying landmark topology version | `93-node-v1` |
| `preprocessing_version` | String | No | Preprocessing pipeline version | `landmarks-v1.0` |
| `dataset_version` | String | No | Final sequence dataset version | `signTalk-seq-v1.0.0` |
| `quality_score` | Float | No | Overall multi-factor sequence quality metric $[0.0, 1.0]$ | `0.6604` |
| `quality_status` | String | No | Multi-tier classification (`GOOD`, `ACCEPTABLE`, `REVIEW`, `REJECT`) | `ACCEPTABLE` |
| `missing_landmark_ratio` | Float | No | Ratio of unobserved or interpolated joint coordinates | `0.214` |
| `pose_detection_rate` | Float | No | Percentage of frames with valid pose tracking | `100.0` |
| `left_hand_detection_rate` | Float | No | Percentage of frames with detected left hand | `42.22` |
| `right_hand_detection_rate` | Float | No | Percentage of frames with detected right hand | `51.11` |
| `sequence_generation_method`| String | No | Generation strategy (`full_recording`, `sliding_window`) | `full_recording` |
| `window_size` | Integer | No | Window length in frames ($T = 45$) | `45` |
| `stride` | Integer | No | Window evaluation step ($S = 5$ for sliding windows) | `45` |
| `padding_length` | Integer | No | Number of padded frames added | `0` |
| `mask_available` | Boolean | No | Indicates presence of auxiliary binary validity mask | `True` |
| `augmentation_status` | String | No | Applied augmentation policy (`none`, `training_jitter`) | `none` |
| `augmentation_seed` | Integer | Yes | Deterministic seed for reproducible augmentation | `null` |
| `file_path` | String | No | Relative filesystem path to serialized `.npz` archive | `data/processed/sequences/train/seq_0001.npz` |

---

## 3. Binary Tensor Schema (`.npz`)

Each sequence archive contains serialized NumPy arrays configured for direct conversion to PyTorch tensors:

```
sequence_archive.npz
├── data:       np.ndarray (float32)  shape=(3, 45, 93)   # Normalized [C, T, V] coordinates
├── mask:       np.ndarray (float32)  shape=(1, 45, 93)   # Binary validity mask [1, T, V]
├── raw_coords: np.ndarray (float32)  shape=(45, 93, 3)   # Native image-space coordinates [T, V, C]
├── label:      np.int64                                  # Class ID [0..9]
├── gloss_id:   np.int64                                  # Vocabulary token index
├── sample_id:  str                                       # Source sample ID (e.g. 'raw_0001')
├── sequence_id:str                                       # Sequence ID (e.g. 'seq_0001')
└── signer_id:  str                                       # Signer ID (e.g. 'signer_01')
```

### 3.1 Dimension Specifications
- **$C = 3$ (Channels):** Cartesian coordinates $(x, y, z)$. Torso-centered and scaled by Euclidean shoulder distance.
- **$T = 45$ (Time):** Standardized temporal window representing $1.8\text{ seconds}$ at $25.0\text{ FPS}$.
- **$V = 93$ (Vertices / Graph Nodes):**
  - $0 \le v \le 20$: Left Hand ($21$ joints)
  - $21 \le v \le 41$: Right Hand ($21$ joints)
  - $42 \le v \le 52$: Upper-Body Pose ($11$ joints)
  - $53 \le v \le 92$: Non-Manual Facial Anchors ($40$ markers)
- **Mask Tensor:** $\mathbf{M} \in \{0, 1\}^{1 \times T \times V}$ where $1$ denotes a genuine or validly interpolated joint observation, and $0$ indicates a missing, occluded, or padded node.
