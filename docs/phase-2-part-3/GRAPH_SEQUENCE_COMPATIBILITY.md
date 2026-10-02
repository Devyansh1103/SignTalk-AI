# Graph and Sequence Compatibility: SignTalk AI

**Document ID:** STAI-P2P3-015  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead Computer Vision Engineer & Lead Architect  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Topological Equivalence Verification

In graph convolutional neural networks, any discrepancy between landmark coordinate indexing and adjacency matrix node definitions corrupts the spatial convolution kernel. We formally verify that:

$$\text{Landmark Extraction Node Order} \equiv \text{Sequence Dataset Node Order} \equiv \text{ST-GCN Adjacency Node Order}$$

---

## 2. Graph Node Invariant Check

The 93 nodes map identically across all components:

| Node Range | Anatomical Component | Preprocessing Index (`landmark_extractor.py`) | Sequence Dataset Index (`sign_sequence_dataset.py`) | Adjacency Matrix Index (`kinematic_adjacency_93.npy`) |
| :---: | :--- | :---: | :---: | :---: |
| **0 – 20** | Left Hand Joints | $0 \dots 20$ | $0 \dots 20$ | $0 \dots 20$ |
| **21 – 41** | Right Hand Joints | $21 \dots 41$ | $21 \dots 41$ | $21 \dots 41$ |
| **42 – 52** | Upper-Body Pose | $42 \dots 52$ | $42 \dots 52$ | $42 \dots 52$ |
| **53 – 92** | Facial Markers | $53 \dots 92$ | $53 \dots 92$ | $53 \dots 92$ |

---

## 3. Kinematic Cross-Modality Verification

1. **Left Arm-to-Hand Bridge:**  
   Node 51 (Left Pose Wrist) connects to Node 0 (Left Hand Wrist).  
   Adjacency check: $\mathbf{A}[51, 0] = \mathbf{A}[0, 51] = 1.0$.
2. **Right Arm-to-Hand Bridge:**  
   Node 52 (Right Pose Wrist) connects to Node 21 (Right Hand Wrist).  
   Adjacency check: $\mathbf{A}[52, 21] = \mathbf{A}[21, 52] = 1.0$.
3. **Upper Pose Torso Anchor:**  
   Node 47 (Left Shoulder) connects to Node 48 (Right Shoulder).  
   Adjacency check: $\mathbf{A}[47, 48] = \mathbf{A}[48, 47] = 1.0$.

---

## 4. Automated Compatibility Enforcement

In [`src/data/stgcn_tensor.py`](file:///d:/SignAI/src/data/stgcn_tensor.py), the function `validate_stgcn_shape()` and automated unit tests in `tests/data/test_stgcn_tensor.py` verify that:
1. Every input tensor has shape $(\dots, 93)$ along the vertex dimension.
2. The loaded adjacency matrix has shape $(3, 93, 93)$.
3. No self-loops exist in the directional partitions (Centripetal and Centrifugal), and exactly $I_{93}$ exists in the Root partition.
4. Any dimension mismatch triggers an immediate non-zero exception before batches reach model layers.
