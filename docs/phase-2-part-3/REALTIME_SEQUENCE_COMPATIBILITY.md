# Real-Time Sequence Compatibility Specification: SignTalk AI

**Document ID:** STAI-P2P3-034  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead System Architect & Real-Time Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Architectural Alignment: Offline Training vs. Online Inference

A critical failure mode in video AI systems is the discrepancy between offline training data formatting and live edge streaming. SignTalk AI enforces strict architectural isomorphism between the offline sequence builder and the online real-time inference engine:

```
[OFFLINE TRAINING PIPELINE]
Raw MP4 Video ──► Uniform Resample (T=45) ──► Torso Normalization ──► Tensor [C=3, T=45, V=93] ──► ST-GCN

[ONLINE STREAMING ENGINE]
Webcam (25 FPS) ──► MediaPipe Tasks ──► Circular FIFO Buffer (W=45) ──► Stride S=5 ──► Tensor [1, 3, 45, 93] ──► ST-GCN
```

---

## 2. Parameter Isomorphism

| Architectural Parameter | Offline Dataset Configuration | Online Real-Time Inference Engine | Consistency Status |
| :--- | :---: | :---: | :---: |
| **Operational Frame Rate** | $25.0\text{ FPS}$ | $25.0\text{ FPS}$ | Exact Match |
| **Window Length ($W$)** | $45\text{ frames}$ ($1.80\text{ s}$) | $45\text{ frames}$ ($1.80\text{ s}$) | Exact Match |
| **Evaluation Stride ($S$)** | Evaluated on full sequences | $5\text{ frames}$ ($200\text{ ms}$) | Isomorphic |
| **Spatial Normalization** | Torso-centered + shoulder distance | Torso-centered + running shoulder distance | Isomorphic |
| **Graph Topology ($V$)** | $93\text{ skeletal nodes}$ | $93\text{ skeletal nodes}$ | Exact Match |
| **Feature Channels ($C$)** | $3\text{ channels } (x, y, z)$ | $3\text{ channels } (x, y, z)$ | Exact Match |

---

## 3. Real-Time Latency Budget (Target Allocation)

To achieve smooth interactive translation without lag, total latency per sliding window must remain strictly below the window hop time ($S = 5\text{ frames} = 200\text{ ms}$):

| Processing Pipeline Stage | Target Latency Budget | Verification Mechanism |
| :--- | :---: | :--- |
| **Frame Capture & Ingestion** | $\le 10\text{ ms}$ | OpenCV VideoCapture thread pool |
| **MediaPipe Tasks Landmark Extraction** | $\le 25\text{ ms}$ | Benchmarked at $22.4\text{ ms}$ in Phase 2 Part 1 |
| **Circular Buffer FIFO Append & Torso Norm** | $\le 2\text{ ms}$ | NumPy ring buffer |
| **ST-GCN Forward Pass (CPU / INT8 ONNX)** | $\le 25\text{ ms}$ | Phase 3 model optimization target |
| **Transformer / Token Output Generation** | $\le 15\text{ ms}$ | Phase 3 inference benchmark |
| **Total Turnaround Latency** | **$\le 77\text{ ms} < 200\text{ ms}$** | **$123\text{ ms}$ Headroom for UI Rendering** |
