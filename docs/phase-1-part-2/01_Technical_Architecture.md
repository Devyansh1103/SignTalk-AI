# 01. Technical Architecture Specification: SignTalk AI

**Document ID:** STAI-P1P2-001  
**Project Name:** SignTalk AI  
**Academic Context:** B.Tech CSE (AI/ML) Final Year Project | GLA University  
**Document Status:** Approved Engineering Blueprint (Phase 1 — Part 2)  

---

## 1. Executive Technical Summary

SignTalk AI is an end-to-end computer vision and sequence translation system designed to translate continuous Indian Sign Language (ISL) signing into fluent natural-language English text captions. 

The technical architecture is structured around a **two-stage spatial-temporal deep learning network**:
1. **Kinematic Feature Extraction:** Monocular RGB video frames are transformed into normalized 3D skeletal landmark graphs using Google MediaPipe Holistic.
2. **Spatial-Temporal Representation:** An 6-block Spatial-Temporal Graph Convolutional Network (ST-GCN) models biological bone connectivity and cross-frame trajectory motion.
3. **Sequence Translation:** A 3-layer autoregressive Transformer decoder translates continuous latent sign representations into natural-language English sentences.
4. **Real-Time Serving:** A Python FastAPI modular monolith connects to a React + TypeScript web application over low-latency WebSockets, achieving an end-to-end design latency budget $< 500\text{ ms}$.

```mermaid
graph LR
    A[Webcam Feed 720p @ 30 FPS] --> B[MediaPipe Holistic 3D Extraction]
    B --> C[Torso Normalization & Scaling]
    C --> D[Sliding Window Tensor Buffer T=45, S=5]
    D --> E[ST-GCN Kinematic Spatial-Temporal Encoder]
    E --> F[Transformer Autoregressive Decoder]
    F --> G[NLP Detokenizer & Confidence Gating]
    G --> H[High-Contrast Live Captions < 500ms]
```

---

## 2. Technical Stack & Architectural Standards

| Subsystem Layer | Core Technology | Version Specification | Architectural Role & Purpose |
| :--- | :--- | :--- | :--- |
| **Client Frontend** | React + TypeScript + Vite | React 18.3, TS 5.4, Vite 5.2 | High-performance, type-safe web client rendering video, canvas overlays, and live captions. |
| **Real-Time Transport**| WebSockets (`ws://`, `wss://`) | Starlette / WebSocket RFC 6455 | Full-duplex continuous frame streaming with sub-15ms transport latency. |
| **Application Backend** | Python + FastAPI + Uvicorn | Python 3.11, FastAPI 0.111, Uvicorn 0.30 | High-performance ASGI modular monolith managing sessions, inference queues, and safety policies. |
| **Computer Vision** | OpenCV + MediaPipe Holistic | OpenCV 4.9, MediaPipe 0.10.14 | Real-time 3D landmark regression across hands (42 pts), upper pose (11 pts), and face (40 pts). |
| **Deep Learning Engine** | PyTorch + TorchScript / ONNX | PyTorch 2.3.1 (CUDA 12.1 / CPU) | Tensor construction, ST-GCN graph convolution, and Transformer translation decoding. |
| **Inference Acceleration**| ONNX Runtime | ONNX Runtime 1.18.0 (CPU / DirectML)| Optimized model serving with INT8/FP16 dynamic quantization for sub-500ms latency. |
| **Packaging & Dev** | Docker + Docker Compose | Docker Engine 26.1, Compose v2 | Multi-container reproducible development and campus lab deployment topology. |

---

## 3. Core Architectural Subsystems

### Subsystem 1: Client Application (Layer 1)
- Implements camera access via standard browser Web APIs (`getUserMedia()`).
- Renders an interactive camera preview with a toggleable HTML5 landmark overlay canvas.
- Displays live captions with dynamic visual confidence indicators and accessible controls (WCAG 2.1 AA compliant).

### Subsystem 2: Real-Time Transport (Layer 2)
- Bi-directional WebSocket endpoint (`/api/v1/stream/ws`) streaming structured landmark packets at $\ge 25\text{ FPS}$ ($< 50\text{ KB/s}$ bandwidth).
- Manages connection heartbeats, packet serialization, and automatic reconnects.

### Subsystem 3: Feature Preprocessing & Normalization (Layer 3 & 4)
- Normalizes landmark coordinates relative to a mid-shoulder origin $\mathbf{p}_{origin} = \frac{\mathbf{p}_{L.Shoulder} + \mathbf{p}_{R.Shoulder}}{2}$.
- Applies torso-span Euclidean distance normalization to eliminate camera distance and signer body size variations.
- Discards raw RGB video frames immediately in volatile memory, guaranteeing absolute biometric privacy.

### Subsystem 4: Kinematic Spatial-Temporal Graph Engine (Layer 5)
- Constructs an anatomical graph tensor $\mathbf{X} \in \mathbb{R}^{3 \times 45 \times 93}$ over a rolling temporal window ($T=45$).
- Executes partitioned spatial graph convolutions along biological bone edges and temporal convolutions along frame trajectories.

### Subsystem 5: Transformer Sequence Translator (Layer 5)
- Cross-attends over latent ST-GCN kinematic sequence tokens to decode coherent natural-language sentences.
- Bridges the structural grammatical mismatch between ISL (SOV) and English (SVO).

### Subsystem 6: Safety, Confidence & Error Handling (Layer 3)
- Computes calibrated prediction confidence.
- Flags predictions below $\theta_{conf} = 0.50$ as uncertain, prompting user repetition rather than emitting hallucinations.
