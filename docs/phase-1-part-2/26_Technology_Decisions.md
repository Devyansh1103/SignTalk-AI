# 26. Technology Decision Matrix & Architectural Evaluations: SignTalk AI

**Document ID:** STAI-P1P2-026  
**Project Name:** SignTalk AI  
**Document Status:** Approved Engineering Blueprint (Phase 1 — Part 2)  

---

## 1. Decision Evaluation Framework

In accordance with strict academic guidelines, all technical selections are evaluated on verified engineering trade-offs rather than subjective terminology (avoiding terms like "best" or "winner").

Each technology candidate is assessed against five core criteria:
1. **Latency & Throughput:** Capability to satisfy the sub-500 ms conversational latency budget.
2. **Academic & Research Rigor:** Reproducibility, peer-reviewed benchmarking, and community standardization.
3. **Data Privacy & Ethics:** Capability to operate locally without mandatory cloud video streaming.
4. **Implementation Feasibility:** Suitability for completion within university academic project constraints.
5. **Portability:** Flexibility to migrate between development, local servers, and future edge runtimes.

---

## 2. Master Technology Decision Matrix

| Subsystem Domain | Candidate Option | Key Technical Advantages | Documented Disadvantages | Project Fit & Academic Decision | Engineering Justification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Frontend Framework** | **React + TypeScript (Selected)** | Component modularity; strong type safety; vast ecosystem for canvas rendering. | Requires Node.js build tooling; larger bundle size than vanilla JS. | **APPROVED** (Primary Client Framework) | Enforces strict type checking across WebSocket message contracts; cleanly isolates camera, caption, and settings components. |
| | Vanilla JS / HTML5 | Zero build dependencies; minimal footprint. | Lacks structured component state management; prone to spaghetti code in complex streaming UIs. | Rejected | Difficult to maintain complex asynchronous WebSocket, video canvas, and accessibility states. |
| | PyQt / Electron | Direct local OS access; avoids browser sandbox. | Heavy distribution footprint ($> 150\text{ MB}$); poor cross-platform web accessibility. | Rejected | Fails the requirement for zero-install, browser-based accessibility in public service desks. |
| **Backend Framework** | **FastAPI (Selected)** | Native `asyncio` WebSocket support; high ASGI throughput; automated OpenAPI docs. | Python GIL requires thread-pool offloading for heavy PyTorch forward passes. | **APPROVED** (Primary Application Backend) | Native async WebSocket handling combined with direct in-process PyTorch tensor execution without IPC overhead. |
| | Flask | Simple, minimalist synchronous framework. | Lacks native async WebSocket primitives; requires heavy external gevent/socketio wrappers. | Rejected | High latency penalties under concurrent 30 FPS coordinate streaming. |
| | Django | Full-featured batteries-included framework with ORM. | Excessive architectural overhead and database coupling; unnecessary for in-memory streaming inference. | Rejected | Violates the principle of architectural simplicity; introduces unwanted database dependencies. |
| **Real-Time Transport**| **WebSockets (Selected)**| Full-duplex bidirectional streaming; sub-15ms transport latency; native browser and ASGI support. | Operates over TCP (subject to head-of-line blocking on lossy networks). | **APPROVED** (Primary Transport Protocol) | Delivers sub-15ms transport latency for compact coordinate arrays ($< 50\text{ KB/s}$) without signaling complexity. |
| | WebRTC Data Channels | Extremely low UDP latency; peer-to-peer data transport. | Requires complex signaling servers (STUN/TURN/ICE); heavy connection setup negotiation. | Rejected for MVP | Unjustified architectural complexity when streaming lightweight coordinates over local/LAN networks. |
| | HTTP Long-Polling | Simple RESTful compatibility. | Massive TCP handshake overhead; latency $> 150\text{ ms}$ per cycle. | Rejected | Cannot sustain real-time 30 FPS streaming. |
| **Computer Vision** | **MediaPipe Holistic (Selected)** | Sub-35ms CPU landmark tracking; extracts 3D hands, pose, and face; zero raw-video retention. | Prone to tracking loss during severe hand occlusions; coordinate jitter in poor lighting. | **APPROVED** (Primary Vision Extractor) | Enables real-time CPU feature extraction on consumer laptops while preserving user biometric privacy. |
| | Dense 3D-CNNs (I3D) | End-to-end video classification without explicit landmark errors. | Massive parameter footprint ($> 25\text{M}$); requires high-end server GPUs; forces raw video streaming. | Rejected | Prohibitive compute requirements for consumer laptops; violates data minimization privacy rules. |
| | OpenPose | High landmark accuracy across crowded scenes. | Extremely slow on CPUs ($> 150\text{ ms}$ per frame); heavy multi-gigabyte dependency footprint. | Rejected | Completely incompatible with sub-500ms real-time CPU latency budgets. |
| **Deep Learning Engine**| **PyTorch (Selected)** | Dynamic computational graph; native Python scientific integration; dominant in ST-GCN research. | Requires compilation optimization (TorchScript / ONNX) for high-efficiency edge CPU serving. | **APPROVED** (Primary Training & Model Backbone)| Established academic standard for ST-GCN and Transformer research; flexible debugging and tensor profiling. |
| | TensorFlow / Keras | Strong production serving tools (TF-Serving). | Less flexible dynamic graph manipulation; diminishing research adoption in graph neural networks. | Rejected | Harder to adapt custom ST-GCN spatial configuration partitioning kernels. |
| **Inference Acceleration**| **ONNX Runtime (Selected)**| Cross-platform hardware acceleration (AVX-512, DirectML, WebGPU); INT8 dynamic quantization. | Requires model export and operator compatibility validation. | **APPROVED** (Primary Optimization Target) | Delivers $2.0\times - 3.5\times$ CPU inference speedup, shrinking ST-GCN forward times to $< 25\text{ ms}$. |
| | NVIDIA TensorRT | Maximum theoretical GPU inference throughput. | Strictly locked to proprietary NVIDIA hardware; incompatible with AMD, Intel, and Apple Silicon. | Future Scope (If GPU mandated) | Excludes users without discrete NVIDIA hardware. |
| **Packaging & Env** | **Docker Compose (Selected)**| Fully reproducible environment; identical execution across OS platforms; isolated volumes. | Slight container virtualization overhead on Windows/macOS. | **APPROVED** (Primary Deployment Topology) | Eliminates "works on my machine" dependency mismatches during academic grading and lab demos. |
