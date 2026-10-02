# 28. Environment & Dependency Governance: SignTalk AI

**Document ID:** STAI-P1P2-028  
**Project Name:** SignTalk AI  
**Document Status:** Approved Engineering Blueprint (Phase 1 — Part 2)  

---

## 1. Runtime Environment Specifications

To prevent dependency drift across development workstations, all core runtimes and library versions are explicitly locked:

| Runtime / Tool | Authoritative Locked Version | Justification |
| :--- | :--- | :--- |
| **Python** | `3.11.9` (64-bit) | Excellent performance optimizations in 3.11; stable compatibility with MediaPipe 0.10 and PyTorch 2.3. |
| **Node.js** | `20.14.0 LTS` | Active Long-Term Support release ensuring stable Vite, React, and npm package builds. |
| **Package Manager (Python)**| `pip 24.1` / `virtualenv` | Standard, universal Python package installation without proprietary lock-in. |
| **Package Manager (JS)** | `npm 10.7` | Native Node package manager with deterministic `package-lock.json`. |
| **PyTorch** | `2.3.1` (CUDA 12.1 or CPU) | Stable computational graph execution, TorchScript tracing, and ONNX export support. |
| **OpenCV** | `4.9.0.80` (`opencv-python-headless`)| Lightweight headless image ingestion without X11/GUI display server dependencies. |
| **MediaPipe** | `0.10.14` | Latest stable release featuring Holistic tracking and calibrated landmark visibility. |
| **FastAPI** | `0.111.0` | Modern async ASGI web framework with Pydantic v2 validation. |
| **React** | `18.3.1` | Stable concurrent rendering and virtual DOM performance. |
| **TypeScript** | `5.4.5` | Strict static typing and enhanced generics for WebSocket message framing. |

---

## 2. Environment Configuration Template (`.env.example`)

```ini
# ==============================================================================
#                      SIGNTALK AI: ENVIRONMENT CONFIGURATION
# ==============================================================================

# Application Environment: development | testing | production
SIGNTALK_ENV=development

# Network Bindings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS="http://localhost:3000,http://localhost:80,http://127.0.0.1:3000"

# Model Checkpoint & Architecture Paths
MODEL_CHECKPOINT_PATH="models/checkpoints/stgcn_trans_v1.pt"
MODEL_ONNX_PATH="models/checkpoints/stgcn_trans_int8.onnx"
VOCABULARY_PATH="assets/vocabularies/mvp_50.json"
GRAPH_ADJACENCY_PATH="assets/graphs/kinematic_adjacency_93.npy"

# Inference Engine Configuration
INFERENCE_DEVICE=cpu                 # Options: cpu | cuda | mps
USE_ONNX_RUNTIME=false               # Set true to enable INT8 ONNX acceleration
SLIDING_WINDOW_LENGTH=45             # Temporal window in frames (T=45 ~ 1.5s)
INFERENCE_STRIDE=5                   # Window evaluation step (S=5 ~ 166ms)
CONFIDENCE_THRESHOLD=0.50            # Minimum certainty for output validation

# Safety & Logging
LOG_LEVEL=INFO                       # Options: DEBUG | INFO | WARNING | ERROR
ENABLE_TELEMETRY=true                # Log latency and FPS metrics to disk
ALLOW_ANONYMOUS_SESSIONS=true        # Permits unauthenticated local sessions
```
