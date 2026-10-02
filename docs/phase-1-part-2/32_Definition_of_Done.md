# 32. Technical Definition of Done (DoD): SignTalk AI

**Document ID:** STAI-P1P2-032  
**Project Name:** SignTalk AI  
**Document Status:** Approved Engineering Blueprint (Phase 1 — Part 2)  

---

## 1. Quality Governance & DoD Philosophy

In accordance with professional engineering standards, no feature or milestone is marked as complete based on subjective impressions. A subsystem is declared **DONE** only when all verifiable criteria in its Definition of Done are satisfied and confirmed by automated tests.

---

## 2. Subsystem Definitions of Done

### 2.1 Landmark Extraction & Preprocessing Pipeline
The landmark extraction pipeline is **DONE** when:
- [ ] Ingests standard 720p/1080p MP4/AVI video files and extracts 3D coordinates $(x, y, z)$ across all 93 nodes.
- [ ] Torso-centric normalization maps the mid-shoulder point exactly to $(0, 0, 0)$ and scales by shoulder width.
- [ ] Handles frames with undetected hands via zero-masking and visibility flags without throwing uncaught exceptions.
- [ ] Outputs feature arrays serialized into standardized Apache Parquet files.
- [ ] Unit tests in `tests/unit/test_normalizer.py` achieve $100\%$ pass rate.
- [ ] Processing speed achieves $\le 40\text{ ms}$ per frame on local CPU.

### 2.2 Baseline Sequence Classifier (Bi-LSTM)
The baseline model is **DONE** when:
- [ ] Implemented in PyTorch with verified parameter scale ($\approx 2.45\text{M}$ parameters).
- [ ] Successfully trains on the signer-independent INCLUDE-50 training split without NaN loss.
- [ ] Early stopping terminates cleanly upon validation loss plateau.
- [ ] Top-1, Top-5, and Macro-F1 evaluation scores on the held-out test split are logged in `experiments/experiment_ledger.csv`.
- [ ] PyTorch model checkpoint and manifest SHA-256 hash are archived in `models/checkpoints/`.

### 2.3 ST-GCN Kinematic Graph Encoder
The ST-GCN model is **DONE** when:
- [ ] 6-block residual architecture compiles and validates against tensor shape $(B, 3, 45, 93) \rightarrow (B, 12, 256)$.
- [ ] Symmetrically normalized spatial adjacency matrix $\mathbf{\Lambda} \in \mathbb{R}^{3 \times 93 \times 93}$ verifies spatial configuration partitioning.
- [ ] Demonstrates a statistically significant Top-1 accuracy gain ($\ge 8\%$) over the Bi-LSTM baseline on identical test splits.
- [ ] Latency benchmark verifies forward pass duration $\le 60\text{ ms}$ on local CPU.

### 2.4 Transformer Sequence Translator
The Transformer sequence translation engine is **DONE** when:
- [ ] 3-layer autoregressive decoder successfully cross-attends over ST-GCN latent sequence memory.
- [ ] Greedy decoding terminates upon generating `<EOS>` or reaching max length ($U=20$).
- [ ] Evaluated on held-out continuous test split of ISL-CSLTR with BLEU 1-4 and WER scores logged in the experiment ledger.
- [ ] Sentence confidence scoring correctly outputs geometric mean token probabilities.

### 2.5 Real-Time Inference & Backend Engine
The backend engine is **DONE** when:
- [ ] FastAPI application cleanly initializes model weights in memory during lifespan startup.
- [ ] WebSocket streaming endpoint `/ws/translate` ingests continuous landmark JSON frames at $\ge 25\text{ FPS}$.
- [ ] In-memory ring buffer updates asynchronously without blocking the Uvicorn ASGI event loop.
- [ ] Gating logic suppresses duplicate captions within $1.5\text{ seconds}$ and flags predictions with confidence $< 0.50$.
- [ ] Automated integration tests in `tests/integration/test_stream_api.py` pass cleanly.

### 2.6 React Frontend Web Application
The frontend application is **DONE** when:
- [ ] Connects to browser webcam via `getUserMedia()` and renders smooth 30 FPS video preview.
- [ ] Landmark overlay toggle renders skeletal joints and bones on HTML5 canvas at 60 Hz.
- [ ] Live captions update dynamically with color-coded confidence indicators (Green/Amber/Red).
- [ ] All primary controls (Start, Pause, Stop, Clear) operate via mouse clicks and keyboard shortcuts (`Spacebar`, `Esc`).
- [ ] Visual contrast adheres to WCAG 2.1 AA ($21:1$ contrast ratio).

### 2.7 Packaging, Privacy & Deployment
Deployment packaging is **DONE** when:
- [ ] Multi-container Docker Compose (`frontend` + `backend`) builds and runs with a single command (`docker compose up`).
- [ ] Network packet inspection verifies zero raw video bytes transmitted over external network adapters.
- [ ] Disk audit verifies zero `.mp4` or image files saved to host storage after an active 5-minute translation session.
