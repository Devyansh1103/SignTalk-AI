# 27. GitHub Repository Architecture & Directory Blueprint: SignTalk AI

**Document ID:** STAI-P1P2-027  
**Project Name:** SignTalk AI  
**Document Status:** Approved Engineering Blueprint (Phase 1 — Part 2)  

---

## 1. Authoritative Repository Directory Tree

The SignTalk AI codebase is organized to maintain a strict separation between core machine learning research, backend API serving, frontend presentation, and reproducible documentation:

```
SignTalk-AI/
│
├── README.md                           # Master project documentation, quickstart, and viva summary
├── LICENSE                             # Open-source academic research license (MIT / Apache 2.0)
├── .gitignore                          # Excludes raw video, virtualenvs, checkpoints, and cache
├── .env.example                        # Template for environment variables
├── docker-compose.yml                  # Multi-container local orchestration (frontend + backend)
├── requirements.txt                    # Pinned Python dependencies for local development
├── package.json                        # Root script manager and frontend dependency definitions
│
├── docs/                               # Formal academic project documentation
│   ├── phase-1-part-1/                 # Project definition, user personas, research gaps, specs
│   ├── phase-1-part-2/                 # Technical architecture, blueprints, data pipelines
│   ├── architecture/                   # Detailed architectural specifications, Mermaid diagrams
│   │   └── adr/                        # Architectural Decision Records (ADR-001 to ADR-008)
│   └── research/                       # Primary citations, academic bibliography, dataset audits
│
├── data/                               # Dataset management, registries, and manifests
│   ├── README.md                       # Data hygiene, directory schemas, and ethics guidelines
│   ├── dataset_registry.csv            # Authoritative machine-readable dataset registry
│   ├── raw/                            # Unprocessed video clips (Git-ignored)
│   ├── landmarks/                      # Pre-extracted Parquet / HDF5 coordinate arrays (Git-ignored)
│   ├── splits/                         # Reproducible signer-independent train/val/test split indices
│   └── supplementary/                  # Local supplementary recording manifests and consent hashes
│
├── notebooks/                          # Exploratory data analysis, prototyping, and visualization
│   ├── 01_explore_isl_datasets.ipynb   # Visualizing frame rates, class distributions, and signers
│   ├── 02_mediapipe_extraction.ipynb   # Prototyping landmark extraction and coordinate plots
│   ├── 03_graph_construction.ipynb     # Validating kinematic adjacency matrices and Laplacian math
│   └── 04_baseline_benchmarking.ipynb  # Running baseline MLP and Bi-LSTM evaluations
│
├── experiments/                        # Experiment tracking and reproducible evaluation ledgers
│   ├── experiment_ledger.csv           # Master CSV recording all runs, hyperparameters, and scores
│   └── runs/                           # Individual run folders with TensorBoard logs and configs
│
├── models/                             # Serialized model artifacts and export tooling
│   ├── checkpoints/                    # Saved PyTorch (.pt) weights and ONNX binaries (Git-ignored)
│   └── registry.json                   # Machine-readable registry of validated model checkpoints
│
├── src/                                # Core reusable AI / ML scientific Python package
│   ├── __init__.py
│   ├── data/                           # Dataset loaders, PyTorch Dataset classes, split generators
│   ├── preprocessing/                  # Torso centering, scale normalization, sliding buffer math
│   ├── landmarks/                      # MediaPipe wrapper, joint selection, missing point filters
│   ├── graph/                          # Kinematic adjacency matrix constructors, graph partitioning
│   ├── models/                         # PyTorch model definitions (MLP, Bi-LSTM, ST-GCN, Transformer)
│   ├── inference/                      # Optimized ONNX / PyTorch inference execution wrappers
│   └── evaluation/                     # Metric calculators: Top-k Acc, BLEU 1-4, WER, ROUGE-L
│
├── backend/                            # FastAPI modular monolith application
│   ├── Dockerfile                      # Backend container build specification
│   ├── requirements.txt                # Production backend dependencies
│   └── app/
│       ├── main.py                     # FastAPI app factory, lifespan handlers, CORS setup
│       ├── api/                        # REST and WebSocket route definitions
│       ├── core/                       # App settings, environment configs, structured logging
│       ├── services/                   # Session manager, inference scheduler, safety evaluator
│       └── nlp/                        # English vocabulary detokenizer and sentence formatter
│
├── frontend/                           # React 18 + TypeScript web client
│   ├── Dockerfile                      # Production Nginx multi-stage container build
│   ├── package.json                    # Frontend package dependencies
│   ├── vite.config.ts                  # Vite build tool and development server configuration
│   ├── tsconfig.json                   # Strict TypeScript compiler options
│   └── src/
│       ├── main.tsx                    # React application entrypoint
│       ├── App.tsx                     # Top-level application shell and providers
│       ├── components/                 # Reusable UI components (Camera, Canvas, Captions, Controls)
│       ├── views/                      # Route screens (TranslatorView, HomeView, SettingsView)
│       ├── hooks/                      # Custom hooks (useWebSocketStream, useCamera, useLandmarks)
│       └── styles/                     # CSS Modules and high-contrast WCAG 2.1 AA tokens
│
├── tests/                              # Comprehensive automated test suite
│   ├── unit/                           # Tests for normalizers, graph math, tokenizers, buffers
│   ├── model/                          # Tests for PyTorch tensor shapes, forward passes, determinism
│   ├── integration/                    # Tests for FastAPI endpoints and WebSocket streaming
│   ├── e2e/                            # Playwright end-to-end browser translation tests
│   └── performance/                    # Latency benchmarks, memory leak detectors, stress tests
│
├── scripts/                            # Operational utility and automation scripts
│   ├── extract_landmarks.py            # Offline batch landmark extraction for raw datasets
│   ├── train_baseline.py               # Training entrypoint for MLP / Bi-LSTM baselines
│   ├── train_stgcn.py                  # Training entrypoint for ST-GCN isolated / continuous models
│   ├── export_onnx.py                  # PyTorch to ONNX quantization and export utility
│   └── evaluate_checkpoint.py          # Standalone test set evaluation and metric report generator
│
├── configs/                            # Configuration YAML files for models, training, and streaming
│   ├── base_config.yaml                # Default paths, buffer sizes, and logging configs
│   ├── train_baseline.yaml             # Hyperparameters for baseline training
│   ├── train_stgcn.yaml                # Hyperparameters for ST-GCN training
│   └── stream_config.yaml              # Sliding window stride, buffer capacity, and thresholds
│
└── deployment/                         # Production orchestration and infrastructure templates
    ├── docker-compose.prod.yml         # Production multi-container composition
    └── nginx/                          # Nginx reverse proxy and TLS termination configurations
```

---

## 2. Version Control Hygiene & `.gitignore` Architecture

The `.gitignore` strictly blocks large binaries, biometric videos, and volatile artifacts from polluting git history:
- **Raw Video Files:** `data/raw/**/*.mp4`, `data/raw/**/*.avi`
- **Extracted Feature Arrays:** `data/landmarks/**/*.parquet`, `data/landmarks/**/*.h5`
- **Model Checkpoints:** `models/checkpoints/*.pt`, `models/checkpoints/*.onnx` (tracked via checksum manifests in git, with actual weights stored in local releases or shared storage).
- **Environment & Secrets:** `.env`, `.env.local`
- **Virtual Environments & Cache:** `venv/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `node_modules/`, `dist/`
