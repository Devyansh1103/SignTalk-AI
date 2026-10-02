# ADR-007: Cloud vs. Edge Deployment Strategy

**Status:** APPROVED  
**Date:** October 2026  
**Context:** Sign language translation systems frequently face trade-offs between centralized cloud server compute and localized edge execution, with significant ramifications for latency, data privacy, internet dependency, and operational cost.

## Decision
Adopt a **Local Modular Monolith (Mode A)** for the academic MVP, architected with a strict separation of concerns to enable seamless migration to **Decentralized On-Device Edge (Mode B)** via ONNX Runtime in future phases.

## Evaluated Alternatives
1. **Centralized Cloud SaaS:** Stream client video or landmarks to remote cloud GPUs (AWS/GCP). *(Rejected: Severe privacy violations during confidential healthcare/banking consultations; network bandwidth reliance; recurring operational cloud hosting costs).*
2. **Immediate Pure Edge (Wasm/WebGPU):** Compile entire pipeline into browser WebAssembly/WebGPU immediately. *(Deferred: Immature PyTorch-to-WebGPU tooling for custom graph convolutions; steep debugging friction during research model iteration).*

## Consequences
- **Positive:** Enables self-contained execution on student/campus workstations; zero cloud compute costs; 100% offline air-gapped viability; absolute patient biometric privacy.
- **Negative:** Dependent on local host machine compute capability (CPU/GPU) for running PyTorch inference.
