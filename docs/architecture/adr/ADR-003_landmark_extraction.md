# ADR-003: Adoption of MediaPipe Holistic for Visual Feature Extraction

**Status:** APPROVED  
**Date:** October 2026  
**Context:** Extracting accurate manual and non-manual visual features from monocular RGB video is the foundational computer vision step. The extractor must execute in real time on commodity CPUs while preserving user biometric privacy.

## Decision
Adopt **Google MediaPipe Holistic** as the core landmark extraction pipeline, filtering the raw output into a curated multimodal topological subset ($N = 53$ or $N = 93$ nodes).

## Evaluated Alternatives
1. **Dense 3D-CNNs on Raw Pixels (I3D, SlowFast):** Eliminates explicit landmark extraction, but requires massive GPU compute, overfits heavily to background pixels and signer clothing, and violates privacy by requiring raw video cloud streaming.
2. **OpenPose:** High accuracy, but extremely compute-heavy ($> 150\text{ ms}$ per frame on CPU), requiring dedicated multi-GPU setups.
3. **MMPose / HRNet:** Strong research benchmarks, but higher dependency footprint and slower CPU inference times than MediaPipe.

## Consequences
- **Positive:** Sub-40 ms CPU inference per frame; robust 3D hand and pose tracking; enables zero raw-video retention by extracting coordinates in volatile memory.
- **Negative:** Susceptible to tracking jitter under severe bilateral hand overlap or extreme low light, requiring temporal smoothing and confidence monitoring.
