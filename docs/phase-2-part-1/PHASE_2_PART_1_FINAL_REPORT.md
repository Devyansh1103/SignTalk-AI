# Phase 2 Part 1 — Final Report: Dataset Selection & Acquisition

**Document ID:** STAI-P2P1-FINAL  
**Phase:** Phase 2 — Part 1  
**Status:** Approved  
**Author:** Lead AI/ML Data & Research Engineer  

---

## 1. Executive Summary

Phase 2 Part 1 has successfully established the dataset foundation for SignTalk AI:
- **Primary Isolated Sign Benchmark:** AI4Bharat INCLUDE-50 (50 vocabulary classes, 15 semantic domains, 958 total video instances across 7 signers).
- **Primary Continuous Candidates:** ISLTranslate (31,117 sentences registered) and ISL-CSLTR (700 sentences).
- **Official Partitions Acquired:** Train split (689 samples, 71.9%), Validation split (77 samples, 8.0%), Test split (192 samples, 20.1%).
- **Landmark Extraction Feasibility:** Validated across 6 video conditions with MediaPipe Tasks (Pose: 100%, Hands: 76.3% RH / 45.6% LH, Latency: 22.4 ms/frame on CPU).
- **Manifests & Records:** Fully synchronized in `data/metadata/dataset_manifest.csv`, `splits.csv`, `class_distribution.csv`.

Phase 2 Part 1 is declared **COMPLETE** and serves as the authoritative source of truth for Phase 2 Part 2.
