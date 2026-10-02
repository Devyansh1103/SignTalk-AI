# SignTalk AI — Dataset Registry & Data Management

**Directory Location:** `/data/`  
**Purpose:** Centralized index and specification for all external benchmark datasets, internal supplementary collections, and pre-extracted skeletal landmark corpora utilized in SignTalk AI.

---

## 1. Overview

SignTalk AI maintains a strict, reproducible data registry. In accordance with the academic truthfulness protocol:
- No dataset is marked as downloaded, processed, or trained upon until raw assets are stored in local storage and verified via cryptographic checksums.
- Clear distinctions are maintained between isolated gesture datasets (used for feature extractor pretraining and baselines) and continuous sign corpora (used for spatial-temporal sequence translation).
- All external datasets must have documented origins, licenses, access mechanisms, and compatibility assessments.

---

## 2. Directory Structure Conventions

When datasets are fetched during Phase 1 Part 2 and Phase 2, they will be organized according to the following canonical schema:

```
/data/
├── README.md                      # This documentation file
├── dataset_registry.csv           # Authoritative machine-readable registry of candidate datasets
├── raw/                           # Unprocessed external video assets (excluded from git)
│   ├── include_50/
│   ├── isl_csltr/
│   └── isl_translate/
├── landmarks/                     # Pre-extracted MediaPipe coordinate parquet / HDF5 files
│   ├── include_50_landmarks/
│   └── isl_csltr_landmarks/
├── splits/                        # Reproducible train / validation / test partition indices
│   ├── include_50_splits.json
│   └── isl_csltr_splits.json
└── supplementary/                 # Internal supplementary recordings collected under ethical consent
    ├── recordings/
    ├── metadata.csv
    └── consent_forms/
```

> [!CAUTION]
> **Data Privacy & Git Hygiene:** Raw video files, biometric facial landmarks of human subjects, and participant consent forms must **never** be committed to version control. Add `/data/raw/` and `/data/supplementary/recordings/` to `.gitignore`.

---

## 3. Dataset Registry Summary

The authoritative registry is tracked in [`dataset_registry.csv`](file:///d:/SignAI/data/dataset_registry.csv).

Current Candidate Status Summary:
- **INCLUDE / INCLUDE-50:** Primary Candidate for isolated sign baseline benchmarking and ST-GCN spatial feature pretraining.
- **ISL-CSLTR:** Primary Candidate for continuous phrase translation MVP (700 sentences, 1,036 words, CC BY 4.0 license).
- **ISLTranslate:** Primary Candidate for continuous sequence-to-sequence translation scaling (31,117 pairs).
- **iSign:** Secondary Candidate for multi-domain benchmark scale-up.
- **ISLRTC Open Dictionary:** Supplementary Candidate for lexical verification and standardized sign morphology.
- **RWTH-PHOENIX-Weather-2014T & WLASL:** Marked *Not Suitable* for primary training (used solely for external literature comparison).
