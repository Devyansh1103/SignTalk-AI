# Dataset Versioning Specification: SignTalk AI

**Document ID:** STAI-P2P3-030  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** ML Data Engineer & System Architect  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Lineage & Provenance Graph

```
Raw Video Archive (AI4Bharat INCLUDE-50 Subset)
                  │
                  ▼
Frame Extraction & Landmark Tracking (MediaPipe Tasks Vision 1.0.1)
                  │
                  ▼
Preprocessed Landmark Dataset: 'landmarks-v1.0' (T=45, V=93)
                  │
                  ▼
Temporal Standardization, Quality Filtering & Vocabulary Tokenization
                  │
                  ▼
Canonical Sequence Dataset: 'signTalk-seq-v1.0.0'
```

---

## 2. Version Release Card: `signTalk-seq-v1.0.0`

| Attribute | Specification | Verification Checksum / Reference |
| :--- | :--- | :--- |
| **Dataset Identifier** | `signTalk-seq-v1.0.0` | Registered in master manifest |
| **Parent Landmark Version**| `landmarks-v1.0` | [`docs/phase-2-part-2/`](file:///d:/SignAI/docs/phase-2-part-2/) |
| **Source Benchmark** | AI4Bharat INCLUDE-50 | ACM MM 2020 |
| **Vocabulary Size** | 10 classes | `assets/vocabularies/mvp_10.json` |
| **Total Sequences** | 60 canonical sequences | `data/manifests/sequence_manifest.csv` |
| **Train Sequences** | 36 sequences ($60.0\%$) | `data/manifests/train.csv` |
| **Validation Sequences** | 12 sequences ($20.0\%$) | `data/manifests/val.csv` |
| **Test Sequences** | 12 sequences ($20.0\%$) | `data/manifests/test.csv` |
| **Rejected Sequences** | 16 sequences ($26.7\%$) | `data/manifests/rejected_sequences.csv` |
| **Signer Count** | 3 deaf signers | `signer_01`, `signer_02`, `signer_03` (20 each) |
| **Sequence Dimensions** | $[C, T, V] = [3, 45, 93]$ | PyTorch float32 tensor |
| **Frame Rate** | $25.0\text{ FPS}$ | $1.80\text{ seconds duration}$ |
| **Adjacency Matrix** | $\mathbf{\Lambda} \in \mathbb{R}^{3 \times 93 \times 93}$ | `assets/graphs/kinematic_adjacency_93.npy` |
| **Release Date** | October 2026 | SignTalk AI Phase 2 Part 3 Finalization |
| **License** | Non-Commercial Academic Research License | Adheres to INCLUDE dataset license |
