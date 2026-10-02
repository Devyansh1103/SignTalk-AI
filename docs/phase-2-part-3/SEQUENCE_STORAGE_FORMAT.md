# Sequence Storage Format Specification: SignTalk AI

**Document ID:** STAI-P2P3-028  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead ML Data Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Storage Evaluation Matrix

Before selecting the persistent serialization format for sequence objects, we evaluated four candidate formats across five operational criteria:

| Candidate Format | Random Access Latency | Compression Ratio | PyTorch Native Support | Platform Portability | Manifest Separation | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **NumPy `.npz` (Compressed)** | **< 1.0 ms** | **High ($35.6\times$)** | **Direct via `np.load`** | **Universal** | **Optimal with CSV** | **SELECTED** |
| **HDF5 (`.h5`)** | < 0.5 ms | Medium ($15\times$) | Requires `h5py` wrapper | Complex multi-threading | Embedded metadata | Avoided (Lock contention) |
| **Apache Parquet** | ~ 2.0 ms | Very High ($40\times$) | Requires tensor flattening | Universal | Integrated | Avoided (Nested tensor overhead) |
| **Raw `.pt` Tensors** | < 0.8 ms | Low ($8\times$) | Native `torch.load` | PyTorch-only | Poor | Avoided (Version incompatibility) |

---

## 2. Selected Architecture: Compressed NumPy Archives (`.npz`) + Canonical CSV Manifests

### 2.1 File Organization
```
data/
├── manifests/
│   ├── sequence_manifest.csv       # Master sequence catalog (60 entries)
│   ├── train.csv                   # Train partition manifest (36 entries)
│   ├── val.csv                     # Validation partition manifest (12 entries)
│   ├── test.csv                    # Test partition manifest (12 entries)
│   └── rejected_sequences.csv      # Audit trail for filtered samples (16 entries)
└── processed/
    └── sequences/
        ├── train/
        │   ├── seq_0001.npz
        │   └── ...
        ├── val/
        │   ├── seq_0037.npz
        │   └── ...
        └── test/
            ├── seq_0049.npz
            └── ...
```

### 2.2 Storage Footprint
- **Total Raw Video:** 62.95 MB
- **Total Processed Sequences (`.npz`):** 1.84 MB
- **Compression Efficiency:** $34.2\times$ reduction in disk storage.
- **Zero Video Decoding at Training Time:** Training completely eliminates CPU-bound FFmpeg/OpenCV video decompression, shifting pipeline throughput from 0.22 video/sec to over 50 sequences/sec.
