# Reproducibility Specification: SignTalk AI

**Document ID:** STAI-P2P3-031  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** ML Data Engineer & Research Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Computational Environment

All dataset transformations and validation checks were executed in the following verified environment:

- **Operating System:** Microsoft Windows 11 Enterprise (AMD64 architecture)
- **Python Version:** 3.13.12 (Miniconda distribution)
- **Core ML Framework:** PyTorch 2.13.0
- **Computer Vision:** OpenCV (`opencv-python`) 5.0.0.93
- **Landmark Engine:** Google MediaPipe Tasks Vision 1.0.1
- **Numerical Libraries:** NumPy 2.2.6, Pandas 2.3.2, SciPy 1.15.2
- **Hardware Platform:** Intel Core i7 / 16 GB DDR4 RAM

---

## 2. Deterministic Reproduction Protocol

The sequence dataset `signTalk-seq-v1.0.0` can be deterministically reproduced from scratch using the following single command:

```bash
python scripts/reproduce_sequences.py
```

### 2.1 Step-by-Step Execution Sequence
1. **Adjacency Matrix Generation:**
   ```bash
   python -c "from src.data.stgcn_tensor import build_spatial_adjacency_matrix; import numpy as np; np.save('assets/graphs/kinematic_adjacency_93.npy', build_spatial_adjacency_matrix())"
   ```
2. **Sequence Dataset Serialization:**
   ```bash
   python scripts/build_sequences.py --config configs/sequence_generation.yaml
   ```
3. **Partition Isolation & Leakage Verification:**
   ```bash
   python scripts/validate_sequence_splits.py --manifest data/manifests/sequence_manifest.csv
   ```
4. **PyTorch DataLoader Integrity Test:**
   ```bash
   python scripts/test_dataloader.py --manifest data/manifests/sequence_manifest.csv
   ```

---

## 3. Seed & Parameter Registry

- Global Random Seed: `42`
- Target FPS: `25.0`
- Target Sequence Length: `45` frames ($1.80\text{ s}$)
- Graph Topology: `93` nodes, `102` kinematic bone edges
- Normalization: Sequence-smoothed torso-center and shoulder Euclidean distance
