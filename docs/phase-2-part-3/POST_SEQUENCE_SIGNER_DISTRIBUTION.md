# Post-Sequence Signer Distribution: SignTalk AI

**Document ID:** STAI-P2P3-022  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** ML Data Engineer & Research Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Signer Representation Overview

To ensure that the sequence dataset does not introduce demographic or morphological bias toward any individual participant, we audit signer representation following sequence construction.

---

## 2. Signer Breakdown Across Evaluation Partitions

| Signer Identifier | Train Sequences | Val Sequences | Test Sequences | Total Sequences | Proportion | Representation Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `signer_01` | 12 | 4 | 4 | 20 | 33.33% | Perfectly Balanced |
| `signer_02` | 12 | 4 | 4 | 20 | 33.33% | Perfectly Balanced |
| `signer_03` | 12 | 4 | 4 | 20 | 33.33% | Perfectly Balanced |
| **Total** | **36** | **12** | **12** | **60** | **100.00%** | **Uniform Parity** |

---

## 3. Classes Articulated per Signer

Every signer executed exactly 2 independent takes of each of the 10 vocabulary classes:
- $10\text{ classes} \times 2\text{ repetitions} = 20\text{ sequences per signer}$.
- Class coverage per signer: $10 / 10$ ($100.0\%$).
- Zero signer-specific vocabulary gaps.

---

## 4. Assessment of Overlap and Disproportion

1. **No Participant Skew:** Each participant contributes exactly one-third of the total dataset.
2. **Homogeneous Partitioning:** The $60/20/20$ partition ratio is maintained strictly at the individual signer level (12 train / 4 val / 4 test for every signer).
3. **Implication for Model Generalization:** The model will encounter diverse arm lengths, hand spans, and signing tempos during training while evaluating on distinct unseen video performances from the same demographic pool.
