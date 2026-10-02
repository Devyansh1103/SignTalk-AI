# Sequence Quality Scoring Specification: SignTalk AI

**Document ID:** STAI-P2P3-018  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead ML Data Engineer & Research Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Quality Scoring Formulation

To avoid arbitrary heuristic decisions, SignTalk AI evaluates sequence quality via a mathematically grounded, multi-factor visibility formulation:

$$Q_{seq} = w_{pose} \cdot r_{pose} + w_{dom} \cdot r_{dom} + w_{nondom} \cdot r_{nondom} + w_{face} \cdot r_{face}$$

where:
- $r_{pose} \in [0.0, 1.0]$: Upper-body pose landmark detection coverage.
- $r_{dom} \in [0.0, 1.0]$: Detection rate of the dominant hand (the hand executing the primary gesture stroke).
- $r_{nondom} \in [0.0, 1.0]$: Detection rate of the non-dominant hand.
- $r_{face} \in [0.0, 1.0]$: Facial anchor detection rate.

### 1.1 Configured Weights
- $w_{pose} = 0.40$: Guarantees continuous torso centering and shoulder scale normalization.
- $w_{dom} = 0.40$: Preserves primary lexical handshape and spatial trajectory signals.
- $w_{nondom} = 0.15$: Captures secondary bilateral interaction without penalizing one-handed signs.
- $w_{face} = 0.05$: Rewards non-manual facial marker presence.

$$\sum w_i = 0.40 + 0.40 + 0.15 + 0.05 = 1.00$$

---

## 2. Multi-Tier Classification Taxonomy

| Category | Quality Threshold | Valid Articulation Frames | Pose Detection Rate | Pipeline Action |
| :--- | :---: | :---: | :---: | :--- |
| **`GOOD`** | $Q_{seq} \ge 0.75$ | $\ge 15$ frames | $\ge 90.0\%$ | Unconditionally included in training and benchmarking. |
| **`ACCEPTABLE`** | $Q_{seq} \ge 0.50$ | $\ge 10$ frames | $\ge 80.0\%$ | Included in standard training and validation. |
| **`REVIEW`** | $0.35 \le Q_{seq} < 0.50$ | $\ge 8$ frames | Any | Retained; flagged for manual validation or adversarial testing. |
| **`REJECT`** | $Q_{seq} < 0.35$ | $< 8$ frames | Any | Excluded from primary training; archived in `rejected_sequences.csv`. |

---

## 3. Rejection & Archival Policy

In accordance with Phase 2 rules, rejected samples are NEVER permanently deleted. Instead:
1. Every rejected sample is serialized to `data/processed/sequences/` with its status recorded as `REJECT`.
2. A dedicated rejection audit log is generated at [`data/manifests/rejected_sequences.csv`](file:///d:/SignAI/data/manifests/rejected_sequences.csv) recording:
   - `sequence_id`
   - `source_recording_id`
   - `quality_score`
   - `reason`
   - `valid_frames`
3. The PyTorch `SignSequenceDataset` loader provides a configurable flag `filter_rejects=True` to exclude rejected instances during standard model training while retaining them for stress testing.
