# Background and Unknown Gesture Strategy: SignTalk AI

**Document ID:** STAI-P2P3-037  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead AI Architect & Safety Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. The Background & "Null Class" Challenge

In closed-set isolated sign benchmarks (such as AI4Bharat INCLUDE-50), every recorded clip corresponds to a valid sign class $[0..K-1]$. The dataset contains zero explicit "idle", "resting", or "arbitrary conversational gesture" samples.

However, in continuous real-world webcam deployment, users spend substantial periods:
- Resting hands at their sides or in their lap.
- Adjusting spectacles, scratching their face, or drinking water.
- Executing transitional motion between distinct lexical signs.

If an isolated classifier is deployed without an explicit background rejection mechanism, it will falsely force every non-sign gesture into one of the 10 vocabulary classes.

---

## 2. Multi-Stage Background Mitigation Strategy

To overcome the absence of explicit background training samples, SignTalk AI designs a 3-tier rejection architecture:

```
Sliding Window Input
         │
         ▼
[Stage 1: Kinematic Energy Gating]
  ├─ If Hand Speed ||v_wrist|| < 0.02 (Resting) ──────► [IDLE / STANDBY]
  └─ If Hand Speed >= 0.02 (Active Gesture) ──────────► [Proceed to Stage 2]
                                                              │
                                                              ▼
                                              [Stage 2: Neural ST-GCN Model]
                                                              │
                                                              ▼
                                              [Stage 3: Out-of-Distribution Gating]
                                                ├─ If Softmax Entropy H(P) > 1.8 ──► [UNKNOWN / REJECT]
                                                ├─ If Max Posterior P_max < 0.45 ──► [UNKNOWN / REJECT]
                                                └─ If P_max >= 0.70 ──────────────► [VALID SIGN DETECTED]
```

---

## 3. Explicit Data Gap Identification

> [!WARNING]
> **Data Gap Documentation:**  
> The absence of curated negative / non-sign gesture recordings in the INCLUDE-50 benchmark represents a formal data limitation.  
> In Phase 4 (System Integration), we recommend collecting an auxiliary "Negative / Idle" dataset comprising resting poses, grooming motions, and natural non-signing communication to train an explicit background rejection head.
