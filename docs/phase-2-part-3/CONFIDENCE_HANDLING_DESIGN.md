# Confidence Handling Design: SignTalk AI

**Document ID:** STAI-P2P3-036  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** Lead AI Architect & Safety Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Dual-Layer Confidence Architecture

SignTalk AI dissociates confidence into two distinct operational layers:
1. **Layer 1: Input Data Quality Confidence ($Q_{in}$)**  
   Evaluates physical landmark visibility, tracking continuity, and occlusion.
2. **Layer 2: Model Prediction Confidence ($P_{model}$)**  
   Evaluates softmax posterior certainty emitted by neural classifiers in Phase 3.

```
Incoming Sliding Window
           │
           ▼
[Layer 1: Input Data Quality Check]
  ├─ If Q_in < 0.35 ────────► [SUPPRESS INFERENCE] (UI: "Occlusion / Check Lighting")
  └─ If Q_in >= 0.35 ───────► [Execute Neural Network Forward Pass]
                                    │
                                    ▼
                      [Layer 2: Model Posterior Check]
                        ├─ If P >= 0.70 ────────► [COMMIT TRANSLATION]
                        ├─ If 0.45 <= P < 0.70 ─► [HOLD STATE / STABILIZE]
                        └─ If P < 0.45 ─────────► [EMIT UNKNOWN / IDLE]
```

---

## 2. Decision Logic Matrix

| Input Quality Score ($Q_{in}$) | Model Posterior ($P_{model}$) | System State | UI Action |
| :---: | :---: | :--- | :--- |
| $< 0.35$ | *Bypassed* | `TRACKING_DEGRADED` | Displays visual prompt to adjust camera/lighting; suppresses prediction. |
| $\ge 0.35$ | $< 0.45$ | `UNKNOWN_GESTURE` | Displays neutral standby indicator; does not append text. |
| $\ge 0.35$ | $0.45 \le P < 0.70$ | `LOW_CERTAINTY_HOLD` | Retains previous translation token or displays provisional italicized preview. |
| $\ge 0.35$ | $\ge 0.70$ | `COMMITTED_PREDICTION`| Emits high-confidence sign translation to active output text box. |

---

## 3. Preservation of Data Fields in Manifest

In sequence generation, the sequence manifest (`sequence_manifest.csv`) embeds:
- `quality_score`: Float $[0.0, 1.0]$ representing $Q_{in}$.
- `quality_status`: String (`GOOD`, `ACCEPTABLE`, `REVIEW`, `REJECT`).
- Future model evaluation scripts in Phase 3 will populate `prediction_confidence` dynamically at inference time without data tampering.
