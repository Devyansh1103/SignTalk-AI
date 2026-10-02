# 34. Open Technical Decisions & Phase 2 Prerequisite Checklist: SignTalk AI

**Document ID:** STAI-P1P2-034  
**Project Name:** SignTalk AI  
**Document Status:** Approved Engineering Blueprint (Phase 1 — Part 2)  

---

## 1. Decision Status Overview

Phase 1 Part 2 provides the complete technical blueprint. Before Phase 2 ("Dataset Acquisition & Ingestion") executes full-scale data downloads and training routines, five specific operational decisions must be locked:

```mermaid
graph TD
    D1[1. Local Host GPU Hardware Specifications]
    D2[2. Primary Training Dataset Confirmation]
    D3[3. 40-Point Facial Marker Latency Profiling]
    D4[4. Multilingual Hindi Display Priority]
    D5[5. Institutional Ethical Review Horizon]
```

---

## 2. Open Technical Decisions Ledger

### DEC-OPEN-01: Development Workstation GPU & Acceleration Target
* **Context:** Dictates batch size ($B=32$ vs $B=64$), training epoch duration, and PyTorch execution target (`cuda` vs. `cpu`).
* **Current Default Blueprint:** Architecture defaults to `INFERENCE_DEVICE=cpu` for maximum portability, but training scripts auto-detect CUDA:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
* **Required Input from User:** Specific local GPU model and VRAM (e.g., RTX 3060 6GB, RTX 4060 8GB, or CPU-only).

### DEC-OPEN-02: Primary Training Dataset Lock
* **Context:** Choice between lightweight local baseline (`INCLUDE-50` + `ISL-CSLTR`) and large-scale academic corpus (`ISLTranslate`).
* **Current Blueprint Recommendation:** Lock **INCLUDE-50** (isolated, 50 classes) for baseline and ST-GCN pretraining, and **ISL-CSLTR** (continuous, 700 sentences) for continuous Transformer fine-tuning.
* **Status:** Recommended for approval.

### DEC-OPEN-03: Salient Facial Node CPU Latency Confirmation
* **Context:** The blueprint specifies a 93-node multimodal graph ($N=53$ hands+pose + 40 face points). If MediaPipe Face Mesh exceeds $15\text{ ms}$ on local student hardware, fallback to the 53-node graph is triggered.
* **Resolution Action:** Phase 2 will execute a 100-frame CPU benchmark script on the host machine to make the final determination.

### DEC-OPEN-04: Target Output Language Granularity
* **Context:** Evaluates whether bilingual Hindi text is required for academic viva defense.
* **Current Blueprint Recommendation:** English text output is the primary academic target. Hindi translation is supported via a secondary frontend dictionary mapping without complicating the core sign-to-text sequence model.

### DEC-OPEN-05: Institutional Review Board (IRB) Supplementary Data Scope
* **Context:** Determines whether live campus student recordings will be conducted under university ethics review, or if evaluation will rely entirely on verified public benchmarks.
* **Current Blueprint Recommendation:** Rely on published academic benchmarks (INCLUDE, ISL-CSLTR) for primary model training and viva defense, using demonstrative webcam test passes by the project team for live validation.
