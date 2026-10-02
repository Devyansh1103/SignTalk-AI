# Project Decision Log: SignTalk AI

**Document ID:** STAI-DOC-P1P1-DEC-001  
**Project Name:** SignTalk AI  
**Document Status:** Official Architectural Decision Records (ADRs)  

---

## 1. Decision Log Protocol

The Decision Log records all major architectural, scientific, and engineering choices made across the SignTalk AI project lifecycle. 

Each entry follows the Architectural Decision Record (ADR) standard:
- **Decision ID:** Unique sequential identifier (`DEC-XXX`).
- **Date:** Date of formalization.
- **Decision:** Clear statement of the adopted architectural or technical choice.
- **Reason:** Engineering, academic, or ethical rationale driving the choice.
- **Evidence:** Peer-reviewed literature, benchmark data, or empirical testing supporting the choice.
- **Alternatives Considered:** Options evaluated and explicitly rejected.
- **Status:** Current operational state (`APPROVED`, `PROPOSED`, `DEFERRED`, `SUPERSEDED`).

> [!NOTE]
> **Integrity Guard:** Hypotheses, unverified design ideas, and open questions are not entered as `APPROVED` decisions. Only choices backed by technical justification and approved project boundaries are marked `APPROVED`.

---

## 2. Architectural Decision Records (ADRs)

| Decision ID | Date | Architectural Decision | Rationale & Engineering Justification | Authoritative Evidence Base | Alternatives Evaluated & Rejected | Current Status |
| :---: | :---: | :--- | :--- | :--- | :--- | :---: |
| **DEC-001** | Oct 2026 | **Exclusive Target on Indian Sign Language (ISL)** | Concentrate research strictly on ISL rather than attempting universal multi-language support. | Societal necessity in India; distinct linguistic grammar; avoidance of unmanageable cross-lingual dataset sprawl. | Supporting ASL and BSL simultaneously *(Rejected: Dilutes academic focus; ASL datasets do not generalize to ISL).* | **APPROVED** |
| **DEC-002** | Oct 2026 | **Standard Monocular RGB Webcam Input** | Rely entirely on consumer 2D webcams; eliminate depth sensors (Kinect) and wearable data gloves. | Real-world accessibility and zero-barrier deployment; hardware sensors restrict adoption in public counters. | Microsoft Kinect / Intel RealSense *(Rejected: Expensive, discontinued, unavailable in rural clinics)*; Wearable IMU data gloves *(Rejected: Intrusive, costly, degrades natural signing).* | **APPROVED** |
| **DEC-003** | Oct 2026 | **Local MediaPipe Skeletal Landmark Extraction** | Extract 3D skeletal landmarks locally in memory via MediaPipe Holistic; discard raw video immediately. | Radical reduction in bandwidth ($< 50\text{ KB/s}$ vs. $> 2\text{ MB/s}$); absolute preservation of patient/citizen visual biometric privacy. | Streaming raw RGB video to cloud vision server *(Rejected: Massive privacy violation under DPDP Act; network bandwidth bottleneck).* | **APPROVED** |
| **DEC-004** | Oct 2026 | **Spatial-Temporal Graph Convolutional Network (ST-GCN)** | Adopt ST-GCN as the core feature encoder to model anatomical bone connectivity and motion trajectories. | Biological kinematic graph provides strong structural inductive bias, preventing catastrophic overfitting on small ISL datasets. | Flattened 1D coordinate vectors fed to LSTM *(Rejected: Discards bone topology)*; Dense 3D-CNNs like I3D *(Rejected: Massive compute footprint, prone to background overfitting).* | **APPROVED** |
| **DEC-005** | Oct 2026 | **Transformer Sequence Translation for Sign-to-Text** | Couple ST-GCN encoder with an autoregressive Transformer decoder to synthesize fluent natural-language sentences. | Bridges linguistic word-order divergence between ISL (Subject-Object-Verb) and English (Subject-Verb-Object). | Isolated gloss concatenation without grammar decoder *(Rejected: Produces broken, unreadable gloss strings)*; CTC-only decoding *(Rejected: Inflexible for non-monotonic syntax reordering).* | **APPROVED** |
| **DEC-006** | Oct 2026 | **Curated 50-Class MVP Domain Vocabulary** | Restrict the Phase 1/2 implementation vocabulary to 50 high-impact classes across healthcare, emergencies, and civic desks. | Ensures academic feasibility within university timelines; matches verified classes in INCLUDE-50 and ISL-CSLTR. | Attempting full 10,000-word ISL dictionary *(Rejected: Impossible due to lack of training data and compute)*; 10-word toy demo *(Rejected: Lacks research value).* | **APPROVED** |
| **DEC-007** | Oct 2026 | **Stratified Signer-Independent Evaluation Splits** | Enforce strict partitioning where test signers never appear in the training split. | Guarantees that accuracy and BLEU metrics evaluate true generalization to novel signers rather than memorization. | Random frame-level shuffle splits *(Rejected: Massive data leakage; falsely inflates accuracy by $20\%-30\%$ across identical video frames).* | **APPROVED** |
| **DEC-008** | Oct 2026 | **Sub-500 ms End-to-End Latency Target Budget** | Establish a strict latency ceiling of 500 ms for the complete pipeline from visual capture to UI caption render. | Human conversational cadence breaks down when conversational latency exceeds 500 ms. | Unbounded batch inference *(Rejected: Incompatible with live interactive communication).* | **APPROVED** |
| **DEC-009** | Oct 2026 | **Confidence-Gated Low-Confidence Output Handling** | Explicitly display uncertainty warnings when prediction confidence falls below $\theta_{conf} = 0.50$. | Eliminates dangerous hallucinated text outputs in critical medical triage and administrative counters. | Emitting top-1 prediction unconditionally *(Rejected: High risk of false positive guidance in healthcare/banking).* | **APPROVED** |
| **DEC-010** | Oct 2026 | **Decoupled FastAPI + React / Web Architecture** | Implement a decoupled architecture communicating over low-latency WebSockets. | Enables clean separation of computer vision / ML inference from UI rendering; facilitates future on-device edge migration. | Monolithic desktop application (Tkinter/PyQt) *(Rejected: Poor accessibility, inflexible deployment)*; Pure cloud microservice *(Rejected: Violates local privacy).* | **APPROVED** |
| **DEC-011** | Oct 2026 | **Salient 40-Point Facial Marker Filtering** | Filter MediaPipe Face Mesh from 468 points down to 40 salient points covering eyebrows, lips, and jaw contours. | Captures essential grammatical non-manual markers without incurring the $16\times$ graph convolution compute penalty. | Full 468-point face mesh *(Rejected: Exceeds CPU latency budget)*; Omitting face completely *(Rejected: Loses question-marking and negation grammar).* | **PROPOSED**<br>*(Pending Phase 2 Benchmark)* |
| **DEC-012** | Oct 2026 | **Secondary Bilingual Hindi Output Generation** | Add Hindi translation as an optional secondary tokenization layer. | Enhances accessibility for non-English speaking citizens at public service counters across India. | Translating to English only *(Current Phase 1 target)*; Translating directly to 22 scheduled languages *(Rejected: Far exceeds project timeline).* | **PROPOSED**<br>*(Phase 2 Evaluation)* |
| **DEC-013** | Oct 2026 | **Bidirectional 3D Signing Avatar Generation** | Synthesize text-to-ISL visual gestures using a 3D animated virtual avatar. | Complete two-way communication parity. | Sign-to-Text unidirectional pipeline *(Current Scope)*. | **FUTURE SCOPE**<br>*(Excluded from current release)* |
