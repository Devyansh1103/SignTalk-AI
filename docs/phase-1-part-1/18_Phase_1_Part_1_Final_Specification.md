# 18. Phase 1 — Part 1 Consolidated Final Project Specification

**Document ID:** STAI-DOC-P1P1-018  
**Project Name:** SignTalk AI  
**Project Title:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Technical Subtitle:** A Spatial-Temporal Graph and Transformer-Based Sign-to-Text System  
**Lead AI/ML Research Engineer & Product Architect:** Academic Engineering Team (GLA University / SignTalk AI Project)  
**Document Status:** Approved Master Specification (End of Phase 1 — Part 1)  

---

```
================================================================================
                       SIGNTALK AI: MASTER SPECIFICATION
================================================================================
```

### PROJECT:
**SignTalk AI** (SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform)

### PRIMARY LANGUAGE:
**Indian Sign Language (ISL)** exclusively. *(American Sign Language [ASL] and British Sign Language [BSL] are strictly excluded from current implementation).*

### INPUT:
- Continuous monocular 2D RGB video stream captured via standard consumer webcam (720p/1080p at $\ge 20\text{ FPS}$).
- Zero requirement for specialized depth sensors, infrared hardware, or wearable data gloves.

### PROCESSING:
1. **Local Vision Ingestion:** Real-time frame capture and buffering via OpenCV / browser Web APIs.
2. **Local Landmark Extraction:** Google MediaPipe Holistic extracting 3D $(x, y, z)$ keypoints:
   - Left Hand: 21 landmarks
   - Right Hand: 21 landmarks
   - Upper-Body Pose: 11 salient keypoints (shoulders, elbows, wrists, nose, eyes, ears)
   - Salient Facial Non-Manual Markers: Curated 40-point subset (eyebrows, lips, jawline)
3. **Normalization & Fusion:** Torso-centric spatial centering with shoulder-width scale invariance; raw video frames purged immediately from volatile memory.
4. **Temporal Sequence Assembly:** Rolling temporal sliding window of $T \in [30, 60]$ frames with stride $S \in [5, 10]$ frames.
5. **Spatial-Temporal Graph Modeling (ST-GCN):** Graph convolution along anatomical kinematic edges ($\mathcal{E}_{spatial}$) and cross-frame joint trajectory edges ($\mathcal{E}_{temporal}$).
6. **Sequence Translation (Transformer):** Autoregressive Transformer decoder mapping spatial-temporal graph embeddings into natural-language English token sequences.
7. **Post-Processing & Rendering:** Confidence calibration, low-confidence threshold gating ($\theta_{conf} = 0.50$), and low-latency WebSocket streaming to web client.

### OUTPUT:
- Large-font, high-contrast real-time natural-language English text captions.
- Visual confidence indicator (Green: $\ge 0.75$, Amber: $0.50 - 0.74$, Red / Repeat prompt: $< 0.50$).
- Real-time performance metrics (FPS, inference latency).
- Transient session text transcript.

### CORE AI:
- **Feature Backbone:** Google MediaPipe Holistic (Client-side / local Python execution).
- **Spatial-Temporal Encoder:** Custom Spatial-Temporal Graph Convolutional Network (ST-GCN; 6 ST-GCN residual blocks; $\approx 2.5\text{M}$ parameters).
- **Sequence Translator:** Lightweight 3-layer Transformer Decoder with multi-head self-attention and cross-attention ($n_{heads} = 4, d_{model} = 256, d_{ff} = 1024$).
- **Serving Architecture:** Python FastAPI backend communicating via low-latency bi-directional WebSockets.

### TARGET USERS:
1. Deaf and hard-of-hearing native ISL signers seeking independent communication.
2. Hearing non-signers across customer service, retail, and transit environments.
3. Healthcare professionals (triage nurses, emergency physicians) conducting clinical intake.
4. Civic administrative officers at government service desks (Aadhaar, pensions, municipal counters).
5. Educators and hearing students in inclusive academic classrooms.
6. First responders and emergency personnel in field crisis triage.

### MVP (ACADEMIC MINIMUM VIABLE PRODUCT):
- **Vocabulary Scope:** 50 curated, linguistically verified ISL signs and phrases covering Healthcare Triage, Emergency Distress, Civic Desks, and Daily Greetings.
- **Continuity Scope:** Translation of continuous multi-sign phrases (e.g., *"I have fever since yesterday"*, *"Please help, take me to hospital"*).
- **Runtime Environment:** 100% self-contained local execution (React/JS frontend + FastAPI backend) on standard consumer PC without cloud dependencies.
- **Latency Budget:** End-to-end processing latency $\le 500\text{ ms}$ at $\ge 20\text{ FPS}$.

### OUT OF SCOPE:
- Universal unrestricted ISL translation across the entire 10,000+ sign lexicon.
- American Sign Language (ASL), British Sign Language (BSL), or other non-ISL sign languages.
- Bidirectional speech/text-to-sign 3D animated avatar generation.
- Autonomous medical diagnosis or legally binding courtroom interpretation.
- Cloud-based raw video streaming or persistent biometric facial recording storage.
- Native mobile app binary distribution (iOS/Android) in the initial release.

### RESEARCH CONTRIBUTION:
1. Designing an anatomically grounded multimodal graph schema combining dual-hand kinematics, upper-body pose anchors, and a computationally efficient 40-point facial non-manual subset tailored for ISL.
2. Empirically evaluating whether spatial-temporal graph convolutions act as an effective structural regularizer against overfitting on low-resource continuous ISL datasets compared to standard vector sequence encoders.
3. Demonstrating an end-to-end continuous sign-to-text pipeline that couples spatial-temporal graph representations with Transformer sequence decoding under a sub-500 ms latency budget on consumer-grade hardware.
4. Benchmarking a privacy-preserving edge architecture that streams lightweight skeletal coordinates ($< 50\text{ KB/s}$) rather than privacy-invasive raw video ($> 2\text{ MB/s}$).

### DATA STRATEGY:
- **Baseline & ST-GCN Pretraining:** AI4Bharat INCLUDE-50 dataset (50 isolated word signs, 854 videos, 7 native Deaf signers, ACM Multimedia 2020).
- **Continuous Phrase MVP Benchmark:** Mendeley ISL-CSLTR dataset (700 continuous sentence videos, 1,036 word vocabulary, CC BY 4.0 license).
- **Scale-Up Corpus (Phase 2):** IIT Kanpur ISLTranslate dataset (~31,117 video-sentence pairs, Findings of ACL 2023).
- **Linguistic Standardization:** Indian Sign Language Research and Training Centre (ISLRTC) official national dictionary (3,000+ words).
- **Supplementary Recordings:** Tier B research protocol (50 classes, 6–8 signers, controlled environmental variations, institutional ethical consent).

### SUCCESS METRICS:
- **Isolated Classification (INCLUDE-50):** Top-1 Accuracy $\ge 85.0\%$, Top-5 Accuracy $\ge 95.0\%$, Macro F1 $\ge 82.0\%$.
- **Continuous Translation (ISL-CSLTR):** BLEU-4 $\ge 22.0$, ROUGE-L $\ge 40.0$, Sentence WER $\le 30.0\%$, BERTScore $\ge 0.80$.
- **System Latency:** End-to-end latency $< 500\text{ ms}$, Throughput $\ge 25\text{ FPS}$, Resident RAM $< 2.0\text{ GB}$.
- **Robustness:** Signer-independent test split performance drop $\le 15\%$; lighting degradation drop $\le 20\%$.
- **Usability:** System Usability Scale (SUS) $\ge 75/100$; Task Completion Rate $\ge 85\%$.

### MAJOR RISKS & MITIGATIONS:
1. *Scarcity of continuous ISL training data* $\rightarrow$ Mitigated by pretraining ST-GCN feature extractor on isolated INCLUDE dataset, applying spatial-temporal augmentations, and utilizing ISL-CSLTR.
2. *Real-time latency budget overrun ($> 500\text{ ms}$)* $\rightarrow$ Mitigated by sliding window striding ($S=5$), greedy/narrow-beam Transformer decoding, and INT8 ONNX runtime quantization.
3. *Hand self-occlusion during bilateral signing* $\rightarrow$ Mitigated by ST-GCN temporal kinematic edges that propagate joint momentum across occluded frames, coupled with One-Euro temporal smoothing.
4. *Scope creep beyond academic timeline* $\rightarrow$ Mitigated by locking the 50-class MVP scope in Phase 1 Part 1 and strictly separating Future Scope items.

### FUTURE SCOPE:
- Multilingual translation output directly synthesizing Hindi, Marathi, and Tamil text captions.
- Bidirectional communication incorporating a 3D animated signing avatar for text-to-ISL synthesis.
- On-device edge deployment executing natively in browser runtimes via WebGPU / WebAssembly.
- Progressive Web App (PWA) packaging for smartphone accessibility.
- Active learning interface enabling certified ISL signers to contribute validated training traces.
