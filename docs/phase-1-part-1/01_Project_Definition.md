
# 01. Project Definition: SignTalk AI

**Document ID:** STAI-DOC-P1P1-001  
**Project Name:** SignTalk AI  
**Project Title:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Technical Subtitle:** A Spatial-Temporal Graph and Transformer-Based Sign-to-Text System  
**Lead AI/ML Research Engineer & Product Architect:** Academic Engineering Team (GLA University / SignTalk AI Project)  
**Target Domain:** AI / Machine Learning / Computer Vision / Natural Language Processing / Assistive Technology  
**Primary Target Language:** Indian Sign Language (ISL)  
**Primary Output Modality:** Natural-Language Text Captions  
**Date of Document:** October 2026  
**Document Status:** Approved Research Specification (Phase 1 — Part 1)  

---

## 1. Project Truthfulness & Status Classification

In accordance with strict academic integrity rules, every subsystem, dataset, model, and metric in SignTalk AI is classified under one of four explicit operational states:

| Status Tag | Operational Definition | Current Applicability in SignTalk AI |
| :--- | :--- | :--- |
| **`IMPLEMENTED`** | Code is written, tested, benchmarked, and verified in this repository. | *None in Phase 1 Part 1 (Strictly design & research phase).* |
| **`PLANNED`** | Concrete architecture, component specification, and execution plan defined for Phase 1 Part 2 / Phase 2. | MediaPipe landmark extraction, normalization pipeline, baseline LSTM classifier, ST-GCN graph encoder, Transformer sequence translator, FastAPI backend, React UI. |
| **`PROPOSED`** | Architectural concept or feature under theoretical evaluation; requires experimental validation before commitment. | Dynamic graph adjacency learning, multimodal feature fusion with 40-keypoint facial mesh subset, WebRTC low-latency streaming pipeline, offline ONNX quantization. |
| **`FUTURE SCOPE`** | Deliberately excluded from the current academic MVP/v1.0 release; deferred to subsequent development phases. | Bidirectional text-to-ISL 3D avatar generation, American Sign Language (ASL) / British Sign Language (BSL) support, native iOS/Android mobile apps, speech-to-ISL synthesis, multi-signer conversational tracking. |

> [!IMPORTANT]
> **Academic Integrity Rule:** No accuracy figure, inference speed, frame rate, dataset dimension, or survey result is treated as an achieved fact unless backed by verified experimental logs or cited peer-reviewed publications.

---

## 2. Context and Motivation

Indian Sign Language (ISL) is the primary visual-gestural language utilized by millions of Deaf and hard-of-hearing individuals across the Indian subcontinent. Despite its linguistic richness and formal recognition (promoted nationally by the Indian Sign Language Research and Training Centre - ISLRTC), severe societal communication barriers persist between ISL signers and the predominantly hearing, non-signing public.

These friction points are especially pronounced in high-stakes environments:
- **Healthcare:** Describing symptoms, medical history, dosages, and emergency triage.
- **Banking and Financial Services:** Conducting confidential transactions, KYC verification, and loan processing.
- **Public Administration & Legal Services:** Submitting official grievances, civil registrations, and police inquiries.
- **Education:** Mainstream classroom integration for Deaf students without dedicated educational interpreters.
- **Daily Commercial & Transit Interactions:** Counter interactions at transit hubs, retail stores, and customer desks.

### 2.1 Why Existing Communication Methods Fall Short
1. **Human Interpreters:** Professional ISL interpreters in India are in critically short supply (estimated in academic and governmental reports at fewer than a few hundred certified interpreters for a Deaf population numbering several million). Furthermore, human interpretation is logistically difficult to schedule for spontaneous daily needs, expensive, and compromises personal privacy during medical consultations and banking.
2. **Pen and Paper / Written Text:** Written language (e.g., written English or Hindi) is not the native language of Deaf individuals whose first language is ISL. ISL has an autonomous grammar distinct from spoken Indian languages. Consequently, written interaction creates cognitive fatigue, severe friction, and frequently leads to miscommunication.
3. **Smart Keyboard / Manual Typing:** Typing requires signers to abandon their natural visual-gestural mode of expression and transition to linear textual keyboards, which fails during dynamic real-time exchanges.
4. **Isolated Hand-Gesture Classifiers:** Prior hobbyist and introductory academic projects commonly classify isolated static hand poses (e.g., finger-spelling the English alphabet 'A' through 'Z'). However, natural human communication does not occur in isolated static characters; natural ISL consists of continuous fluid phrases.

---

## 3. Technical Complexity of Continuous ISL Translation

Automated sign language translation (SLT) is substantially more difficult than standard computer vision or spoken language translation tasks due to several compounding technical factors:

### 3.1 Continuous Signing vs. Isolated Gesture Recognition
- **Co-articulation (Movement Epenthesis):** In continuous signing, the trajectory and hand posture entering and exiting a sign are heavily influenced by the preceding and succeeding signs. Isolated models trained on isolated snapshots fail completely on continuous streams because there are no physical pauses between words.
- **Absence of Natural Delimiters:** Unlike speech (which contains acoustic pauses) or text (which contains spaces and punctuation), continuous video contains no explicit delimiter between lexical tokens.
- **Signer Speed and Trajectory Variance:** Variations in signing velocity, signing radius, and emotional expressiveness cause non-linear temporal deformations across the video stream.

### 3.2 The Critical Role of Spatial Relationships
ISL relies heavily on spatial syntax:
- Signs are articulated within a 3D signing space extending from the waist to above the head, and across the shoulders.
- The relative coordinate position of the hands with respect to anatomical anchors (forehead, temple, chin, chest, contralateral shoulder, and non-dominant hand) encodes distinct lexical meanings. A hand shape executed at the forehead signifies a completely different concept when articulated at the chest.
- Dual-hand coordination involves complex spatial interplay where the non-dominant hand often acts as a base or grammatical reference while the dominant hand executes the movement.

### 3.3 The Critical Role of Temporal Movement
Signs are fundamentally dynamic spatio-temporal trajectories:
- Hand orientation, finger flexion, linear trajectories, circular motions, and abrupt stops distinguish minimal lexical pairs.
- Repetition frequency and movement dynamics often indicate grammatical nuances such as aspect, pluralization, or intensity.

### 3.4 Non-Manual Markers (Facial Expressions and Body Posture)
Linguistically, sign language is multimodal:
- **Facial Cues:** Eyebrow raising/furrowing, eye aperture changes, and mouth movements (mouthing / mouth gestures) differentiate statements from questions (polar questions vs. wh-questions), convey negation, and modulate semantic intensity.
- **Body Posture:** Torso tilts and shoulder shifts establish conversational perspective (role shifting) and contrastive references in the spatial discourse.
- Vision pipelines that rely exclusively on hand keypoints lose these critical grammatical markers.

### 3.5 Real-Time Processing Demands
Conversational exchange necessitates immediate feedback. In live human communication, delays exceeding 500 ms disrupt interaction cadence, introduce conversational collisions, and degrade user trust. Achieving real-time inference requires a streamlined architecture capable of extracting multimodal landmarks, constructing spatial-temporal graphs, and autoregressively or non-autoregressively decoding text within tight latency budgets on standard consumer hardware.

### 3.6 Privacy, Bandwidth, and Offline Viability
Standard deep learning architectures that stream full-resolution video to cloud-hosted computer vision backends introduce significant vulnerabilities:
- **Severe Privacy Invasions:** Capturing and transmitting uncompressed facial and environmental video of signers during confidential medical examinations or banking transactions violates basic privacy norms and statutory data protection frameworks (such as the Digital Personal Data Protection Act - DPDP in India).
- **Network Dependency and Bandwidth Overhead:** Continuous video streaming requires sustained upstream bandwidth (1.5–3.0 Mbps per client), making cloud-dependent systems brittle in rural or low-connectivity Indian regions.
- **Edge / On-Device Viability:** By extracting compact skeletal coordinate landmarks locally on client devices (~few kilobytes per second) or performing end-to-end local inference, privacy is maintained, bandwidth requirements drop by orders of magnitude, and offline operation becomes technically feasible.

---

## 4. Formal Problem Statements

### A. One-Sentence Problem Statement
> **SignTalk AI addresses the communication barrier between Indian Sign Language (ISL) users and non-signers by developing a real-time, privacy-preserving computer vision and deep learning system that translates continuous ISL video into natural-language text captions using spatial-temporal graph neural networks and Transformer sequence modeling.**

### B. Short Problem Statement
Deaf and hard-of-hearing individuals in India who communicate through Indian Sign Language (ISL) face severe isolation when interacting with hearing non-signers in vital environments including healthcare, banking, education, and public administration. Existing automated solutions are predominantly limited to static, isolated alphabet gesture classification, ignore essential facial and grammatical body cues, demand high-bandwidth cloud video transmission that compromises user privacy, or fail to achieve the low-latency processing necessary for fluid live conversation. SignTalk AI solves this challenge by engineering a continuous sign-to-text pipeline that extracts skeletal landmark representations, models multi-joint spatial-temporal dependencies via ST-GCN, and translates continuous sign representations into coherent natural-language sentences via a Transformer architecture under real-time constraints.

### C. Detailed Problem Statement
Sign language translation is fundamentally distinct from speech-to-text processing because visual signing operates in four dimensions (three spatial dimensions plus time) and employs simultaneous multimodal channels: manual articulators (hand shape, palm orientation, location, and motion) and non-manual articulators (facial expression, head movement, and body posture). In the context of Indian Sign Language (ISL), automated technological support remains severely underdeveloped compared to American Sign Language (ASL) or German Sign Language (DGS). 

Most prior research suffers from critical deficiencies:
1. **Gesture Oversimplification:** Confining recognition to single static frames or isolated dictionary words, rendering the system incapable of handling the continuous co-articulated signing typical of natural communication.
2. **Computational Inefficiency and Background Sensitivity:** Relying on heavy 3D Convolutional Neural Networks (3D-CNNs) on raw video frames, which are computationally expensive, prone to overfitting on visual backgrounds and signer clothing, and incapable of executing on client-side edge devices.
3. **Loss of Structural Topology:** Flattening extracted skeletal coordinates into unstructured 1D vectors, thereby discarding the natural kinematic graph topology of the human skeletal system.
4. **Grammar Mismatch:** Treating sign recognition merely as classification of individual glosses without performing sequence-to-sequence translation into grammatically fluent spoken/written language (accounting for the Subject-Object-Verb (SOV) structure of ISL versus target spoken languages).
5. **Privacy Vulnerability:** Transmitting continuous user video to remote cloud servers, which presents unacceptable security and privacy risks in sensitive environments such as hospitals, banks, and government service counters.

SignTalk AI directly tackles these interrelated challenges by building an end-to-end framework that couples lightweight landmark extraction (MediaPipe Hands, Pose, and Face), topological spatial-temporal graph modeling (ST-GCN), and Transformer-based sequence translation, targeting continuous sign-to-text captioning with low latency, robust generalization, and edge-friendly inference.

### D. Research-Oriented Problem Formulation

Let an input video stream representing continuous Indian Sign Language signing be denoted as a sequence of $T$ consecutive video frames:
$$\mathbf{V} = \left( \mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_T \right), \quad \mathbf{v}_t \in \mathbb{R}^{H \times W \times 3}$$

A landmark extraction function $\mathcal{F}_{ext}$ maps each frame $\mathbf{v}_t$ to a set of $N$ spatial keypoint coordinates representing hand joints, upper-body pose landmarks, and salient facial markers:
$$\mathbf{X}_t = \mathcal{F}_{ext}(\mathbf{v}_t) \in \mathbb{R}^{N \times D}$$
where $D \in \{2, 3\}$ represents the normalized coordinate dimension $(x, y, z)$ relative to anatomical anchor points (e.g., mid-shoulder or wrist base).

The continuous signing sequence over temporal window $T$ is modeled as a connected spatial-temporal graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where:
- $\mathcal{V} = \{ v_{t, i} \mid t \in \{1, \dots, T\}, i \in \{1, \dots, N\} \}$ represents the set of all skeletal landmark nodes across all frames.
- The edge set $\mathcal{E} = \mathcal{E}_{spatial} \cup \mathcal{E}_{temporal}$ consists of:
  - Intra-frame spatial edges $\mathcal{E}_{spatial} = \{ (v_{t, i}, v_{t, j}) \mid (i, j) \in \mathcal{E}_{skeleton} \}$, representing biological kinematic connectivity (e.g., finger bone segments, wrist-elbow-shoulder linkages).
  - Inter-frame temporal edges $\mathcal{E}_{temporal} = \{ (v_{t, i}, v_{t+1, i}) \mid t \in \{1, \dots, T-1\}, i \in \{1, \dots, N\} \}$, connecting identical anatomical joints across consecutive time steps.

The graph tensor $\mathbf{X} \in \mathbb{R}^{C_{in} \times T \times N}$ is processed by a Spatial-Temporal Graph Convolutional Network (ST-GCN) parameterized by weights $\mathbf{\Theta}_{GCN}$:
$$\mathbf{H} = \operatorname{ST-GCN}(\mathbf{X}; \mathbf{\Theta}_{GCN}) \in \mathbb{R}^{T' \times d_{model}}$$
where $\mathbf{H} = (\mathbf{h}_1, \dots, \mathbf{h}_{T'})$ is the sequence of latent continuous signing representations extracted from the spatial-temporal graph convolutions.

The latent sequence $\mathbf{H}$ is then passed into a sequence-to-sequence Transformer translation model parameterized by $\mathbf{\Theta}_{trans}$. The decoder autoregressively generates the target natural-language textual sequence $\mathbf{Y} = (y_1, y_2, \dots, y_U)$ from a target vocabulary $\mathcal{V}_{text}$:
$$P(\mathbf{Y} \mid \mathbf{V}) = \prod_{u=1}^{U} P\left(y_u \mid y_{<u}, \mathbf{H}; \mathbf{\Theta}_{trans}\right)$$

**Optimization Objective:**
The objective during training is to minimize the empirical risk over a labeled dataset $\mathcal{D} = \{(\mathbf{V}^{(k)}, \mathbf{Y}^{(k)})\}_{k=1}^{K}$:
$$\min_{\mathbf{\Theta}_{GCN}, \mathbf{\Theta}_{trans}} \frac{1}{K} \sum_{k=1}^{K} \left[ \mathcal{L}_{trans}\left(\mathbf{Y}^{(k)}, \hat{\mathbf{Y}}^{(k)}\right) + \lambda \mathcal{L}_{CTC}\left(\mathbf{G}^{(k)}, \hat{\mathbf{G}}^{(k)}\right) \right]$$
where $\mathcal{L}_{trans}$ is the cross-entropy translation loss with label smoothing, $\mathcal{L}_{CTC}$ is an optional auxiliary Connectionist Temporal Classification loss over intermediate sign gloss sequences $\mathbf{G}^{(k)}$ to regularize temporal feature alignment, and $\lambda \ge 0$ is a balancing hyperparameter.

**Inference Constraint:**
The complete end-to-end processing pipeline must satisfy the temporal deadline:
$$\Delta t_{pipeline} = \Delta t_{capture} + \Delta t_{landmark} + \Delta t_{GCN} + \Delta t_{Transformer} + \Delta t_{render} \le \tau_{target}$$
where the target latency bound is $\tau_{target} = 500\text{ ms}$, operating continuously at $\ge 20\text{ frames per second (FPS)}$ on non-specialized consumer edge computing hardware.
