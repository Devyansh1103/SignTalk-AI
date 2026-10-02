# Dataset Selection Specification: SignTalk AI

**Document ID:** STAI-P2P1-001  
**Phase:** Phase 2 — Part 1 (Dataset Selection & Validation)  
**Status:** Approved  
**Language Focus:** Indian Sign Language (ISL)  

---

## 1. Selected Primary Datasets

In accordance with Phase 1 architecture and feasibility criteria, the project selects the following datasets:

| Dataset Name | Domain / Mode | Num Classes / Sentences | Total Samples | Signers | Primary Purpose | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **INCLUDE-50** | Isolated Gloss / Word | 50 classes (15 categories) | 958 videos | 7 deaf signers | Baseline model, ST-GCN spatial feature extraction | **Active / Acquired** |
| **ISLTranslate** | Continuous Sentence | 31,117 sentence pairs | 31,117 | Multi-signer | Sequence translation scaling | **Registered / Metadata Acquired** |
| **ISL-CSLTR** | Continuous Phrase | 1,036 words / 700 sentences | 700 videos | Unknown | Phrase translation MVP | **Primary Candidate** |

---

## 2. Source Provenance: INCLUDE-50

- **Authors:** Advaith Sridhar, Rohith Gandhi Ganesan, Pratyush Kumar, Mitesh Khapra (AI4Bharat / IIT Madras).
- **Publication:** *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition*, ACM Multimedia (MM '20), October 2020.
- **Recording Origin:** Recorded with students and instructors from the St. Louis School for the Deaf in Adyar, Chennai.
- **Visual Parameters:** 
  - Resolution: 1920×1080 full HD downscaled for web/model processing.
  - Video format: MOV/MP4.
  - Frame Rate: 25.0 to 30.0 FPS.
  - Lighting: Controlled classroom and studio illumination.
  - Camera setup: Static tripod, frontal upper-body shot framing head, torso, and full arm extents.

---

## 3. Strict Scope Constraints

1. **Exclusion of Non-ISL Datasets:**
   - WLASL (American Sign Language) is strictly excluded from training.
   - RWTH-PHOENIX-Weather-2014T (German Sign Language) is used strictly for architectural reference, not for ISL training.
2. **License Adherence:**
   - INCLUDE is utilized under its non-commercial Academic Research License.
   - Video assets and skeletal coordinates are processed with privacy preservation.
