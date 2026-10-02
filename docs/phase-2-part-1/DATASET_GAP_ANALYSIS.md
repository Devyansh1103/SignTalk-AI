# Dataset Gap Analysis: SignTalk AI

**Document ID:** STAI-P2P1-004  
**Phase:** Phase 2 — Part 1  
**Status:** Approved  

---

## 1. Identified Data Gaps in Indian Sign Language (ISL)

While international sign languages (such as ASL via WLASL/How2Sign or DGS via PHOENIX) possess mature continuous sentence corpora, ISL research faces distinct historical data scarcity:

1. **Continuous vs. Isolated Disparity:**
   - INCLUDE-50 provides robust isolated lexical coverage (958 videos, 50 classes) but lacks continuous linguistic transitions, co-articulation, and grammatical facial markers found in running conversation.
   - ISLTranslate provides parallel text but video downloads require manual institutional credentialing.
   - **Mitigation:** Two-stage model strategy where isolated signs in INCLUDE-50 pretrain the spatial landmark backbone (ST-GCN), followed by continuous sequence translation (Transformer) on continuous phrase datasets (ISL-CSLTR).

2. **Signer & Demographic Representation:**
   - INCLUDE-50 utilizes 7 student signers from Chennai. While regional lexical variation across North vs South India exists in ISL, standard vocabulary items in INCLUDE-50 are verified against ISLRTC standards.
   - **Mitigation:** Strict signer-independent validation partition to evaluate generalization on unseen signers.

3. **Occlusion & Rapid Motion Bottlenecks:**
   - In rapid signs (e.g. `monday`, `cellphone`), hands frequently cross the face or touch the torso, causing MediaPipe landmark jitter or hand-loss.
   - **Mitigation:** Missing-landmark handling via carry-forward/temporal linear interpolation and confidence masking in Phase 2 Part 2.
