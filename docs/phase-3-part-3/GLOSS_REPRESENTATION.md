# SignTalk AI — Gloss Representation Specification

**Document ID:** `DOC-P3P3-GLOSS-REP-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** COMPLETE & DOCUMENTED  

---

## 1. Gloss Definition & Separation

In Sign Language Linguistics:
- A **gloss** is a typographic label written in spoken-language words (traditionally capital letters) representing the lexical identity of a sign gesture.
- A gloss is **not** an English word translation; sign languages have distinct syntax, morphology, and spatial grammar separate from spoken English.
- We maintain strict separation between:
  1. `source_label`: Original dataset category (e.g., `Greetings/48. Hello`).
  2. `canonical_gloss`: Standardized uppercase linguistic token (`HELLO`).
  3. `translation`: Spoken English phrase (`Hello`).

---

## 2. Token Sequence Format for Autoregressive Modeling

To train an autoregressive sequence model:
- **Decoder Input:** Starts with `<BOS>` prompt followed by target tokens:
  $$\mathbf{y}_{\text{in}} = [\langle\text{BOS}\rangle, y_1, y_2, \dots, y_{S-1}]$$
  For isolated signs ($S=2$): $[\langle\text{BOS}\rangle, \text{token\_id}] = [2, \text{token\_id}]$.
- **Decoder Target:** Shifted target sequence ending with `<EOS>`:
  $$\mathbf{y}_{\text{tgt}} = [y_1, y_2, \dots, \langle\text{EOS}\rangle]$$
  For isolated signs ($S=2$): $[\text{token\_id}, \langle\text{EOS}\rangle] = [\text{token\_id}, 3]$.
- **Attention Mask:** Boolean tensor indicating valid target positions: $[1, 1]$.
- **Padding:** Unused positions are padded with $\langle\text{PAD}\rangle = 0$.
