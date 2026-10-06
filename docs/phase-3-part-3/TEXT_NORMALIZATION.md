# SignTalk AI — Text Normalization Specification

**Document ID:** `DOC-P3P3-TEXT-NORM-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** COMPLETE & DOCUMENTED  

---

## 1. Objective and Policy

Text normalization standardizes linguistic strings across data loading, training target preparation, and inference evaluation. The objective is to eliminate extraneous whitespace, formatting variance, and non-canonical representations without discarding crucial linguistic distinctions.

---

## 2. Normalization Passes

1. **Unicode Canonical Decomposition & Composition (NFKC):**
   - Applies `unicodedata.normalize('NFKC', text)` to standardize accented characters, typographic ligatures, and width variants.
2. **Whitespace Harmonization:**
   - Converts multiple spaces, non-breaking spaces, tabs, and newlines to a single standard ASCII space (`\x20`) and strips outer leading/trailing spaces.
3. **Punctuation Policy:**
   - For gloss tokens: All punctuation is removed except underscores (`_`) representing multi-word compound glosses (e.g., `THANK_YOU`).
   - For English translations: Standard capitalization and basic punctuation are preserved during evaluation, with optional punctuation stripping for token metrics.
4. **Casing Policy:**
   - Glosses: Strictly upper-case (`HELLO`, `GOOD`, `TEACHER`).
   - Natural English text: Standard sentence casing (`Hello`, `Thank you`, `Good`).
