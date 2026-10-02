# Label Normalization Specification: SignTalk AI

**Document ID:** STAI-P2P3-010  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 2 — Part 3 (Sequence Dataset Construction)  
**Author:** ML Data Engineer & NLP Specialist  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Overview & Objectives

In raw sign language benchmarks (such as AI4Bharat INCLUDE), label annotations often exhibit inconsistent formatting, irregular casing (e.g. `Adjectives/94. good/` vs `Places/23. Court/`), concatenated tokens (e.g. `storeorshop`, `trainticket`, `biglarge`), and special character variations.

Label normalization establishes a deterministic mapping pipeline that converts arbitrary raw source strings into unified canonical identifiers while preserving full bidirectional provenance.

---

## 2. Normalization Rules

1. **Lower-Case Standardization:** All canonical labels are converted to lower-case ASCII alphanumeric characters with underscores for token separation (`[a-z0-9_]`).
2. **Numeric Prefix Stripping:** Raw index prefixes (e.g. `94. good` $\to$ `good`) are stripped.
3. **Compound Word Splitting:** Concatenated labels from source datasets are formally disambiguated:
   - `thankyou` $\to$ `thankyou` (canonical label) / `THANK_YOU` (gloss) / `"Thank you"` (translation).
   - `storeorshop` $\to$ `store_or_shop` / `STORE_OR_SHOP`.
   - `trainticket` $\to$ `train_ticket` / `TRAIN_TICKET`.
   - `goodmorning` $\to$ `good_morning` / `GOOD_MORNING`.
4. **No Semantically Unjustified Merges:** Classes that share visual or conceptual similarities (e.g. `smalllittle` and `short`) are strictly preserved as distinct classes with unique class IDs unless linguistic authority explicitly mandates a merger.

---

## 3. Authoritative Label Mapping Registry

The canonical label mapping for the active 10-class dataset is registered in [`data/processed/label_mapping.csv`](file:///d:/SignAI/data/processed/label_mapping.csv):

| Source Dataset | Source Class ID | Source Label | Canonical Class ID | Canonical Label | Linguistic Gloss | Natural Language Translation | Notes |
| :--- | :---: | :--- | :---: | :--- | :--- | :--- | :--- |
| `INCLUDE-50` | 22 | `Greetings/48. Hello` | 0 | `hello` | `HELLO` | `"Hello"` | Validated isolated sign |
| `INCLUDE-50` | 42 | `Greetings/95. thankyou` | 1 | `thankyou` | `THANK_YOU` | `"Thank you"` | Validated isolated sign |
| `INCLUDE-50` | 18 | `Adjectives/94. good` | 2 | `good` | `GOOD` | `"Good"` | Validated isolated sign |
| `INCLUDE-50` | 20 | `Adjectives/3. happy` | 3 | `happy` | `HAPPY` | `"Happy"` | Validated isolated sign |
| `INCLUDE-50` | 29 | `Days_and_Time/67. Monday`| 4 | `monday` | `MONDAY` | `"Monday"` | Validated isolated sign |
| `INCLUDE-50` | 6 | `Means_of_Transportation/11. Car` | 5 | `car` | `CAR` | `"Car"` | Validated isolated sign |
| `INCLUDE-50` | 2 | `Animals/4. Bird` | 6 | `bird` | `BIRD` | `"Bird"` | Validated isolated sign |
| `INCLUDE-50` | 24 | `Home/20. House` | 7 | `house` | `HOUSE` | `"House"` | Validated isolated sign |
| `INCLUDE-50` | 43 | `Days_and_Time/86. Time` | 8 | `time` | `TIME` | `"Time"` | Validated isolated sign |
| `INCLUDE-50` | 41 | `Jobs/84. Teacher` | 9 | `teacher` | `TEACHER` | `"Teacher"` | Validated isolated sign |
