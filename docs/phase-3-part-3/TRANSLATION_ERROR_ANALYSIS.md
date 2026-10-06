# SignTalk AI — Translation Error Analysis Framework

**Document ID:** `DOC-P3P3-ERR-ANALYSIS-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** SPECIFIED & DOCUMENTED  

---

## 1. Error Categorization Taxonomy

In an integrated visual-to-language translation system, errors originate from distinct failure modes across the pipeline. We define a 9-category error taxonomy to classify failures:

| Category | Description | Primary Origin |
| :--- | :--- | :--- |
| **1. Recognition Error** | ST-GCN visual encoder fails to separate hand trajectory or joint motion. | Visual Frontend |
| **2. Missing Sign** | Visual gesture was present but omitted in decoded token sequence. | Cross-Attention / Decoder |
| **3. Extra Sign** | Decoder emits additional lexical tokens not present in visual input. | Autoregressive Decoder |
| **4. Wrong Sign / Confusion** | Similar kinematic gesture confused for an alternative class. | Visual Encoder / Vocabulary |
| **5. Wrong Ordering** | Signs recognized correctly but emitted in incorrect sequential order. | Decoder Positional Modeling |
| **6. Tokenization Error** | Out-of-vocabulary gloss mapped to `<UNK>` or improperly decomposed. | Tokenizer / Vocabulary |
| **7. Translation Error** | Sign recognized correctly, but translated to an inappropriate English word. | Lexical Lookup / Mapping |
| **8. Grammar Issue** | English translation is ungrammatical or awkwardly phrased. | Language Generation |
| **9. Hallucination** | Fluent English generated that contradicts or invents information. | Language Decoder |

---

## 2. Decoupling Visual Errors from Language Errors

A crucial research requirement is determining:
$$\text{Was the visual representation wrong, OR was the visual representation correct and the Transformer failed?}$$

- If the ST-GCN classification head from Phase 3 Part 2 predicted correctly ($y_{\text{GCN}} = y_{\text{true}}$) but the Transformer decoder generates an incorrect token ($\hat{y}_{\text{TF}} \ne y_{\text{true}}$):
  $\implies$ **Transformer Cross-Attention / Decoding Failure**.
- If the ST-GCN representation was ambiguous or misclassified in Phase 3 Part 2 ($y_{\text{GCN}} \ne y_{\text{true}}$) and the Transformer generates the same error:
  $\implies$ **Propagated Visual Recognition Error**.
- If the Transformer corrects a low-confidence visual error through linguistic conditioning:
  $\implies$ **Linguistic Error Correction**.
