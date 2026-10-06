# SignTalk AI — Comprehensive Data Leakage Audit

**Document ID:** `DOC-P3P4-LEAKAGE-001`  
**Phase:** Phase 3 — Part 4 (Model Evaluation, Ablation & Final Model Selection)  
**System Title:** SignTalk AI: Real-Time Indian Sign Language Translation Platform  
**Date:** October 2026  
**Status:** PASSED WITH DOCUMENTED CONSTRAINTS  

---

## 1. Executive Summary

A rigorous audit of the complete data processing, tokenization, training, and evaluation pipelines was conducted to verify that no information from the test partition leaked into model training or validation procedures. 

| Audit Area | Risk Level | Verified Status | Result |
| :--- | :---: | :---: | :---: |
| **Sequence ID Overlap** | High | Checked across all splits | **PASS (0% Overlap)** |
| **Raw Video Clip Overlap**| High | Checked across all splits | **PASS (0% Overlap)** |
| **Temporal Window Leaks** | High | Isolated full recordings used | **PASS (0% Overlap)** |
| **Signer Overlap** | Medium | Signer IDs inspected | **DOCUMENTED OVERLAP (Repetition Split)** |
| **Preprocessing & Normalization** | Medium | Sample-independent root-centering | **PASS (No Global Leakage)** |
| **Data Augmentation Leakage** | High | Only applied at train time | **PASS (Test Data Unaugmented)** |
| **Tokenizer Fitting** | High | Vocabulary pre-fixed | **PASS (No Dynamic Test Fitting)** |
| **Hyperparameter Tuning on Test** | Critical | Tuned exclusively on val split | **PASS (Test Partition Held Out)** |

---

## 2. Leakage Vectors & Audit Details

### Vector 1: Sequence & Recording Partitioning
- **Verification Method:** Set intersection test between `sequence_id` and `source_video_id` across `data/manifests/train.csv`, `val.csv`, and `test.csv`.
- **Finding:**
  $$\text{Train Seqs} \cap \text{Val Seqs} = \emptyset, \quad \text{Train Seqs} \cap \text{Test Seqs} = \emptyset, \quad \text{Val Seqs} \cap \text{Test Seqs} = \emptyset$$
  $$\text{Train Videos} \cap \text{Test Videos} = \emptyset$$
- **Verdict:** **CLEAN.** No identical video recording or landmark file is duplicated across partitions.

### Vector 2: Signer Identity Isolation
- **Verification Method:** Checked whether signers in test partition (`signer_01`, `signer_02`, `signer_03`) appear in training.
- **Finding:** All 3 signers appear in training, validation, and testing.
- **Verdict:** **CONSTRAINED.** The split is a repetition-isolated split, not a signer-independent split.

### Vector 3: Normalization Statistics
- **Verification Method:** Inspected `src/preprocessing/normalizer.py` and `SignSTGCN.data_bn`.
- **Finding:** Landmark normalization in the pipeline does NOT compute global mean/standard deviation across the entire dataset. Instead, normalization is computed **instance-by-instance** using each sequence's own root joint (nose / mid-shoulder) and inter-shoulder distance. BatchNorm layers during test evaluation operate in `eval()` mode using running statistics accumulated strictly during training epochs.
- **Verdict:** **CLEAN.** No test-set statistical moments contaminated the training process.

### Vector 4: Tokenizer & Vocabulary Fitting
- **Verification Method:** Inspected `src/nlp/tokenizer.py` and vocabulary generation script.
- **Finding:** The vocabulary `mvp_10.json` is a fixed 14-token schema defined a priori from the 10 target ISL glosses plus 4 special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`). The tokenizer was NOT fitted on frequency statistics of the test set.
- **Verdict:** **CLEAN.** Zero vocabulary or tokenizer leakage.

### Vector 5: Augmentation Leakage
- **Verification Method:** Inspected dataloader instantiation in `src/data/dataloader.py` and training scripts.
- **Finding:** Augmentations (coordinate jitter, temporal dropouts) are gated strictly behind the `augment=True` argument, which is enabled ONLY during training and explicitly set to `False` for validation and testing dataloaders.
- **Verdict:** **CLEAN.** Test sequences were evaluated in their pristine, unaugmented form.

### Vector 6: Decision Threshold Optimization
- **Verification Method:** Inspected confidence threshold tuning scripts.
- **Finding:** Confidence thresholds for rejection ($\tau=0.70$) were analyzed across validation partitions and are evaluated post-hoc on test without retraining or hyperparameter search on test.
- **Verdict:** **CLEAN.**

---

## 3. Final Leakage Audit Conclusion

The data pipeline and evaluation protocols adhere to rigorous scientific standards. With the explicit documentation that the current dataset represents a **repetition split across 3 signers** rather than a zero-shot unseen-signer split, the codebase is cleared of all blocking data leakage hazards.
