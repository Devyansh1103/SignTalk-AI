# Phase 3 Part 3 Final Report: Transformer/NLP Translation Layer

**Project:** SignTalk AI — Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Document ID:** `DOC-P3P3-FINAL-REPORT-001`  
**Status:** COMPLETE, VERIFIED & BENCHMARKED  

---

## 1. Objective

The objective of Phase 3 Part 3 was to build, train, validate, and evaluate the language-understanding layer that consumes the spatial-temporal representations produced by the ST-GCN visual encoder and converts them into an appropriate linguistic representation (sign glosses and natural language English translations).

---

## 2. Language Task

The language task was audited and bounded according to the available dataset supervision:
- **Formal Task Classification:** **Isolated Gloss Recognition & Single-Token Translation Alignment** (Case C of architectural requirements).
- **Scope & Boundaries:** The model consumes spatiotemporal landmark sequences $\mathbf{X} \in \mathbb{R}^{B \times 3 \times 45 \times 93}$ and autoregressively generates token sequences $(\langle\text{BOS}\rangle \to \text{token\_id} \to \langle\text{EOS}\rangle)$ mapped to discrete ISL glosses and English translation phrases. Continuous multi-word sentence translation is explicitly documented as requiring expanded continuous datasets.

---

## 3. Dataset Supervision

A formal audit of `signTalk-seq-v1.0.0` confirmed:
- Available signals: 10 categorical sign classes, canonical gloss tokens (`HELLO` through `TEACHER`), and English word translations ("Hello" through "Teacher") across 60 standardized sequence archives (36 train, 12 val, 12 test).
- Missing signals: Continuous multi-sign transcripts, word-level grammatical syntax, and continuous signing transitions.
- Compliance: No artificial multi-word sentences were fabricated. All models were optimized strictly on real dataset supervision.

---

## 4. Architecture

We implemented the complete **SignTranslationModel** comprising:
1. **ST-GCN Visual Encoder (`SignSTGCN`):** 6-block partitioned graph convolutional network extracting spatiotemporal representations over the 93-node canonical MediaPipe skeleton.
2. **Feature Projection Layer (`FeatureProjection`):** Projects visual sequence features from $D_{\text{in}} = 256$ to Transformer embedding dimension $D_{\text{model}} = 128$ with LayerNorm and Dropout.
3. **Temporal Positional Encoding (`SinusoidalPositionalEncoding`):** Analytical frequency-based temporal order injection.
4. **Sign Language Transformer (`SignLanguageTransformer`):**
   - 2-layer multi-head Transformer Encoder (4 attention heads, $D_{\text{ff}} = 512$, pre-LayerNorm).
   - 2-layer multi-head Transformer Decoder with causal self-attention and cross-attention over visual memory.
   - Vocabulary Linear Head projecting to token probabilities ($|\mathcal{V}| = 14$).

---

## 5. ST-GCN Interface

- **Input Dimension:** $[B, 3, 45, 93]$.
- **Temporal Subsampling:** Strides $[1, 1, 2, 1, 2, 1]$ reduce $T=45$ to $T'=12$.
- **Spatial Node Pooling:** Average pooling over 93 nodes produces temporal feature sequence $[B, 12, 256]$.
- **Projection Interface:** `SignTranslationModel.extract_visual_embeddings(x, mask)` yields $[B, 12, 128]$ and downsampled boolean padding mask $[B, 12]$.

---

## 6. Transformer Configuration

- **Vocabulary Size:** 14 tokens (4 special control tokens + 10 lexical sign classes)
- **Embedding Dimension ($D_{\text{model}}$):** 128
- **Attention Heads:** 4
- **Encoder Layers:** 2
- **Decoder Layers:** 2
- **Feedforward Dimension:** 512
- **Dropout:** 0.1
- **Optimizer:** AdamW ($\text{lr} = 0.001$, $\text{weight\_decay} = 0.0001$)
- **Scheduler:** CosineAnnealingLR ($T_{\max} = 60$, $\eta_{\min} = 10^{-5}$)
- **Batch Size:** 8

---

## 7. Tokenizer

- Implemented in `src/nlp/tokenizer.py` with vocabulary `assets/vocabularies/mvp_10.json`.
- Special tokens: `<PAD>` (0), `<UNK>` (1), `<BOS>` (2), `<EOS>` (3).
- Lexical tokens: `HELLO` (4) through `TEACHER` (13).
- Integrated `TextNormalizer` handling Unicode NFKC normalization, whitespace collapsing, and punctuation policy.

---

## 8. Training Strategy

- **Two-Stage Decoupled Strategy:** ST-GCN visual encoder was initialized from the best validation checkpoint of Phase 3 Part 2 (`experiments/stgcn/checkpoints/best_checkpoint.pt`) and frozen.
- Only the `FeatureProjection` and `SignLanguageTransformer` layers were trained, ensuring fast convergence and preventing catastrophic forgetting of spatial graph structures.
- Trained with teacher forcing using shifted prefix tokens ($[\langle\text{BOS}\rangle, y_1] \to [y_1, \langle\text{EOS}\rangle]$).

---

## 9. Validation Strategy

- Validation split ($N=12$) evaluated every epoch for both teacher-forced sequence accuracy and full autoregressive greedy generation accuracy.
- Early stopping monitored validation generated sequence accuracy (`val_gen_seq_acc`) with a patience of 20 epochs.
- Best validation model saved at Epoch 4 (Early stopping triggered at Epoch 24).

---

## 10. Decoding Strategy

- **Primary Mode:** Autoregressive greedy decoding step-by-step from $\langle\text{BOS}\rangle$ until $\langle\text{EOS}\rangle$ or maximum decoding length.
- **Confidence Gating Policy:** Minimum confidence threshold $\theta_{\text{conf}} = 0.50$. Predictions below threshold are flagged as uncertain ("Uncertain sign" / "Please repeat").

---

## 11. Evaluation Metrics

- Token-level: Accuracy, Macro/Weighted Precision, Recall, F1.
- Sequence-level: Exact Match (EM) Sequence Accuracy.
- Translation-level: Corpus BLEU-1, BLEU-2, BLEU-4, ROUGE-L.
- Operational: Model inference latency, decoding latency, parameter count, checkpoint size.

---

## 12. Results

*Empirical measurements on held-out test split ($N=12$ sequences):*
- **Generated Sequence Exact Match:** **83.33%** (10 of 12 test sequences correct)
- **Token Accuracy:** **91.67%**
- **Macro Precision:** **83.33%** (+8.33 pp over ST-GCN alone; +54.16 pp over Baseline)
- **Macro Recall:** **86.36%** (+6.36 pp over ST-GCN alone; +41.36 pp over Baseline)
- **Macro F1-Score:** **83.03%** (+6.36 pp over ST-GCN alone; +52.36 pp over Baseline)
- **Weighted F1-Score:** **90.00%** (+12.22 pp over ST-GCN alone; +58.89 pp over Baseline)
- **Test Loss:** **0.6558**
- **Corpus BLEU-1:** **84.62%**
- **ROUGE-L F1:** **83.33%**

---

## 13. Error Analysis

Exactly 2 errors occurred on the test set:
1. `seq_0052` (`HAPPY` $\to$ `TEACHER`, confidence 0.8912): Propagated visual confusion originating from ST-GCN feature overlap on two-handed torso strokes.
2. `seq_0054` (`MONDAY` $\to$ `BIRD`, confidence 0.6892): Cross-attention misallocation where the decoder attended to index finger trajectory similarity with the `BIRD` sign.
- Significantly, `seq_0058` (`TIME`), which was misclassified by the ST-GCN linear head in Phase 3 Part 2, was **successfully rectified by the Transformer** with 0.9399 confidence.

---

## 14. Full Pipeline Results

The complete pipeline `Landmarks -> ST-GCN -> FeatureProjection -> Transformer -> Tokens -> Policy` executed cleanly via `scripts/evaluate_full_pipeline.py`:
- 10/12 predictions accepted and correct.
- Pipeline processed every sequence with detailed confidence and latency tracking.

---

## 15. Model Size

- **Total Parameters:** **3,100,776**
  - Frozen ST-GCN: 2,137,818 (68.9%)
  - Feature Projection: 33,152 (1.1%)
  - Sign Language Transformer: 929,806 (30.0%)
- **Trainable Parameters:** **962,958**
- **Checkpoint Size on Disk:** **19.46 MB**

---

## 16. Inference Performance

*Benchmarked on CPU over 100 iterations (Batch Size = 1):*
- **ST-GCN Visual Feature Extraction:** Mean: 117.67 ms | Median: 101.06 ms | P95: 189.28 ms
- **Transformer Autoregressive Decoder:** Mean: 12.63 ms | Median: 7.27 ms | P95: 16.48 ms
- **Total Pipeline Latency:** Mean: **130.31 ms** | Median: **109.12 ms** | P95: **203.05 ms**
- **Throughput:** **7.67 sequences/sec**

---

## 17. Reproducibility

- Deterministic twin training runs from random seed 42 verified zero numerical drift:
  - Max loss discrepancy: **`0.00e+00`**
  - Max parameter discrepancy: **`0.00e+00`**
- All 32 automated unit tests passing in `tests/`.

---

## 18. Limitations

1. **Dataset Scope:** Current benchmark is bounded to 60 isolated sequences across 10 sign lemmas.
2. **Translation Task Boundary:** Single-token gloss and phrase prediction; continuous discourse translation remains unsupported by current supervision.
3. **CPU Execution Speed:** End-to-end latency of ~130 ms provides ~7.7 FPS; mobile deployment will benefit from INT8 quantization or ONNX runtime graphs.

---

## 19. Dataset Gaps

Transitioning to continuous multi-sign translation requires:
- Paired continuous sign video recordings with multi-word sentence transcripts.
- Co-articulation and movement epenthesis annotations.
- Scaled sign vocabulary ($\ge 250$ to $1,000$ sign classes).
- Multi-signer cohorts ($\ge 20$ native signers).

---

## 20. Readiness for Phase 3 Part 4

All prerequisites for Phase 3 Part 4 are satisfied:
- [x] ST-GCN visual encoder verified and checkpointed.
- [x] Transformer translation layer built, trained, and benchmarked.
- [x] Full inference pipeline and confidence policy operational.
- [x] Architectural handoff and performance boundaries documented.
- [x] Unit test suite passing 100%.

SignTalk AI is fully prepared for Phase 3 Part 4 (Model Selection, Comparative Ablations & Production Readiness).
