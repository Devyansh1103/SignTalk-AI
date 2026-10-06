# SignTalk AI — Transformer Translation Layer Empirical Results

**Document ID:** `DOC-P3P3-RESULTS-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** MEASURED & BENCHMARKED  

---

## 1. Experimental Summary

All metrics reported below are actual empirical measurements obtained on the held-out test partition of `signTalk-seq-v1.0.0` ($N=12$ sequences) using the finalized checkpoint (`experiments/transformer/checkpoints/best_checkpoint.pt`).

---

## 2. Quantitative Model Metrics

### A. Recognition & Translation Metrics

| Metric | Measured Value | Notes |
| :--- | :---: | :--- |
| **Generated Sequence Exact Match** | **83.33%** | 10 of 12 test sequences decoded perfectly |
| **Token Accuracy** | **91.67%** | Non-padding token accuracy |
| **Macro Precision** | **83.33%** | Unweighted average precision across vocabulary |
| **Macro Recall** | **86.36%** | Unweighted average recall across vocabulary |
| **Macro F1-Score** | **83.03%** | Harmonic mean of macro precision and recall |
| **Weighted F1-Score** | **90.00%** | Support-weighted F1 across test instances |
| **Test Cross-Entropy Loss** | **0.6558** | Token cross-entropy loss (ignore_index=0) |
| **Corpus BLEU-1** | **84.62%** | Unigram lexical precision on translation text |
| **ROUGE-L Precision** | **83.33%** | Longest common subsequence precision |
| **ROUGE-L Recall** | **83.33%** | Longest common subsequence recall |
| **ROUGE-L F1-Score** | **83.33%** | Harmonic mean of ROUGE-L precision and recall |

---

## 3. Parameter Footprint and Model Size

| Module | Parameters | Trainable? | Proportion |
| :--- | :---: | :---: | :---: |
| **ST-GCN Visual Graph Encoder** | 2,137,818 | No (Frozen) | 68.9% |
| **Feature Projection Layer** | 33,152 | Yes | 1.1% |
| **Sign Language Transformer** | 929,806 | Yes | 30.0% |
| **Total Model Footprint** | **3,100,776** | **962,958** | **100.0%** |
| **Checkpoint Size on Disk** | **19.46 MB** | — | — |

---

## 4. Inference Latency & Throughput Benchmark

*Measured over 100 consecutive benchmark iterations on CPU (Batch Size = 1):*

| Component | Mean Latency | Median Latency | P95 Latency |
| :--- | :---: | :---: | :---: |
| **Visual Feature Extraction (ST-GCN)** | 117.67 ms | 101.06 ms | 189.28 ms |
| **Autoregressive Decoder (Transformer)** | 12.63 ms | 7.27 ms | 16.48 ms |
| **Total End-to-End Latency** | **130.31 ms** | **109.12 ms** | **203.05 ms** |
| **Inference Throughput** | **7.67 seq/sec** | — | — |

---

## 5. Architectural Progression Comparison

| Benchmark Dimension | Phase 3 Part 1 Baseline (BiGRU) | Phase 3 Part 2 (ST-GCN Head) | Phase 3 Part 3 (ST-GCN + Transformer) |
| :--- | :---: | :---: | :---: |
| **Top-1 / Exact Match Accuracy** | 41.67% | 83.33% | **83.33%** |
| **Macro Precision** | 29.17% | 75.00% | **83.33% (+8.33 pp)** |
| **Macro Recall** | 45.00% | 80.00% | **86.36% (+6.36 pp)** |
| **Macro F1-Score** | 30.67% | 76.67% | **83.03% (+6.36 pp)** |
| **Weighted F1-Score** | 31.11% | 77.78% | **90.00% (+12.22 pp)** |
| **Autoregressive Generation** | No | No | **YES** |
| **Translation BLEU-1** | N/A | N/A | **84.62%** |
| **Total Parameters** | 1,489,418 | 2,137,818 | 3,100,776 |
| **Checkpoint Size** | 17.06 MB | 25.88 MB | 19.46 MB |
| **CPU Latency (batch=1)** | 6.87 ms | 105.83 ms | 130.31 ms |
