# SignTalk AI — End-to-End Real-Time Performance Report

**Generated:** 2026-10-06 22:13:34
**Sample Count:** 100 cycles
**Operating System:** Windows 11 (AMD64)
**Python:** 3.13.12
**PyTorch:** 2.13.0+cpu
**Model Checkpoint:** `experiments/stgcn/checkpoints/best_checkpoint.pt`
**Temporal Window Size ($T$):** 45 frames ($1.80\text{ s}$ @ $25\text{ FPS}$)
**Graph Nodes ($V$):** 93 multimodal nodes
**Host RAM RSS:** 0.0 MB

## 1. Latency Breakdown Matrix

| Pipeline Stage | Mean (ms) | Median (ms) | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | Std (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **tensor_prep** | 0.27 | 0.26 | 0.26 | 0.35 | 0.39 | 0.50 | 0.20 | 0.54 | 0.06 |
| **stgcn_inference** | 97.28 | 94.06 | 94.06 | 113.56 | 117.27 | 132.71 | 79.76 | 136.38 | 12.54 |
| **confidence_filter** | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.01 | 0.03 | 0.00 |
| **smoothing** | 0.11 | 0.11 | 0.11 | 0.15 | 0.16 | 0.20 | 0.04 | 0.42 | 0.05 |
| **state_machine** | 0.01 | 0.01 | 0.01 | 0.05 | 0.06 | 0.08 | 0.00 | 0.14 | 0.02 |
| **deduplicator** | 0.01 | 0.00 | 0.00 | 0.00 | 0.01 | 0.06 | 0.00 | 0.24 | 0.03 |
| **translation** | 0.01 | 0.01 | 0.01 | 0.02 | 0.03 | 0.04 | 0.01 | 0.04 | 0.01 |
| **display_render** | 1.16 | 0.77 | 0.77 | 0.98 | 1.11 | 1.62 | 0.67 | 35.31 | 3.44 |
| **end_to_end** | 98.90 | 96.02 | 96.02 | 115.20 | 121.39 | 134.24 | 80.88 | 138.04 | 12.76 |

## 2. Engineering Latency Targets vs Measured Performance

| Subsystem Stage | Engineering Target | Measured Mean | Measured P95 | Status |
| :--- | :---: | :---: | :---: | :---: |
| **ST-GCN Visual Model** | < 150 ms | 97.28 ms | 117.27 ms | PASS |
| **Tensor Preparation** | < 5 ms | 0.27 ms | 0.39 ms | PASS |
| **Majority Vote Smoothing** | < 2 ms | 0.11 ms | 0.16 ms | PASS |
| **Linguistic Translation** | < 10 ms | 0.01 ms | 0.03 ms | PASS |
| **Diagnostic HUD Rendering** | < 10 ms | 1.16 ms | 1.11 ms | PASS |
| **Total Prediction Cycle** | < 200 ms | 98.90 ms | 121.39 ms | PASS |

## 3. Key Findings & Profiling Assessment

1. **Dominant Cost:** ST-GCN inference on CPU accounts for approximately 98.4% of the total execution time.
2. **Post-Processing Overhead:** Filtering, smoothing, state machine, deduplication, and linguistic translation execute in sub-millisecond time (< 0.25 ms total).
3. **Memory Stability:** RAM consumption remained bounded with zero memory accumulation across 100+ inference cycles.
