# 12. Performance Engineering & Optimization Strategy: SignTalk AI

**Document ID:** STAI-P1P2-012  
**Project Name:** SignTalk AI  
**Document Status:** Approved Engineering Blueprint (Phase 1 — Part 2)  

---

## 1. Performance Engineering Lifecycle

SignTalk AI enforces an empirical performance engineering lifecycle:
$$\text{Baseline Measurement} \longrightarrow \text{Bottleneck Identification} \longrightarrow \text{Targeted Optimization} \longrightarrow \text{Verification Against Accuracy}$$

No optimization technique (e.g., quantization, model pruning) is retained if it degrades translation accuracy (BLEU-4) by more than $5\%$ relative to the FP32 baseline.

---

## 2. Resource Envelopes & Profiling Limits

| System Resource | Nominal Target Budget | Maximum Hard Ceiling | Verification Metric |
| :--- | :--- | :--- | :--- |
| **End-to-End Latency** | **$< 500\text{ ms}$** | $800\text{ ms}$ | Wall-clock sign-to-caption duration |
| **Throughput (FPS)** | **$\ge 25\text{ FPS}$** | Minimum $15\text{ FPS}$ | Sustained frame ingestion rate |
| **Host System RAM** | **$< 2.0\text{ GB}$ resident** | $3.5\text{ GB}$ resident | Python process Resident Set Size (RSS) |
| **GPU VRAM (If Active)** | **$< 2.0\text{ GB}$** | $4.0\text{ GB}$ | PyTorch `torch.cuda.memory_allocated()` |
| **Host CPU Utilization** | **$\le 60\%$ (4 threads)** | $85\%$ across 4 threads | Multi-core thread utilization |

---

## 3. Targeted Optimization Techniques

### 3.1 ONNX Runtime & INT8 Dynamic Quantization
- **Mechanism:** Export trained PyTorch ST-GCN and Transformer models to Open Neural Network Exchange (ONNX) format, followed by dynamic range quantization:
  ```python
  from onnxruntime.quantization import quantize_dynamic, QuantType

  quantize_dynamic(
      model_input="models/checkpoints/stgcn_fp32.onnx",
      model_output="models/checkpoints/stgcn_int8.onnx",
      weight_type=QuantType.QInt8
  )
  ```
- **Performance Impact:** Reduces ST-GCN CPU forward pass latency by $\approx 2.5\times$ (from $\approx 60\text{ ms}$ to $< 25\text{ ms}$) via Intel AVX-512 / AMD Zen VNNI vectorized integer instructions.
- **Model Footprint:** Model checkpoint size compresses from $\approx 17.5\text{ MB}$ to $< 5\text{ MB}$.

### 3.2 Temporal Stride Optimization ($S = 5$)
- Evaluating continuous sliding windows on every frame ($30\text{ inferences/sec}$) causes CPU thermal throttling.
- By triggering inference once every $S = 5\text{ frames}$ ($\approx 6\text{ inferences/sec}$), computational overhead drops by $83.3\%$ while maintaining a fresh caption update every $166\text{ ms}$, which comfortably matches human visual reading cadence.

### 3.3 Zero-Allocation Circular Buffers
- Avoid dynamic memory allocations (`np.append()`, `torch.cat()`) within the per-frame loop.
- Pre-allocate a contiguous memory block `np.zeros((45, 93, 3), dtype=np.float32)`. Frames are inserted using pointer indexing with modulo arithmetic (`idx = frame_count % 45`), eliminating Python garbage collection latency spikes.
