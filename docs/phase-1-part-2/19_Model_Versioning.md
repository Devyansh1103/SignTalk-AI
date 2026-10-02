# 19. Model Versioning & Artifact Governance: SignTalk AI

**Document ID:** STAI-P1P2-019  
**Project Name:** SignTalk AI  
**Document Status:** Approved Engineering Blueprint (Phase 1 — Part 2)  

---

## 1. Semantic Versioning Protocol

SignTalk AI enforces a semantic model versioning standard:

$$\mathbf{MODEL\_TAG} = \text{<Architecture>}-\text{v<Major>.<Minor>.<Patch>}$$

- **Major Version ($X.0.0$):** Breaking architectural changes (e.g., modifying graph node count $N=53 \rightarrow N=93$, altering Transformer attention dimensions, or changing vocabulary size).
- **Minor Version ($0.Y.0$):** Dataset expansion, new training splits, or hyperparameter updates retaining identical tensor shapes.
- **Patch Version ($0.0.Z$):** Weight fine-tuning, bug fixes, or ONNX quantization optimizations.

---

## 2. Model Version Registry

| Model Tag | Architecture | Input Shape $(C, T, N)$ | Target Dataset | Checkpoint Path | Format | Status |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- |
| **`baseline-mlp-v1.0.0`** | 2-Layer MLP Baseline | $(1, 45, 279)$ | INCLUDE-50 | `models/checkpoints/baseline_mlp_v1.pt` | PyTorch | `PLANNED` |
| **`baseline-lstm-v1.0.0`**| 2-Layer Bi-LSTM Baseline| $(1, 45, 279)$ | INCLUDE-50 | `models/checkpoints/baseline_bilstm_v1.pt` | PyTorch | `PLANNED` |
| **`stgcn-iso-v1.0.0`** | 6-Block ST-GCN Isolated | $(3, 45, 53)$ | INCLUDE-50 | `models/checkpoints/stgcn_iso_v1.pt` | PyTorch | `PLANNED` |
| **`stgcn-trans-v1.0.0`** | ST-GCN + Transformer | $(3, 45, 93)$ | ISL-CSLTR | `models/checkpoints/stgcn_trans_v1.pt` | PyTorch | `PLANNED` |
| **`stgcn-trans-int8`** | ST-GCN + Transformer | $(3, 45, 93)$ | ISL-CSLTR | `models/checkpoints/stgcn_trans_int8.onnx`| ONNX INT8 | `PLANNED` |

---

## 3. Cryptographic Checksum & Reproducibility Verification

Every model artifact saved to disk must be accompanied by an authoritative metadata manifest:
```json
{
  "model_version": "stgcn-trans-v1.0.0",
  "created_at_utc": "2026-10-15T12:00:00Z",
  "git_commit_hash": "a4b7c9d1e2f3",
  "framework": "PyTorch 2.3.1",
  "file_name": "stgcn_trans_v1.pt",
  "file_size_bytes": 17482340,
  "sha256_checksum": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "random_seed": 42,
  "metrics": {
    "val_bleu4": 23.4,
    "val_wer": 28.2,
    "inference_latency_cpu_ms": 142.5
  }
}
```
During server startup, FastAPI calculates the SHA-256 hash of the loaded `.pt` file. If the checksum mismatches the manifest, startup aborts to prevent corrupted weights from entering production serving.
