# SignTalk AI — ST-GCN to Transformer Interface Handoff

**Document ID:** `DOC-P3P2-TRANSFORMER-HANDOFF-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Status:** ARCHITECTURAL SPECIFICATION (PRE-TRANSFORMER HANDOFF)  

---

## 1. Architectural Role and Interface Pipeline

In the full SignTalk AI technical direction, the system architecture couples a Spatial-Temporal Graph Convolutional Network (ST-GCN) visual frontend with a Transformer language decoder:

```
Raw Frames [B, 3, H, W]
       ↓ (MediaPipe Vision Tasks)
Landmark Tensor [B, 3, T, V]
       ↓ (ST-GCN Spatial-Temporal Encoder)
Latent Spatiotemporal Sequence Embedding [B, T', D_model]
       ↓ (Cross-Attention Conditioning)
Transformer Autoregressive Decoder / NLP Generator
       ↓
Translated Text Tokens / English Gloss Sentences
```

> [!IMPORTANT]
> **Current Dataset Linguistic Boundary:** The current dataset release (`signTalk-seq-v1.0.0`) contains **isolated sign tokens** (single-word glosses: `HELLO`, `THANK_YOU`, `CAR`, etc.). Continuous sign language sentences with multi-word grammar and sequential sentence annotations are **NOT** present in this dataset release. Consequently, training an autoregressive continuous-sentence Transformer decoder at this stage is unsupported by the available supervision. Full Transformer sentence generation belongs to subsequent dataset expansions.

---

## 2. Spatiotemporal Feature Tensor Interface

The ST-GCN visual encoder exposes a dedicated feature extraction method (`SignSTGCN.extract_features`) producing standardized representations for Transformer conditioning:

### A. Frame-Wise Sequence Output (for Autoregressive Transformer Decoders)
When serving as a visual encoder for a Transformer cross-attention mechanism, the global temporal pooling step is bypassed:
- **Tensor Shape:** $[\mathbf{B}, \mathbf{T}', \mathbf{D}_{\text{model}}]$
  - $B$: Batch size
  - $T'$: Subsampled temporal frame count ($T' = 12$ after two stride-2 blocks from $T=45$)
  - $D_{\text{model}}$: Feature embedding dimension ($D_{\text{model}} = 256$, spatial nodes averaged over $V$)
- **Spatial Reduction:** $\mathbf{X}_{\text{temp\_seq}} = \frac{1}{V} \sum_{v=1}^{V} \mathbf{X}_{\text{block6}}(:, :, :, v) \in \mathbb{R}^{B \times 256 \times T'}$, transposed to $[B, T', 256]$.
- **Sequence Mask:** Boolean mask $[B, T']$ indicating valid vs zero-padded frames.

### B. Global Sequence Descriptor Output (for Single-Token Embedding Classifiers)
- **Tensor Shape:** $[\mathbf{B}, \mathbf{D}_{\text{model}}] = [B, 256]$
  - Spatially and temporally averaged global visual descriptor used by the linear classification head.

---

## 3. Token Vocabulary Interface

The tokenization contract with the Transformer decoder is defined by [`assets/vocabularies/mvp_10.json`](file:///d:/SignAI/assets/vocabularies/mvp_10.json):
- Special Tokens: `<PAD>` (0), `<UNK>` (1), `<BOS>` (2), `<EOS>` (3).
- Lexical Tokens: `HELLO` (4) through `TEACHER` (13).
- Target Decoder Tensor Format: `decoder_input` ($[B, S]$) and `decoder_target` ($[B, S]$).
