# SignTalk AI — Positional Encoding Strategy

**Document ID:** `DOC-P3P3-POS-ENC-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** COMPLETE & DOCUMENTED  

---

## 1. Motivation

Standard self-attention operations in Transformers are permutation-equivariant:
$$\text{Attention}(\mathbf{Q}\mathbf{P}, \mathbf{K}\mathbf{P}, \mathbf{V}\mathbf{P}) = \mathbf{P} \cdot \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})$$
for any permutation matrix $\mathbf{P}$. In sign language, however, gesture semantics are fundamentally determined by temporal ordering (e.g., hand approaching the face vs moving away from the face, or sequential hand transitions). Injecting temporal order into both the visual representation sequence and target token sequence is essential.

---

## 2. Evaluation of Positional Encoding Alternatives

| Scheme | Description | Parameter Overhead | Length Extrapolation | Suitability for SignTalk AI |
| :--- | :--- | :---: | :---: | :---: |
| **Sinusoidal Encoding** | Fixed analytical sine and cosine functions across geometric progression of frequencies. | **0 params** | Strong (analytical formula) | **PRIMARY CHOICE** (Zero parameters, prevents overfitting on small datasets) |
| **Learned Positional Embeddings** | Trainable embedding table $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{T_{\max} \times D_{\text{model}}}$. | $T_{\max} \times D_{\text{model}}$ | Poor (cannot exceed $T_{\max}$) | Alternative for large datasets |
| **Relative Positional Encoding (Shaw/T5)** | Biases attention logits by relative index $(i - j)$. | Small ($\mathcal{O}(2K)$) | Moderate | Overly complex for current sequence lengths ($T'=12$) |

---

## 3. Mathematical Formulation

We adopt standard **Sinusoidal Positional Encoding** (Vaswani et al. 2017) as the primary encoding:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/D_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/D_{\text{model}}}}\right)$$
where $pos \in [0, T'-1]$ and $i \in [0, D_{\text{model}}/2 - 1]$.

For any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$, allowing the model to easily attend to relative temporal offsets in signing dynamics without adding trainable parameters.
