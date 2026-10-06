# SignTalk AI — Decoding Strategy and Generation Algorithms

**Document ID:** `DOC-P3P3-DECODING-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** COMPLETE & DOCUMENTED  

---

## 1. Generation Paradigms

During inference, ground truth token inputs are unavailable. The language decoder must generate tokens sequentially given visual memory $\mathbf{Z} \in \mathbb{R}^{B \times T' \times D_{\text{model}}}$. We evaluate two decoding strategies:

### A. Greedy Decoding
- At each step $s$, the token with the highest predicted probability is selected:
  $$\hat{y}_s = \arg\max_{v \in \mathcal{V}} P(y_s = v \mid y_{<s}, \mathbf{Z})$$
- **Properties:**
  - Computationally efficient ($\mathcal{O}(S)$ steps).
  - Deterministic and reproducible.
  - Well-suited for real-time inference pipelines.
  - Optimal for isolated signs and short token sequences ($S \le 3$).

### B. Beam Search Decoding
- Maintains the top-$K$ most probable sequence hypotheses at each step:
  $$\mathcal{H}_s = \text{Top-}K \left( \left\{ (\mathbf{y}_{<s}, v) \mid \mathbf{y}_{<s} \in \mathcal{H}_{s-1}, v \in \mathcal{V} \right\} \right)$$
- **Properties:**
  - Explores wider combinatorial path spaces.
  - Essential for complex multi-word continuous sign translation.
  - Adds computational latency ($\approx K \times$ greedy cost).

---

## 2. Selection for Phase 3 Part 3

For the current isolated sequence dataset ($S=2$: $[\text{token\_id}, \langle\text{EOS}\rangle]$), **Greedy Decoding** is mathematically sufficient and guarantees optimal single-step selection without incurring beam search overhead. Both greedy decoding and configurable beam search interfaces are implemented in the translation model.
