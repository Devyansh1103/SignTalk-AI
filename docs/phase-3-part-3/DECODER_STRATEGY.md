# SignTalk AI — Decoder Strategy and Teacher Forcing

**Document ID:** `DOC-P3P3-DECODER-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** COMPLETE & DOCUMENTED  

---

## 1. Autoregressive Factorization

The target token sequence $\mathbf{y} = (y_1, y_2, \dots, y_S)$ is modeled conditionally given the encoded visual memory $\mathbf{Z} \in \mathbb{R}^{B \times T' \times D}$:
$$P(\mathbf{y} \mid \mathbf{Z}) = \prod_{s=1}^{S} P(y_s \mid y_{<s}, \mathbf{Z})$$
where $y_{<s} = (y_0, y_1, \dots, y_{s-1})$ represents preceding target tokens, anchored by $y_0 = \langle\text{BOS}\rangle$.

---

## 2. Teacher Forcing During Training

To ensure stable gradient flow and parallelized sequence processing during training, we employ **Teacher Forcing**:
- The decoder input is the prefix shifted sequence:
  $$\mathbf{y}_{\text{in}} = [\langle\text{BOS}\rangle, y_1, y_2, \dots, y_{S-1}]$$
- The loss targets are the unshifted ground truth tokens ending with $\langle\text{EOS}\rangle$:
  $$\mathbf{y}_{\text{tgt}} = [y_1, y_2, \dots, y_{S-1}, \langle\text{EOS}\rangle]$$

### Causal Masking
To prevent the self-attention mechanism in the decoder from attending to future tokens ($j > i$), an upper-triangular causal mask $\mathbf{M}_{\text{causal}} \in \mathbb{R}^{S \times S}$ is applied:
$$\mathbf{M}_{\text{causal}}(i, j) = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
This allows parallel forward evaluation across all $S$ positions during training while preserving strict causal autoregression.

---

## 3. Inference Autoregression

During inference, ground-truth target tokens are unavailable. The model executes step-by-step sequential generation:
1. Initialize $\mathbf{y}^{(0)} = [\langle\text{BOS}\rangle]$.
2. For step $s = 1, \dots, S_{\max}$:
   a. Compute decoder hidden state for last position: $\mathbf{h}_s = \text{Decoder}(\mathbf{y}^{(s-1)}, \mathbf{Z})[-1]$.
   b. Compute vocabulary probability distribution: $\mathbf{p}_s = \text{softmax}(\mathbf{W}_{\text{vocab}} \mathbf{h}_s)$.
   c. Select next token $\hat{y}_s = \arg\max \mathbf{p}_s$ (Greedy) or sample from top- $K$.
   d. Append $\hat{y}_s$ to sequence: $\mathbf{y}^{(s)} = [\mathbf{y}^{(s-1)}, \hat{y}_s]$.
   e. Terminate if $\hat{y}_s = \langle\text{EOS}\rangle$ or $s = S_{\max}$.
