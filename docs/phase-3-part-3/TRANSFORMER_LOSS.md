# SignTalk AI — Transformer Loss Strategy

**Document ID:** `DOC-P3P3-LOSS-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** COMPLETE & DOCUMENTED  

---

## 1. Objective Function

The objective for autoregressive language sequence generation is the token-level cross-entropy loss over the predicted vocabulary distributions, with sequence padding explicitly masked out:

$$\mathcal{L}_{\text{token}}(\theta) = -\frac{1}{\sum_{b=1}^{B} \sum_{s=1}^{S} \mathbb{I}(y_{b,s} \ne \text{pad\_id})} \sum_{b=1}^{B} \sum_{s=1}^{S} \mathbb{I}(y_{b,s} \ne \text{pad\_id}) \log P_\theta(y_{b,s} \mid y_{b,<s}, \mathbf{Z}_b)$$

In PyTorch, this is implemented natively via:
```python
criterion = nn.CrossEntropyLoss(
    ignore_index=pad_idx, label_smoothing=label_smoothing
)
```

---

## 2. Padding Masking Policy

The padding token `<PAD>` (ID 0) is an artifact of batch collation across sequences of different lengths. Computing gradients with respect to padding predictions artificially inflates gradients on non-semantic positions and degrades lexical calibration.
Setting `ignore_index=0` ensures:
- Zero loss contribution from padded positions.
- Normalization denominator counts only valid non-padding tokens.

---

## 3. Label Smoothing Policy

Label smoothing ($\epsilon \in [0.0, 0.1]$) replaces hard 1-hot target distributions with:
$$q(k) = (1 - \epsilon) \delta_{k, y} + \frac{\epsilon}{|\mathcal{V}|}$$
- **Current Configuration:** We set `label_smoothing = 0.0` (standard cross-entropy) by default to maintain parity with the Phase 3 Part 1 and Part 2 baselines.
- An optional ablation setting of $\epsilon = 0.05$ is parameterized in `configs/transformer.yaml` for testing regularized sequence generation.
