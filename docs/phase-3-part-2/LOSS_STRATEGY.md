# SignTalk AI — ST-GCN Loss Function Strategy

**Document ID:** `DOC-P3P2-LOSS-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Task:** 10-Class Isolated Sign Recognition  
**Loss Function:** Standard Multi-Class Cross-Entropy Loss  
**Comparison Reference:** Baseline Loss Specification ([`docs/phase-3-part-1/LOSS_FUNCTION.md`](file:///d:/SignAI/docs/phase-3-part-1/LOSS_FUNCTION.md))  

---

## 1. Objective and Comparative Rigor

To guarantee a scientifically sound, fair comparison between the Phase 3 Part 1 baseline model (`SignBaselineModel`) and the ST-GCN model (`SignSTGCN`), the training loss objective must remain **identical** unless empirical data necessitates an adjustment.

Altering the loss formulation (e.g. introducing Focal Loss, label smoothing, or arbitrary class re-weighting) would conflate architectural improvements with loss regularization artifacts.

---

## 2. Loss Formulation: Multi-Class Cross-Entropy Loss

Given a batch of $B$ sequences, model logits $\mathbf{z} \in \mathbb{R}^{B \times C}$, and ground-truth integer class labels $y \in \{0, \dots, C-1\}$:

$$\mathcal{L}_{\text{CE}}(\mathbf{z}, y) = -\frac{1}{B} \sum_{b=1}^{B} \log \left( \frac{\exp(z_{b, y_b})}{\sum_{c=0}^{C-1} \exp(z_{b, c})} \right)$$

PyTorch implementation:
```python
loss_fn = torch.nn.CrossEntropyLoss()
```

---

## 3. Class Imbalance Re-Evaluation

The training partition (`data/manifests/train.csv`) contains 36 sequences distributed across the 10 classes:
- Classes 0 to 9 each contain between 3 and 4 sequences.
- Max-to-min class representation ratio is $4 / 3 = 1.33:1$.
- Imbalance ratio is minimal ($< 1.5$), so class weighting is unnecessary and would risk destabilizing batch gradients on this small training set.

---

## 4. Decision Log

1. **Standard Cross-Entropy Retained:** CrossEntropyLoss is maintained as the primary objective function.
2. **No Label Smoothing in Primary Run:** Label smoothing is omitted to evaluate the raw discriminative capacity of spatial graph convolutions.
3. **Parity with Baseline:** Ensures that any accuracy or F1 improvements observed are strictly attributable to the ST-GCN spatiotemporal graph modeling.
