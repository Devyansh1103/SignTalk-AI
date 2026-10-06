# Loss Function Specification: SignTalk AI

**Document ID:** STAI-P3P1-008  
**Project:** SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase:** Phase 3 — Part 1 (Baseline Model & Training Infrastructure)  
**Author:** Lead AI/ML Data Engineer & Research Engineer  
**Date:** October 2026  
**Status:** Approved  

---

## 1. Mathematical Formulation

For isolated sign gesture classification across $K = 10$ mutually exclusive vocabulary classes, the primary objective function is standard categorical **Cross-Entropy Loss**:

$$\mathcal{L}_{CE} = -\frac{1}{B}\sum_{i=1}^B \ln\left(\frac{\exp(z_{i, y_i})}{\sum_{j=0}^{K-1}\exp(z_{i, j})}\right)$$

where:
- $B$: Batch size.
- $z_{i, j}$: Model output logit for sample $i$ and class $j$.
- $y_i \in \{0, 1, \dots, K-1\}$: Ground truth integer class label.

---

## 2. Class Weighting & Imbalance Assessment

### 2.1 Empirical Distribution
- In the complete canonical sequence dataset (`sequence_manifest.csv`), all 10 vocabulary classes have **exactly 6 instances** ($100.0\%$ perfectly balanced).
- In the training partition (`train.csv`, 36 samples), each class has between 3 and 4 samples.
- When `filter_rejects=True` is enabled, samples with severe motion blur ($< 8$ valid hand frames) are excluded, leaving 28 clean training samples with class support ranging from 2 to 4 samples per class.

### 2.2 Decision: Standard Unweighted Cross-Entropy
Given the tightly bounded distribution (support ratio $\max/\min = 4/2 = 2.0$), extreme class reweighting or focal loss is unwarranted and risks destabilizing gradients on compact batches. Standard Cross-Entropy provides smooth, unbiased gradient descent.
