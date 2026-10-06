# SignTalk AI — ST-GCN Optimization & Scheduling Strategy

**Document ID:** `DOC-P3P2-OPT-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Configuration Reference:** [`configs/stgcn.yaml`](file:///d:/SignAI/configs/stgcn.yaml)  

---

## 1. Optimizer Selection: Decoupled Weight Decay (AdamW)

Spatial-temporal graph convolutional networks feature multiple heterogeneous parameter groups:
1. Spatial 1x1 convolutions ($\mathbf{W}$)
2. Learnable edge importance masks ($\mathbf{M}$)
3. Temporal 1D convolutions ($K_t = 9$)
4. Batch normalization scale and shift parameters ($\gamma, \beta$)
5. Residual projection weights

To prevent $L_2$ weight regularization from interfering with gradient moment tracking across these heterogeneous layers, **AdamW** (Loshchilov & Hutter, ICLR 2019) is chosen.

### Hyperparameter Settings:
- **Base Learning Rate ($\eta_0$):** $1.0 \times 10^{-3}$ ($0.001$)
- **Momentum Coefficients:** $\beta_1 = 0.9, \beta_2 = 0.999$
- **Numerical Stability ($\epsilon$):** $1.0 \times 10^{-8}$
- **Weight Decay ($\lambda$):** $1.0 \times 10^{-4}$ ($0.0001$)
- **Gradient Clipping:** Max norm $1.0$ (via `torch.nn.utils.clip_grad_norm_`)

---

## 2. Learning Rate Scheduler: Cosine Annealing

Rather than step decay (which requires heuristic milestone definitions), learning rate scheduling uses **Cosine Annealing** without restarts (Loshchilov & Hutter, ICLR 2017):

$$\eta_t = \eta_{\min} + \frac{1}{2} (\eta_0 - \eta_{\min}) \left( 1 + \cos\left( \frac{t}{T_{\max}} \pi \right) \right)$$

- **Maximum Epochs ($T_{\max}$):** $80$
- **Minimum Learning Rate ($\eta_{\min}$):** $1.0 \times 10^{-5}$ ($0.00001$)

The smooth cosine curve maintains sufficient exploratory gradient step sizes during early epochs, gradually transitioning to fine-grained parameter refinement in later epochs.

---

## 3. Early Stopping Policy

To prevent overfitting on the training partition:
- **Monitored Metric:** `val_macro_f1` (Validation Macro-Averaged F1-Score).
- **Patience:** 20 epochs.
- **Action:** If `val_macro_f1` fails to exceed the historical peak for 20 consecutive epochs, training terminates and `best_checkpoint.pt` is loaded for final evaluation.
