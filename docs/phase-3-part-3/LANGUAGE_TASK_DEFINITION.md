# SignTalk AI — Language Task Definition

**Document ID:** `DOC-P3P3-TASK-DEF-001`  
**Phase:** Phase 3 — Part 3 (Transformer/NLP Translation Layer)  
**Date:** October 2026  
**Status:** DEFINED & BOUNDED  

---

## 1. Task Taxonomy Classification

In computational sign language processing, language tasks span a structured hierarchy:
1. **Isolated Sign Classification:** Given a segment $X \in \mathbb{R}^{C \times T \times V}$, predict a single categorical label $y \in \{1, \dots, K\}$.
2. **Continuous Sign Language Recognition (CSLR):** Given an unsegmented continuous video $X_{1:T}$, predict an ordered sequence of glosses $(g_1, g_2, \dots, g_M)$ (where $M \le T$) without explicit frame-level boundary alignments (typically solved via CTC or Transformer decoders).
3. **Sign Language Translation (SLT):** Given video or landmarks $X_{1:T}$, generate a grammatically correct natural language sentence in a spoken target language (e.g., English text $(w_1, w_2, \dots, w_N)$).

### Classification of the Current SignTalk AI Dataset
- **Supported Task:** **Isolated Gloss Recognition & Single-Token Translation Alignment**.
- **Supervision Available:** Each sample is an isolated recording mapped to a single semantic gloss token $g \in \mathcal{V}_{\text{gloss}}$ and a single English translation phrase $w \in \mathcal{V}_{\text{trans}}$.
- **Explicit Boundary:** The current task is **NOT** continuous sentence-level translation. We do not fabricate artificial multi-word sentences or claim continuous machine translation, adhering strictly to **CASE C** of the architectural specification.

---

## 2. Formal Task Definition

### Input
- Spatiotemporal landmark sequence $\mathbf{X} \in \mathbb{R}^{B \times C \times T \times V}$ ($C=3, T=45, V=93$).
- Temporal validity mask $\mathbf{M}_{\text{src}} \in \{0, 1\}^{B \times 1 \times T \times V}$.

### Intermediate Visual Latent Representation
- ST-GCN spatiotemporal feature representation $\mathbf{Z}_{\text{visual}} = \text{ST-GCN}(\mathbf{X}) \in \mathbb{R}^{B \times T' \times D_{\text{visual}}}$ where $T'=12$ and $D_{\text{visual}}=256$.

### Autoregressive Language Task Interface
- Target Gloss/Word Token Sequence: $\mathbf{y} = (y_0, y_1, y_2) = (\langle\text{BOS}\rangle, \text{token\_id}, \langle\text{EOS}\rangle)$.
- Decoder conditioning:
  $$P(\mathbf{y} \mid \mathbf{Z}_{\text{visual}}) = \prod_{s=1}^{S} P(y_s \mid y_{<s}, \mathbf{Z}_{\text{visual}})$$
- Output: Token probability distribution over the vocabulary $\mathcal{V}$ ($|\mathcal{V}| = 14$).

---

## 3. Evaluation Metrics
1. **Token Accuracy:** Percentage of generated target tokens matching ground truth.
2. **Sequence Exact Match:** Percentage of sequences where the entire decoded token sequence $(\hat{y}_1, \dots, \hat{y}_S)$ matches ground truth exactly.
3. **Macro and Weighted F1-Score:** Per-token and per-class harmonic mean of precision and recall.
4. **BLEU-1 / ROUGE-L:** Automatic translation concordance scores between decoded tokens and target reference strings.
