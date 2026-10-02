# 09. Transformer Sequence Translation Architecture: SignTalk AI

**Document ID:** STAI-P1P2-009  
**Project Name:** SignTalk AI  
**Document Status:** Approved Engineering Blueprint (Phase 1 — Part 2)  

---

## 1. Architectural Role & Design Rationale

While the ST-GCN encoder compresses raw spatial-temporal joint graphs into a compact sequence of latent visual-gestural tokens $\mathbf{H} = (\mathbf{h}_1, \dots, \mathbf{h}_{T'}) \in \mathbb{R}^{12 \times 256}$, it does not generate natural language. 

The **Transformer Decoder** bridges the structural gap between the non-linear, spatial grammar of Indian Sign Language and the linear, syntactically structured grammar of written English:
- **Reordering Capability:** Handles non-monotonic reordering between ISL (Subject-Object-Verb) and English (Subject-Verb-Object).
- **Lightweight Footprint:** A compact 3-layer architecture avoids the multi-gigabyte memory overhead and hallucination tendencies of massive pre-trained LLMs, executing autoregressive decoding in $< 100\text{ ms}$ on local CPUs.

```mermaid
graph TD
    subgraph TransformerDecoder [Transformer Sequence Translation Decoder]
        MEM[ST-GCN Memory: H in R^B x 12 x 256] --> CROSS_ATTN[Multi-Head Cross-Attention: 4 Heads, Dim 256]
        
        TOK_IN[Target Tokens: y_1..y_u-1] --> EMB[Word Embedding: Dim 256]
        EMB --> POS[Sinusoidal Positional Encoding]
        POS --> MASK_ATTN[Causal Masked Self-Attention]
        MASK_ATTN --> ADD_NORM1[Add & LayerNorm]
        
        ADD_NORM1 --> CROSS_ATTN
        CROSS_ATTN --> ADD_NORM2[Add & LayerNorm]
        
        ADD_NORM2 --> FFN[Feed-Forward Network: 256 -> 1024 -> 256 GeLU]
        FFN --> ADD_NORM3[Add & LayerNorm]
        
        ADD_NORM3 --> REPEAT[Stack of 3 Decoder Layers]
        REPEAT --> PROJ[Linear Projection Head: 256 -> |V_text|=500]
        PROJ --> SOFTMAX[Softmax Vocabulary Distribution P_u]
        SOFTMAX --> ARMAX[Greedy / Beam Search k=3 Token Selection]
    end
```

---

## 2. Transformer Decoder Layer Configurations

| Component / Layer | Hyperparameter Specification | Mathematical Description / Purpose | Parameter Count |
| :--- | :--- | :--- | :---: |
| **Target Word Embedding** | Dimension $d_{model} = 256$, Vocab Size $|\mathcal{V}_{text}| = 500$ | Maps discrete token IDs into dense semantic vectors $\mathbf{E} \in \mathbb{R}^{500 \times 256}$. | $128,000$ |
| **Positional Encoding** | Fixed Sinusoidal, Max Length $L_{max} = 64$ | Injects word order information without adding learnable parameters. | $0$ (Static) |
| **Causal Self-Attention** | $n_{heads} = 4, d_k = d_v = 64$, Dropout $p = 0.1$ | Masked attention preventing token $u$ from attending to future tokens $u' > u$. | $263,168$ |
| **Cross-Attention** | $n_{heads} = 4, d_k = d_v = 64$, Dropout $p = 0.1$ | Queries attend to encoder latent visual memory $\mathbf{H} \in \mathbb{R}^{12 \times 256}$. | $263,168$ |
| **Feed-Forward Network** | $d_{ff} = 1024$, GeLU Activation, Dropout $p = 0.1$ | Two-layer MLP with expansion factor $4\times$: $\operatorname{GeLU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$. | $526,336$ |
| **Layer Normalization** | 3 LayerNorms per block, $\epsilon = 1 \times 10^{-5}$ | Post-LN stabilization with residual connections. | $3,072$ |
| **Stack Multiplier** | **3 Identical Decoder Blocks** | Multiplies layer parameters by $3\times$. | $\times 3 \approx 3,168,000$ |
| **Final Vocabulary Head** | Linear $(256 \rightarrow 500)$ (Tied with embedding weights) | Generates unnormalized log-probability distribution over vocabulary. | $0$ (Weight Tied) |
| **TOTAL TRANSFORMER** | **3 Layers, 4 Heads, $d_{model} = 256$** | — | **$\approx \mathbf{1.85\text{M}}$ Parameters** |

---

## 3. Vocabulary Design & Autoregressive Decoding Protocol

### 3.1 Vocabulary Tokenization Scheme
The translation vocabulary $|\mathcal{V}_{text}| = 500$ tokens encompasses:
- **Special Control Tokens (4):** `<PAD>` (0), `<SOS>` (1, Start-of-Sequence), `<EOS>` (2, End-of-Sequence), `<UNK>` (3, Unknown/Out-of-Vocabulary).
- **Core MVP Vocabulary (120 words):** Lemmatized words and functional grammar words (articles, prepositions, auxiliary verbs: *is, are, have, since, to, in, at, please, doctor, pain, hospital*).
- **Domain Phrases & Punctuation (20 tokens):** Period, comma, question mark.

### 3.2 Autoregressive Decoding with Caching
During inference, sentence generation executes step-by-step:
1. Initialize input sequence with start token: $\mathbf{y}_0 = [\text{<SOS>}]$.
2. At step $u$, compute next-token probability distribution:
   $$P(y_u \mid y_{<u}, \mathbf{H}) = \operatorname{Softmax}\left( \operatorname{TransformerDecoder}(y_{<u}, \mathbf{H}) \right)$$
3. **Greedy Selection (Real-Time Mode):**
   $$\hat{y}_u = \arg\max_{w \in \mathcal{V}_{text}} P(y_u = w \mid y_{<u}, \mathbf{H})$$
4. Append $\hat{y}_u$ to the generated sequence.
5. Terminate when $\hat{y}_u = \text{<EOS>}$ or when maximum length $u = U_{max} = 20$ tokens is reached.

### 3.3 Calibrated Confidence Calculation
Overall sentence confidence $\bar{c}_{sentence}$ is computed as the geometric mean of individual token likelihoods, penalized for brevity:
$$\bar{c}_{sentence} = \left( \prod_{u=1}^{U} P(\hat{y}_u \mid \hat{y}_{<u}, \mathbf{H}) \right)^{\frac{1}{U}} \times \min\left(1.0, \frac{U}{3}\right)$$
If $\bar{c}_{sentence} < 0.50$, the system triggers the low-confidence safety protocol rather than rendering unreliable medical/civic text.
