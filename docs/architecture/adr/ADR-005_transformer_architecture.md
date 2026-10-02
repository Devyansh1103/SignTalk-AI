# ADR-005: Selection of Transformer Sequence-to-Sequence Translation Decoder

**Status:** APPROVED  
**Date:** October 2026  
**Context:** Indian Sign Language follows a visual-spatial grammar with Subject-Object-Verb (SOV) sentence structure, whereas target English text follows Subject-Verb-Object (SVO) order. A pure sign classifier or linear gloss mapper cannot produce grammatically coherent sentences.

## Decision
Deploy a **lightweight 3-layer Transformer Decoder** (Camgoz et al., CVPR 2020) that applies multi-head cross-attention over the ST-GCN latent sequence representations to autoregressively decode natural-language English tokens.

## Evaluated Alternatives
1. **Rule-Based Gloss-to-Text Mapping:** Brittle; requires exhaustive hand-crafted grammar rules; fails on unmodeled co-articulations and complex tense agreements.
2. **Heavy Pretrained Large Language Models (LLMs - e.g., Llama-3, GPT-4):** Massive computational footprint ($> 8\text{ GB}$ VRAM); impossible to run with $< 100\text{ ms}$ decoding latency on consumer laptops; introduces severe text hallucination risks in healthcare.
3. **Seq2Seq LSTM with Bahdanau Attention:** Viable, but suffers from vanishing gradients over longer sequences and lacks the parallel training efficiency and expressive multi-head attention of Transformers.

## Consequences
- **Positive:** Models non-monotonic word reordering between ISL and English; generates fluent natural language; compact parameter footprint ($< 4\text{M}$ parameters) enabling fast inference.
- **Negative:** Requires parallel continuous sign-to-text datasets (ISL-CSLTR) for effective sequence training; requires careful regularization (label smoothing, dropout) to prevent overfitting.
