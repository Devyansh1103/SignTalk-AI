"""
SignTalk AI: gloss -> sentence Transformer.

Encoder-decoder Transformer (2 + 2 layers, 4 heads, d_model 128) that turns the glosses the
ST-GCN recognised ("<2hi> i mother sick") into a sentence ("मेरी माँ बीमार हैं ।"). One model
serves both languages; the first source token picks the output language. Trained on the
synthetic corpus of src/nlp/gloss_grammar.py by scripts/train_gloss_translator.py.
"""
from typing import List, Tuple

import torch
import torch.nn as nn

from src.models.layers.positional_encoding import SinusoidalPositionalEncoding

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
LANG_TOKENS = {"en": "<2en>", "hi": "<2hi>"}
PUNCT = ".,!?।"


def tokenize_sentence(s: str) -> List[str]:
    """'Hello, how are you?' -> ['Hello', ',', 'how', 'are', 'you', '?']"""
    for p in PUNCT:
        s = s.replace(p, f" {p} ")
    return s.split()


def detokenize(tokens: List[str]) -> str:
    out = ""
    for t in tokens:
        out += t if (t in PUNCT or not out) else " " + t
    return out


class GlossTranslator(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int = 128, nhead: int = 4,
                 encoder_layers: int = 2, decoder_layers: int = 2, ff_dim: int = 512,
                 dropout: float = 0.1, max_len: int = 32, pad_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_idx)
        self.pos = SinusoidalPositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers, dim_feedforward=ff_dim, dropout=dropout, batch_first=True,
        )
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """src [B, S], tgt [B, T] (starts with <bos>) -> logits [B, T, tgt_vocab]."""
        T = tgt.size(1)
        pos = torch.arange(T, device=tgt.device)
        causal = pos[None, :] > pos[:, None]  # True above the diagonal; not triu: onnxruntime has no bool Trilu
        src_pad = src == self.pad_idx
        # No sqrt(d_model) scaling: with N(0, 1) embeddings it makes them ~11x the positional
        # encoding, and the model can hardly tell "I BLUE SHIRT" from "I SHIRT BLUE"
        x = self.pos(self.src_emb(src))
        y = self.pos(self.tgt_emb(tgt))
        h = self.transformer(x, y, tgt_mask=causal, src_key_padding_mask=src_pad,
                             tgt_key_padding_mask=tgt == self.pad_idx, memory_key_padding_mask=src_pad)
        return self.out(h)

    @torch.no_grad()
    def greedy(self, src: torch.Tensor, bos: int, eos: int, max_len: int = 24) -> Tuple[List[int], float]:
        """Greedy decoding for one source [1, S]: (token ids without <bos>/<eos>, lowest step probability)."""
        ys = torch.tensor([[bos]], device=src.device)
        min_p = 1.0
        for _ in range(max_len):
            probs = self(src, ys)[0, -1].softmax(-1)
            p, nxt = probs.max(-1)
            min_p = min(min_p, float(p))
            if int(nxt) == eos:
                break
            ys = torch.cat([ys, nxt.view(1, 1)], dim=1)
        return ys[0, 1:].tolist(), min_p


class NextTokenProbs(nn.Module):
    """ONNX wrapper: next-token probabilities at every target position, for fixed-size inputs
    padded with <pad> (the TorchScript exporter bakes sequence lengths into the attention
    reshapes, so the browser pads instead of using dynamic axes). Read row len(prefix) - 1."""

    def __init__(self, model: GlossTranslator):
        super().__init__()
        self.model = model

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self.model(src, tgt).softmax(-1)
