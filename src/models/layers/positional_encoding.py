"""
SignTalk AI: Temporal Positional Encoding Layer.

Provides temporal order representations for Transformer architectures:
  - SinusoidalPositionalEncoding (fixed analytical, Vaswani et al. 2017)
  - LearnedPositionalEncoding (trainable positional embedding table)
"""

from typing import Optional
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
        max_len: int = 512
    ):
        """
        Args:
            embed_dim: Model embedding dimension (d_model).
            dropout: Dropout probability applied to position-augmented embeddings.
            max_len: Maximum supported temporal sequence length.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, embed_dim]

        # Registered as persistent buffer (moves automatically with model.to(device))
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, T, embed_dim].
            
        Returns:
            Tensor of shape [B, T, embed_dim] with positional encodings added.
        """
        if x.dim() != 3:
            raise ValueError(
                f"SinusoidalPositionalEncoding expects a 3D tensor [B, T, D], got {list(x.shape)}."
            )
        B, T, D = x.shape
        if D != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embed_dim}, got {D}."
            )
        if T > self.pe.shape[1]:
            raise ValueError(
                f"Sequence length T={T} exceeds maximum supported length max_len={self.pe.shape[1]}."
            )

        # Broadcast addition across batch dimension
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Trainable lookup table for temporal positional embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
        max_len: int = 512
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, T, embed_dim].
            
        Returns:
            Tensor of shape [B, T, embed_dim] with learned positional embeddings added.
        """
        if x.dim() != 3:
            raise ValueError(
                f"LearnedPositionalEncoding expects a 3D tensor [B, T, D], got {list(x.shape)}."
            )
        B, T, D = x.shape
        if D != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embed_dim}, got {D}."
            )

        device = x.device
        positions = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embedding(positions)
        return self.dropout(x + pos_emb)
