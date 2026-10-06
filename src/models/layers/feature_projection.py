"""
SignTalk AI: Feature Projection Layer.

Projects ST-GCN spatiotemporal latent sequence representations into the
Transformer model embedding space:
  - Input:  [B, T, D_in] (e.g. [B, 12, 256])
  - Output: [B, T, D_out] (e.g. [B, 12, 128] or [B, 12, 256])
  - Incorporates linear projection, LayerNorm, and Dropout.
"""

from typing import Optional
import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    """
    Projects visual feature representations into Transformer model embedding dimension.
    """

    def __init__(
        self,
        in_dim: int = 256,
        embed_dim: int = 128,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Args:
            in_dim: Input feature dimensionality from visual encoder (e.g., ST-GCN final channels).
            embed_dim: Target embedding dimensionality for Transformer layers.
            dropout: Dropout probability applied to projected features.
            use_layer_norm: If True, applies LayerNorm after projection.
        """
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim

        self.projection = nn.Linear(in_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier normal initialization for projection matrix."""
        nn.init.xavier_normal_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.constant_(self.projection.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, T, in_dim].
            
        Returns:
            Projected tensor of shape [B, T, embed_dim].
        """
        if x.dim() != 3:
            raise ValueError(
                f"FeatureProjection expects a 3D tensor of shape [B, T, in_dim], "
                f"but received tensor with shape {list(x.shape)} (dim={x.dim()})."
            )

        B, T, D = x.shape
        if D != self.in_dim:
            raise ValueError(
                f"FeatureProjection input feature dimension mismatch: "
                f"expected in_dim={self.in_dim}, but got {D}."
            )

        out = self.projection(x)
        out = self.norm(out)
        out = self.dropout(out)
        return out
