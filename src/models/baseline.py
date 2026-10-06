"""
SignTalk AI: Baseline Sequence Model Implementation.

Implements a Bidirectional Gated Recurrent Unit (BiGRU) with temporal
pooling as the empirical reference baseline for spatial-temporal sign classification.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignBaselineModel(nn.Module):
    """
    Bidirectional GRU Baseline Model for Indian Sign Language gesture recognition.
    
    Accepts standardized input tensors of shape [B, C, T, V] = [B, 3, 45, 93],
    flattens spatial nodes per frame into [B, T, C * V] = [B, 45, 279],
    projects through a feed-forward layer, processes via multi-layer BiGRU,
    pools temporally across time, and emits categorical class logits.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_nodes: int = 93,
        sequence_length: int = 45,
        proj_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 10,
        pooling_type: str = "mean_max"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.raw_feat_dim = in_channels * num_nodes  # 3 * 93 = 279
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_classes = num_classes
        self.pooling_type = pooling_type

        # 1. Feature Projection Layer
        self.input_projection = nn.Sequential(
            nn.Linear(self.raw_feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout if num_layers > 1 else 0.0)
        )

        # 2. Recurrent Core (GRU)
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 3. Pooling output dimension calculation
        gru_out_dim = hidden_dim * self.num_directions  # e.g. 128 * 2 = 256
        if pooling_type == "mean_max":
            pooled_dim = gru_out_dim * 2  # 512
        else:
            pooled_dim = gru_out_dim      # 256

        # 4. Multi-Layer Perceptron Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, T, V] or [B, T, C*V]
            mask: Optional binary mask [B, 1, T, V]
            
        Returns:
            logits: Class logits tensor of shape [B, num_classes]
        """
        # Reshape [B, C, T, V] -> [B, T, C * V]
        if x.ndim == 4:
            B, C, T, V = x.shape
            # Permute to [B, T, C, V] and flatten spatial dims
            x_seq = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * V)
        elif x.ndim == 3:
            x_seq = x
            B, T, _ = x.shape
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape: {x.shape}")

        # 1. Project input frame features
        proj = self.input_projection(x_seq)  # [B, T, proj_dim]

        # 2. Process temporal sequence via BiGRU
        gru_out, _ = self.gru(proj)  # [B, T, hidden_dim * num_directions]

        # 3. Temporal Pooling
        if self.pooling_type == "mean_max":
            mean_pooled = torch.mean(gru_out, dim=1)  # [B, 256]
            max_pooled, _ = torch.max(gru_out, dim=1)  # [B, 256]
            pooled = torch.cat([mean_pooled, max_pooled], dim=-1)  # [B, 512]
        elif self.pooling_type == "mean":
            pooled = torch.mean(gru_out, dim=1)
        elif self.pooling_type == "max":
            pooled, _ = torch.max(gru_out, dim=1)
        elif self.pooling_type == "last":
            pooled = gru_out[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # 4. Class logits
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits
