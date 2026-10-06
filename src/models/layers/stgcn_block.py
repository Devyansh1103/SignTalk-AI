"""
SignTalk AI: ST-GCN Unit Block.

Combines Spatial Graph Convolution, Temporal 1D Convolution, Batch Normalization,
Non-linear Activation, Dropout, and learned or identity Residual Connections.
"""

from typing import Optional
import torch
import torch.nn as nn

from src.models.layers.graph_conv import SpatialGraphConv
from src.models.layers.temporal_conv import TemporalConv


class STGCNBlock(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Unit Block.
    
    Structure:
      x -> SpatialGraphConv -> BatchNorm -> ReLU ->
           TemporalConv -> BatchNorm -> Dropout -> (+) Residual -> ReLU -> output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_subsets: int = 3,
        num_nodes: int = 93,
        temporal_kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.3,
        residual: bool = True,
        use_learnable_edge_weights: bool = True
    ):
        """
        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            num_subsets: Number of graph partition subsets (K).
            num_nodes: Number of graph nodes (V=93).
            temporal_kernel_size: Window length for temporal convolution.
            stride: Temporal stride factor (1 for preservation, 2 for downsampling).
            dropout: Dropout probability.
            residual: Whether to include residual connection.
            use_learnable_edge_weights: Learnable edge attention weighting.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.residual = residual

        # 1. Spatial Graph Convolution
        self.sgcn = SpatialGraphConv(
            in_channels=in_channels,
            out_channels=out_channels,
            num_subsets=num_subsets,
            num_nodes=num_nodes,
            use_learnable_edge_weights=use_learnable_edge_weights
        )
        self.bn_spat = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2. Temporal 1D Convolution
        self.tcn = TemporalConv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=temporal_kernel_size,
            stride=stride
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 3. Residual Connection
        if not residual:
            self.res = None
        elif in_channels == out_channels and stride == 1:
            self.res = nn.Identity()
        else:
            # Learned projection to match channel count and temporal stride
            self.res = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=(stride, 1),
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ST-GCN block.
        
        Args:
            x: Input tensor [B, C_in, T, V].
            A: Partitioned adjacency tensor [K, V, V].
            
        Returns:
            Output tensor [B, C_out, T', V].
        """
        # Residual branch
        if self.res is not None:
            res_val = self.res(x)
        else:
            res_val = 0.0

        # Spatial Graph Convolution
        h = self.sgcn(x, A)
        h = self.bn_spat(h)
        h = self.relu(h)

        # Temporal Convolution
        h = self.tcn(h)
        h = self.dropout(h)

        # Merge with residual
        h = h + res_val
        out = self.relu(h)

        return out
