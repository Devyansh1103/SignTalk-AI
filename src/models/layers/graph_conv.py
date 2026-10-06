"""
SignTalk AI: Spatial Graph Convolution Layer.

Implements spatial graph convolution operating over partitioned anatomical
skeletal graphs for tensor shape [B, C, T, V].
"""

from typing import Optional
import torch
import torch.nn as nn


class SpatialGraphConv(nn.Module):
    """
    Spatial Graph Convolution layer.
    
    Operates on spatiotemporal tensors [B, C_in, T, V] using a partitioned
    adjacency matrix [K, V, V] and optional learnable edge importance weighting.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_subsets: int = 3,
        num_nodes: int = 93,
        use_learnable_edge_weights: bool = True
    ):
        """
        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            num_subsets: Number of partition subsets in adjacency matrix (K).
            num_nodes: Number of graph nodes (V=93).
            use_learnable_edge_weights: Whether to learn an element-wise edge mask M.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = num_subsets
        self.num_nodes = num_nodes
        self.use_learnable_edge_weights = use_learnable_edge_weights

        # 1x1 convolution projecting in_channels -> K * out_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * num_subsets,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )

        # Learnable edge importance mask [K, V, V]
        if self.use_learnable_edge_weights:
            self.edge_importance = nn.Parameter(torch.ones(num_subsets, num_nodes, num_nodes))
        else:
            self.register_parameter("edge_importance", None)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initializes weights using Kaiming normal distribution."""
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, C_in, T, V].
            A: Normalized adjacency tensor of shape [K, V, V].
            
        Returns:
            Output tensor of shape [B, C_out, T, V].
        """
        if x.dim() != 4:
            raise ValueError(f"SpatialGraphConv expects 4D input [B, C, T, V], got shape: {x.shape}")
        
        B, C, T, V = x.shape
        if C != self.in_channels:
            raise ValueError(f"Channel dimension mismatch: Expected {self.in_channels}, got {C}")
        if V != self.num_nodes:
            raise ValueError(f"Node dimension mismatch: Expected {self.num_nodes}, got {V}")
        if A.shape[0] != self.num_subsets or A.shape[1] != V or A.shape[2] != V:
            raise ValueError(f"Adjacency shape mismatch: Expected [{self.num_subsets}, {V}, {V}], got {A.shape}")

        # Compute effective adjacency: A_eff = A * M
        if self.use_learnable_edge_weights and self.edge_importance is not None:
            A_eff = A * self.edge_importance
        else:
            A_eff = A

        # Linear projection over input channels: [B, K * C_out, T, V]
        x_proj = self.conv(x)

        # Reshape to [B, K, C_out, T, V]
        x_proj = x_proj.view(B, self.num_subsets, self.out_channels, T, V)

        # Tensor contraction over partition K and node V:
        # For each subset k: sum_u (x_proj[b, k, c, t, u] * A_eff[k, u, v])
        # Note: A_eff[k, v, u] defines message flow from neighbor u to target v.
        # einsum: 'nkctv,kvw->nctw' where v is source neighbor, w is target node
        y = torch.einsum("nkctv,kvw->nctw", x_proj, A_eff)

        return y
