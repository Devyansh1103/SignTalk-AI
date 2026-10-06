"""
SignTalk AI: Spatial-Temporal Graph Convolutional Network (ST-GCN).

Full deep architecture for Indian Sign Language recognition on 93-node
MediaPipe multimodal landmark sequences [B, C, T, V].
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn

from src.models.graph import SignGraph
from src.models.layers.stgcn_block import STGCNBlock


class SignSTGCN(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network for sign language sequence classification.
    
    Architecture:
      Input [B, C_in, T, V]
        -> Data Batch Normalization
        -> Stack of ST-GCN Blocks (Spatial Graph Conv + Temporal Conv + Residual)
        -> Global Average Pooling (Spatial & Temporal)
        -> Classification Head
        -> Class Logits [B, num_classes]
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        num_nodes: int = 93,
        sequence_length: int = 45,
        graph_strategy: str = "spatial",
        block_channels: Optional[List[int]] = None,
        block_strides: Optional[List[int]] = None,
        temporal_kernel_size: int = 9,
        dropout: float = 0.3,
        residual: bool = True,
        use_learnable_edge_weights: bool = True
    ):
        """
        Args:
            in_channels: Spatial coordinate channels (default: 3 for x, y, z).
            num_classes: Vocabulary size (default: 10).
            num_nodes: Anatomical landmarks (default: 93).
            sequence_length: Temporal frame count (default: 45).
            graph_strategy: Adjacency partitioning ('spatial', 'distance', 'uniform').
            block_channels: Channel progression across ST-GCN blocks.
            block_strides: Temporal downsampling strides per block.
            temporal_kernel_size: Window length for temporal convolutions.
            dropout: Dropout probability.
            residual: Whether to enable residual skip connections.
            use_learnable_edge_weights: Whether to learn edge attention masks.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.graph_strategy = graph_strategy

        # Default 6-block progression
        if block_channels is None:
            block_channels = [64, 64, 128, 128, 256, 256]
        if block_strides is None:
            block_strides = [1, 1, 2, 1, 2, 1]

        if len(block_channels) != len(block_strides):
            raise ValueError(
                f"block_channels ({len(block_channels)}) and block_strides ({len(block_strides)}) "
                "must have matching lengths."
            )

        self.block_channels = block_channels
        self.block_strides = block_strides

        # 1. Initialize Graph & Adjacency Matrix
        self.graph = SignGraph(strategy=graph_strategy)
        self.graph.validate_input_nodes(num_nodes)
        
        # Register adjacency tensor as a persistent model buffer [K, V, V]
        # Automatically moved to correct device when model.to(device) is called
        initial_A = self.graph.get_adjacency_tensor()
        self.register_buffer("A", initial_A)
        self.num_subsets = self.A.shape[0]

        # 2. Input Data Normalization
        # Normalizes raw coordinates [B, C, T, V] along channel and node dimensions
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        # 3. Stacked ST-GCN Blocks
        self.blocks = nn.ModuleList()
        current_in = in_channels
        for out_ch, stride in zip(block_channels, block_strides):
            self.blocks.append(
                STGCNBlock(
                    in_channels=current_in,
                    out_channels=out_ch,
                    num_subsets=self.num_subsets,
                    num_nodes=num_nodes,
                    temporal_kernel_size=temporal_kernel_size,
                    stride=stride,
                    dropout=dropout,
                    residual=residual,
                    use_learnable_edge_weights=use_learnable_edge_weights
                )
            )
            current_in = out_ch

        # 4. Classification Head
        final_dim = block_channels[-1]
        self.head_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(final_dim, num_classes)
        self._reset_head_parameters()

    def _reset_head_parameters(self):
        """Initializes classification head parameters."""
        nn.init.normal_(self.fc.weight, 0, 0.02)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0.0)

    def extract_features(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pool_temporal: bool = True
    ) -> torch.Tensor:
        """
        Extracts high-level pooled spatiotemporal representations before logits projection.
        
        Args:
            x: Input tensor [B, C, T, V].
            mask: Optional visibility mask [B, 1, T, V].
            pool_temporal: If True, pools both time and nodes to return [B, C_final].
                           If False, pools nodes only and returns temporal sequence [B, T', C_final].
            
        Returns:
            Feature tensor: [B, C_final] if pool_temporal=True, or [B, T', C_final] if False.
        """
        B, C, T, V = x.shape
        if C != self.in_channels or V != self.num_nodes:
            raise ValueError(
                f"Input tensor dimension mismatch: Got [B={B}, C={C}, T={T}, V={V}], "
                f"expected C={self.in_channels}, V={self.num_nodes}"
            )

        # Apply visibility mask if present
        if mask is not None:
            x = x * mask.float()

        # Input Normalization: reshape to [B, C*V, T]
        x_norm = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x_norm = self.data_bn(x_norm)
        # Reshape back to [B, C, T, V]
        x = x_norm.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()

        # Propagate through ST-GCN block stack
        h = x
        for block in self.blocks:
            h = block(h, self.A)

        # Spatial average pooling over graph nodes V: [B, C_final, T', V] -> [B, C_final, T']
        spatial_pooled = h.mean(dim=3)

        if pool_temporal:
            # Global Average Pooling: [B, C_final]
            return spatial_pooled.mean(dim=2)
        else:
            # Transpose to sequence format for Transformer: [B, T', C_final]
            return spatial_pooled.transpose(1, 2).contiguous()

    def extract_temporal_sequence(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extracts temporal sequence of spatial node-averaged embeddings [B, T', C_final].
        """
        return self.extract_features(x, mask=mask, pool_temporal=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass producing class logits.
        
        Args:
            x: Input tensor [B, C, T, V].
            mask: Optional visibility mask [B, 1, T, V].
            
        Returns:
            Logits tensor [B, num_classes].
        """
        features = self.extract_features(x, mask)
        features = self.head_dropout(features)
        logits = self.fc(features)
        return logits
