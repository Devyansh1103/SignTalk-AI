"""
SignTalk AI: Temporal Convolution Layer.

Applies 1D temporal convolution across sequence frames T while preserving
or operating independently over graph nodes V.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class TemporalConv(nn.Module):
    """
    Temporal Convolution layer.
    
    Operates on [B, C, T, V] along the temporal axis T with configurable
    kernel size, stride, padding, and dilation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dilation: int = 1,
        padding: Optional[int] = None
    ):
        """
        Args:
            in_channels: Input channel dimension.
            out_channels: Output channel dimension.
            kernel_size: Temporal window receptive field (default: 9).
            stride: Temporal downsampling factor (1 or 2).
            dilation: Temporal dilation rate (default: 1).
            padding: Temporal padding (default: symmetric ((kernel_size - 1) * dilation) // 2).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        if padding is None:
            self.padding = ((kernel_size - 1) * dilation) // 2
        else:
            self.padding = padding

        # Conv2d over (T, V) where spatial kernel is (kernel_size, 1)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(self.kernel_size, 1),
            stride=(self.stride, 1),
            padding=(self.padding, 0),
            dilation=(self.dilation, 1),
            bias=True
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initializes weights using Kaiming normal distribution."""
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C_in, T, V].
            
        Returns:
            Output tensor [B, C_out, T', V].
        """
        if x.dim() != 4:
            raise ValueError(f"TemporalConv expects 4D input [B, C, T, V], got shape: {x.shape}")
        
        y = self.conv(x)
        y = self.bn(y)
        return y
