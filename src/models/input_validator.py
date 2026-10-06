"""
SignTalk AI: ST-GCN Input Validator.

Enforces strict structural and dimensional invariants on input spatiotemporal
tensors before training and inference.
"""

from typing import Tuple, Dict, Any, Optional
import torch


class STGCNInputValidationError(ValueError):
    """Raised when an input tensor violates ST-GCN structural dimensions."""
    pass


class STGCNInputValidator:
    """
    Validates that input tensors match the expected [B, C, T, V] spatiotemporal format.
    """

    def __init__(
        self,
        expected_channels: int = 3,
        expected_temporal_length: int = 45,
        expected_nodes: int = 93,
        allow_variable_length: bool = False
    ):
        """
        Args:
            expected_channels: Number of spatial coordinates (default: 3).
            expected_temporal_length: Number of time frames (default: 45).
            expected_nodes: Anatomical landmarks (default: 93).
            allow_variable_length: Whether temporal length T can vary across batches.
        """
        self.expected_channels = expected_channels
        self.expected_temporal_length = expected_temporal_length
        self.expected_nodes = expected_nodes
        self.allow_variable_length = allow_variable_length

    def validate(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[int, int, int, int]:
        """
        Validates the 4D input tensor shape [B, C, T, V].
        
        Args:
            x: Input PyTorch tensor.
            mask: Optional visibility mask tensor.
            
        Returns:
            Tuple of validated dimensions (B, C, T, V).
            
        Raises:
            STGCNInputValidationError: If any dimension or type constraint is violated.
        """
        if not isinstance(x, torch.Tensor):
            raise STGCNInputValidationError(f"Expected torch.Tensor input, got {type(x).__name__}")

        if x.dim() != 4:
            raise STGCNInputValidationError(
                f"ST-GCN requires a 4D tensor of shape [B, C, T, V], but received {x.dim()}D tensor with shape {list(x.shape)}"
            )

        B, C, T, V = x.shape

        if B <= 0:
            raise STGCNInputValidationError(f"Batch dimension B must be positive, got B={B}")

        if C != self.expected_channels:
            raise STGCNInputValidationError(
                f"Channel dimension mismatch: Expected C={self.expected_channels} (x, y, z coordinates), "
                f"got C={C}"
            )

        if not self.allow_variable_length and T != self.expected_temporal_length:
            raise STGCNInputValidationError(
                f"Temporal dimension mismatch: Expected T={self.expected_temporal_length} frames, "
                f"got T={T}"
            )

        if V != self.expected_nodes:
            raise STGCNInputValidationError(
                f"Node dimension mismatch: Expected V={self.expected_nodes} anatomical landmarks, "
                f"got V={V}"
            )

        # Check mask compatibility if present
        if mask is not None:
            if mask.dim() != 4:
                raise STGCNInputValidationError(
                    f"Visibility mask must be 4D [B, 1, T, V], got {mask.dim()}D shape {list(mask.shape)}"
                )
            if mask.shape[0] != B or mask.shape[2] != T or mask.shape[3] != V:
                raise STGCNInputValidationError(
                    f"Mask dimensions [B={mask.shape[0]}, 1, T={mask.shape[2]}, V={mask.shape[3]}] do not match "
                    f"input tensor dimensions [B={B}, C={C}, T={T}, V={V}]"
                )

        return B, C, T, V


# Alias for general import compatibility
InputValidator = STGCNInputValidator
