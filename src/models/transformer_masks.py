"""
SignTalk AI: Transformer Attention & Padding Masks.

Provides masking utilities for visual and linguistic Transformer layers:
  - Causal / subsequent autoregressive mask for decoders
  - Key padding masks for token sequences
  - Downsampling of temporal frame validity masks for visual memory
"""

from typing import Optional
import torch
import torch.nn.functional as F


def generate_causal_mask(
    size: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Generates an additive upper-triangular causal mask to prevent attention to future tokens.
    
    Args:
        size: Length of target sequence S.
        device: Target torch device.
        dtype: Tensor data type (typically torch.float32).
        
    Returns:
        Tensor of shape [size, size] with 0.0 on and below diagonal, and -inf above.
    """
    mask = torch.triu(torch.full((size, size), float("-inf"), dtype=dtype, device=device), diagonal=1)
    return mask


def generate_causal_bool_mask(
    size: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generates a boolean upper-triangular causal mask for PyTorch Transformer modules.
    True indicates masked positions (future tokens).
    
    Args:
        size: Length of target sequence S.
        device: Target torch device.
        
    Returns:
        Boolean tensor of shape [size, size] where upper triangle is True.
    """
    return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)


def create_padding_mask(
    tokens: torch.Tensor,
    pad_idx: int = 0
) -> torch.Tensor:
    """
    Creates a key padding mask from a 2D token tensor.
    In PyTorch Transformer convention, positions with True are ignored in attention.
    
    Args:
        tokens: Integer token tensor of shape [B, S].
        pad_idx: Token ID corresponding to <PAD> (default: 0).
        
    Returns:
        Boolean tensor of shape [B, S] where True indicates padding.
    """
    if tokens.dim() != 2:
        raise ValueError(f"tokens must be 2D [B, S], got shape {list(tokens.shape)}")
    return tokens == pad_idx


def downsample_temporal_mask(
    input_mask: torch.Tensor,
    target_len: int = 12
) -> torch.Tensor:
    """
    Downsamples a frame-level validity mask [B, 1, T, V] or [B, T] to visual memory length T'.
    
    Args:
        input_mask: Input tensor of shape [B, 1, T, V], [B, T, V], or [B, T].
        target_len: Subsampled temporal length T' (e.g. 12).
        
    Returns:
        Boolean tensor of shape [B, target_len] where True indicates invalid / padded frames.
    """
    if input_mask.dim() == 4:
        # [B, 1, T, V] -> max over V, squeeze channel -> [B, T]
        # Frame is valid if at least one landmark is visible
        frame_validity = (input_mask > 0.5).any(dim=-1).squeeze(1).float()  # [B, T]
    elif input_mask.dim() == 3:
        frame_validity = (input_mask > 0.5).any(dim=-1).float()  # [B, T]
    elif input_mask.dim() == 2:
        frame_validity = input_mask.float()
    else:
        raise ValueError(f"Unsupported mask shape: {list(input_mask.shape)}")

    B, T = frame_validity.shape
    if T == target_len:
        downsampled = frame_validity
    else:
        # Max pool over temporal stride intervals to preserve valid frame signals
        # Reshape to [B, 1, T] for 1D adaptive pooling
        downsampled = F.adaptive_max_pool1d(frame_validity.unsqueeze(1), target_len).squeeze(1)

    # In PyTorch key_padding_mask convention: True indicates padded/invalid frame
    # frame_validity == 0 -> True (ignore)
    key_padding_mask = downsampled < 0.5
    return key_padding_mask
