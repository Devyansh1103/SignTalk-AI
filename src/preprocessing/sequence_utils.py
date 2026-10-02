"""
SignTalk AI - Sequence Utilities & Missing Landmark Handler
Handles missing joints, temporal alignment, interpolation, and padding/truncation
while preserving sequence integrity and temporal trajectory flow.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


def handle_missing_landmarks(
    coords: np.ndarray,          # Shape: (T, V, C) e.g. (T, 93, 3)
    mask: np.ndarray,            # Shape: (T, V) boolean detection mask
    strategy: str = "interpolate_and_mask",
    max_consecutive_missing: int = 10,
    visibility: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies configurable missing landmark strategy across temporal dimension.

    Args:
        coords: (T, V, C) raw skeletal coordinates.
        mask: (T, V) boolean indicating detected (True) or missing (False).
        strategy: 'interpolate_and_mask', 'carry_forward', 'zero_mask', or 'confidence_mask'.
        max_consecutive_missing: Max frame gap to bridge with interpolation.
        visibility: Optional (T, V) float visibility scores.

    Returns:
        Tuple of:
            - processed_coords: (T, V, C) with handled missing values
            - updated_mask: (T, V) binary mask indicating valid/imputed nodes
    """
    T, V, C = coords.shape
    processed = coords.copy()
    updated_mask = mask.copy()

    if strategy == "zero_mask":
        # Missing points remain exactly 0.0, mask preserves detection status
        for t in range(T):
            for v in range(V):
                if not mask[t, v]:
                    processed[t, v] = 0.0
        return processed, updated_mask

    elif strategy == "carry_forward":
        # Forward propagate last known valid coordinates
        for v in range(V):
            last_valid = None
            gap_count = 0
            for t in range(T):
                if mask[t, v]:
                    last_valid = processed[t, v].copy()
                    gap_count = 0
                else:
                    if last_valid is not None and gap_count < max_consecutive_missing:
                        processed[t, v] = last_valid
                        gap_count += 1
                        # Mark as imputed (or remain False in original mask)
                    else:
                        processed[t, v] = 0.0
        return processed, updated_mask

    elif strategy == "interpolate_and_mask":
        # 1D linear interpolation across temporal gaps of length <= max_consecutive_missing
        for v in range(V):
            valid_indices = np.where(mask[:, v])[0]
            if len(valid_indices) == 0:
                processed[:, v] = 0.0
                continue
            if len(valid_indices) == 1:
                # Only 1 observation; carry forward / backward within limit
                idx = valid_indices[0]
                processed[:max(0, idx - max_consecutive_missing), v] = 0.0
                processed[min(T, idx + max_consecutive_missing):, v] = 0.0
                continue

            # Linear interpolation for interior missing points
            for i in range(len(valid_indices) - 1):
                t_start = valid_indices[i]
                t_end = valid_indices[i + 1]
                gap = t_end - t_start - 1

                if 0 < gap <= max_consecutive_missing:
                    # Interpolate
                    p_start = processed[t_start, v]
                    p_end = processed[t_end, v]
                    for step, t in enumerate(range(t_start + 1, t_end), start=1):
                        alpha = step / (gap + 1)
                        processed[t, v] = (1.0 - alpha) * p_start + alpha * p_end
                        updated_mask[t, v] = True  # Imputed valid point

            # Exterior frames before first and after last valid detection
            first_idx = valid_indices[0]
            last_idx = valid_indices[-1]
            if first_idx > 0:
                lead_gap = min(first_idx, max_consecutive_missing)
                processed[first_idx - lead_gap:first_idx, v] = processed[first_idx, v]
                processed[:first_idx - lead_gap, v] = 0.0
            if last_idx < T - 1:
                trail_gap = min(T - 1 - last_idx, max_consecutive_missing)
                processed[last_idx + 1:last_idx + 1 + trail_gap, v] = processed[last_idx, v]
                processed[last_idx + 1 + trail_gap:, v] = 0.0

        return processed, updated_mask

    elif strategy == "confidence_mask":
        # Discard points below visibility threshold
        thresh = 0.3
        if visibility is not None:
            for t in range(T):
                for v in range(V):
                    if visibility[t, v] < thresh:
                        processed[t, v] = 0.0
                        updated_mask[t, v] = False
        return processed, updated_mask

    else:
        # Default fallback: zero masking
        for t in range(T):
            for v in range(V):
                if not mask[t, v]:
                    processed[t, v] = 0.0
        return processed, updated_mask


def pad_or_truncate_sequence(
    coords: np.ndarray,          # (T, V, C)
    mask: np.ndarray,            # (T, V)
    target_length: int = 45,
    pad_mode: str = "zero_padding"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardizes sequence length to target_length T frames.

    Args:
        coords: (T_in, V, C)
        mask: (T_in, V)
        target_length: Desired fixed length (e.g. 45 frames)
        pad_mode: 'zero_padding' or 'repeat_last'

    Returns:
        Tuple of:
            - output_coords: (target_length, V, C)
            - output_mask: (target_length, V)
    """
    T_in, V, C = coords.shape

    if T_in == target_length:
        return coords.copy(), mask.copy()

    if T_in > target_length:
        # Truncate uniformly or center crop
        start_idx = (T_in - target_length) // 2
        end_idx = start_idx + target_length
        return coords[start_idx:end_idx].copy(), mask[start_idx:end_idx].copy()

    # T_in < target_length: Pad
    out_coords = np.zeros((target_length, V, C), dtype=np.float32)
    out_mask = np.zeros((target_length, V), dtype=bool)

    # Place sequence centered or at beginning
    out_coords[:T_in] = coords
    out_mask[:T_in] = mask

    if pad_mode == "repeat_last" and T_in > 0:
        for t in range(T_in, target_length):
            out_coords[t] = coords[-1]
            out_mask[t] = mask[-1]

    return out_coords, out_mask


def temporal_resample_sequence(
    coords: np.ndarray,          # (T_in, V, C)
    mask: np.ndarray,            # (T_in, V)
    target_length: int = 45
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resamples a sequence to exactly target_length frames using linear interpolation
    along the temporal axis, preserving motion velocity.

    Args:
        coords: (T_in, V, C)
        mask: (T_in, V)
        target_length: Target frame count (e.g. 45)

    Returns:
        Tuple of:
            - resampled_coords: (target_length, V, C)
            - resampled_mask: (target_length, V)
    """
    T_in, V, C = coords.shape
    if T_in == target_length:
        return coords.copy(), mask.copy()
    if T_in < 2:
        return pad_or_truncate_sequence(coords, mask, target_length)

    orig_times = np.linspace(0.0, 1.0, T_in)
    target_times = np.linspace(0.0, 1.0, target_length)

    resampled_coords = np.zeros((target_length, V, C), dtype=np.float32)
    resampled_mask = np.zeros((target_length, V), dtype=bool)

    for v in range(V):
        for c in range(C):
            resampled_coords[:, v, c] = np.interp(target_times, orig_times, coords[:, v, c])
        # Mask is True if nearest original frame was True
        nearest_orig_idx = np.round(target_times * (T_in - 1)).astype(int)
        resampled_mask[:, v] = mask[nearest_orig_idx, v]

    return resampled_coords, resampled_mask
