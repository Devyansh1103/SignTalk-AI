"""
SignTalk AI: Robustness Testing and Perturbation Engine.

Applies controlled physical and sensor perturbations:
  1. Additive Coordinate Gaussian Noise (simulating tracking jitter)
  2. Temporal Frame Dropping (simulating dropped camera frames / bandwidth loss)
  3. Temporal Speed Scaling (simulating slow vs fast signing)
  4. Anatomical Subsystem Masking (simulating hand / face occlusion)
  5. Horizontal Mirroring (simulating left-handed vs right-handed variance)
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.classification_metrics import compute_classification_metrics


def apply_gaussian_noise(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """Adds zero-mean Gaussian jitter to coordinates [B, C, T, V]."""
    noise = torch.randn_like(x) * sigma
    return x + noise


def apply_frame_dropout(x: torch.Tensor, drop_rate: float = 0.25) -> torch.Tensor:
    """Randomly zeros out entire temporal frame slices."""
    B, C, T, V = x.shape
    num_drop = int(T * drop_rate)
    if num_drop == 0:
        return x

    x_perturbed = x.clone()
    for b in range(B):
        drop_indices = np.random.choice(T, size=num_drop, replace=False)
        x_perturbed[b, :, drop_indices, :] = 0.0
    return x_perturbed


def apply_speed_scaling(x: torch.Tensor, factor: float = 1.25) -> torch.Tensor:
    """
    Resamples temporal dimension using linear interpolation and pads/truncates back to T=45.
    factor > 1.0: faster signing (accelerated sequence)
    factor < 1.0: slower signing (stretched sequence)
    """
    B, C, T, V = x.shape
    new_T = max(4, int(T / factor))

    # [B, C, T, V] -> [B, C*V, T]
    flat = x.permute(0, 1, 3, 2).reshape(B, C * V, T)
    # Interpolate to new_T
    resampled = torch.nn.functional.interpolate(flat, size=new_T, mode="linear", align_corners=False)

    # Pad or truncate back to original T
    if new_T < T:
        pad_size = T - new_T
        padded = torch.nn.functional.pad(resampled, (0, pad_size), mode="replicate")
    else:
        padded = resampled[:, :, :T]

    out = padded.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()
    return out


def apply_subsystem_masking(x: torch.Tensor, start_node: int, end_node: int) -> torch.Tensor:
    """Zeros out a contiguous slice of landmark nodes [start_node, end_node)."""
    x_perturbed = x.clone()
    x_perturbed[:, :, :, start_node:end_node] = 0.0
    return x_perturbed


def apply_horizontal_mirroring(x: torch.Tensor) -> torch.Tensor:
    """
    Applies horizontal reflection:
    1. Invert X coordinate: x_mirror = 1.0 - x (or -x if normalized around 0)
    2. Swap Left Hand (nodes 0-20) and Right Hand (nodes 21-41)
    """
    x_mirrored = x.clone()
    # Invert x channel (channel 0)
    x_mirrored[:, 0, :, :] = -x_mirrored[:, 0, :, :]

    # Swap left hand (0-20) and right hand (21-41)
    lh = x_mirrored[:, :, :, 0:21].clone()
    rh = x_mirrored[:, :, :, 21:42].clone()
    x_mirrored[:, :, :, 0:21] = rh
    x_mirrored[:, :, :, 21:42] = lh

    return x_mirrored


def evaluate_robustness_suite(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    num_classes: int = 10
) -> List[Dict[str, Any]]:
    """
    Runs an exhaustive test suite comparing clean vs perturbed inputs.
    """
    model.eval()
    model.to(device)

    perturbation_specs = [
        {"name": "Clean (Normal Condition)", "fn": lambda x: x},
        {"name": "Low Noise (sigma=0.01)", "fn": lambda x: apply_gaussian_noise(x, 0.01)},
        {"name": "Medium Noise (sigma=0.03)", "fn": lambda x: apply_gaussian_noise(x, 0.03)},
        {"name": "High Noise (sigma=0.05)", "fn": lambda x: apply_gaussian_noise(x, 0.05)},
        {"name": "Frame Dropout 10%", "fn": lambda x: apply_frame_dropout(x, 0.10)},
        {"name": "Frame Dropout 25%", "fn": lambda x: apply_frame_dropout(x, 0.25)},
        {"name": "Frame Dropout 50%", "fn": lambda x: apply_frame_dropout(x, 0.50)},
        {"name": "Slow Signing (0.75x)", "fn": lambda x: apply_speed_scaling(x, 0.75)},
        {"name": "Fast Signing (1.25x)", "fn": lambda x: apply_speed_scaling(x, 1.25)},
        {"name": "Rapid Signing (1.50x)", "fn": lambda x: apply_speed_scaling(x, 1.50)},
        {"name": "Occluded Left Hand", "fn": lambda x: apply_subsystem_masking(x, 0, 21)},
        {"name": "Occluded Right Hand", "fn": lambda x: apply_subsystem_masking(x, 21, 42)},
        {"name": "Occluded Face", "fn": lambda x: apply_subsystem_masking(x, 53, 93)},
        {"name": "Horizontal Mirroring", "fn": lambda x: apply_horizontal_mirroring(x)},
    ]

    results = []

    for spec in perturbation_specs:
        test_name = spec["name"]
        transform_fn = spec["fn"]

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch.get("x") if "x" in batch else batch.get("landmarks")
                y = batch["label"]
                mask = batch.get("mask", None)

                # Apply perturbation
                x_pert = transform_fn(x).to(device)
                if mask is not None:
                    mask = mask.to(device)

                logits = model(x_pert, mask)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(y.numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        y_t = np.array(all_targets)
        y_p = np.array(all_preds)
        y_pr = np.array(all_probs)

        metrics = compute_classification_metrics(
            y_true=y_t,
            y_pred=y_p,
            y_probs=y_pr,
            num_classes=num_classes,
            class_names=class_names
        )

        acc = metrics["accuracy"]
        macro_f1 = metrics["macro_f1"]
        failure_rate = round(1.0 - acc, 4)

        results.append({
            "condition": test_name,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "failure_rate": failure_rate,
            "sample_count": len(y_t)
        })

    return results
