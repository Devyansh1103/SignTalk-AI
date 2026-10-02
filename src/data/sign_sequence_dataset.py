"""
SignTalk AI: PyTorch Sign Sequence Dataset.

Loads standardized sequence archives (.npz) indexed by sequence manifests:
  - Outputs ST-GCN ready tensors of shape [C, T, V] = [3, 45, 93]
  - Supports auxiliary binary validity masks [1, T, V]
  - Supports first-order temporal velocity expansion [6, 45, 93]
  - Supports Transformer decoder target tokens and gloss IDs
  - Provides deterministic evaluation and optional training-time augmentation
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.stgcn_tensor import (
    landmarks_to_stgcn_tensor,
    compute_temporal_velocity,
    NUM_NODES,
    SEQUENCE_LENGTH,
    CHANNELS
)


class SignSequenceDataset(Dataset):
    """
    Production-ready PyTorch Dataset for SignTalk AI spatial-temporal sequences.
    """

    def __init__(
        self,
        manifest_path: str = "data/manifests/sequence_manifest.csv",
        split: Optional[str] = None,
        filter_rejects: bool = False,
        min_quality: float = 0.0,
        include_velocity: bool = False,
        augment: bool = False,
        transform_seed: Optional[int] = None
    ):
        """
        Args:
            manifest_path: Path to sequence manifest CSV (master or split-specific).
            split: Optional partition filter ('train', 'val', 'test').
            filter_rejects: If True, excludes samples with quality_status == 'REJECT'.
            min_quality: Minimum quality score required to include a sequence.
            include_velocity: If True, expands channels from C=3 to C=6 with temporal differences.
            augment: If True, applies training-time spatial augmentation (jitter, scaling, rotation).
            transform_seed: Deterministic random seed for reproducibility.
        """
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        df = pd.read_csv(manifest_path)

        if split is not None:
            df = df[df["split"] == split].copy()

        if filter_rejects and "quality_status" in df.columns:
            df = df[df["quality_status"] != "REJECT"].copy()

        if min_quality > 0.0 and "quality_score" in df.columns:
            df = df[df["quality_score"] >= min_quality].copy()

        self.df = df.reset_index(drop=True)
        self.include_velocity = include_velocity
        self.augment = augment
        self.rng = np.random.default_rng(transform_seed) if transform_seed is not None else np.random.default_rng()

    def __len__(self) -> int:
        return len(self.df)

    def _apply_augmentation(self, coords: np.ndarray) -> np.ndarray:
        """
        Applies mild training-time landmark augmentation:
          - Gaussian coordinate jitter (sigma=0.005)
          - Mild scale variation (s in [0.95, 1.05])
          - Small 2D rotation in xy-plane (theta in [-5 deg, +5 deg])
        coords: shape (3, T, V)
        """
        c = coords.copy()
        
        # 1. Mild scale
        scale = float(self.rng.uniform(0.95, 1.05))
        c[:2, :, :] *= scale  # scale xy
        
        # 2. Planar rotation (theta in radians)
        theta = float(self.rng.uniform(-0.087, 0.087))  # ~ +/- 5 degrees
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x = c[0, :, :].copy()
        y = c[1, :, :].copy()
        c[0, :, :] = cos_t * x - sin_t * y
        c[1, :, :] = sin_t * x + cos_t * y
        
        # 3. Gaussian jitter
        jitter = self.rng.normal(0.0, 0.005, size=c.shape).astype(np.float32)
        c += jitter
        
        return c

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        file_path = str(row["file_path"])

        # Fallback if relative path needs root prefix
        if not os.path.exists(file_path):
            alt_path = os.path.join(os.getcwd(), file_path)
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                raise FileNotFoundError(f"Sequence archive not found: {file_path}")

        with np.load(file_path) as arc:
            data = arc["data"]  # (3, 45, 93)
            mask = arc["mask"]  # (1, 45, 93)
            class_id = int(arc["label"])
            gloss_id = int(arc["gloss_id"]) if "gloss_id" in arc else class_id + 4

        # Apply augmentation if requested (training mode only)
        if self.augment:
            data = self._apply_augmentation(data)

        # Convert to PyTorch tensors
        tensor_x, tensor_mask = landmarks_to_stgcn_tensor(data, mask)

        # Optional first-order velocity features (expands C=3 to C=6)
        if self.include_velocity:
            tensor_x = compute_temporal_velocity(tensor_x)

        # Transformer autoregressive targets
        # Decoder input: [<BOS>, GLOSS_ID] = [2, gloss_id]
        # Decoder target: [GLOSS_ID, <EOS>] = [gloss_id, 3]
        decoder_input = torch.tensor([2, gloss_id], dtype=torch.long)
        decoder_target = torch.tensor([gloss_id, 3], dtype=torch.long)
        attention_mask = torch.tensor([1, 1], dtype=torch.long)

        metadata = {
            "sequence_id": str(row["sequence_id"]),
            "source_recording_id": str(row["source_recording_id"]),
            "source_video_id": str(row["source_video_id"]),
            "signer_id": str(row["signer_id"]),
            "split": str(row["split"]),
            "label": str(row["label"]),
            "gloss": str(row["gloss"]),
            "translation": str(row["translation"]),
            "quality_score": float(row["quality_score"]),
            "quality_status": str(row["quality_status"])
        }

        return {
            "x": tensor_x,                         # Tensor [C, 45, 93]
            "mask": tensor_mask,                   # Tensor [1, 45, 93]
            "label": torch.tensor(class_id, dtype=torch.long), # Scalar [0..9]
            "gloss_id": torch.tensor(gloss_id, dtype=torch.long), # Scalar token
            "decoder_input": decoder_input,        # Tensor [2]
            "decoder_target": decoder_target,      # Tensor [2]
            "attention_mask": attention_mask,      # Tensor [2]
            "metadata": metadata
        }
