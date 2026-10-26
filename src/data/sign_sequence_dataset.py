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


# Node order after a left <-> right mirror: hands 0-20 <-> 21-41, pose eyes / ears / shoulders /
# elbows / wrists (43-52) pairwise. Nose (42) and the face nodes (unused, face mesh is off) stay.
MIRROR_PERM = np.arange(NUM_NODES)
for _a, _b in [(i, i + 21) for i in range(21)] + [(43, 44), (45, 46), (47, 48), (49, 50), (51, 52)]:
    MIRROR_PERM[_a], MIRROR_PERM[_b] = _b, _a


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
        transform_seed: Optional[int] = None,
        restrict_to_vocab: bool = False,
        mirror_prob: float = 0.0
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
            restrict_to_vocab: If True, keeps only rows whose 'label' is in the canonical
                10-class vocabulary and takes class_id from the label (not from the .npz),
                so stale/colliding class ids from the full INCLUDE manifests cannot leak in.
            mirror_prob: With augment, probability of mirroring a clip left <-> right (see _mirror).
        """
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        df = pd.read_csv(manifest_path)

        self.restrict_to_vocab = restrict_to_vocab
        if restrict_to_vocab:
            from src.data.label_map import LABEL_TO_ID
            before = len(df)
            df = df[df["label"].astype(str).isin(LABEL_TO_ID)].copy()
            print(f"[DATA] restrict_to_vocab: kept {len(df)}/{before} rows from {manifest_path}")
            if len(df) == 0:
                raise ValueError(f"No rows with a canonical label in {manifest_path}")

        if split is not None:
            df = df[df["split"] == split].copy()

        if filter_rejects and "quality_status" in df.columns:
            df = df[df["quality_status"] != "REJECT"].copy()

        if min_quality > 0.0 and "quality_score" in df.columns:
            df = df[df["quality_score"] >= min_quality].copy()

        self.df = df.reset_index(drop=True)
        self.include_velocity = include_velocity
        self.augment = augment
        self.mirror_prob = float(mirror_prob)
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

    def _temporal_crop(self, data: np.ndarray, mask: np.ndarray, p: float = 0.5,
                       min_frac: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """With probability p, keep a random contiguous 70-100% span of the clip and
        resample it back to T frames (nearest frame). Training clips are whole recordings
        squeezed into 45 frames; the live app feeds 1.8 s sliding windows that can start
        or end mid-sign, so the model should not rely on seeing the full clip.
        data: (3, T, V), mask: (1, T, V)"""
        if self.rng.random() >= p:
            return data, mask
        T = data.shape[1]
        span = int(round(T * float(self.rng.uniform(min_frac, 1.0))))
        start = int(self.rng.integers(0, T - span + 1))
        idx = np.round(np.linspace(start, start + span - 1, T)).astype(int)
        return data[:, idx, :].copy(), mask[:, idx, :].copy()

    def _drop_rest_hands(self, data: np.ndarray, mask: np.ndarray, p: float = 0.5,
                         rest_y: float = 1.35) -> Tuple[np.ndarray, np.ndarray]:
        """With probability p, hide hands that are resting below signing space (normalized
        y > rest_y). INCLUDE signers stand with their hands visible at the hips before and after
        a sign; a seated webcam user's resting hands are out of frame. Measured on 150 test clips
        through the live extractor: removing resting hands drops accuracy 81% -> 71%."""
        if self.rng.random() >= p:
            return data, mask
        data, mask = data.copy(), mask.copy()
        for a, b in ((0, 21), (21, 42)):
            hand_mask = mask[0, :, a:b] > 0                                        # (T, 21)
            highest = np.where(hand_mask, data[1, :, a:b], np.inf).min(axis=1)     # smallest y = highest point
            resting = hand_mask.any(axis=1) & (highest > rest_y)
            data[:, resting, a:b] = 0.0
            mask[:, resting, a:b] = 0
        return data, mask

    def _low_fps(self, data: np.ndarray, mask: np.ndarray, p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """With probability p, sample-and-hold every k-th frame (k = 2..4). The CPU backend
        delivers 5-10 FPS, so a live sign resampled to 45 frames repeats each real frame
        2-4 times; at 6 FPS accuracy dropped 81% -> 67% on test clips."""
        if self.rng.random() >= p:
            return data, mask
        k = int(self.rng.integers(2, 5))
        T = data.shape[1]
        idx = np.minimum((np.arange(T) + int(self.rng.integers(0, k))) // k * k, T - 1)
        return data[:, idx, :].copy(), mask[:, idx, :].copy()

    def _mirror(self, data: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """With probability mirror_prob, mirror the clip as a left-handed signer would sign it:
        negate x (data is centred on mid-shoulder) and swap left/right hands and pose points.
        The current model scores 85.5% on the test clips but 24.4% on the same clips mirrored."""
        if self.mirror_prob <= 0 or self.rng.random() >= self.mirror_prob:
            return data, mask
        data, mask = data[:, :, MIRROR_PERM].copy(), mask[:, :, MIRROR_PERM].copy()
        data[0] *= -1.0
        return data, mask

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
            if self.restrict_to_vocab:
                from src.data.label_map import LABEL_TO_ID
                class_id = LABEL_TO_ID[str(row["label"])]
            gloss_id = int(arc["gloss_id"]) if "gloss_id" in arc else class_id + 4

        # Apply augmentation if requested (training mode only)
        if self.augment:
            data, mask = self._temporal_crop(data, mask)
            data, mask = self._low_fps(data, mask)
            data, mask = self._drop_rest_hands(data, mask)
            data, mask = self._mirror(data, mask)
            data = self._apply_augmentation(data) * (mask > 0)  # keep missing nodes at exactly 0, like live input

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
