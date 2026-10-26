"""
SignTalk AI: Sequence DataLoader Factory.

Provides factory functions to instantiate standard PyTorch DataLoaders
for train, val, and test partitions with configurable batching, shuffling,
and worker management.
"""

from typing import Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.sign_sequence_dataset import SignSequenceDataset
from src.data.collate import sign_sequence_collate_fn


def create_sequence_dataloader(
    manifest_path: str = "data/manifests/sequence_manifest.csv",
    split: Optional[str] = None,
    batch_size: int = 8,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    filter_rejects: Optional[bool] = None,
    min_quality: float = 0.0,
    include_velocity: bool = False,
    augment: Optional[bool] = None,
    seed: Optional[int] = 42,
    restrict_to_vocab: bool = False,
    balance_classes: bool = False,
    mirror_prob: float = 0.0
) -> DataLoader:
    """
    Creates an optimized PyTorch DataLoader for sign sequence batches.
    
    Default policies:
      - Shuffle: True for train, False for val/test
      - Filter rejects: True for train, False for val/test
      - Augment: True for train, False for val/test
    """
    is_train = (split == "train")
    
    if shuffle is None:
        shuffle = is_train
    if filter_rejects is None:
        filter_rejects = is_train
    if augment is None:
        augment = is_train

    dataset = SignSequenceDataset(
        manifest_path=manifest_path,
        split=split,
        filter_rejects=filter_rejects,
        min_quality=min_quality,
        include_velocity=include_velocity,
        augment=augment,
        transform_seed=seed,
        restrict_to_vocab=restrict_to_vocab,
        mirror_prob=mirror_prob
    )

    sampler = None
    # Optional per-row weight column (e.g. the user's own recordings drawn more often)
    row_weight = (dataset.df["sample_weight"].fillna(1.0).to_numpy(dtype=np.float64)
                  if "sample_weight" in dataset.df.columns else None)
    if (balance_classes or row_weight is not None) and len(dataset) > 0:
        # Weight each clip by 1/sqrt(clips in its class): classes with 4 train clips
        # (INCLUDE Adjectives) are drawn ~2x as often as 15-clip classes. Full 1/count
        # balancing would replay the same 4 clips ~4x and overfit them.
        counts = dataset.df["label"].map(dataset.df["label"].value_counts()).to_numpy(dtype=np.float64)
        w = 1.0 / np.sqrt(counts) if balance_classes else np.ones(len(dataset))
        weights = torch.as_tensor(w * (row_weight if row_weight is not None else 1.0), dtype=torch.double)
        gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=gen)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=sign_sequence_collate_fn
    )

    return loader
