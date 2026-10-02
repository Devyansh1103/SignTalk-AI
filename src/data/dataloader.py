"""
SignTalk AI: Sequence DataLoader Factory.

Provides factory functions to instantiate standard PyTorch DataLoaders
for train, val, and test partitions with configurable batching, shuffling,
and worker management.
"""

from typing import Optional, Dict, Any
from torch.utils.data import DataLoader

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
    seed: Optional[int] = 42
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
        transform_seed=seed
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=sign_sequence_collate_fn
    )

    return loader
