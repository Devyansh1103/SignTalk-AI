"""
SignTalk AI - PyTorch Landmark Dataset & DataLoader
Provides a production PyTorch Dataset interface for preprocessed skeletal NPZ files.
Validates tensor shapes, splits, labels, signer IDs, and masks for ST-GCN and Transformer.
"""

import os
import glob
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader


class SignLandmarkDataset(Dataset):
    """
    PyTorch Dataset loading preprocessed landmark representations.
    Returns:
        - data_tensor: torch.FloatTensor of shape (C, T, V) e.g. (3, 45, 93)
        - mask_tensor: torch.BoolTensor of shape (1, T, V) e.g. (1, 45, 93)
        - label: torch.LongTensor scalar class ID
        - metadata: Dictionary containing sample_id, signer_id, split, quality_score
    """

    def __init__(
        self,
        data_dir: str = "data/processed/landmarks",
        split: str = "train",
        transform: Optional[Any] = None,
        min_quality: float = 0.0
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.min_quality = float(min_quality)

        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Dataset split directory does not exist: {split_dir}")

        pattern = os.path.join(split_dir, "*.npz")
        all_files = sorted(glob.glob(pattern))

        self.samples: List[str] = []
        for f in all_files:
            try:
                npz = np.load(f, allow_pickle=True)
                quality = float(npz.get("quality_score", 1.0))
                if quality >= self.min_quality:
                    self.samples.append(f)
            except Exception:
                continue

        if len(self.samples) == 0:
            print(f"Warning: Zero samples found in {split_dir} matching min_quality >= {self.min_quality}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]:
        file_path = self.samples[idx]
        with np.load(file_path, allow_pickle=True) as npz:
            data = npz["data"]  # (3, 45, 93)
            mask = npz["mask"]  # (1, 45, 93)
            class_id = int(npz["class_id"])
            sample_id = str(npz["sample_id"])
            signer_id = str(npz["signer_id"])
            quality_score = float(npz["quality_score"])
            class_label = str(npz["class_label"])

        # Convert to PyTorch tensors
        data_tensor = torch.from_numpy(data).float()
        mask_tensor = torch.from_numpy(mask).bool()

        if self.transform is not None:
            data_tensor, mask_tensor = self.transform(data_tensor, mask_tensor)

        meta = {
            "sample_id": sample_id,
            "signer_id": signer_id,
            "class_label": class_label,
            "split": self.split,
            "quality_score": quality_score,
            "file_path": file_path
        }

        return data_tensor, mask_tensor, class_id, meta


def create_landmark_dataloader(
    data_dir: str = "data/processed/landmarks",
    split: str = "train",
    batch_size: int = 8,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    min_quality: float = 0.0
) -> DataLoader:
    """
    Creates a standard PyTorch DataLoader for the requested split.

    Args:
        data_dir: Path to root processed landmarks directory.
        split: 'train', 'val', or 'test'.
        batch_size: Batch size B.
        shuffle: Whether to shuffle (defaults to True for train, False for val/test).
        num_workers: Number of worker processes.
        min_quality: Quality filtering threshold.

    Returns:
        Configured PyTorch DataLoader instance.
    """
    if shuffle is None:
        shuffle = (split == "train")

    dataset = SignLandmarkDataset(data_dir=data_dir, split=split, min_quality=min_quality)

    def collate_fn(batch):
        data_list = [item[0] for item in batch]
        mask_list = [item[1] for item in batch]
        label_list = [item[2] for item in batch]
        meta_list = [item[3] for item in batch]

        # Stack into [B, C, T, V] and [B, 1, T, V]
        batch_data = torch.stack(data_list, dim=0)
        batch_mask = torch.stack(mask_list, dim=0)
        batch_labels = torch.tensor(label_list, dtype=torch.long)

        return batch_data, batch_mask, batch_labels, meta_list

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )


def verify_dataloader_tensor_shapes(
    data_dir: str = "data/processed/landmarks",
    split: str = "train",
    expected_channels: int = 3,
    expected_frames: int = 45,
    expected_nodes: int = 93
) -> Dict[str, Any]:
    """
    Sanity check function to verify DataLoader yields exact [B, C, T, V] tensor shapes.
    """
    loader = create_landmark_dataloader(data_dir=data_dir, split=split, batch_size=4)
    if len(loader.dataset) == 0:
        return {"status": "FAILED", "reason": f"No samples in split {split}"}

    for batch_idx, (data_b, mask_b, labels_b, meta_b) in enumerate(loader):
        B, C, T, V = data_b.shape
        passed = (C == expected_channels and T == expected_frames and V == expected_nodes)
        return {
            "status": "PASSED" if passed else "FAILED",
            "batch_size": B,
            "channels": C,
            "frames": T,
            "nodes": V,
            "data_shape": list(data_b.shape),
            "mask_shape": list(mask_b.shape),
            "labels_shape": list(labels_b.shape),
            "dtype": str(data_b.dtype),
            "is_finite": bool(torch.all(torch.isfinite(data_b)).item())
        }

    return {"status": "FAILED", "reason": "Loader yielded 0 batches"}
