"""
SignTalk AI: Unit tests for complete SignSTGCN model.
"""

import pytest
import os
import torch

from src.models.stgcn import SignSTGCN
from src.data.sign_sequence_dataset import SignSequenceDataset


def test_stgcn_forward_random_tensor():
    model = SignSTGCN(
        in_channels=3,
        num_classes=10,
        num_nodes=93,
        sequence_length=45,
        block_channels=[32, 64, 128],
        block_strides=[1, 2, 2]
    )
    x = torch.randn(2, 3, 45, 93)
    out = model(x)
    assert out.shape == (2, 10)


def test_stgcn_backward_pass():
    model = SignSTGCN(
        in_channels=3,
        num_classes=10,
        num_nodes=93,
        sequence_length=45,
        block_channels=[32, 64],
        block_strides=[1, 1]
    )
    x = torch.randn(2, 3, 45, 93, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    # Check that a sample parameter in first block got gradients
    assert model.blocks[0].sgcn.conv.weight.grad is not None


def test_stgcn_real_dataset_sample():
    manifest_path = "data/manifests/train.csv"
    if not os.path.exists(manifest_path):
        pytest.skip("Train manifest not found.")

    dataset = SignSequenceDataset(
        manifest_path=manifest_path,
        split="train",
        filter_rejects=False,
        augment=False
    )
    sample = dataset[0]
    x = sample["x"].unsqueeze(0)  # [1, 3, 45, 93]
    mask = sample["mask"].unsqueeze(0)  # [1, 1, 45, 93]

    model = SignSTGCN(
        in_channels=3,
        num_classes=10,
        num_nodes=93,
        sequence_length=45,
        block_channels=[32, 64],
        block_strides=[1, 1]
    )
    model.eval()
    with torch.no_grad():
        out = model(x, mask)
    assert out.shape == (1, 10)
