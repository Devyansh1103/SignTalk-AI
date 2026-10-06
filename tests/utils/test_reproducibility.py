"""
Unit tests for Device and Reproducibility utilities.
"""

import pytest
import torch
import numpy as np

from src.utils.device import get_device
from src.utils.reproducibility import set_seed


def test_device_selection():
    dev, info = get_device("auto")
    assert isinstance(dev, torch.device)
    assert info["allocated_device"] in ["cuda:0", "mps", "cpu"]


def test_forced_cpu():
    dev, info = get_device("cpu")
    assert dev.type == "cpu"
    assert info["allocated_device"] == "cpu"


def test_set_seed_determinism():
    info = set_seed(12345)
    assert info["seed"] == 12345

    a1 = np.random.rand(5)
    t1 = torch.rand(5)

    # Re-seed
    set_seed(12345)
    a2 = np.random.rand(5)
    t2 = torch.rand(5)

    assert np.allclose(a1, a2)
    assert torch.allclose(t1, t2)
