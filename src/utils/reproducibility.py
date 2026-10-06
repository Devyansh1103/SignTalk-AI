"""
SignTalk AI: Reproducibility and Determinism Utilities.

Enforces deterministic seed configuration across Python, NumPy,
PyTorch, and CUDA environments.
"""

from typing import Dict, Any
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> Dict[str, Any]:
    """
    Sets deterministic seeds across all random number generators.
    
    Args:
        seed: Integer seed value.
        
    Returns:
        dict summarizing the seed configuration.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return {
        "seed": seed,
        "python_seeded": True,
        "numpy_seeded": True,
        "torch_cpu_seeded": True,
        "torch_cuda_seeded": cuda_available,
        "cudnn_deterministic": cuda_available
    }
