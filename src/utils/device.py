"""
SignTalk AI: Hardware and Device Detection Utility.

Detects available hardware accelerators (CUDA GPUs, Apple MPS, or CPU)
and provides unified device allocation with fallback protection.
"""

from typing import Dict, Any, Tuple
import torch


def get_device(preference: str = "auto") -> Tuple[torch.device, Dict[str, Any]]:
    """
    Selects and returns the compute device along with hardware metadata.
    
    Args:
        preference: 'auto', 'cuda', 'cpu', or 'mps'.
        
    Returns:
        device: torch.device
        info: dict containing hardware diagnostics
    """
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    info = {
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "gpu_name": None,
        "device_count": 0,
        "allocated_device": "cpu"
    }

    if preference == "cuda":
        if cuda_available:
            dev = torch.device("cuda:0")
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
            info["allocated_device"] = "cuda:0"
        else:
            print("[WARNING] CUDA requested but not available; falling back to CPU.")
            dev = torch.device("cpu")
            info["allocated_device"] = "cpu"

    elif preference == "mps":
        if mps_available:
            dev = torch.device("mps")
            info["allocated_device"] = "mps"
        else:
            print("[WARNING] MPS requested but not available; falling back to CPU.")
            dev = torch.device("cpu")
            info["allocated_device"] = "cpu"

    elif preference == "cpu":
        dev = torch.device("cpu")
        info["allocated_device"] = "cpu"

    else:  # 'auto'
        if cuda_available:
            dev = torch.device("cuda:0")
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
            info["allocated_device"] = "cuda:0"
        elif mps_available:
            dev = torch.device("mps")
            info["allocated_device"] = "mps"
        else:
            dev = torch.device("cpu")
            info["allocated_device"] = "cpu"

    return dev, info


def resolve_device(preference: str = "auto") -> torch.device:
    """Convenience helper returning just the torch.device object."""
    dev, _ = get_device(preference)
    return dev
