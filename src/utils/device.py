"""Device management and reproducibility utilities for chord recognition."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch
import torchaudio


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device for computation.
    
    Args:
        device: Specific device to use. If None, auto-detect.
        
    Returns:
        torch.device: The selected device.
    """
    if device is not None:
        return torch.device(device)
    
    # Auto-detect device with fallback order
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> dict:
    """Get information about the current device.
    
    Returns:
        dict: Device information including type, memory, etc.
    """
    device = get_device()
    info = {"device": str(device)}
    
    if device.type == "cuda":
        info.update({
            "device_name": torch.cuda.get_device_name(device),
            "memory_total": torch.cuda.get_device_properties(device).total_memory,
            "memory_allocated": torch.cuda.memory_allocated(device),
            "memory_reserved": torch.cuda.memory_reserved(device),
        })
    elif device.type == "mps":
        info["device_name"] = "Apple Silicon GPU"
    else:
        info["device_name"] = "CPU"
    
    return info


def move_to_device(
    data: Union[torch.Tensor, dict, list, tuple], 
    device: torch.device
) -> Union[torch.Tensor, dict, list, tuple]:
    """Move data to the specified device.
    
    Args:
        data: Data to move to device.
        device: Target device.
        
    Returns:
        Data moved to the target device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, device) for item in data)
    else:
        return data


def clear_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_usage() -> dict:
    """Get current memory usage.
    
    Returns:
        dict: Memory usage information.
    """
    memory_info = {}
    
    if torch.cuda.is_available():
        memory_info.update({
            "cuda_allocated": torch.cuda.memory_allocated(),
            "cuda_reserved": torch.cuda.memory_reserved(),
            "cuda_max_allocated": torch.cuda.max_memory_allocated(),
        })
    
    return memory_info
