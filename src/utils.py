"""
Utility functions for reproducibility and helper functions.
"""
import torch
import random
import numpy as np
import os


def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the available device (CPU or CUDA)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_hardware_info():
    """Get hardware information for reproducibility report."""
    info = {
        'device': 'CPU' if not torch.cuda.is_available() else 'CUDA',
        'cpu_count': os.cpu_count(),
    }
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    return info

