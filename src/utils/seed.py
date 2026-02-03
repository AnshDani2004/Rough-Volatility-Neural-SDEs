"""
Seed utilities for reproducibility.

This module provides deterministic seeding across numpy, torch, and random.
"""

import os
import random
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Parameters
    ----------
    seed : int
        Random seed value. Default is 42.
        
    Notes
    -----
    For exact GPU reproducibility, also set:
        CUBLAS_WORKSPACE_CONFIG=:4096:8
        torch.use_deterministic_algorithms(True)
    
    GPU floating-point operations may still have minor variance.
    For exact reproduction, use CPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for CUBLAS
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    logger.debug(f"Set all random seeds to {seed}")


def get_generator(seed: int = 42) -> torch.Generator:
    """
    Get a seeded PyTorch generator.
    
    Parameters
    ----------
    seed : int
        Random seed value.
        
    Returns
    -------
    torch.Generator
        Seeded generator for reproducible sampling.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
