"""
Perturbation generation for EGGROLL TRM.

Uses deterministic hash-based generation for reproducibility.
Matches the approach from nano-egg/HyperscaleES.
"""

import torch
from torch import Tensor
from typing import Dict, Tuple
import math


def hash_combine(seed: int, thread_id: Tensor) -> Tensor:
    """
    Deterministic hash for reproducible perturbations.
    
    Args:
        seed: Base seed
        thread_id: Population member IDs
        
    Returns:
        Combined seeds (same shape as thread_id)
    """
    tid = thread_id.to(torch.int64)
    combined = (seed ^ (tid + 0x9e3779b9 + (seed << 6) + (seed >> 2))) & 0x7FFFFFFF
    return combined


def generate_vector(seed: Tensor, size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """
    Generate deterministic pseudo-random vector from integer seed.
    
    Uses a simple hash function to generate values in [-1, 1].
    
    Args:
        seed: Seeds tensor (pop_size,)
        size: Vector length
        device: Device
        dtype: Data type
        
    Returns:
        Vectors (pop_size, size)
    """
    pop_size = seed.shape[0]
    
    # Create indices
    indices = torch.arange(size, device=device, dtype=torch.int64).unsqueeze(0)
    
    # Expand seed
    seed_exp = seed.unsqueeze(1) & 0xFFFFFFFF
    
    # Hash function
    h = seed_exp ^ (indices * 2246822519)
    h = h & 0xFFFFFFFF
    h = ((h >> 16) ^ h) & 0xFFFFFFFF
    h = (h * 73244475) & 0xFFFFFFFF
    h = ((h >> 16) ^ h) & 0xFFFFFFFF
    
    # Convert to float [-1, 1]
    result = (h.float() / 2147483648.0) - 1.0
    return result.to(dtype)


def generate_layer_perturbations(
    layer_name: str,
    shape: Tuple[int, int],  # (out_dim, in_dim)
    epoch: int,
    thread_ids: Tensor,  # (pop_size,)
    base_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """
    Generate A, B perturbations for a layer.
    
    For rank-1 EGGROLL:
        W_perturbed = W + σ * A ⊗ B
        Where A: (out_dim,), B: (in_dim,)
    
    Args:
        layer_name: Parameter name for deterministic seeding
        shape: (out_features, in_features)
        epoch: Current epoch
        thread_ids: Population member indices
        base_seed: Base random seed
        device: Device
        dtype: Data type
        
    Returns:
        A: (pop_size, out_dim)
        B: (pop_size, in_dim)
    """
    out_dim, in_dim = shape
    
    # Layer-specific seed offset
    name_hash = hash(layer_name) & 0x7FFFFFFF
    layer_seed = base_seed + name_hash + epoch * 1000003
    
    # Combine with thread_ids
    seeds = hash_combine(layer_seed, thread_ids)
    
    # Generate A and B
    A = generate_vector(seeds, out_dim, device, dtype)
    B = generate_vector(seeds + 1, in_dim, device, dtype)
    
    # Normalize for stable gradients
    A = A / math.sqrt(out_dim)
    B = B / math.sqrt(in_dim)
    
    return A, B


def generate_perturbations(
    param_shapes: Dict[str, Tuple[int, int]],
    epoch: int,
    thread_ids: Tensor,
    base_seed: int = 42,
) -> Dict[str, Tuple[Tensor, Tensor]]:
    """
    Generate all perturbations for a model.
    
    Args:
        param_shapes: Dict from get_param_shapes()
        epoch: Current epoch
        thread_ids: Population member indices
        base_seed: Base random seed
        
    Returns:
        Dict mapping layer names to (A, B) perturbation tuples
    """
    device = thread_ids.device
    dtype = torch.float32
    
    perturbations = {}
    for name, shape in param_shapes.items():
        A, B = generate_layer_perturbations(
            name, shape, epoch, thread_ids, base_seed, device, dtype
        )
        perturbations[name] = (A, B)
    
    return perturbations
