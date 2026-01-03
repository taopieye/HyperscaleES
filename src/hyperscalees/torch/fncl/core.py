"""
EGGROLL Core - PyTorch implementation of low-rank Evolution Strategies.

CORE CONCEPT:
    perturbed_linear(x, W, b, A, B) = x @ W.T + b + x @ B @ A.T
    
    This is a low-rank perturbation of the weight matrix, enabling efficient
    ES gradient estimation with O(r(m+n)) memory instead of O(m×n) per layer.

This module contains the core EGGROLL primitives and Dict-based API.
For experiments and examples, see eggroll_fncl.py.

================================================================================
DICT-BASED API (recommended)
================================================================================

    # 1. Initialize params as a dict
    params = {
        'layer1.weight': torch.randn(256, 4, device="cuda") * 0.1,
        'layer1.bias': torch.zeros(256, device="cuda"),
        'layer2.weight': torch.randn(2, 256, device="cuda") * 0.1,
        'layer2.bias': torch.zeros(2, device="cuda"),
    }
    
    # 2. Get weight shapes for perturbation generation
    shapes = get_weight_shapes(params)
    
    # 3. Training loop
    for epoch in range(max_epochs):
        perts = generate_perturbations(shapes, pop, rank, sigma, gen, dtype)
        
        # Forward pass
        h = torch.relu(perturbed_forward(x, params['layer1.weight'], 
                                         params['layer1.bias'], perts, 'layer1.weight'))
        logits = perturbed_forward(h, params['layer2.weight'],
                                   params['layer2.bias'], perts, 'layer2.weight')
        
        # ES update
        f = normalize_fitnesses(fitness_fn(logits))
        grads = compute_gradients(f, perts, pop)
        update_params(params, grads, lr)

================================================================================
RAW PRIMITIVES API (maximum control)
================================================================================

    # Generate perturbations for each weight matrix
    A1, _, B1 = generate_lowrank_perturbations(pop, 256, 4, rank, sigma, gen, dtype)
    A2, _, B2 = generate_lowrank_perturbations(pop, 2, 256, rank, sigma, gen, dtype)
    
    # Forward pass using perturbed_linear
    h = torch.relu(perturbed_linear(x, W1, b1, A1, B1))
    logits = perturbed_linear(h, W2, b2, A2, B2)
    
    # ES gradient computation + update
    f = normalize_fitnesses(fitnesses)
    W1 = W1 + lr * compute_es_gradient(f, A1, B1, pop)
    W2 = W2 + lr * compute_es_gradient(f, A2, B2, pop)
"""

import torch
import math
from dataclasses import dataclass, field


__all__ = [
    # Configuration
    "EggrollConfig",
    # Dict-Based API
    "get_weight_shapes",
    "generate_perturbations",
    "compute_gradients",
    "update_params",
    "perturbed_forward",
    # Raw Primitives API
    "generate_lowrank_perturbations",
    "perturbed_linear",
    "apply_lowrank_perturbation",
    "compute_weight_perturbation",
    "compute_es_gradient",
    "normalize_fitnesses",
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EggrollConfig:
    """Configuration for EGGROLL training."""
    population_size: int = 2048
    rank: int = 4
    sigma: float = 0.1
    lr: float = 0.1
    lr_decay: float = 1.0
    sigma_decay: float = 0.999
    max_epochs: int = 100
    batch_size: int = 256
    seed: int = 42
    dtype: torch.dtype = field(default=torch.float32)


# =============================================================================
# Dict-Based API
# =============================================================================

def get_weight_shapes(params: dict[str, torch.Tensor]) -> dict[str, tuple[int, int]]:
    """
    Extract shapes of weight tensors for perturbation generation.
    
    Supports:
    - 2D weights (Linear): (out_dim, in_dim)
    - 4D weights (Conv2d): (out_channels, in_channels*k*k) - flattened
    
    Args:
        params: Dict of model parameters
        
    Returns:
        Dict mapping weight names to (out_dim, in_dim) shapes
        
    Example:
        params = {'fc.weight': W_fc, 'conv.weight': W_conv, ...}
        shapes = get_weight_shapes(params)
        # {'fc.weight': (10, 256), 'conv.weight': (32, 144)}
    """
    shapes: dict[str, tuple[int, int]] = {}
    for name, tensor in params.items():
        if not isinstance(tensor, torch.Tensor) or 'weight' not in name:
            continue
        if tensor.dim() == 2:
            shapes[name] = (tensor.shape[0], tensor.shape[1])
        elif tensor.dim() == 4:
            out_ch, in_ch, k1, k2 = tensor.shape
            shapes[name] = (out_ch, in_ch * k1 * k2)
    return shapes


def generate_perturbations(
    shapes: dict[str, tuple[int, int]],
    population_size: int,
    rank: int,
    sigma: float,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate perturbations for all weight matrices.
    
    Args:
        shapes: Dict from get_weight_shapes()
        population_size: Number of population members
        rank: Low-rank perturbation rank
        sigma: Perturbation scale
        generator: Torch random generator
        dtype: Data type
        
    Returns:
        Dict mapping weight names to (A_scaled, B) tuples
    """
    perts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, (out_dim, in_dim) in shapes.items():
        A_scaled, _, B = generate_lowrank_perturbations(
            population_size, out_dim, in_dim, rank, sigma, generator, dtype
        )
        perts[name] = (A_scaled, B)
    return perts


def compute_gradients(
    fitnesses: torch.Tensor,
    perts: dict[str, tuple[torch.Tensor, torch.Tensor]],
    population_size: int,
) -> dict[str, torch.Tensor]:
    """
    Compute ES gradients for all weight matrices.
    
    Args:
        fitnesses: (population,) normalized fitness scores
        perts: Dict from generate_perturbations()
        population_size: Number of population members
        
    Returns:
        Dict mapping weight names to gradient tensors
    """
    grads: dict[str, torch.Tensor] = {}
    for name, (A_scaled, B) in perts.items():
        grads[name] = compute_es_gradient(fitnesses, A_scaled, B, population_size)
    return grads


def update_params(
    params: dict[str, torch.Tensor],
    grads: dict[str, torch.Tensor],
    lr: float,
) -> dict[str, torch.Tensor]:
    """
    Update parameters with gradients.
    
    Args:
        params: Dict of model parameters
        grads: Dict from compute_gradients()
        lr: Learning rate
        
    Returns:
        Updated params dict (in-place modification)
    """
    for name, grad in grads.items():
        param = params[name]
        if grad.shape != param.shape:
            grad = grad.reshape(param.shape)
        params[name] = param + lr * grad
    return params


def perturbed_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    perts: dict[str, tuple[torch.Tensor, torch.Tensor]],
    weight_name: str,
) -> torch.Tensor:
    """
    Apply perturbed linear using perturbations from dict.
    
    Args:
        x: Input tensor (pop, *batch, in_dim)
        weight: Weight matrix (out_dim, in_dim)
        bias: Bias vector (out_dim,) or None
        perts: Perturbations dict from generate_perturbations()
        weight_name: Key in perts dict (e.g., 'layer1.weight')
        
    Returns:
        Output tensor (pop, *batch, out_dim)
    """
    if weight_name in perts:
        A_scaled, B = perts[weight_name]
        return perturbed_linear(x, weight, bias, A_scaled, B)
    else:
        return x @ weight.T + (bias if bias is not None else 0)


# =============================================================================
# Low-Rank Primitives (@torch.compile for zero overhead)
# =============================================================================

def generate_lowrank_perturbations(
    population_size: int,
    out_dim: int,
    in_dim: int,
    rank: int,
    sigma: float,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate antithetic low-rank perturbations for EGGROLL on CUDA.
    
    Memory: O(r(m+n)) per layer where m=out_dim, n=in_dim, r=rank
    
    Returns:
        A_scaled: (population_size, out_dim, rank) - scaled A factors
        A: (population_size, out_dim, rank) - unscaled A factors
        B: (population_size, in_dim, rank) - B factors (unscaled)
        
    The perturbation to weight W is: A_scaled @ B.T (low-rank approximation)
    Antithetic: first half and second half are negatives of each other.
    """
    half_pop = population_size // 2
    scale = sigma / (rank ** 0.5)
    
    # Generate positive half
    A_pos = torch.randn(half_pop, out_dim, rank, device="cuda", dtype=dtype, generator=generator)
    B_pos = torch.randn(half_pop, in_dim, rank, device="cuda", dtype=dtype, generator=generator)
    
    # Antithetic pairs: negate A, keep B
    A = torch.cat([A_pos, -A_pos], dim=0)
    B = torch.cat([B_pos, B_pos], dim=0)
    
    # Scale A by sigma/sqrt(rank)
    A_scaled = A * scale
    
    return A_scaled, A, B


@torch.compile
def apply_lowrank_perturbation(x, B, A_scaled):
    """
    Apply low-rank perturbation to input tensor.
    
    Computes: x @ B @ A.T  (never materializes m×n matrix)
    
    Supports any number of batch dimensions:
        - (pop, in_dim) -> (pop, out_dim)
        - (pop, batch, in_dim) -> (pop, batch, out_dim)
        - (pop, b1, b2, in_dim) -> (pop, b1, b2, out_dim)
    
    Args:
        x: (population, *batch_dims, in_dim) - input tensor
        B: (population, in_dim, rank) - B factors
        A_scaled: (population, out_dim, rank) - scaled A factors
        
    Returns:
        perturbation: (population, *batch_dims, out_dim)
    """
    pop_size = x.shape[0]
    in_dim = x.shape[-1]
    out_dim = A_scaled.shape[1]
    batch_shape = x.shape[1:-1]
    
    x_flat = x.reshape(pop_size, -1, in_dim)
    pert_flat = torch.einsum('pbi,pir,pjr->pbj', x_flat, B, A_scaled)
    return pert_flat.reshape(pop_size, *batch_shape, out_dim)


@torch.compile
def perturbed_linear(x, W, b, A_scaled, B):
    """
    Perturbed linear layer: base linear + low-rank perturbation.
    
    Computes: x @ W.T + b + x @ B @ A.T
    
    THIS IS THE FUNDAMENTAL EGGROLL PRIMITIVE.
    
    The key insight from JAX EGGROLL: we compute x @ B @ A.T (two rank-r matmuls)
    instead of x @ (A @ B.T), which would materialize the full m×n perturbation.
    
    Args:
        x: (population, *batch_dims, in_dim) - input tensor
        W: (out_dim, in_dim) - weight matrix (shared across population)
        b: (out_dim,) - bias vector (shared across population)
        A_scaled: (population, out_dim, rank) - scaled A perturbation factors
        B: (population, in_dim, rank) - B perturbation factors
        
    Returns:
        output: (population, *batch_dims, out_dim)
    """
    base = x @ W.T + b
    pert = apply_lowrank_perturbation(x, B, A_scaled)
    return base + pert


@torch.compile
def compute_weight_perturbation(A_scaled: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Materialize the full weight perturbation matrix from low-rank factors.
    
    WARNING: This explicitly computes A @ B.T, which is O(m*n) memory.
    Only use when materialization is unavoidable (e.g., conv2d via grouped conv).
    
    For linear layers, use perturbed_linear() instead - it computes:
        x @ B @ A.T  (two rank-r matmuls, never materializes m×n matrix)
    
    Args:
        A_scaled: (population, out_dim, rank) - scaled A factors  
        B: (population, in_dim, rank) - B factors
        
    Returns:
        delta_W: (population, out_dim, in_dim) - materialized weight perturbation
    """
    return torch.einsum('pir,pjr->pij', A_scaled, B)


@torch.compile
def compute_es_gradient(fitnesses, A_scaled, B, population_size):
    """
    Compute ES gradient from fitnesses and perturbation factors.
    
    Args:
        fitnesses: (population,) - normalized fitness scores
        A_scaled: (population, out_dim, rank) - scaled A factors
        B: (population, in_dim, rank) - B factors
        population_size: int
        
    Returns:
        gradient: (out_dim, in_dim) - gradient estimate for weight matrix
    """
    sqrt_N = math.sqrt(population_size)
    f = fitnesses[:, None, None]
    return torch.einsum('nir,njr->ij', f * A_scaled, B) / sqrt_N


@torch.compile
def normalize_fitnesses(fitnesses, eps=1e-8):
    """
    Normalize fitness scores to zero mean, unit variance.
    
    Args:
        fitnesses: (population,) - raw fitness scores
        eps: small constant for numerical stability
        
    Returns:
        normalized: (population,) - normalized fitness scores
    """
    return (fitnesses - fitnesses.mean()) / (fitnesses.std() + eps)
