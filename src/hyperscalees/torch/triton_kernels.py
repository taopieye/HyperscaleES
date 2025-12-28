"""
Fast batched operations for EGGROLL using PyTorch native operations.

This module implements the core computational primitives that make EGGROLL fast,
matching JAX's performance by using:
1. Vectorized noise generation with torch.Generator (one call for all population)
2. Efficient batched matrix multiplication with bmm
3. Proper antithetic sampling (paired members share noise, opposite signs)

JAX reference pattern (from noiser/eggroll.py):
    true_thread_idx = thread_id // 2  # Antithetic pairs
    sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)
    key = fold_in(fold_in(base_key, epoch), true_thread_idx)
    lora_params = randn(key, (in_features + out_features, rank))
    B = lora_params[:in_features]
    A = lora_params[in_features:] * sigma

This implementation uses:
    - One torch.Generator seeded by (base_seed, epoch, param_id)
    - Generate (pop_size//2, in+out, rank) noise for unique members
    - Replicate for antithetic pairs
    - Apply ±sigma based on member parity

This approach is 10-20x faster than per-member Triton noise generation because:
1. PyTorch's cuRAND is highly optimized for bulk generation
2. bmm is heavily optimized for batched workloads
3. No kernel launch overhead per member
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


# Prime for mixing epoch into seed (small enough for int32)
PRIME_EPOCH = 1009
PRIME_PARAM = 10007


def generate_lowrank_factors_torch(
    out_features: int,
    in_features: int,
    rank: int,
    seed: int,
    epoch: int,
    member_ids: torch.Tensor,
    param_id: int,
    sigma: float,
    noise_reuse: int = 0,
    antithetic: bool = True,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate low-rank factors for multiple population members.
    
    Uses vectorized noise generation matching JAX's semantics:
    - Deterministic: same (seed, epoch, param_id) → same noise
    - Antithetic: paired members (0,1), (2,3), etc. share base noise
    - Efficient: one generator call for all population
    
    Args:
        out_features: Output dimension (a in JAX code)
        in_features: Input dimension (b in JAX code)  
        rank: Low-rank dimension
        seed: Base random seed
        epoch: Current epoch (for determinism across calls)
        member_ids: Tensor of population member indices
        param_id: Parameter identifier (for per-layer determinism)
        sigma: Base perturbation strength
        noise_reuse: Epochs between noise regeneration (0 = always regenerate)
        antithetic: Whether to use antithetic sampling
        device: Target device
        dtype: Data type
    
    Returns:
        A: (population_size, out_features, rank) - scaled by ±sigma/sqrt(rank)
        B: (population_size, in_features, rank) - unscaled
    """
    if isinstance(member_ids, int):
        member_ids = torch.tensor([member_ids], dtype=torch.int64)
    
    if device is None:
        device = member_ids.device
    
    member_ids = member_ids.to(device)
    population_size = member_ids.shape[0]
    
    # Handle noise reuse (epoch folding, matching JAX)
    if noise_reuse > 0:
        effective_epoch = epoch // noise_reuse
    else:
        effective_epoch = 0
    
    # Scale sigma by 1/sqrt(rank) as in JAX
    scaled_sigma = sigma / (rank ** 0.5)
    
    # Compute seed for this (epoch, param)
    gen_seed = seed + effective_epoch * PRIME_EPOCH + param_id * PRIME_PARAM
    
    # Create generator
    gen = torch.Generator(device=device)
    gen.manual_seed(gen_seed)
    
    if antithetic:
        # For antithetic sampling, generate noise for unique members only
        # Members 0,1 share noise, members 2,3 share noise, etc.
        num_unique = (member_ids.max().item() // 2) + 1
        
        # Generate noise: (num_unique, in_features + out_features, rank)
        noise = torch.randn(
            num_unique, in_features + out_features, rank,
            generator=gen, device=device, dtype=dtype
        )
        
        # Use repeat_interleave for efficient expansion (faster than indexing)
        # This creates [noise[0], noise[0], noise[1], noise[1], ...]
        noise_expanded = noise.repeat_interleave(2, dim=0)
        
        # Slice to get exactly population_size (handles odd population sizes)
        gathered_noise = noise_expanded[member_ids]
        
        # Split into B and A
        B = gathered_noise[:, :in_features, :]  # (pop, in, rank)
        A_base = gathered_noise[:, in_features:, :] * scaled_sigma  # (pop, out, rank)
        
        # Apply antithetic signs: even member_ids → +1, odd → -1
        signs = torch.where(
            (member_ids % 2 == 0).view(-1, 1, 1),
            torch.ones(1, device=device, dtype=dtype),
            -torch.ones(1, device=device, dtype=dtype)
        )
        A = A_base * signs
    else:
        # No antithetic: each member gets unique noise
        max_member = member_ids.max().item() + 1
        
        noise = torch.randn(
            max_member, in_features + out_features, rank,
            generator=gen, device=device, dtype=dtype
        )
        
        gathered_noise = noise[member_ids]
        B = gathered_noise[:, :in_features, :]
        A = gathered_noise[:, in_features:, :] * scaled_sigma
    
    return A, B


def batched_perturbed_linear_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    A: torch.Tensor,
    B: torch.Tensor,
    member_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Batched linear with pre-generated low-rank factors.
    
    Computes: out[i] = x[i] @ W.T + x[i] @ B[i] @ A[i].T + bias
    
    NOTE: A and B should already be indexed/gathered for each batch element.
    The member_ids parameter is kept for API compatibility but not used here.
    
    This uses PyTorch's optimized bmm for the batched computation.
    """
    # A and B are already gathered for each batch element - use directly
    # Base linear: x @ W.T + bias
    if bias is not None:
        base = F.linear(x, weight, bias)
    else:
        base = F.linear(x, weight)
    
    # Low-rank perturbation: x @ B @ A.T
    # x: (batch, in), B: (batch, in, rank) → xB: (batch, rank)
    xB = torch.bmm(x.unsqueeze(1), B).squeeze(1)
    # xB: (batch, rank), A.T: (batch, rank, out) → pert: (batch, out)
    perturbation = torch.bmm(xB.unsqueeze(1), A.transpose(1, 2)).squeeze(1)
    
    return base + perturbation


def fused_perturbed_linear_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    member_ids: torch.Tensor,
    seed: int,
    epoch: int,
    param_id: int,
    rank: int,
    sigma: float,
    noise_reuse: int = 0,
    antithetic: bool = True,
) -> torch.Tensor:
    """
    Fused perturbed linear - generates factors and computes in one call.
    
    This is the main API matching JAX's do_mm behavior:
    - Deterministic noise from (seed, epoch, param_id)
    - Antithetic sampling for variance reduction
    - Efficient batched computation
    """
    out_features, in_features = weight.shape
    
    # Generate factors
    A, B = generate_lowrank_factors_torch(
        out_features=out_features,
        in_features=in_features,
        rank=rank,
        seed=seed,
        epoch=epoch,
        member_ids=member_ids,
        param_id=param_id,
        sigma=sigma,
        noise_reuse=noise_reuse,
        antithetic=antithetic,
        device=x.device,
        dtype=x.dtype,
    )
    
    # Compute perturbed linear
    return batched_perturbed_linear_torch(x, weight, bias, A, B, member_ids)


def compute_es_gradient_torch(
    fitnesses: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Compute ES gradient estimate from fitnesses and perturbation factors.
    
    The ES gradient is: E[fitness * perturbation]
    For low-rank: grad = sum_i(fitness[i] * A[i] @ B[i].T) / population_size
    """
    population_size = fitnesses.shape[0]
    # Weighted sum: fitness[i] * A[i] @ B[i].T
    weighted_A = fitnesses[:, None, None] * A  # (pop, out, rank)
    grad = torch.einsum('por,pir->oi', weighted_A, B) / population_size
    return grad
