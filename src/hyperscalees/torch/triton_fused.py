"""
Fused Triton kernels for EGGROLL low-rank factor generation.

This module provides high-performance Triton kernels that fuse multiple operations:
1. Random number generation (using Philox RNG)
2. Antithetic expansion (in-kernel, no separate pass)
3. Sigma scaling and sign application

Performance improvements over PyTorch native:
- Single kernel launch instead of 5+ separate operations
- No intermediate tensor allocations
- Better memory bandwidth utilization
- Reduced kernel launch overhead

The kernels match the semantics of the PyTorch implementation exactly,
producing deterministic outputs given (seed, epoch, param_id, member_id).
"""
import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


# Primes for mixing seeds (must match triton_kernels.py)
PRIME_EPOCH = 1009
PRIME_PARAM = 10007


@triton.jit
def _philox_round(c0, c1, c2, c3, k0, k1):
    """Single round of Philox 4x32 CBRNG."""
    # Philox constants
    PHILOX_M0: tl.constexpr = 0xD2511F53
    PHILOX_M1: tl.constexpr = 0xCD9E8D57
    
    # Multiply and get high/low parts
    hi0 = ((c0.to(tl.uint64) * PHILOX_M0) >> 32).to(tl.uint32)
    lo0 = (c0 * PHILOX_M0).to(tl.uint32)
    hi1 = ((c2.to(tl.uint64) * PHILOX_M1) >> 32).to(tl.uint32)
    lo1 = (c2 * PHILOX_M1).to(tl.uint32)
    
    # Update counters
    new_c0 = hi1 ^ c1 ^ k0
    new_c1 = lo1
    new_c2 = hi0 ^ c3 ^ k1
    new_c3 = lo0
    
    return new_c0, new_c1, new_c2, new_c3


@triton.jit
def _philox_4x32(seed: tl.uint64, counter: tl.uint32, subsequence: tl.uint32):
    """
    Philox 4x32 random number generator.
    
    Returns 4 uint32 random values that can be converted to floats.
    """
    # Initialize counter
    c0 = counter
    c1 = subsequence
    c2 = tl.zeros_like(counter)
    c3 = tl.zeros_like(counter)
    
    # Initialize key from seed
    k0 = (seed & 0xFFFFFFFF).to(tl.uint32)
    k1 = ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
    
    # Philox bump constants
    PHILOX_W0: tl.constexpr = 0x9E3779B9
    PHILOX_W1: tl.constexpr = 0xBB67AE85
    
    # 10 rounds of Philox
    for _ in range(10):
        c0, c1, c2, c3 = _philox_round(c0, c1, c2, c3, k0, k1)
        k0 = k0 + PHILOX_W0
        k1 = k1 + PHILOX_W1
    
    return c0, c1, c2, c3


@triton.jit
def _uint32_to_normal(u0: tl.uint32, u1: tl.uint32):
    """
    Convert two uint32 to a normally distributed float using Box-Muller.
    
    Returns a single float32 value from N(0, 1).
    """
    # Convert to uniform (0, 1)
    # Use (u + 0.5) / 2^32 to avoid exact 0
    TWO_POW_32_INV: tl.constexpr = 2.3283064365386963e-10  # 1 / 2^32
    
    u = (u0.to(tl.float32) + 0.5) * TWO_POW_32_INV
    v = (u1.to(tl.float32) + 0.5) * TWO_POW_32_INV
    
    # Box-Muller transform
    # z0 = sqrt(-2 * ln(u)) * cos(2 * pi * v)
    import math
    TWO_PI: tl.constexpr = 2.0 * math.pi
    
    r = tl.sqrt(-2.0 * tl.log(u))
    theta = TWO_PI * v
    z0 = r * tl.cos(theta)
    
    return z0


@triton.jit
def generate_lowrank_factors_kernel(
    A_ptr,  # Output: (pop_size, out_features, rank)
    B_ptr,  # Output: (pop_size, in_features, rank)
    member_ids_ptr,  # Input: (pop_size,)
    seed: tl.uint64,
    effective_epoch: tl.int32,
    param_id: tl.int32,
    scaled_sigma: tl.float32,
    in_features: tl.int32,
    out_features: tl.int32,
    rank: tl.int32,
    pop_size: tl.int32,
    antithetic: tl.int32,  # 1 = true, 0 = false
    A_stride_pop: tl.int32,
    A_stride_feat: tl.int32,
    B_stride_pop: tl.int32,
    B_stride_feat: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for low-rank factor generation.
    
    Each program instance handles one (member, rank_idx) pair.
    We generate all in_features + out_features values for that pair.
    """
    # Get program IDs
    pid_member = tl.program_id(0)
    pid_rank = tl.program_id(1)
    
    if pid_member >= pop_size or pid_rank >= rank:
        return
    
    # Load member_id
    member_id = tl.load(member_ids_ptr + pid_member)
    
    # Compute true member index for antithetic sampling
    if antithetic == 1:
        true_member = member_id // 2
        sign = tl.where(member_id % 2 == 0, 1.0, -1.0)
    else:
        true_member = member_id
        sign = 1.0
    
    # Compute unique seed for this (epoch, param, member, rank)
    # This ensures deterministic output
    member_seed = seed + effective_epoch * PRIME_EPOCH + param_id * PRIME_PARAM
    
    # Generate random values for B (in_features values)
    total_features = in_features + out_features
    
    # Process B features
    for feat_idx in range(0, in_features, BLOCK_SIZE):
        feat_offsets = feat_idx + tl.arange(0, BLOCK_SIZE)
        mask = feat_offsets < in_features
        
        # Generate random numbers using Philox
        # Use (true_member, rank_idx, feat_idx) as counter/subsequence
        counter = (true_member * total_features * rank + pid_rank * total_features + feat_offsets).to(tl.uint32)
        subseq = tl.zeros_like(counter)
        
        r0, r1, r2, r3 = _philox_4x32(member_seed, counter, subseq)
        
        # Convert to normal distribution
        normal_val = _uint32_to_normal(r0, r1)
        
        # Store B (unscaled)
        B_offset = pid_member * B_stride_pop + feat_offsets * B_stride_feat + pid_rank
        tl.store(B_ptr + B_offset, normal_val, mask=mask)
    
    # Process A features
    for feat_idx in range(0, out_features, BLOCK_SIZE):
        feat_offsets = feat_idx + tl.arange(0, BLOCK_SIZE)
        mask = feat_offsets < out_features
        
        # Generate random numbers - continue counter from where B left off
        counter = (true_member * total_features * rank + pid_rank * total_features + in_features + feat_offsets).to(tl.uint32)
        subseq = tl.zeros_like(counter)
        
        r0, r1, r2, r3 = _philox_4x32(member_seed, counter, subseq)
        
        # Convert to normal distribution
        normal_val = _uint32_to_normal(r0, r1)
        
        # Apply sigma scaling and antithetic sign
        scaled_val = normal_val * scaled_sigma * sign
        
        # Store A
        A_offset = pid_member * A_stride_pop + feat_offsets * A_stride_feat + pid_rank
        tl.store(A_ptr + A_offset, scaled_val, mask=mask)


def generate_lowrank_factors_triton(
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
    Generate low-rank factors using fused Triton kernel.
    
    This is a drop-in replacement for generate_lowrank_factors_torch with
    identical semantics but better performance through kernel fusion.
    
    Args:
        out_features: Output dimension
        in_features: Input dimension
        rank: Low-rank dimension
        seed: Base random seed
        epoch: Current epoch
        member_ids: Tensor of population member indices
        param_id: Parameter identifier
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
    pop_size = member_ids.shape[0]
    
    # Handle noise reuse
    if noise_reuse > 0:
        effective_epoch = epoch // noise_reuse
    else:
        effective_epoch = 0
    
    # Scale sigma by 1/sqrt(rank)
    scaled_sigma = sigma / (rank ** 0.5)
    
    # Allocate output tensors
    A = torch.empty(pop_size, out_features, rank, device=device, dtype=dtype)
    B = torch.empty(pop_size, in_features, rank, device=device, dtype=dtype)
    
    # Configure kernel launch
    BLOCK_SIZE = 64
    grid = (pop_size, rank)
    
    # Launch kernel
    generate_lowrank_factors_kernel[grid](
        A, B, member_ids,
        seed, effective_epoch, param_id, scaled_sigma,
        in_features, out_features, rank, pop_size,
        1 if antithetic else 0,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return A, B


# =============================================================================
# Simplified Triton kernel using built-in random
# =============================================================================

@triton.jit
def generate_factors_simple_kernel(
    A_ptr,  # Output: (pop_size, out_features, rank)
    B_ptr,  # Output: (pop_size, in_features, rank)
    member_ids_ptr,  # Input: (pop_size,)
    seed,  # uint64 seed
    param_id,  # int32 param id
    scaled_sigma,  # float32
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    rank: tl.constexpr,
    pop_size,
    antithetic: tl.constexpr,  # bool
    BLOCK_FEAT: tl.constexpr,
):
    """
    Simplified kernel using Triton's built-in random functions.
    
    Each program handles one population member.
    """
    pid = tl.program_id(0)
    
    if pid >= pop_size:
        return
    
    # Load member_id
    member_id = tl.load(member_ids_ptr + pid)
    
    # Antithetic handling
    if antithetic:
        true_member = member_id // 2
        sign = tl.where(member_id % 2 == 0, 1.0, -1.0).to(tl.float32)
    else:
        true_member = member_id
        sign = tl.full([], 1.0, dtype=tl.float32)
    
    # Compute unique offset for this member's random sequence
    # Multiply by large prime to spread out seeds
    member_offset = true_member * 999983 + param_id * 1000003
    
    # Generate B values (in_features x rank)
    for r in range(rank):
        for feat_start in range(0, in_features, BLOCK_FEAT):
            feat_offsets = feat_start + tl.arange(0, BLOCK_FEAT)
            mask = feat_offsets < in_features
            
            # Generate random offsets for this block
            rand_offset = member_offset + r * (in_features + out_features) + feat_offsets
            
            # Use Triton's randn
            rand_vals = tl.randn(seed, rand_offset)
            
            # Store B (unscaled)
            B_offset = pid * in_features * rank + feat_offsets * rank + r
            tl.store(B_ptr + B_offset, rand_vals, mask=mask)
    
    # Generate A values (out_features x rank)
    for r in range(rank):
        for feat_start in range(0, out_features, BLOCK_FEAT):
            feat_offsets = feat_start + tl.arange(0, BLOCK_FEAT)
            mask = feat_offsets < out_features
            
            # Generate random offsets (continue from B)
            rand_offset = member_offset + r * (in_features + out_features) + in_features + feat_offsets
            
            # Use Triton's randn
            rand_vals = tl.randn(seed, rand_offset)
            
            # Apply sigma and sign
            scaled_vals = rand_vals * scaled_sigma * sign
            
            # Store A
            A_offset = pid * out_features * rank + feat_offsets * rank + r
            tl.store(A_ptr + A_offset, scaled_vals, mask=mask)


def generate_lowrank_factors_triton_simple(
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
    Generate low-rank factors using simplified Triton kernel with built-in randn.
    
    This version uses Triton's built-in random number generation which is
    simpler and often faster than manual Philox implementation.
    """
    if isinstance(member_ids, int):
        member_ids = torch.tensor([member_ids], dtype=torch.int64)
    
    if device is None:
        device = member_ids.device
    
    member_ids = member_ids.to(device).to(torch.int32)
    pop_size = member_ids.shape[0]
    
    # Handle noise reuse - fold epoch into seed
    if noise_reuse > 0:
        effective_epoch = epoch // noise_reuse
    else:
        effective_epoch = 0
    
    # Combine seed with epoch
    combined_seed = seed + effective_epoch * PRIME_EPOCH
    
    # Scale sigma
    scaled_sigma = sigma / (rank ** 0.5)
    
    # Allocate outputs
    A = torch.empty(pop_size, out_features, rank, device=device, dtype=dtype)
    B = torch.empty(pop_size, in_features, rank, device=device, dtype=dtype)
    
    # Launch kernel
    BLOCK_FEAT = min(64, max(in_features, out_features))
    grid = (pop_size,)
    
    generate_factors_simple_kernel[grid](
        A, B, member_ids,
        combined_seed, param_id, scaled_sigma,
        in_features, out_features, rank, pop_size,
        antithetic,
        BLOCK_FEAT=BLOCK_FEAT,
    )
    
    return A, B


# =============================================================================
# Highly optimized kernel with proper parallelization
# =============================================================================

@triton.jit  
def generate_factors_optimized_kernel(
    A_ptr,
    B_ptr,
    member_ids_ptr,
    seed,
    param_id,
    scaled_sigma,
    in_features,
    out_features,
    rank,
    pop_size,
    antithetic: tl.constexpr,
    BLOCK_POP: tl.constexpr,
    BLOCK_FEAT: tl.constexpr,
):
    """
    Optimized kernel that processes multiple members in parallel.
    
    Grid: (ceil(pop_size/BLOCK_POP), ceil(max_features/BLOCK_FEAT), rank)
    """
    pid_pop = tl.program_id(0)
    pid_feat = tl.program_id(1)
    pid_rank = tl.program_id(2)
    
    # Population indices for this block
    pop_offsets = pid_pop * BLOCK_POP + tl.arange(0, BLOCK_POP)
    pop_mask = pop_offsets < pop_size
    
    # Load member IDs
    member_ids = tl.load(member_ids_ptr + pop_offsets, mask=pop_mask, other=0)
    
    # Antithetic handling
    if antithetic:
        true_members = member_ids // 2
        signs = tl.where(member_ids % 2 == 0, 1.0, -1.0).to(tl.float32)
    else:
        true_members = member_ids
        signs = tl.full([BLOCK_POP], 1.0, dtype=tl.float32)
    
    # Feature indices
    feat_offsets = pid_feat * BLOCK_FEAT + tl.arange(0, BLOCK_FEAT)
    
    # Process B features (feat_idx < in_features)
    B_mask = pop_mask[:, None] & (feat_offsets[None, :] < in_features)
    
    # Generate random offset for B
    rand_offset_B = (
        true_members[:, None] * 999983 + 
        param_id * 1000003 + 
        pid_rank * (in_features + out_features) + 
        feat_offsets[None, :]
    )
    
    # Generate random values
    rand_B = tl.randn(seed, rand_offset_B)
    
    # Store B
    B_offsets = (
        pop_offsets[:, None] * in_features * rank +
        feat_offsets[None, :] * rank +
        pid_rank
    )
    tl.store(B_ptr + B_offsets, rand_B, mask=B_mask)
    
    # Process A features (feat_idx < out_features)
    A_mask = pop_mask[:, None] & (feat_offsets[None, :] < out_features)
    
    # Generate random offset for A (continue from B)
    rand_offset_A = (
        true_members[:, None] * 999983 +
        param_id * 1000003 +
        pid_rank * (in_features + out_features) +
        in_features +
        feat_offsets[None, :]
    )
    
    # Generate and scale random values
    rand_A = tl.randn(seed, rand_offset_A)
    scaled_A = rand_A * scaled_sigma * signs[:, None]
    
    # Store A
    A_offsets = (
        pop_offsets[:, None] * out_features * rank +
        feat_offsets[None, :] * rank +
        pid_rank
    )
    tl.store(A_ptr + A_offsets, scaled_A, mask=A_mask)


def generate_lowrank_factors_triton_optimized(
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
    Generate low-rank factors using optimized Triton kernel.
    
    This is the recommended implementation for production use.
    """
    if isinstance(member_ids, int):
        member_ids = torch.tensor([member_ids], dtype=torch.int64)
    
    if device is None:
        device = member_ids.device
    
    member_ids = member_ids.to(device).to(torch.int32)
    pop_size = member_ids.shape[0]
    
    # Handle noise reuse
    if noise_reuse > 0:
        effective_epoch = epoch // noise_reuse
    else:
        effective_epoch = 0
    
    combined_seed = seed + effective_epoch * PRIME_EPOCH
    scaled_sigma = sigma / (rank ** 0.5)
    
    # Allocate outputs
    A = torch.empty(pop_size, out_features, rank, device=device, dtype=dtype)
    B = torch.empty(pop_size, in_features, rank, device=device, dtype=dtype)
    
    # Configure grid
    BLOCK_POP = 32
    BLOCK_FEAT = 64
    max_features = max(in_features, out_features)
    
    grid = (
        triton.cdiv(pop_size, BLOCK_POP),
        triton.cdiv(max_features, BLOCK_FEAT),
        rank,
    )
    
    generate_factors_optimized_kernel[grid](
        A, B, member_ids,
        combined_seed, param_id, scaled_sigma,
        in_features, out_features, rank, pop_size,
        antithetic,
        BLOCK_POP=BLOCK_POP,
        BLOCK_FEAT=BLOCK_FEAT,
    )
    
    return A, B


# Default to the optimized version
generate_lowrank_factors_fused = generate_lowrank_factors_triton_optimized
