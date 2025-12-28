"""
Triton kernels for EGGROLL's parallel noise generation and batched operations.

This module implements the core computational kernels that make EGGROLL fast:
1. Philox-based parallel PRNG for deterministic, on-the-fly noise generation
2. Batched low-rank perturbed matrix multiplication: out = x @ W.T + x @ B @ A.T
3. Gradient accumulation for ES updates

The key insight is that each population member can generate its own noise
independently using Philox PRNG seeded with (base_seed, epoch, member_id).
This eliminates the need to store noise tensors - we regenerate them as needed.
"""
import torch
import triton
import triton.language as tl
from typing import Tuple
import math
import numpy as np


# =============================================================================
# Philox PRNG Implementation
# =============================================================================
# Philox is a counter-based PRNG that's perfect for GPU parallelism:
# - Each thread can independently compute random numbers
# - Deterministic: same (key, counter) always produces same output  
# - No global state or synchronization needed

PHILOX_ROUNDS = 10
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57


@triton.jit
def philox_round(c0, c1, c2, c3, k0, k1):
    """Single round of Philox 4x32."""
    # Multiply and split
    hi0 = tl.extra.cuda.mulhi(PHILOX_M0, c0)
    lo0 = c0 * PHILOX_M0
    hi1 = tl.extra.cuda.mulhi(PHILOX_M1, c2)
    lo1 = c2 * PHILOX_M1
    
    # Permute and XOR
    new_c0 = hi1 ^ c1 ^ k0
    new_c1 = lo1
    new_c2 = hi0 ^ c3 ^ k1
    new_c3 = lo0
    
    return new_c0, new_c1, new_c2, new_c3


@triton.jit
def philox_4x32(counter0, counter1, counter2, counter3, key0, key1):
    """
    Philox 4x32-10 PRNG.
    
    Takes 4 32-bit counters and 2 32-bit keys, produces 4 32-bit random outputs.
    Counters encode (epoch, member_id, param_id, position).
    Keys encode the base seed.
    """
    c0, c1, c2, c3 = counter0, counter1, counter2, counter3
    k0, k1 = key0, key1
    
    # 10 rounds of Philox
    for _ in tl.static_range(PHILOX_ROUNDS):
        c0, c1, c2, c3 = philox_round(c0, c1, c2, c3, k0, k1)
        # Bump the key
        k0 = k0 + PHILOX_W0
        k1 = k1 + PHILOX_W1
    
    return c0, c1, c2, c3


@triton.jit
def uint32_to_float_01(x):
    """Convert uint32 to float in [0, 1)."""
    # Divide by 2^32
    return x.to(tl.float32) * (1.0 / 4294967296.0)


@triton.jit  
def box_muller(u1, u2):
    """Box-Muller transform: convert two uniform [0,1) to two standard normals."""
    # Clamp u1 away from 0 to avoid log(0)
    u1_safe = tl.maximum(u1, 1e-10)
    r = tl.sqrt(-2.0 * tl.log(u1_safe))
    theta = 2.0 * 3.14159265358979323846 * u2
    z0 = r * tl.cos(theta)
    z1 = r * tl.sin(theta)
    return z0, z1


# =============================================================================
# Low-Rank Factor Generation Kernel
# =============================================================================

@triton.jit
def generate_lowrank_factors_kernel(
    A_ptr,  # Output: (out_features, rank)
    B_ptr,  # Output: (in_features, rank)
    seed0: tl.constexpr,  # Base seed (low 32 bits)
    seed1: tl.constexpr,  # Base seed (high 32 bits)
    epoch,
    member_id,
    param_id,
    out_features,
    in_features,
    rank: tl.constexpr,
    sigma,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Generate low-rank factors A (out_features x rank) and B (in_features x rank).
    
    Uses Philox PRNG seeded with (seed, epoch, member_id, param_id) so the same
    inputs always produce the same factors - enabling noise regeneration.
    """
    pid = tl.program_id(0)
    
    # Determine if we're generating for A or B based on pid
    total_A_elements = out_features * rank
    total_B_elements = in_features * rank
    
    # Element offset within this block
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Generate A elements (first total_A_elements)
    mask_A = offs < total_A_elements
    
    # Counter: use offset as part of counter
    c0 = offs.to(tl.uint32)
    c1 = (epoch & 0xFFFFFFFF).to(tl.uint32)  # epoch as uint32
    c2 = (member_id & 0xFFFFFFFF).to(tl.uint32)  # member as uint32
    c3 = ((param_id << 16) | 0).to(tl.uint32)  # param_id + marker for A
    
    # Generate 4 random uint32s
    r0, r1, r2, r3 = philox_4x32(c0, c1, c2, c3, seed0, seed1)
    
    # Convert to floats and apply Box-Muller
    u0 = uint32_to_float_01(r0)
    u1 = uint32_to_float_01(r1)
    z0, z1 = box_muller(u0, u1)
    
    # Scale by sigma and store A
    tl.store(A_ptr + offs, z0 * sigma, mask=mask_A)
    
    # Now generate B elements (offset by total_A_elements)
    offs_B = offs
    mask_B = offs_B < total_B_elements
    
    # Different counter for B (change c3 marker)
    c3_B = ((param_id << 16) | 1).to(tl.uint32)  # param_id + marker for B
    
    r0_B, r1_B, r2_B, r3_B = philox_4x32(c0, c1, c2, c3_B, seed0, seed1)
    u0_B = uint32_to_float_01(r0_B)
    u1_B = uint32_to_float_01(r1_B)
    z0_B, z1_B = box_muller(u0_B, u1_B)
    
    tl.store(B_ptr + offs_B, z0_B, mask=mask_B)


# =============================================================================
# Batched Low-Rank Perturbed MatMul Kernel
# =============================================================================

@triton.jit
def batched_perturbed_linear_kernel(
    # Inputs
    X_ptr,        # (batch, in_features)
    W_ptr,        # (out_features, in_features)  
    bias_ptr,     # (out_features,) or None
    member_ids_ptr,  # (batch,) - which population member for each sample
    # Outputs
    Out_ptr,      # (batch, out_features)
    # Noise generation params
    seed0: tl.constexpr,
    seed1: tl.constexpr,
    epoch,
    param_id,
    # Dimensions
    batch_size,
    in_features,
    out_features,
    rank: tl.constexpr,
    sigma,
    # Flags
    HAS_BIAS: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    Compute batched linear with low-rank perturbations:
        out[i] = x[i] @ W.T + x[i] @ B[member_ids[i]] @ A[member_ids[i]].T + bias
    
    Each sample uses a different perturbation based on its member_id.
    Perturbations are generated on-the-fly using Philox PRNG.
    
    This kernel computes the result WITHOUT materializing the full perturbed weight matrix.
    """
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # output feature dimension
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # batch indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output indices
    offs_k = tl.arange(0, BLOCK_K)  # input feature indices
    offs_r = tl.arange(0, BLOCK_R)  # rank indices
    
    # Masks
    mask_m = offs_m < batch_size
    mask_n = offs_n < out_features
    
    # Load member_ids for this batch block
    member_ids = tl.load(member_ids_ptr + offs_m, mask=mask_m, other=0)
    
    # Accumulator for output
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Part 1: Compute x @ W.T (standard matmul)
    for k in range(0, in_features, BLOCK_K):
        offs_k_curr = k + offs_k
        mask_k = offs_k_curr < in_features
        
        # Load X block: (BLOCK_M, BLOCK_K)
        x_ptrs = X_ptr + offs_m[:, None] * in_features + offs_k_curr[None, :]
        x_block = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load W block: (BLOCK_N, BLOCK_K) -> transpose to (BLOCK_K, BLOCK_N)
        w_ptrs = W_ptr + offs_n[:, None] * in_features + offs_k_curr[None, :]
        w_block = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Accumulate: x @ W.T
        acc += tl.dot(x_block, tl.trans(w_block))
    
    # Part 2: Compute x @ B @ A.T (low-rank perturbation)
    # We need to generate A and B on-the-fly for each unique member_id in this block
    # This is the tricky part - we process each sample with its own perturbation
    
    # For simplicity in this kernel, we'll loop over unique member_ids
    # A more optimized version would batch samples with the same member_id
    
    for sample_idx in range(BLOCK_M):
        if offs_m[sample_idx] >= batch_size:
            continue
            
        member_id = tl.load(member_ids_ptr + offs_m[sample_idx])
        
        # Generate noise for this member's A and B factors
        # A: (out_features, rank), B: (in_features, rank)
        # We'll compute x @ B @ A.T element by element
        
        # Load x for this sample
        x_sample = tl.zeros((BLOCK_K,), dtype=tl.float32)
        
        # Compute x @ B first -> (rank,)
        xB = tl.zeros((BLOCK_R,), dtype=tl.float32)
        for k in range(0, in_features, BLOCK_K):
            offs_k_curr = k + offs_k
            mask_k = offs_k_curr < in_features
            
            x_val = tl.load(X_ptr + offs_m[sample_idx] * in_features + offs_k_curr, 
                           mask=mask_k, other=0.0)
            
            # Generate B[k, :] for this member
            for r in range(rank):
                # Philox counter for B[k, r]
                c0 = (offs_k_curr).to(tl.uint32)
                c1 = (epoch & 0xFFFFFFFF).to(tl.uint32)
                c2 = (member_id & 0xFFFFFFFF).to(tl.uint32)
                c3 = ((param_id << 16) | (r << 8) | 1).to(tl.uint32)  # B marker
                
                r0, r1, _, _ = philox_4x32(c0, c1, c2, c3, seed0, seed1)
                u0 = uint32_to_float_01(r0)
                u1 = uint32_to_float_01(r1)
                b_val, _ = box_muller(u0, u1)
                
                xB[r] += tl.sum(x_val * b_val * mask_k)
        
        # Now compute xB @ A.T -> (out_features,)
        for n_idx in range(BLOCK_N):
            if offs_n[n_idx] >= out_features:
                continue
            
            pert_val = 0.0
            for r in range(rank):
                # Generate A[n, r] for this member
                c0 = (offs_n[n_idx]).to(tl.uint32)
                c1 = (epoch & 0xFFFFFFFF).to(tl.uint32)
                c2 = (member_id & 0xFFFFFFFF).to(tl.uint32)
                c3 = ((param_id << 16) | (r << 8) | 0).to(tl.uint32)  # A marker
                
                r0, r1, _, _ = philox_4x32(c0, c1, c2, c3, seed0, seed1)
                u0 = uint32_to_float_01(r0)
                u1 = uint32_to_float_01(r1)
                a_val, _ = box_muller(u0, u1)
                
                pert_val += xB[r] * a_val * sigma
            
            # Add perturbation to accumulator
            # This is inefficient - in practice we'd vectorize this better
            acc[sample_idx, n_idx] += pert_val
    
    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias[None, :]
    
    # Store output
    out_ptrs = Out_ptr + offs_m[:, None] * out_features + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# =============================================================================
# Python Interface Functions
# =============================================================================

def generate_lowrank_factors(
    out_features: int,
    in_features: int, 
    rank: int,
    seed: int,
    epoch: int,
    member_id: int,
    param_id: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate low-rank factors A and B using Philox PRNG.
    
    Args:
        out_features: Number of output features (m)
        in_features: Number of input features (n)
        rank: Rank of the perturbation
        seed: Base PRNG seed
        epoch: Current epoch (for noise scheduling)
        member_id: Population member index
        param_id: Parameter index (for multi-param models)
        sigma: Noise standard deviation
        device: Torch device
        dtype: Output dtype
        
    Returns:
        A: (out_features, rank) tensor
        B: (in_features, rank) tensor
        
    The perturbation matrix is A @ B.T, but we never compute this directly.
    """
    A = torch.empty(out_features, rank, device=device, dtype=dtype)
    B = torch.empty(in_features, rank, device=device, dtype=dtype)
    
    seed0 = seed & 0xFFFFFFFF
    seed1 = (seed >> 32) & 0xFFFFFFFF
    
    # Total elements to generate
    total_elements = max(out_features * rank, in_features * rank)
    BLOCK_SIZE = 256
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    generate_lowrank_factors_kernel[grid](
        A, B,
        seed0, seed1,
        epoch, member_id, param_id,
        out_features, in_features, rank,
        sigma,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return A, B


def batched_perturbed_linear(
    x: torch.Tensor,           # (batch, in_features)
    weight: torch.Tensor,       # (out_features, in_features)
    bias: torch.Tensor | None,  # (out_features,) or None
    member_ids: torch.Tensor,   # (batch,) - int tensor
    seed: int,
    epoch: int,
    param_id: int,
    rank: int,
    sigma: float,
) -> torch.Tensor:
    """
    Batched linear layer with per-sample low-rank perturbations.
    
    Computes: out[i] = x[i] @ W.T + x[i] @ B[member_ids[i]] @ A[member_ids[i]].T + bias
    
    The A and B factors are generated on-the-fly using Philox PRNG, so we never
    need to store them. This is the key to EGGROLL's memory efficiency.
    
    Args:
        x: Input tensor (batch, in_features)
        weight: Weight matrix (out_features, in_features)
        bias: Optional bias (out_features,)
        member_ids: Population member index for each sample (batch,)
        seed: Base PRNG seed
        epoch: Current epoch
        param_id: Parameter index
        rank: Perturbation rank
        sigma: Noise scale
        
    Returns:
        Output tensor (batch, out_features)
    """
    batch_size, in_features = x.shape
    out_features = weight.shape[0]
    
    output = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype)
    
    seed0 = seed & 0xFFFFFFFF
    seed1 = (seed >> 32) & 0xFFFFFFFF
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    BLOCK_R = min(rank, 16)
    
    grid = (
        (batch_size + BLOCK_M - 1) // BLOCK_M,
        (out_features + BLOCK_N - 1) // BLOCK_N,
    )
    
    batched_perturbed_linear_kernel[grid](
        x, weight, bias, member_ids, output,
        seed0, seed1, epoch, param_id,
        batch_size, in_features, out_features, rank, sigma,
        bias is not None,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_R=BLOCK_R,
    )
    
    return output


# =============================================================================
# Vectorized Python Implementation (Fallback / Reference)
# =============================================================================
# This implementation uses PyTorch ops and is easier to understand/debug.
# It's not as memory-efficient as the Triton version but still avoids
# sequential iteration over population members.

def generate_lowrank_factors_torch(
    out_features: int,
    in_features: int,
    rank: int,
    seed: int,
    epoch: int,
    member_ids: torch.Tensor,  # (population_size,) or single int
    param_id: int,
    sigma: float,
    noise_reuse: int = 0,
    antithetic: bool = True,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate low-rank factors for multiple population members at once.
    
    Uses a vectorized approach: generate a unique seed for each (epoch, member, param)
    combination and use it to seed independent generators.
    
    For antithetic sampling:
    - member_ids 2k and 2k+1 share the same base noise
    - member 2k gets +sigma, member 2k+1 gets -sigma
    - This is implemented by using member_id // 2 as the "true" index
    
    Returns:
        A: (population_size, out_features, rank) tensor - scaled by ±sigma
        B: (population_size, in_features, rank) tensor - unscaled
    """
    if isinstance(member_ids, int):
        member_ids = torch.tensor([member_ids], device=device)
    
    population_size = member_ids.shape[0]
    member_ids_np = member_ids.cpu().numpy()
    
    # Handle noise reuse: epoch 0-4 all use epoch 0 if noise_reuse=5
    if noise_reuse > 0:
        effective_epoch = epoch // noise_reuse
    else:
        effective_epoch = 0  # Same noise for all epochs
    
    # For antithetic sampling, pairs (2k, 2k+1) share the same base noise
    # The "true" member index is member_id // 2
    if antithetic:
        true_member_ids = member_ids_np // 2
        # Sigma sign: even members get +sigma, odd members get -sigma
        sigma_signs = np.where(member_ids_np % 2 == 0, 1.0, -1.0)
    else:
        true_member_ids = member_ids_np
        sigma_signs = np.ones(population_size)
    
    # Create deterministic seeds for each member
    # Use a simple hash combining seed, epoch, member_id, param_id
    # This ensures reproducibility: same inputs -> same outputs
    base_seeds = (
        seed 
        + effective_epoch * 1000000 
        + true_member_ids * 1000 
        + param_id
    )
    
    # Allocate output tensors
    A = torch.empty(population_size, out_features, rank, device=device, dtype=dtype)
    B = torch.empty(population_size, in_features, rank, device=device, dtype=dtype)
    
    # Generate noise for all members (still uses a loop but over members only, not params)
    # In production, this would use Triton for true parallelism
    for i, (s, sign) in enumerate(zip(base_seeds, sigma_signs)):
        gen = torch.Generator(device='cpu').manual_seed(int(s))
        
        # Generate A and B from the same random stream for consistency
        combined = torch.randn(out_features + in_features, rank, generator=gen, dtype=dtype)
        B[i] = combined[:in_features].to(device)
        # A is scaled by sigma with appropriate sign for antithetic
        A[i] = combined[in_features:].to(device) * sigma * sign
        
    return A, B


def batched_perturbed_linear_torch(
    x: torch.Tensor,            # (batch, in_features)
    weight: torch.Tensor,       # (out_features, in_features)
    bias: torch.Tensor | None,  # (out_features,) or None
    A: torch.Tensor,            # (population_size, out_features, rank)
    B: torch.Tensor,            # (population_size, in_features, rank)
    member_ids: torch.Tensor,   # (batch,) - indices into population
) -> torch.Tensor:
    """
    Batched linear with pre-generated low-rank factors.
    
    Computes: out[i] = x[i] @ W.T + x[i] @ B[member_ids[i]] @ A[member_ids[i]].T + bias
    
    Uses efficient gather + batched matmul to avoid sequential iteration.
    """
    batch_size = x.shape[0]
    
    # Gather the A and B factors for each sample's member_id
    # A_gather: (batch, out_features, rank)
    # B_gather: (batch, in_features, rank)
    A_gather = A[member_ids]
    B_gather = B[member_ids]
    
    # Compute base: x @ W.T -> (batch, out_features)
    base = torch.mm(x, weight.t())
    
    # Compute perturbation: x @ B @ A.T
    # Step 1: x @ B -> (batch, rank)
    # x: (batch, in_features), B_gather: (batch, in_features, rank)
    xB = torch.einsum('bi,bir->br', x, B_gather)
    
    # Step 2: xB @ A.T -> (batch, out_features)
    # xB: (batch, rank), A_gather: (batch, out_features, rank)
    perturbation = torch.einsum('br,bor->bo', xB, A_gather)
    
    # Combine
    out = base + perturbation
    
    if bias is not None:
        out = out + bias
    
    return out


def compute_es_gradient_torch(
    fitnesses: torch.Tensor,     # (population_size,)
    A: torch.Tensor,             # (population_size, out_features, rank)
    B: torch.Tensor,             # (population_size, in_features, rank)
    sigma: float,
) -> torch.Tensor:
    """
    Compute ES gradient estimate from fitnesses and perturbation factors.
    
    The ES gradient is: (1/σ²) * E[fitness * perturbation]
    
    For low-rank perturbations ΔW = A @ B.T, we compute:
        grad = sum_i(fitness[i] * A[i] @ B[i].T) / population_size
        
    We use einsum to compute this WITHOUT materializing the full perturbation matrices.
    """
    population_size = fitnesses.shape[0]
    
    # Normalize fitnesses (already done in convert_fitnesses, but just in case)
    # Shape broadcasting: fitnesses[:, None, None] for (pop, 1, 1)
    weighted_A = fitnesses[:, None, None] * A  # (pop, out, rank)
    
    # Compute gradient: einsum over population and rank dimensions
    # grad[o, i] = sum_p sum_r (weighted_A[p, o, r] * B[p, i, r]) / pop_size
    grad = torch.einsum('por,pir->oi', weighted_A, B) / population_size
    
    return grad
