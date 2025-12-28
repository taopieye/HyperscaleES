"""
Triton kernels for EGGROLL's parallel noise generation and batched operations.

This module implements the core computational kernels that make EGGROLL fast:
1. Philox-based parallel PRNG for deterministic, on-the-fly noise generation
2. Fused perturbed linear: out = x @ W.T + x @ B @ A.T + bias in ONE kernel
3. Gradient accumulation for ES updates

The key insight is that each population member can generate its own noise
independently using Philox PRNG seeded with (base_seed, epoch, member_id).
This eliminates the need to store noise tensors - we regenerate them as needed.

This implementation follows the JAX reference in noiser/eggroll.py:
- B comes from first `in_features` elements of combined noise
- A comes from last `out_features` elements  
- Only A is scaled by sigma (with antithetic sign)
- true_member = member_id // 2 for antithetic pairing
"""
import torch
import triton
import triton.language as tl
from typing import Tuple, Optional
import math


# =============================================================================
# Philox PRNG Implementation
# =============================================================================
# Philox is a counter-based PRNG that's perfect for GPU parallelism:
# - Each thread can independently compute random numbers
# - Deterministic: same (key, counter) always produces same output  
# - No global state or synchronization needed

PHILOX_ROUNDS: tl.constexpr = 10
PHILOX_W0: tl.constexpr = 0x9E3779B9
PHILOX_W1: tl.constexpr = 0xBB67AE85
PHILOX_M0: tl.constexpr = 0xD2511F53
PHILOX_M1: tl.constexpr = 0xCD9E8D57


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
    return x.to(tl.float32) * (1.0 / 4294967296.0)


@triton.jit  
def box_muller(u1, u2):
    """Box-Muller transform: convert two uniform [0,1) to two standard normals."""
    u1_safe = tl.maximum(u1, 1e-10)
    r = tl.sqrt(-2.0 * tl.log(u1_safe))
    theta = 2.0 * 3.14159265358979323846 * u2
    z0 = r * tl.cos(theta)
    z1 = r * tl.sin(theta)
    return z0, z1


# =============================================================================
# Factor Generation Kernel (for step() and testing)
# =============================================================================

@triton.jit
def generate_factors_kernel(
    A_ptr,           # (population, out_features, rank)
    B_ptr,           # (population, in_features, rank)
    member_ids_ptr,  # (population,) int32
    key0, key1,
    epoch,
    param_id,
    sigma,
    population_size,
    out_features,
    in_features,
    stride_ap, stride_ao, stride_ar,
    stride_bp, stride_bi, stride_br,
    rank: tl.constexpr,
    ANTITHETIC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Generate A and B factors for all population members.
    
    Each program generates one (population, dimension, rank) element.
    """
    pid = tl.program_id(0)
    
    # Total elements to generate
    total_A_elements = population_size * out_features * rank
    total_B_elements = population_size * in_features * rank
    
    # Offsets for this block
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # First generate B elements
    mask_B = offs < total_B_elements
    
    # Decompose linear index into (p, i, r) for B
    b_p = offs // (in_features * rank)
    b_remainder = offs % (in_features * rank)
    b_i = b_remainder // rank
    b_r = b_remainder % rank
    
    # Load member_ids
    member_ids_B = tl.load(member_ids_ptr + b_p, mask=mask_B, other=0)
    
    # Compute true_member and sigma_sign for antithetic sampling
    true_members_B = member_ids_B // 2 if ANTITHETIC else member_ids_B
    
    # Generate B values (unscaled)
    c0_B = b_i.to(tl.uint32)
    c1_B = b_r.to(tl.uint32)
    c2_B = true_members_B.to(tl.uint32)
    c3_B = ((epoch << 20) | (param_id << 4) | 1).to(tl.uint32)  # 1 = B marker
    
    r0_B, r1_B, _, _ = philox_4x32(c0_B, c1_B, c2_B, c3_B, key0, key1)
    u0_B = uint32_to_float_01(r0_B)
    u1_B = uint32_to_float_01(r1_B)
    z0_B, _ = box_muller(u0_B, u1_B)
    
    # Store B
    B_offsets = b_p * stride_bp + b_i * stride_bi + b_r * stride_br
    tl.store(B_ptr + B_offsets, z0_B, mask=mask_B)
    
    # Now generate A elements
    mask_A = offs < total_A_elements
    
    # Decompose linear index into (p, o, r) for A
    a_p = offs // (out_features * rank)
    a_remainder = offs % (out_features * rank)
    a_o = a_remainder // rank
    a_r = a_remainder % rank
    
    # Load member_ids for A
    member_ids_A = tl.load(member_ids_ptr + a_p, mask=mask_A, other=0)
    
    # Compute true_member and sigma_sign
    true_members_A = member_ids_A // 2 if ANTITHETIC else member_ids_A
    sigma_signs_A = tl.where(member_ids_A % 2 == 0, 1.0, -1.0) if ANTITHETIC else tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
    
    # Scale factor: sigma / sqrt(rank)
    scaled_sigma = sigma / tl.sqrt(rank.to(tl.float32))
    
    # Generate A values (scaled)
    c0_A = a_o.to(tl.uint32)
    c1_A = a_r.to(tl.uint32)
    c2_A = true_members_A.to(tl.uint32)
    c3_A = ((epoch << 20) | (param_id << 4) | 0).to(tl.uint32)  # 0 = A marker
    
    r0_A, r1_A, _, _ = philox_4x32(c0_A, c1_A, c2_A, c3_A, key0, key1)
    u0_A = uint32_to_float_01(r0_A)
    u1_A = uint32_to_float_01(r1_A)
    z0_A, _ = box_muller(u0_A, u1_A)
    
    # Scale by sigma * sign
    z0_A = z0_A * scaled_sigma * sigma_signs_A
    
    # Store A
    A_offsets = a_p * stride_ap + a_o * stride_ao + a_r * stride_ar
    tl.store(A_ptr + A_offsets, z0_A, mask=mask_A)


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
    Generate low-rank factors for multiple population members using Triton.
    
    This is the GPU-parallel version for generating factors for step()
    or for testing purposes.
    
    Returns:
        A: (population_size, out_features, rank) - scaled by ±sigma/sqrt(rank)
        B: (population_size, in_features, rank) - unscaled
    """
    if isinstance(member_ids, int):
        member_ids = torch.tensor([member_ids], device=device)
    
    population_size = member_ids.shape[0]
    
    if device is None:
        device = member_ids.device
    
    member_ids = member_ids.contiguous().to(dtype=torch.int32, device=device)
    
    # Handle noise reuse
    if noise_reuse > 0:
        effective_epoch = epoch // noise_reuse
    else:
        effective_epoch = 0
    
    # Allocate output
    A = torch.empty(population_size, out_features, rank, device=device, dtype=dtype)
    B = torch.empty(population_size, in_features, rank, device=device, dtype=dtype)
    
    key0 = seed & 0xFFFFFFFF
    key1 = (seed >> 32) & 0xFFFFFFFF
    
    # Grid configuration
    BLOCK_SIZE = 256
    total_elements = max(
        population_size * out_features * rank,
        population_size * in_features * rank
    )
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    generate_factors_kernel[grid](
        A, B, member_ids,
        key0, key1,
        effective_epoch, param_id, sigma,
        population_size, out_features, in_features,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        rank=rank,
        ANTITHETIC=antithetic,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return A, B


# Alias for backwards compatibility
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
    
    This is the main API for generating factors. It uses Triton GPU kernels
    for efficient parallel generation.
    
    Returns:
        A: (population_size, out_features, rank) - scaled by ±sigma/sqrt(rank)
        B: (population_size, in_features, rank) - unscaled
    """
    return generate_lowrank_factors_triton(
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
        device=device,
        dtype=dtype,
    )


# =============================================================================
# Fused Perturbed Linear Kernel
# =============================================================================

@triton.jit
def fused_perturbed_linear_kernel(
    # Inputs
    X_ptr,           # (batch, in_features)
    W_ptr,           # (out_features, in_features)
    bias_ptr,        # (out_features,) or None
    member_ids_ptr,  # (batch,) int32
    Out_ptr,         # (batch, out_features)
    # Noise params  
    key0,            # Base seed low 32 bits
    key1,            # Base seed high 32 bits
    epoch,
    param_id,
    sigma,
    # Dimensions
    batch_size,
    in_features,
    out_features,
    # Strides
    stride_xb, stride_xi,
    stride_wo, stride_wi,
    stride_ob, stride_oo,
    # Config
    rank: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ANTITHETIC: tl.constexpr,
    # Block sizes
    BLOCK_B: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused perturbed linear with on-the-fly noise generation.
    
    Computes: out[b] = x[b] @ W.T + x[b] @ B[m] @ A[m].T + bias
    where m = member_ids[b], and A, B are generated on-the-fly via Philox.
    
    Following JAX reference:
    - B is unscaled random normal
    - A = random_normal * sigma * sign / sqrt(rank)
    """
    pid_b = tl.program_id(0)  # Batch block
    pid_o = tl.program_id(1)  # Output feature block
    
    # Batch indices for this block
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < batch_size
    
    # Output indices for this block
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    mask_o = offs_o < out_features
    
    # Load member_ids for this batch block
    member_ids = tl.load(member_ids_ptr + offs_b, mask=mask_b, other=0)
    
    # Compute true_member and sigma_sign for antithetic sampling
    if ANTITHETIC:
        true_members = member_ids // 2
        sigma_signs = tl.where(member_ids % 2 == 0, 1.0, -1.0)
    else:
        true_members = member_ids
        sigma_signs = tl.full((BLOCK_B,), 1.0, dtype=tl.float32)
    
    # Accumulator for output: (BLOCK_B, BLOCK_O)
    acc = tl.zeros((BLOCK_B, BLOCK_O), dtype=tl.float32)
    
    # =========================================================================
    # Part 1: Base matmul x @ W.T
    # =========================================================================
    for k_start in range(0, in_features, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < in_features
        
        # Load X block: (BLOCK_B, BLOCK_K)
        x_ptrs = X_ptr + offs_b[:, None] * stride_xb + offs_k[None, :] * stride_xi
        x_block = tl.load(x_ptrs, mask=mask_b[:, None] & mask_k[None, :], other=0.0)
        
        # Load W block: (BLOCK_O, BLOCK_K)
        w_ptrs = W_ptr + offs_o[:, None] * stride_wo + offs_k[None, :] * stride_wi
        w_block = tl.load(w_ptrs, mask=mask_o[:, None] & mask_k[None, :], other=0.0)
        
        # Accumulate: x @ W.T -> (BLOCK_B, BLOCK_O)
        acc += tl.dot(x_block, tl.trans(w_block), allow_tf32=True)
    
    # =========================================================================
    # Part 2: Low-rank perturbation x @ B @ A.T
    # =========================================================================
    # Scale factor: sigma / sqrt(rank) as in JAX reference
    scaled_sigma = sigma / tl.sqrt(rank.to(tl.float32))
    
    # For each rank component
    for r in tl.static_range(rank):
        # Compute xB[b] = sum_k(x[b,k] * B[true_member[b], k, r])
        xB = tl.zeros((BLOCK_B,), dtype=tl.float32)
        
        for k_start in range(0, in_features, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < in_features
            
            # Load X block
            x_ptrs = X_ptr + offs_b[:, None] * stride_xb + offs_k[None, :] * stride_xi
            x_block = tl.load(x_ptrs, mask=mask_b[:, None] & mask_k[None, :], other=0.0)
            
            # Generate B values for this k range
            # B[true_member, k, r] - need (BLOCK_B, BLOCK_K) values
            c0_B = offs_k[None, :].to(tl.uint32)  # (1, BLOCK_K)
            c1_B = tl.full((1,), r, dtype=tl.uint32)  # rank index
            c2_B = true_members[:, None].to(tl.uint32)  # (BLOCK_B, 1)
            c3_B = ((epoch << 20) | (param_id << 4) | 1).to(tl.uint32)  # B marker
            
            # Broadcast for Philox - we compute one (b, k) pair at a time
            # For efficiency, we'll vectorize over k
            B_vals = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
            
            for b_idx in tl.static_range(BLOCK_B):
                c2_scalar = true_members[b_idx].to(tl.uint32)
                r0, r1, _, _ = philox_4x32(
                    offs_k.to(tl.uint32),
                    tl.full((BLOCK_K,), r, dtype=tl.uint32),
                    tl.full((BLOCK_K,), c2_scalar, dtype=tl.uint32),
                    tl.full((BLOCK_K,), c3_B, dtype=tl.uint32),
                    key0, key1
                )
                u0 = uint32_to_float_01(r0)
                u1 = uint32_to_float_01(r1)
                z0, _ = box_muller(u0, u1)
                B_vals = tl.where(
                    tl.arange(0, BLOCK_B)[:, None] == b_idx,
                    z0[None, :],
                    B_vals
                )
            
            # Accumulate x @ B for this k block
            xB += tl.sum(x_block * B_vals * mask_k[None, :], axis=1)
        
        # Now compute A values and accumulate: acc += xB[:, None] * A
        # Generate A[true_member, o, r] for each output position
        c3_A = ((epoch << 20) | (param_id << 4) | 0).to(tl.uint32)  # A marker
        
        for b_idx in tl.static_range(BLOCK_B):
            c2_scalar = true_members[b_idx].to(tl.uint32)
            r0, r1, _, _ = philox_4x32(
                offs_o.to(tl.uint32),
                tl.full((BLOCK_O,), r, dtype=tl.uint32),
                tl.full((BLOCK_O,), c2_scalar, dtype=tl.uint32),
                tl.full((BLOCK_O,), c3_A, dtype=tl.uint32),
                key0, key1
            )
            u0 = uint32_to_float_01(r0)
            u1 = uint32_to_float_01(r1)
            a_vals, _ = box_muller(u0, u1)  # (BLOCK_O,)
            
            # Scale and accumulate
            a_scaled = a_vals * scaled_sigma * sigma_signs[b_idx]
            pert_contrib = xB[b_idx] * a_scaled
            
            acc = tl.where(
                tl.arange(0, BLOCK_B)[:, None] == b_idx,
                acc + pert_contrib[None, :],
                acc
            )
    
    # =========================================================================
    # Part 3: Add bias
    # =========================================================================
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_o, mask=mask_o, other=0.0)
        acc += bias[None, :]
    
    # =========================================================================
    # Store output
    # =========================================================================
    out_ptrs = Out_ptr + offs_b[:, None] * stride_ob + offs_o[None, :] * stride_oo
    tl.store(out_ptrs, acc, mask=mask_b[:, None] & mask_o[None, :])


def fused_perturbed_linear(
    x: torch.Tensor,           # (batch, in_features)
    weight: torch.Tensor,       # (out_features, in_features)
    bias: Optional[torch.Tensor],
    member_ids: torch.Tensor,   # (batch,) int32
    seed: int,
    epoch: int,
    param_id: int,
    rank: int,
    sigma: float,
    antithetic: bool = True,
) -> torch.Tensor:
    """
    Fused perturbed linear with on-the-fly noise generation.
    
    Computes: out[i] = x[i] @ W.T + x[i] @ B[member_ids[i]] @ A[member_ids[i]].T + bias
    
    The A and B factors are generated on-the-fly using Philox PRNG.
    This matches the JAX reference implementation semantics.
    """
    batch_size, in_features = x.shape
    out_features = weight.shape[0]
    
    # Ensure contiguous for Triton
    x = x.contiguous()
    weight = weight.contiguous()
    member_ids = member_ids.contiguous().to(torch.int32)
    
    output = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype)
    
    # Extract seed components
    key0 = seed & 0xFFFFFFFF
    key1 = (seed >> 32) & 0xFFFFFFFF
    
    # Block sizes - tuned for typical GPU architectures
    BLOCK_B = min(32, triton.next_power_of_2(batch_size))
    BLOCK_O = min(64, triton.next_power_of_2(out_features))
    BLOCK_K = min(64, triton.next_power_of_2(in_features))
    
    # Ensure minimum block sizes for Triton
    BLOCK_B = max(BLOCK_B, 1)
    BLOCK_O = max(BLOCK_O, 1)
    BLOCK_K = max(BLOCK_K, 1)
    
    grid = (
        triton.cdiv(batch_size, BLOCK_B),
        triton.cdiv(out_features, BLOCK_O),
    )
    
    fused_perturbed_linear_kernel[grid](
        x, weight, bias, member_ids, output,
        key0, key1, epoch, param_id, sigma,
        batch_size, in_features, out_features,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        rank=rank,
        HAS_BIAS=bias is not None,
        ANTITHETIC=antithetic,
        BLOCK_B=BLOCK_B,
        BLOCK_O=BLOCK_O,
        BLOCK_K=BLOCK_K,
    )
    
    return output


# =============================================================================
# Batched Perturbed Linear (uses pre-generated factors)
# =============================================================================

def batched_perturbed_linear_torch(
    x: torch.Tensor,            # (batch, in_features)
    weight: torch.Tensor,       # (out_features, in_features)
    bias: Optional[torch.Tensor],
    A: torch.Tensor,            # (population_size, out_features, rank)
    B: torch.Tensor,            # (population_size, in_features, rank)
    member_ids: torch.Tensor,   # (batch,) - indices into population
) -> torch.Tensor:
    """
    Batched linear with pre-generated low-rank factors.
    
    Computes: out[i] = x[i] @ W.T + x[i] @ B[member_ids[i]] @ A[member_ids[i]].T + bias
    
    Uses efficient gather + batched matmul to avoid sequential iteration.
    """
    # Gather the A and B factors for each sample's member_id
    A_gather = A[member_ids]  # (batch, out_features, rank)
    B_gather = B[member_ids]  # (batch, in_features, rank)
    
    # Compute base: x @ W.T -> (batch, out_features)
    base = torch.mm(x, weight.t())
    
    # Compute perturbation: x @ B @ A.T
    # Step 1: x @ B -> (batch, rank)
    xB = torch.einsum('bi,bir->br', x, B_gather)
    
    # Step 2: xB @ A.T -> (batch, out_features)
    perturbation = torch.einsum('br,bor->bo', xB, A_gather)
    
    # Combine
    out = base + perturbation
    
    if bias is not None:
        out = out + bias
    
    return out


# =============================================================================
# ES Gradient Computation
# =============================================================================

def compute_es_gradient_torch(
    fitnesses: torch.Tensor,     # (population_size,)
    A: torch.Tensor,             # (population_size, out_features, rank)
    B: torch.Tensor,             # (population_size, in_features, rank)
    sigma: float,
) -> torch.Tensor:
    """
    Compute ES gradient estimate from fitnesses and perturbation factors.
    
    The ES gradient is: E[fitness * perturbation]
    
    For low-rank perturbations ΔW = A @ B.T, we compute:
        grad = sum_i(fitness[i] * A[i] @ B[i].T) / population_size
        
    We use einsum to compute this WITHOUT materializing the full perturbation matrices.
    """
    population_size = fitnesses.shape[0]
    
    # Shape broadcasting: fitnesses[:, None, None] for (pop, 1, 1)
    weighted_A = fitnesses[:, None, None] * A  # (pop, out, rank)
    
    # Compute gradient: einsum over population and rank dimensions
    # grad[o, i] = sum_p sum_r (weighted_A[p, o, r] * B[p, i, r]) / pop_size
    grad = torch.einsum('por,pir->oi', weighted_A, B) / population_size
    
    return grad
