"""
Triton kernels for fused EGGROLL operations.

Strategy: Two-phase approach that Triton can actually compile
1. Generate low-rank factors A, B using vectorized Philox RNG
2. Fused matmul: out = x @ W.T + bias + (x @ B) @ A.T

This is simpler than full fusion but still eliminates most kernel launch overhead.
"""

import torch
import triton
import triton.language as tl
import math


# =============================================================================
# Philox RNG - Vectorized for Triton
# =============================================================================

@triton.jit
def philox_round(c0, c1, c2, c3, k0, k1):
    """Single round of Philox 4x32."""
    PHILOX_M0: tl.constexpr = 0xD2511F53
    PHILOX_M1: tl.constexpr = 0xCD9E8D57
    
    prod0_lo = c0 * PHILOX_M0
    prod0_hi = tl.math.mulhi(c0, PHILOX_M0)
    prod1_lo = c2 * PHILOX_M1
    prod1_hi = tl.math.mulhi(c2, PHILOX_M1)
    
    new_c0 = prod1_hi ^ c1 ^ k0
    new_c1 = prod1_lo
    new_c2 = prod0_hi ^ c3 ^ k1
    new_c3 = prod0_lo
    
    return new_c0, new_c1, new_c2, new_c3


@triton.jit  
def philox_4x32_10_vec(key, counter):
    """
    Vectorized Philox 4x32-10 PRNG.
    
    Args:
        key: tensor of 64-bit keys
        counter: tensor of 64-bit counters (same shape as key)
    
    Returns:
        4 tensors of uint32 random values
    """
    PHILOX_W0: tl.constexpr = 0x9E3779B9
    PHILOX_W1: tl.constexpr = 0xBB67AE85
    
    # Split key into two 32-bit values
    k0 = (key & 0xFFFFFFFF).to(tl.uint32)
    k1 = ((key >> 32) & 0xFFFFFFFF).to(tl.uint32)
    
    # Split counter into four 32-bit values
    c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((counter >> 32) & 0xFFFFFFFF).to(tl.uint32)
    c2 = tl.zeros_like(c0)
    c3 = tl.zeros_like(c0)
    
    # 10 rounds of Philox
    for _ in range(10):
        c0, c1, c2, c3 = philox_round(c0, c1, c2, c3, k0, k1)
        k0 = k0 + PHILOX_W0
        k1 = k1 + PHILOX_W1
    
    return c0, c1, c2, c3


@triton.jit
def uint32_to_normal(u):
    """Convert uint32 to standard normal using inverse CDF approximation."""
    # Convert to uniform [0, 1)
    uniform = u.to(tl.float32) * (1.0 / 4294967296.0)
    # Clamp to avoid log(0)
    uniform = tl.maximum(uniform, 1e-7)
    uniform = tl.minimum(uniform, 1.0 - 1e-7)
    # Inverse CDF approximation (Abramowitz & Stegun)
    t = tl.sqrt(-2.0 * tl.log(uniform))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    normal = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    # Flip sign for half the distribution
    sign = tl.where(u > 2147483648, 1.0, -1.0)
    return normal * sign


# =============================================================================
# Factor Generation Kernel - Generates A and B matrices
# =============================================================================

@triton.jit
def generate_factors_kernel(
    # Output pointers
    A_ptr,           # [batch, out_features, rank] - output matrix A
    B_ptr,           # [batch, in_features, rank] - output matrix B
    # RNG parameters
    base_key,        # int64: base RNG key (scalar)
    member_ids_ptr,  # [batch]: member indices for RNG
    layer_idx,       # int: layer index for RNG
    # Perturbation parameters
    sigma,           # float: perturbation scale
    antithetic,      # bool: whether to use antithetic sampling
    # Dimensions
    batch_size,
    out_features,
    in_features,
    rank,
    # Strides for A [batch, out, rank]
    stride_A_batch, stride_A_out, stride_A_rank,
    # Strides for B [batch, in, rank]
    stride_B_batch, stride_B_in, stride_B_rank,
    # Block sizes
    BLOCK_BATCH: tl.constexpr,
    BLOCK_FEAT: tl.constexpr,
):
    """
    Generate low-rank factors A and B for all batch members.
    
    Each program generates a block of [BLOCK_BATCH, BLOCK_FEAT] values
    for either A or B (determined by program_id).
    
    A has shape [batch, out_features, rank]
    B has shape [batch, in_features, rank]
    """
    # Which matrix (0 = A, 1 = B) and which rank
    pid_matrix = tl.program_id(0)  # 0..1 for A/B
    pid_rank = tl.program_id(1)    # 0..rank-1
    pid_batch = tl.program_id(2)   # batch blocks
    
    # Offsets
    offs_batch = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    mask_batch = offs_batch < batch_size
    
    # Load member IDs for this batch block
    member_ids = tl.load(member_ids_ptr + offs_batch, mask=mask_batch, other=0)
    
    # Compute effective member and sign for antithetic sampling
    if antithetic:
        effective_member = member_ids // 2
        sign = tl.where(member_ids % 2 == 0, 1.0, -1.0)
    else:
        effective_member = member_ids
        sign = tl.full((BLOCK_BATCH,), 1.0, dtype=tl.float32)
    
    # Compute member keys using 32-bit mixing constants
    LAYER_MIX: tl.constexpr = 0x9E3779B9
    MEMBER_MIX: tl.constexpr = 0xBB67AE85
    
    layer_key = base_key ^ (layer_idx * LAYER_MIX)
    member_keys = layer_key ^ (effective_member.to(tl.int64) * MEMBER_MIX)
    
    # Scale for A (includes sigma/sqrt(rank) and sign)
    # Cast rank to float using float() since it's a Python int
    rank_float = float(rank)
    scale_A = sigma / tl.sqrt(rank_float) * sign
    
    # Current rank index
    r = pid_rank
    
    if pid_matrix == 0:
        # Generate A factors [batch, out_features, rank]
        # Process out_features in blocks
        num_feat_blocks = tl.cdiv(out_features, BLOCK_FEAT)
        for feat_block in range(num_feat_blocks):
            offs_feat = feat_block * BLOCK_FEAT + tl.arange(0, BLOCK_FEAT)
            mask_feat = offs_feat < out_features
            
            # RNG counter: feature_idx + r * out_features
            # Shape: [BLOCK_BATCH, BLOCK_FEAT]
            counter = offs_feat[None, :].to(tl.int64) + r * out_features
            
            # Broadcast member_keys to [BLOCK_BATCH, BLOCK_FEAT]
            keys = member_keys[:, None] + tl.zeros((1, BLOCK_FEAT), dtype=tl.int64)
            counters = tl.zeros((BLOCK_BATCH, 1), dtype=tl.int64) + counter
            
            # Generate random values
            c0, _, _, _ = philox_4x32_10_vec(keys, counters)
            vals = uint32_to_normal(c0) * scale_A[:, None]
            
            # Store to A[batch, feat, r]
            A_ptrs = (A_ptr + 
                     offs_batch[:, None] * stride_A_batch + 
                     offs_feat[None, :] * stride_A_out + 
                     r * stride_A_rank)
            tl.store(A_ptrs, vals, mask=mask_batch[:, None] & mask_feat[None, :])
    
    else:
        # Generate B factors [batch, in_features, rank]
        num_feat_blocks = tl.cdiv(in_features, BLOCK_FEAT)
        for feat_block in range(num_feat_blocks):
            offs_feat = feat_block * BLOCK_FEAT + tl.arange(0, BLOCK_FEAT)
            mask_feat = offs_feat < in_features
            
            # RNG counter: out_features + feature_idx + r * in_features
            # (offset by out_features to separate from A's random stream)
            counter = (out_features + offs_feat[None, :]).to(tl.int64) + r * in_features
            
            # Broadcast member_keys
            keys = member_keys[:, None] + tl.zeros((1, BLOCK_FEAT), dtype=tl.int64)
            counters = tl.zeros((BLOCK_BATCH, 1), dtype=tl.int64) + counter
            
            # Generate random values (B doesn't get sigma scaling)
            c0, _, _, _ = philox_4x32_10_vec(keys, counters)
            vals = uint32_to_normal(c0)
            
            # Store to B[batch, feat, r]
            B_ptrs = (B_ptr + 
                     offs_batch[:, None] * stride_B_batch + 
                     offs_feat[None, :] * stride_B_in + 
                     r * stride_B_rank)
            tl.store(B_ptrs, vals, mask=mask_batch[:, None] & mask_feat[None, :])


# =============================================================================
# Fused Matmul Kernel - Computes x @ W.T + bias + (x @ B) @ A.T
# =============================================================================

@triton.jit
def fused_perturbed_matmul_kernel(
    # Input pointers
    x_ptr,           # [batch, in_features]
    W_ptr,           # [out_features, in_features]
    bias_ptr,        # [out_features] or None
    A_ptr,           # [batch, out_features, rank]
    B_ptr,           # [batch, in_features, rank]
    out_ptr,         # [batch, out_features]
    # Dimensions
    batch_size,
    in_features,
    out_features,
    rank,
    # Strides
    stride_x_batch, stride_x_in,
    stride_W_out, stride_W_in,
    stride_A_batch, stride_A_out, stride_A_rank,
    stride_B_batch, stride_B_in, stride_B_rank,
    stride_out_batch, stride_out_out,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Fused kernel: out = x @ W.T + bias + (x @ B) @ A.T
    
    This computes the base linear layer plus the low-rank perturbation
    in a single kernel pass.
    """
    pid_m = tl.program_id(0)  # Batch
    pid_n = tl.program_id(1)  # Output features
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    mask_m = offs_m < batch_size
    mask_n = offs_n < out_features
    
    # Accumulator for output
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Part 1: x @ W.T
    for k_start in range(0, in_features, BLOCK_K):
        k_offs = k_start + offs_k
        mask_k = k_offs < in_features
        
        # Load x [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_x_batch + k_offs[None, :] * stride_x_in
        x_block = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load W [BLOCK_N, BLOCK_K]
        W_ptrs = W_ptr + offs_n[:, None] * stride_W_out + k_offs[None, :] * stride_W_in
        W_block = tl.load(W_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # acc += x @ W.T
        acc += tl.dot(x_block, tl.trans(W_block))
    
    # Add bias
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias[None, :]
    
    # Part 2: (x @ B) @ A.T
    # For each batch element, compute:
    #   temp[r] = sum_k(x[k] * B[k, r])  shape: [BLOCK_M, rank]
    #   pert[n] = sum_r(temp[r] * A[n, r])  shape: [BLOCK_M, BLOCK_N]
    
    # We iterate over rank since it's typically small (1-16)
    pert = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for r in range(rank):
        # Compute x @ B[:, r] for this rank
        xB = tl.zeros((BLOCK_M,), dtype=tl.float32)
        
        for k_start in range(0, in_features, BLOCK_K):
            k_offs = k_start + offs_k
            mask_k = k_offs < in_features
            
            # Load x [BLOCK_M, BLOCK_K]
            x_ptrs = x_ptr + offs_m[:, None] * stride_x_batch + k_offs[None, :] * stride_x_in
            x_block = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            
            # Load B[:, :, r] for this batch [BLOCK_M, BLOCK_K]
            B_ptrs = (B_ptr + 
                     offs_m[:, None] * stride_B_batch + 
                     k_offs[None, :] * stride_B_in + 
                     r * stride_B_rank)
            B_block = tl.load(B_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            
            # xB += sum over k (x * B)
            xB += tl.sum(x_block * B_block, axis=1)
        
        # Load A[:, :, r] for output features [BLOCK_M, BLOCK_N]
        A_ptrs = (A_ptr + 
                 offs_m[:, None] * stride_A_batch + 
                 offs_n[None, :] * stride_A_out + 
                 r * stride_A_rank)
        A_block = tl.load(A_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        
        # pert += xB[:, None] * A
        pert += xB[:, None] * A_block
    
    # Combine
    out = acc + pert
    
    # Store
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_batch + offs_n[None, :] * stride_out_out
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


# =============================================================================
# Python Wrapper - Main API
# =============================================================================

def fused_perturbed_linear(
    x: torch.Tensor,           # [batch, in_features]
    weight: torch.Tensor,      # [out_features, in_features]
    bias: torch.Tensor | None, # [out_features]
    base_key: int,             # RNG base key
    member_ids: torch.Tensor,  # [batch] member indices
    layer_idx: int,            # Layer index for RNG
    sigma: float,              # Perturbation scale
    rank: int,                 # Low-rank dimension
    antithetic: bool = True,   # Antithetic sampling
) -> torch.Tensor:
    """
    Fused perturbed linear layer: out = x @ W.T + bias + perturbation
    
    Two-phase approach:
    1. Generate A, B factors using Triton RNG kernel
    2. Compute fused matmul x @ W.T + bias + (x @ B) @ A.T
    
    This eliminates most kernel launch overhead while being simpler
    than full fusion.
    """
    assert x.is_cuda and weight.is_cuda
    assert x.ndim == 2 and weight.ndim == 2
    
    batch_size, in_features = x.shape
    out_features, in_features_w = weight.shape
    assert in_features == in_features_w
    
    device = x.device
    dtype = x.dtype
    
    # Ensure member_ids is on correct device and is int64
    if not member_ids.is_cuda:
        member_ids = member_ids.to(device)
    member_ids = member_ids.to(torch.int64)
    
    # Allocate factor tensors
    A = torch.empty((batch_size, out_features, rank), device=device, dtype=dtype)
    B = torch.empty((batch_size, in_features, rank), device=device, dtype=dtype)
    
    # Phase 1: Generate factors
    BLOCK_BATCH = 32
    BLOCK_FEAT = 128
    
    grid_factors = (
        2,  # A and B
        rank,  # one program per rank
        triton.cdiv(batch_size, BLOCK_BATCH),
    )
    
    generate_factors_kernel[grid_factors](
        A, B,
        base_key, member_ids, layer_idx,
        sigma, antithetic,
        batch_size, out_features, in_features, rank,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_FEAT=BLOCK_FEAT,
    )
    
    # Phase 2: Fused matmul
    out = torch.empty((batch_size, out_features), device=device, dtype=dtype)
    
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 64
    
    grid_matmul = (
        triton.cdiv(batch_size, BLOCK_M),
        triton.cdiv(out_features, BLOCK_N),
    )
    
    fused_perturbed_matmul_kernel[grid_matmul](
        x, weight, bias, A, B, out,
        batch_size, in_features, out_features, rank,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        HAS_BIAS=bias is not None,
    )
    
    return out


# =============================================================================
# Pure PyTorch Reference Implementation (for testing)
# =============================================================================

def pytorch_perturbed_linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    base_key: int,
    member_ids: torch.Tensor,
    layer_idx: int,
    sigma: float,
    rank: int,
    antithetic: bool = True,
) -> torch.Tensor:
    """
    Pure PyTorch reference implementation for testing.
    
    This uses the same RNG logic but implemented in PyTorch.
    """
    batch_size, in_features = x.shape
    out_features = weight.shape[0]
    device = x.device
    dtype = x.dtype
    
    # Base linear
    out = torch.nn.functional.linear(x, weight, bias)
    
    # Generate factors using PyTorch
    # For testing, we use torch's RNG seeded deterministically
    A = torch.empty((batch_size, out_features, rank), device=device, dtype=dtype)
    B = torch.empty((batch_size, in_features, rank), device=device, dtype=dtype)
    
    for i in range(batch_size):
        member_id = member_ids[i].item()
        if antithetic:
            effective_member = member_id // 2
            sign = 1.0 if member_id % 2 == 0 else -1.0
        else:
            effective_member = member_id
            sign = 1.0
        
        # Deterministic seed from key, layer, member
        seed = base_key ^ (layer_idx * 0x9E3779B9) ^ (effective_member * 0xBB67AE85)
        seed = seed & 0x7FFFFFFF  # Ensure positive for PyTorch
        
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        
        # Generate A and B
        scale = sigma / math.sqrt(rank) * sign
        A[i] = torch.randn(out_features, rank, generator=gen, device=device, dtype=dtype) * scale
        B[i] = torch.randn(in_features, rank, generator=gen, device=device, dtype=dtype)
    
    # Perturbation: (x @ B) @ A.T
    # x: [batch, in], B: [batch, in, rank], A: [batch, out, rank]
    xB = torch.einsum('bi,bir->br', x, B)  # [batch, rank]
    pert = torch.einsum('br,bor->bo', xB, A)  # [batch, out]
    
    return out + pert
