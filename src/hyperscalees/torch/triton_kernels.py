"""
Triton kernels for fused EGGROLL operations.

Strategy: Simple two-phase approach using Triton's built-in RNG
1. Generate low-rank factors A, B using tl.rand with seeded offsets
2. Fused matmul: out = x @ W.T + bias + (x @ B) @ A.T
"""

import torch
import triton
import triton.language as tl
import math


# =============================================================================
# Factor Generation Kernel - Using Triton's built-in RNG
# =============================================================================

@triton.jit
def generate_A_kernel(
    # Output pointer
    A_ptr,           # [batch, out_features, rank]
    # RNG seed (different per member)
    seeds_ptr,       # [batch] - precomputed seeds
    # Parameters
    scale,           # float: sigma / sqrt(rank) - precomputed
    signs_ptr,       # [batch] - precomputed signs for antithetic
    # Dimensions
    batch_size,
    out_features,
    RANK: tl.constexpr,
    # Strides
    stride_batch, stride_out, stride_rank,
    # Block sizes
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    """Generate A factors [batch, out_features, rank]."""
    pid_batch = tl.program_id(0)
    pid_out = tl.program_id(1)
    
    offs_batch = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    offs_out = pid_out * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    
    mask_batch = offs_batch < batch_size
    mask_out = offs_out < out_features
    
    # Load seeds and signs for this batch block
    seeds = tl.load(seeds_ptr + offs_batch, mask=mask_batch, other=0)
    signs = tl.load(signs_ptr + offs_batch, mask=mask_batch, other=1.0)
    
    # Generate for each rank
    for r in range(RANK):
        # Compute unique offset for each (batch, out, r) combination
        # offset = seed * large_prime + out_idx * rank + r
        offsets = seeds[:, None] * 1000003 + offs_out[None, :] * RANK + r
        
        # Generate uniform random in [0, 1)
        uniform = tl.rand(0, offsets.to(tl.uint32))
        
        # Box-Muller transform to get normal (approximate with simple method)
        # Use inverse CDF approximation
        uniform = tl.maximum(uniform, 1e-7)
        uniform = tl.minimum(uniform, 1.0 - 1e-7)
        
        # Approximate inverse normal CDF
        t = tl.sqrt(-2.0 * tl.log(uniform))
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308
        normal = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
        
        # Apply sign from second half of uniform distribution
        normal = tl.where(uniform > 0.5, normal, -normal)
        
        # Apply scale and antithetic sign
        vals = normal * scale * signs[:, None]
        
        # Store to A[batch, out, r]
        A_ptrs = (A_ptr + 
                 offs_batch[:, None] * stride_batch + 
                 offs_out[None, :] * stride_out + 
                 r * stride_rank)
        tl.store(A_ptrs, vals, mask=mask_batch[:, None] & mask_out[None, :])


@triton.jit
def generate_B_kernel(
    # Output pointer
    B_ptr,           # [batch, in_features, rank]
    # RNG seed (different per member)
    seeds_ptr,       # [batch] - precomputed seeds
    # Dimensions
    batch_size,
    in_features,
    out_features,    # needed for offset calculation
    RANK: tl.constexpr,
    # Strides
    stride_batch, stride_in, stride_rank,
    # Block sizes
    BLOCK_BATCH: tl.constexpr,
    BLOCK_IN: tl.constexpr,
):
    """Generate B factors [batch, in_features, rank]."""
    pid_batch = tl.program_id(0)
    pid_in = tl.program_id(1)
    
    offs_batch = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    offs_in = pid_in * BLOCK_IN + tl.arange(0, BLOCK_IN)
    
    mask_batch = offs_batch < batch_size
    mask_in = offs_in < in_features
    
    # Load seeds for this batch block
    seeds = tl.load(seeds_ptr + offs_batch, mask=mask_batch, other=0)
    
    # Offset to make B different from A
    B_offset = out_features * RANK
    
    # Generate for each rank
    for r in range(RANK):
        # Compute unique offset for each (batch, in, r) combination
        offsets = seeds[:, None] * 1000003 + B_offset + offs_in[None, :] * RANK + r
        
        # Generate uniform random in [0, 1)
        uniform = tl.rand(0, offsets.to(tl.uint32))
        
        # Box-Muller transform to get normal (approximate with simple method)
        uniform = tl.maximum(uniform, 1e-7)
        uniform = tl.minimum(uniform, 1.0 - 1e-7)
        
        # Approximate inverse normal CDF
        t = tl.sqrt(-2.0 * tl.log(uniform))
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308
        normal = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
        
        # Apply sign from second half of uniform distribution
        normal = tl.where(uniform > 0.5, normal, -normal)
        
        # Store to B[batch, in, r] (no scaling for B, scale only on A)
        B_ptrs = (B_ptr + 
                 offs_batch[:, None] * stride_batch + 
                 offs_in[None, :] * stride_in + 
                 r * stride_rank)
        tl.store(B_ptrs, normal, mask=mask_batch[:, None] & mask_in[None, :])


# =============================================================================
# Fused Matmul Kernel
# =============================================================================

@triton.jit
def fused_perturbed_matmul_kernel(
    # Input pointers
    X_ptr,           # [batch, in_features]
    W_ptr,           # [out_features, in_features]
    bias_ptr,        # [out_features] or None
    A_ptr,           # [batch, out_features, rank]
    B_ptr,           # [batch, in_features, rank]
    # Output pointer
    out_ptr,         # [batch, out_features]
    # Dimensions
    batch_size,
    in_features,
    out_features,
    RANK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    # Strides for X [batch, in]
    stride_X_batch, stride_X_in,
    # Strides for W [out, in]
    stride_W_out, stride_W_in,
    # Strides for A [batch, out, rank]
    stride_A_batch, stride_A_out, stride_A_rank,
    # Strides for B [batch, in, rank]
    stride_B_batch, stride_B_in, stride_B_rank,
    # Strides for out [batch, out]
    stride_out_batch, stride_out_out,
    # Block sizes
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_IN: tl.constexpr,
):
    """
    Compute: out = x @ W.T + bias + (x @ B) @ A.T
    
    For each output element out[b, o]:
        out[b, o] = sum_i(x[b,i] * W[o,i]) + bias[o] + sum_r(sum_i(x[b,i]*B[b,i,r]) * A[b,o,r])
    """
    pid_batch = tl.program_id(0)
    pid_out = tl.program_id(1)
    
    offs_batch = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    offs_out = pid_out * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    
    mask_batch = offs_batch < batch_size
    mask_out = offs_out < out_features
    
    # Accumulator for output
    acc = tl.zeros((BLOCK_BATCH, BLOCK_OUT), dtype=tl.float32)
    
    # First compute x @ W.T by iterating over in_features
    for in_block in range(0, in_features, BLOCK_IN):
        offs_in = in_block + tl.arange(0, BLOCK_IN)
        mask_in = offs_in < in_features
        
        # Load X[batch, in]
        X_ptrs = X_ptr + offs_batch[:, None] * stride_X_batch + offs_in[None, :] * stride_X_in
        x_vals = tl.load(X_ptrs, mask=mask_batch[:, None] & mask_in[None, :], other=0.0)
        
        # Load W[out, in]
        W_ptrs = W_ptr + offs_out[:, None] * stride_W_out + offs_in[None, :] * stride_W_in
        w_vals = tl.load(W_ptrs, mask=mask_out[:, None] & mask_in[None, :], other=0.0)
        
        # Accumulate: out[b,o] += sum_i(x[b,i] * W[o,i])
        # x_vals: [BLOCK_BATCH, BLOCK_IN]
        # w_vals: [BLOCK_OUT, BLOCK_IN]
        # We need [BLOCK_BATCH, BLOCK_OUT]
        acc += tl.dot(x_vals, tl.trans(w_vals))
    
    # Add bias if present
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_out, mask=mask_out, other=0.0)
        acc += bias_vals[None, :]
    
    # Now compute low-rank perturbation: (x @ B) @ A.T
    # For each rank r:
    #   proj[b, r] = sum_i(x[b,i] * B[b,i,r])
    #   perturb[b, o] += proj[b, r] * A[b, o, r]
    
    for r in range(RANK):
        # Compute proj = x @ B[:,:,r]
        proj = tl.zeros((BLOCK_BATCH,), dtype=tl.float32)
        for in_block in range(0, in_features, BLOCK_IN):
            offs_in = in_block + tl.arange(0, BLOCK_IN)
            mask_in = offs_in < in_features
            
            # Load X[batch, in]
            X_ptrs = X_ptr + offs_batch[:, None] * stride_X_batch + offs_in[None, :] * stride_X_in
            x_vals = tl.load(X_ptrs, mask=mask_batch[:, None] & mask_in[None, :], other=0.0)
            
            # Load B[batch, in, r]
            B_ptrs = (B_ptr + 
                     offs_batch[:, None] * stride_B_batch + 
                     offs_in[None, :] * stride_B_in + 
                     r * stride_B_rank)
            b_vals = tl.load(B_ptrs, mask=mask_batch[:, None] & mask_in[None, :], other=0.0)
            
            # proj[b] += sum_i(x[b,i] * B[b,i,r])
            proj += tl.sum(x_vals * b_vals, axis=1)
        
        # Load A[batch, out, r]
        A_ptrs = (A_ptr + 
                 offs_batch[:, None] * stride_A_batch + 
                 offs_out[None, :] * stride_A_out + 
                 r * stride_A_rank)
        a_vals = tl.load(A_ptrs, mask=mask_batch[:, None] & mask_out[None, :], other=0.0)
        
        # acc[b, o] += proj[b] * A[b, o, r]
        acc += proj[:, None] * a_vals
    
    # Store output
    out_ptrs = out_ptr + offs_batch[:, None] * stride_out_batch + offs_out[None, :] * stride_out_out
    tl.store(out_ptrs, acc, mask=mask_batch[:, None] & mask_out[None, :])


# =============================================================================
# Python Wrapper
# =============================================================================

def fused_perturbed_linear(
    x: torch.Tensor,           # [batch, in_features]
    weight: torch.Tensor,      # [out_features, in_features]
    bias: torch.Tensor | None, # [out_features] or None
    base_key: int,             # RNG base key
    member_ids: torch.Tensor,  # [batch] member indices
    layer_idx: int,            # Layer index for RNG
    sigma: float,              # Perturbation scale
    rank: int,                 # Low-rank dimension
    antithetic: bool = True,   # Whether to use antithetic sampling
) -> torch.Tensor:
    """
    Fused perturbed linear layer using Triton kernels.
    
    Computes: out = x @ W.T + bias + sigma * (x @ B) @ A.T
    where A, B are random low-rank factors.
    """
    batch_size, in_features = x.shape
    out_features = weight.shape[0]
    device = x.device
    dtype = x.dtype
    
    # Compute seeds for each member (deterministic based on base_key, layer_idx, member_id)
    if antithetic:
        effective_members = member_ids // 2
        signs = torch.where(member_ids % 2 == 0, 1.0, -1.0).to(dtype)
    else:
        effective_members = member_ids
        signs = torch.ones(batch_size, dtype=dtype, device=device)
    
    # Simple seed computation: combine base_key, layer_idx, and member_id
    seeds = (base_key + layer_idx * 1000003 + effective_members * 999983).to(torch.int64)
    
    # Precompute scale = sigma / sqrt(rank)
    scale = sigma / math.sqrt(rank)
    
    # Allocate factor matrices
    A = torch.empty((batch_size, out_features, rank), dtype=dtype, device=device)
    B = torch.empty((batch_size, in_features, rank), dtype=dtype, device=device)
    
    # Block sizes
    BLOCK_BATCH = 32
    BLOCK_OUT = 64
    BLOCK_IN = 64
    
    # Launch A generation kernel
    grid_A = (
        triton.cdiv(batch_size, BLOCK_BATCH),
        triton.cdiv(out_features, BLOCK_OUT),
    )
    generate_A_kernel[grid_A](
        A, seeds, scale, signs,
        batch_size, out_features, rank,
        A.stride(0), A.stride(1), A.stride(2),
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OUT=BLOCK_OUT,
    )
    
    # Launch B generation kernel
    grid_B = (
        triton.cdiv(batch_size, BLOCK_BATCH),
        triton.cdiv(in_features, BLOCK_IN),
    )
    generate_B_kernel[grid_B](
        B, seeds,
        batch_size, in_features, out_features, rank,
        B.stride(0), B.stride(1), B.stride(2),
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_IN=BLOCK_IN,
    )
    
    # Allocate output
    out = torch.empty((batch_size, out_features), dtype=dtype, device=device)
    
    # Launch fused matmul kernel
    grid_matmul = (
        triton.cdiv(batch_size, BLOCK_BATCH),
        triton.cdiv(out_features, BLOCK_OUT),
    )
    fused_perturbed_matmul_kernel[grid_matmul](
        x, weight, bias if bias is not None else x,  # dummy for None
        A, B, out,
        batch_size, in_features, out_features, rank,
        bias is not None,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OUT=BLOCK_OUT,
        BLOCK_IN=BLOCK_IN,
    )
    
    return out


# =============================================================================
# Reference Implementation
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
    """Pure PyTorch reference implementation for testing."""
    batch_size, in_features = x.shape
    out_features = weight.shape[0]
    device = x.device
    dtype = x.dtype
    
    # Base matmul
    out = torch.nn.functional.linear(x, weight, bias)
    
    # Generate random factors using PyTorch's generator for each member
    for b in range(batch_size):
        member_id = member_ids[b].item()
        if antithetic:
            effective_member = member_id // 2
            sign = 1.0 if member_id % 2 == 0 else -1.0
        else:
            effective_member = member_id
            sign = 1.0
        
        # Create deterministic generator
        gen = torch.Generator(device=device)
        seed = base_key + layer_idx * 1000003 + effective_member * 999983
        gen.manual_seed(seed)
        
        # Generate A and B
        A = torch.randn(out_features, rank, generator=gen, device=device, dtype=dtype)
        B = torch.randn(in_features, rank, generator=gen, device=device, dtype=dtype)
        
        # Apply perturbation: sigma/sqrt(rank) * sign * (x @ B) @ A.T
        scale = sigma / math.sqrt(rank) * sign
        proj = x[b:b+1] @ B  # [1, rank]
        perturb = proj @ A.T  # [1, out_features]
        out[b:b+1] += scale * perturb
    
    return out
