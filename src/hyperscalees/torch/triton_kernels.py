"""
Triton kernels for fused EGGROLL operations.

The core insight: generate low-rank perturbation factors ON-THE-FLY inside the kernel
using the EXACT same RNG as PyTorch (_fold_in + splitmix32 + Box-Muller).

This ensures that:
1. batched_forward (Triton) produces identical perturbations as
2. _sample_perturbation (PyTorch) used during gradient computation

Formula: out = x @ W.T + bias + (x @ B) @ A.T
where A, B are generated deterministically from (base_key, member_id, layer_idx).
"""

import torch
import triton
import triton.language as tl
import math


# =============================================================================
# Fused Kernel: RNG + Matmul + Perturbation all in one
# =============================================================================

@triton.jit
def fused_eggroll_kernel(
    # Input pointers
    X_ptr,           # [batch, in_features]
    W_ptr,           # [out_features, in_features]
    bias_ptr,        # [out_features] or None
    # RNG inputs
    layer_keys_ptr,  # [batch] - precomputed fold_in(fold_in(base, member), layer)
    signs_ptr,       # [batch] - precomputed antithetic signs
    # Output pointer
    out_ptr,         # [batch, out_features]
    # Dimensions
    batch_size,
    in_features,
    out_features,
    # Parameters
    scale,           # sigma / sqrt(rank), precomputed
    RANK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    # Strides
    stride_X_batch, stride_X_in,
    stride_W_out, stride_W_in,
    stride_out_batch, stride_out_out,
    # Block sizes
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_IN: tl.constexpr,
):
    """
    Fully fused EGGROLL forward: generates A, B on-the-fly and computes output.
    
    RNG uses splitmix32 to match PyTorch _random_normal_batched exactly.
    """
    pid_batch = tl.program_id(0)
    pid_out = tl.program_id(1)
    
    offs_batch = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    offs_out = pid_out * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    
    mask_batch = offs_batch < batch_size
    mask_out = offs_out < out_features
    
    # Load layer keys and signs
    layer_keys = tl.load(layer_keys_ptr + offs_batch, mask=mask_batch, other=0)
    signs = tl.load(signs_ptr + offs_batch, mask=mask_batch, other=1.0)
    
    # =========================================================================
    # Phase 1: Compute base matmul x @ W.T + bias
    # =========================================================================
    acc = tl.zeros((BLOCK_BATCH, BLOCK_OUT), dtype=tl.float32)
    
    for in_block in range(0, in_features, BLOCK_IN):
        offs_in = in_block + tl.arange(0, BLOCK_IN)
        mask_in = offs_in < in_features
        
        X_ptrs = X_ptr + offs_batch[:, None] * stride_X_batch + offs_in[None, :] * stride_X_in
        x_vals = tl.load(X_ptrs, mask=mask_batch[:, None] & mask_in[None, :], other=0.0)
        
        W_ptrs = W_ptr + offs_out[:, None] * stride_W_out + offs_in[None, :] * stride_W_in
        w_vals = tl.load(W_ptrs, mask=mask_out[:, None] & mask_in[None, :], other=0.0)
        
        acc += tl.dot(x_vals, tl.trans(w_vals))
    
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_out, mask=mask_out, other=0.0)
        acc += bias_vals[None, :]
    
    # =========================================================================
    # Phase 2: Generate factors and compute perturbation on-the-fly
    # =========================================================================
    # PyTorch layout: factors = normal(key, (m+n, r))
    # A = factors[:m, :], B = factors[m:, :]
    # Counter for factors[i, j] = i * r + j (row-major flatten)
    # But we need to match Box-Muller pairs...
    #
    # Actually PyTorch does: counters = arange(numel_even), seed = key + counter
    # So for shape (m+n, r), counter for element [i,j] in flattened = i * r + j
    
    # Probit constants
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    
    for r in range(RANK):
        # Compute proj[b] = sum_i(x[b,i] * B[b,i,r])
        proj = tl.zeros((BLOCK_BATCH,), dtype=tl.float32)
        
        for in_block in range(0, in_features, BLOCK_IN):
            offs_in = in_block + tl.arange(0, BLOCK_IN)
            mask_in = offs_in < in_features
            
            X_ptrs = X_ptr + offs_batch[:, None] * stride_X_batch + offs_in[None, :] * stride_X_in
            x_vals = tl.load(X_ptrs, mask=mask_batch[:, None] & mask_in[None, :], other=0.0)
            
            # B[i, r] -> counter = (out_features + i) * RANK + r
            B_counters = (out_features + offs_in[None, :]) * RANK + r
            B_seeds = (layer_keys[:, None] + B_counters) & 0xFFFFFFFF
            
            # Splitmix32 (matching PyTorch)
            B_seeds = ((B_seeds ^ (B_seeds >> 17)) * 0x6D5AD4BB) & 0xFFFFFFFF
            B_seeds = ((B_seeds ^ (B_seeds >> 11)) * 0x4C4C1B51) & 0xFFFFFFFF
            B_seeds = ((B_seeds ^ (B_seeds >> 15)) * 0x31848BAB) & 0xFFFFFFFF
            B_seeds = (B_seeds ^ (B_seeds >> 14)) & 0xFFFFFFFF
            
            B_uniform = B_seeds.to(tl.float32) / 4294967296.0
            B_uniform = tl.maximum(B_uniform, 1e-7)
            B_uniform = tl.minimum(B_uniform, 1.0 - 1e-7)
            
            # Probit approximation
            t = tl.sqrt(-2.0 * tl.log(tl.where(B_uniform < 0.5, B_uniform, 1.0 - B_uniform)))
            B_normal = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
            B_normal = tl.where(B_uniform < 0.5, -B_normal, B_normal)
            
            proj += tl.sum(x_vals * B_normal, axis=1)
        
        # Generate A[o, r] -> counter = o * RANK + r
        A_counters = offs_out[None, :] * RANK + r
        A_seeds = (layer_keys[:, None] + A_counters) & 0xFFFFFFFF
        
        A_seeds = ((A_seeds ^ (A_seeds >> 17)) * 0x6D5AD4BB) & 0xFFFFFFFF
        A_seeds = ((A_seeds ^ (A_seeds >> 11)) * 0x4C4C1B51) & 0xFFFFFFFF
        A_seeds = ((A_seeds ^ (A_seeds >> 15)) * 0x31848BAB) & 0xFFFFFFFF
        A_seeds = (A_seeds ^ (A_seeds >> 14)) & 0xFFFFFFFF
        
        A_uniform = A_seeds.to(tl.float32) / 4294967296.0
        A_uniform = tl.maximum(A_uniform, 1e-7)
        A_uniform = tl.minimum(A_uniform, 1.0 - 1e-7)
        
        t = tl.sqrt(-2.0 * tl.log(tl.where(A_uniform < 0.5, A_uniform, 1.0 - A_uniform)))
        A_normal = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
        A_normal = tl.where(A_uniform < 0.5, -A_normal, A_normal)
        
        A_scaled = A_normal * scale * signs[:, None]
        acc += proj[:, None] * A_scaled
    
    # =========================================================================
    # Phase 3: Store output
    # =========================================================================
    out_ptrs = out_ptr + offs_batch[:, None] * stride_out_batch + offs_out[None, :] * stride_out_out
    tl.store(out_ptrs, acc, mask=mask_batch[:, None] & mask_out[None, :])


# =============================================================================
# Python Wrapper
# =============================================================================

def _fold_in_torch(key: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """JAX-style fold_in - matches strategy.py _fold_in exactly."""
    mixed = (key.to(torch.int64) + data.to(torch.int64) * 2654435761) & 0xFFFFFFFF
    mixed = (mixed ^ (mixed >> 15)) & 0xFFFFFFFF
    mixed = (mixed * 2246822519) & 0xFFFFFFFF
    mixed = (mixed ^ (mixed >> 13)) & 0xFFFFFFFF
    mixed = (mixed * 3266489917) & 0xFFFFFFFF
    mixed = (mixed ^ (mixed >> 16)) & 0xFFFFFFFF
    return mixed


def fused_perturbed_linear(
    x: torch.Tensor,           # [batch, in_features]
    weight: torch.Tensor,      # [out_features, in_features]
    bias: torch.Tensor | None, # [out_features] or None
    base_key: int,             # RNG base key (already combined with epoch)
    member_ids: torch.Tensor,  # [batch] member indices
    layer_idx: int,            # Layer index for RNG (param_key)
    sigma: float,              # Perturbation scale
    rank: int,                 # Low-rank dimension
    antithetic: bool = True,   # Whether to use antithetic sampling
) -> torch.Tensor:
    """
    Fused perturbed linear: out = x @ W.T + bias + σ(x @ B) @ A.T
    
    A, B are generated on-the-fly inside the Triton kernel using the same
    RNG as PyTorch's _generate_lowrank_factors_batched.
    """
    batch_size, in_features = x.shape
    out_features = weight.shape[0]
    device = x.device
    dtype = x.dtype
    
    # Compute antithetic effective member IDs and signs
    if antithetic:
        effective_members = member_ids // 2
        signs = torch.where(member_ids % 2 == 0, 1.0, -1.0).to(dtype)
    else:
        effective_members = member_ids
        signs = torch.ones(batch_size, dtype=dtype, device=device)
    
    # Compute layer keys using fold_in pattern (matching PyTorch exactly)
    base_key_t = torch.tensor(base_key, dtype=torch.int64, device=device)
    layer_idx_t = torch.tensor(layer_idx, dtype=torch.int64, device=device)
    
    member_keys = _fold_in_torch(base_key_t, effective_members.to(torch.int64))
    layer_keys = _fold_in_torch(member_keys, layer_idx_t)
    
    scale = sigma / math.sqrt(rank)
    out = torch.empty((batch_size, out_features), dtype=dtype, device=device)
    
    BLOCK_BATCH = 32
    BLOCK_OUT = 64
    BLOCK_IN = 64
    
    grid = (
        triton.cdiv(batch_size, BLOCK_BATCH),
        triton.cdiv(out_features, BLOCK_OUT),
    )
    
    fused_eggroll_kernel[grid](
        x, weight, bias if bias is not None else x,
        layer_keys, signs,
        out,
        batch_size, in_features, out_features,
        scale, rank, bias is not None,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OUT=BLOCK_OUT,
        BLOCK_IN=BLOCK_IN,
    )
    
    return out
