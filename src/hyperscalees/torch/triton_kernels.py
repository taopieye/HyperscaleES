"""
Triton kernels for fused EGGROLL operations.

This is THE implementation - no PyTorch RNG fallback.
All random number generation happens on-the-fly in Triton using tl.rand().

Core computation: out = x @ W.T + bias + sign * (x @ B) @ A.T
where A, B are generated deterministically from (seed, member_id, layer_idx).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_forward_kernel(
    # Inputs
    X_ptr, W_ptr, bias_ptr, out_ptr,
    # RNG seeds and signs (one per batch element)
    seeds_ptr, signs_ptr,
    # Dimensions
    batch_size, in_features, out_features,
    # ES params
    sigma_scale,  # sigma / sqrt(rank), prescaled
    RANK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    # Strides
    stride_xb, stride_xi,
    stride_wo, stride_wi,
    stride_ob, stride_oo,
    # Block sizes
    BLOCK_B: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """
    Fused EGGROLL forward pass.
    
    For each batch element b with seed s and sign sgn:
      out[b] = x[b] @ W.T + bias + sgn * sigma_scale * (x[b] @ B[b]) @ A[b].T
    
    A[b], B[b] are generated on-the-fly using tl.rand with seed s.
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    
    mask_b = offs_b < batch_size
    mask_o = offs_o < out_features
    
    # Load per-element seeds and signs
    seeds = tl.load(seeds_ptr + offs_b, mask=mask_b, other=0)
    signs = tl.load(signs_ptr + offs_b, mask=mask_b, other=1.0)
    
    # =========================================================================
    # Phase 1: Base matmul  x @ W.T + bias
    # =========================================================================
    acc = tl.zeros((BLOCK_B, BLOCK_O), dtype=tl.float32)
    
    for i_start in range(0, in_features, BLOCK_I):
        offs_i = i_start + tl.arange(0, BLOCK_I)
        mask_i = offs_i < in_features
        
        # Load X block: (BLOCK_B, BLOCK_I)
        x_ptrs = X_ptr + offs_b[:, None] * stride_xb + offs_i[None, :] * stride_xi
        x_block = tl.load(x_ptrs, mask=mask_b[:, None] & mask_i[None, :], other=0.0)
        
        # Load W block: (BLOCK_O, BLOCK_I)
        w_ptrs = W_ptr + offs_o[:, None] * stride_wo + offs_i[None, :] * stride_wi
        w_block = tl.load(w_ptrs, mask=mask_o[:, None] & mask_i[None, :], other=0.0)
        
        # Accumulate x @ W.T
        acc += tl.dot(x_block, tl.trans(w_block))
    
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_o, mask=mask_o, other=0.0)
        acc += bias[None, :]
    
    # =========================================================================
    # Phase 2: Low-rank perturbation sign * (x @ B) @ A.T, generated on-the-fly
    # =========================================================================
    # Probit constants for normal approximation
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    
    for r in range(RANK):
        # Compute projection: proj[b] = sum_i(x[b,i] * B[b,i,r])
        proj = tl.zeros((BLOCK_B,), dtype=tl.float32)
        
        for i_start in range(0, in_features, BLOCK_I):
            offs_i = i_start + tl.arange(0, BLOCK_I)
            mask_i = offs_i < in_features
            
            # Load X block
            x_ptrs = X_ptr + offs_b[:, None] * stride_xb + offs_i[None, :] * stride_xi
            x_block = tl.load(x_ptrs, mask=mask_b[:, None] & mask_i[None, :], other=0.0)
            
            # Generate B[i,r] on-the-fly: tl.rand(seed, offset)
            # offset for B[i,r] = i * RANK + r
            B_offsets = offs_i[None, :] * RANK + r
            B_rand = tl.rand(seeds[:, None], B_offsets)  # uniform [0,1)
            
            # Probit approximation: uniform -> normal
            u = tl.maximum(B_rand, 1e-7)
            u = tl.minimum(u, 1.0 - 1e-7)
            t = tl.sqrt(-2.0 * tl.log(tl.where(u < 0.5, u, 1.0 - u)))
            B_normal = t - (c0 + c1*t + c2*t*t) / (1.0 + d1*t + d2*t*t + d3*t*t*t)
            B_normal = tl.where(u < 0.5, -B_normal, B_normal)
            
            # Project: x * B summed over input dimension
            proj += tl.sum(x_block * B_normal, axis=1)
        
        # Generate A[o,r] on-the-fly
        # offset for A[o,r] = (in_features + o) * RANK + r
        A_offsets = (in_features + offs_o[None, :]) * RANK + r
        A_rand = tl.rand(seeds[:, None], A_offsets)
        
        u = tl.maximum(A_rand, 1e-7)
        u = tl.minimum(u, 1.0 - 1e-7)
        t = tl.sqrt(-2.0 * tl.log(tl.where(u < 0.5, u, 1.0 - u)))
        A_normal = t - (c0 + c1*t + c2*t*t) / (1.0 + d1*t + d2*t*t + d3*t*t*t)
        A_normal = tl.where(u < 0.5, -A_normal, A_normal)
        
        # Add perturbation: sign * proj[b] * A[b,o,r] * sigma_scale
        acc += (signs[:, None] * proj[:, None]) * A_normal * sigma_scale
    
    # =========================================================================
    # Store output
    # =========================================================================
    out_ptrs = out_ptr + offs_b[:, None] * stride_ob + offs_o[None, :] * stride_oo
    tl.store(out_ptrs, acc, mask=mask_b[:, None] & mask_o[None, :])


def fused_perturbed_forward(
    x: torch.Tensor,           # (batch, in_features)
    weight: torch.Tensor,      # (out_features, in_features)
    bias: torch.Tensor | None, # (out_features,) or None
    seeds: torch.Tensor,       # (batch,) int32 seeds per element
    signs: torch.Tensor,       # (batch,) float32 +1/-1 for antithetic
    sigma: float,
    rank: int,
) -> torch.Tensor:
    """
    Fused forward pass with low-rank perturbation.
    
    Args:
        x: Input tensor (batch, in_features)
        weight: Weight matrix (out_features, in_features)
        bias: Optional bias (out_features,)
        seeds: Per-element RNG seeds (batch,) - determines A, B for each element
        signs: Per-element signs (batch,) - +1 or -1 for antithetic
        sigma: Perturbation scale
        rank: Low-rank dimension
    
    Returns:
        Output tensor (batch, out_features)
    """
    batch, in_features = x.shape
    out_features = weight.shape[0]
    
    # Allocate output
    out = torch.empty(batch, out_features, device=x.device, dtype=x.dtype)
    
    # Compute sigma_scale
    r = min(rank, in_features, out_features)
    sigma_scale = sigma / (r ** 0.5)
    
    # Grid
    BLOCK_B = 32
    BLOCK_O = 64
    BLOCK_I = 64
    
    grid = (triton.cdiv(batch, BLOCK_B), triton.cdiv(out_features, BLOCK_O))
    
    _fused_forward_kernel[grid](
        x, weight, bias, out,
        seeds, signs,
        batch, in_features, out_features,
        sigma_scale,
        r,  # RANK
        bias is not None,  # HAS_BIAS
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_B, BLOCK_O, BLOCK_I,
    )
    
    return out


@triton.jit
def _generate_factors_kernel(
    # Output
    A_ptr, B_ptr,
    # RNG
    seeds_ptr,
    # Dimensions
    batch_size, out_features, in_features, rank,
    # Params
    sigma_scale,
    # Strides
    stride_Ab, stride_Ao, stride_Ar,
    stride_Bb, stride_Bi, stride_Br,
    # Blocks
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Generate A and B factors for gradient computation."""
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_r = tl.program_id(2)
    
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    r = pid_r
    
    mask_b = offs_b < batch_size
    
    seeds = tl.load(seeds_ptr + offs_b, mask=mask_b, other=0)
    
    # Probit constants
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    
    # Generate A[b, d, r] where d indexes out_features
    mask_o = offs_d < out_features
    A_offsets = (in_features + offs_d[None, :]) * rank + r
    A_rand = tl.rand(seeds[:, None], A_offsets)
    
    u = tl.maximum(A_rand, 1e-7)
    u = tl.minimum(u, 1.0 - 1e-7)
    t = tl.sqrt(-2.0 * tl.log(tl.where(u < 0.5, u, 1.0 - u)))
    A_normal = t - (c0 + c1*t + c2*t*t) / (1.0 + d1*t + d2*t*t + d3*t*t*t)
    A_normal = tl.where(u < 0.5, -A_normal, A_normal)
    A_normal = A_normal * sigma_scale
    
    A_ptrs = A_ptr + offs_b[:, None] * stride_Ab + offs_d[None, :] * stride_Ao + r * stride_Ar
    tl.store(A_ptrs, A_normal, mask=mask_b[:, None] & mask_o[None, :])
    
    # Generate B[b, d, r] where d indexes in_features
    mask_i = offs_d < in_features
    B_offsets = offs_d[None, :] * rank + r
    B_rand = tl.rand(seeds[:, None], B_offsets)
    
    u = tl.maximum(B_rand, 1e-7)
    u = tl.minimum(u, 1.0 - 1e-7)
    t = tl.sqrt(-2.0 * tl.log(tl.where(u < 0.5, u, 1.0 - u)))
    B_normal = t - (c0 + c1*t + c2*t*t) / (1.0 + d1*t + d2*t*t + d3*t*t*t)
    B_normal = tl.where(u < 0.5, -B_normal, B_normal)
    
    B_ptrs = B_ptr + offs_b[:, None] * stride_Bb + offs_d[None, :] * stride_Bi + r * stride_Br
    tl.store(B_ptrs, B_normal, mask=mask_b[:, None] & mask_i[None, :])


def generate_factors(
    seeds: torch.Tensor,       # (batch,)
    out_features: int,
    in_features: int,
    rank: int,
    sigma: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate low-rank factors A, B using same RNG as fused_perturbed_forward.
    
    Used during step() to reconstruct perturbations for gradient computation.
    
    Returns:
        A: (batch, out_features, rank) with sigma scaling applied
        B: (batch, in_features, rank) without scaling
    """
    batch = seeds.shape[0]
    r = min(rank, in_features, out_features)
    sigma_scale = sigma / (r ** 0.5)
    
    A = torch.empty(batch, out_features, r, device=device, dtype=dtype)
    B = torch.empty(batch, in_features, r, device=device, dtype=dtype)
    
    BLOCK_B = 32
    BLOCK_D = 64
    
    grid = (
        triton.cdiv(batch, BLOCK_B),
        triton.cdiv(max(out_features, in_features), BLOCK_D),
        r,
    )
    
    _generate_factors_kernel[grid](
        A, B,
        seeds,
        batch, out_features, in_features, r,
        sigma_scale,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        BLOCK_B, BLOCK_D,
    )
    
    return A, B


def compute_layer_seeds(
    base_seed: int,
    epoch: int,
    member_ids: torch.Tensor,  # (batch,)
    layer_idx: int,
    antithetic: bool,
    noise_reuse: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-element seeds and signs for a layer.
    
    Seed derivation:
        effective_epoch = epoch // noise_reuse if noise_reuse > 0 else 0
        effective_member = member_id // 2 if antithetic else member_id
        seed = hash(base_seed, effective_epoch, effective_member, layer_idx)
    
    Returns:
        seeds: (batch,) int32 seeds
        signs: (batch,) float +1 or -1 for antithetic
    """
    device = member_ids.device
    batch = member_ids.shape[0]
    
    effective_epoch = epoch // noise_reuse if noise_reuse > 0 else 0
    
    if antithetic:
        effective_members = member_ids // 2
        signs = 1.0 - 2.0 * (member_ids % 2).float()
    else:
        effective_members = member_ids
        signs = torch.ones(batch, device=device, dtype=torch.float32)
    
    # Hash: base_seed * P1 + epoch * P2 + member * P3 + layer * P4
    # Using different primes for good mixing
    seeds = (
        (base_seed * 2654435761) ^
        (effective_epoch * 2246822519) ^
        (effective_members * 3266489917) ^
        (layer_idx * 668265263)
    ) & 0x7FFFFFFF  # Keep positive for int32
    
    return seeds.to(torch.int32), signs
