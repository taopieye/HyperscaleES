"""
Triton kernels for fused EGGROLL operations.

The key insight: JAX achieves 15x speedup by fusing RNG + perturbation + matmul
into a single XLA kernel. We replicate this with Triton.

Main kernel: fused_perturbed_linear
- Computes: out = x @ W.T + sigma * (x @ B) @ A.T
- Where A, B are generated on-the-fly from deterministic RNG
- No intermediate tensors materialized
"""

import torch
import triton
import triton.language as tl
import math


# =============================================================================
# Philox RNG - Must match our _fold_in and _random_normal exactly
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
def philox_4x32_10(key, counter):
    """
    Philox 4x32-10 PRNG.
    
    Args:
        key: 64-bit key (will be split into two 32-bit keys)
        counter: 64-bit counter (will be split into 4 32-bit counters)
    
    Returns:
        4 uint32 random values
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
    """Convert uint32 to standard normal using Box-Muller (approximate)."""
    # Convert to uniform [0, 1)
    uniform = u.to(tl.float32) * (1.0 / 4294967296.0)
    # Clamp to avoid log(0)
    uniform = tl.maximum(uniform, 1e-7)
    uniform = tl.minimum(uniform, 1.0 - 1e-7)
    # Inverse CDF approximation (fast but less accurate than Box-Muller)
    # Using a rational approximation to the normal quantile function
    # For better accuracy, we'd use proper Box-Muller with pairs
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


@triton.jit
def generate_normal_from_key(key, idx):
    """Generate a single normal random value from key and index."""
    c0, c1, c2, c3 = philox_4x32_10(key, idx)
    return uint32_to_normal(c0)


# =============================================================================
# Fused Perturbed Linear Kernel
# =============================================================================

@triton.jit
def fused_perturbed_linear_kernel(
    # Input/output pointers
    x_ptr,           # [batch, in_features]
    W_ptr,           # [out_features, in_features]
    bias_ptr,        # [out_features] or None
    out_ptr,         # [batch, out_features]
    # RNG parameters
    base_key,        # int64: base RNG key
    member_ids_ptr,  # [batch]: member indices for RNG
    layer_idx,       # int: layer index for RNG
    # Perturbation parameters
    sigma,           # float: perturbation scale
    rank,            # int: rank of low-rank perturbation
    antithetic,      # bool: whether to use antithetic sampling
    # Dimensions
    batch_size,
    in_features,
    out_features,
    # Strides
    stride_x_batch, stride_x_in,
    stride_W_out, stride_W_in,
    stride_out_batch, stride_out_out,
    # Block sizes
    BLOCK_M: tl.constexpr,  # Batch block size
    BLOCK_N: tl.constexpr,  # Output features block size
    BLOCK_K: tl.constexpr,  # Input features block size
    BLOCK_R: tl.constexpr,  # Rank block size (must be >= rank)
    HAS_BIAS: tl.constexpr,
):
    """
    Fused kernel computing: out = x @ W.T + bias + sigma * (x @ B) @ A.T
    
    Where A[out_features, rank] and B[in_features, rank] are generated on-the-fly
    from deterministic RNG based on (base_key, member_id, layer_idx).
    
    This fuses what would otherwise be:
    1. Generate random factors (kernel 1)
    2. x @ B (kernel 2)
    3. result @ A.T (kernel 3)
    4. Scale by sigma (kernel 4)
    5. Add to base matmul (kernel 5)
    
    Into a single kernel with no intermediate tensor materialization.
    """
    # Program ID
    pid_m = tl.program_id(0)  # Batch dimension
    pid_n = tl.program_id(1)  # Output dimension
    
    # Compute batch and output ranges for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Masks for bounds checking
    mask_m = offs_m < batch_size
    mask_n = offs_n < out_features
    
    # Load member IDs for this batch block
    member_ids = tl.load(member_ids_ptr + offs_m, mask=mask_m, other=0)
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # =========================================================================
    # Part 1: Compute base matmul x @ W.T
    # =========================================================================
    for k_start in range(0, in_features, BLOCK_K):
        k_offs = k_start + offs_k
        mask_k = k_offs < in_features
        
        # Load x block: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_x_batch + k_offs[None, :] * stride_x_in
        x_block = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load W block: [BLOCK_N, BLOCK_K] (W is stored as [out, in])
        W_ptrs = W_ptr + offs_n[:, None] * stride_W_out + k_offs[None, :] * stride_W_in
        W_block = tl.load(W_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Accumulate: [BLOCK_M, BLOCK_N] += [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(x_block, tl.trans(W_block))
    
    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias[None, :]
    
    # =========================================================================
    # Part 2: Compute low-rank perturbation (x @ B) @ A.T
    # This is the tricky part - we generate A, B on-the-fly per member
    # =========================================================================
    
    # For each member in the batch, we need different A, B factors
    # But they're generated from the same key pattern
    # 
    # The perturbation for member m is:
    #   delta[m] = sigma * (x[m] @ B[m]) @ A[m].T
    # 
    # Where A[m], B[m] are generated from fold_in(fold_in(base_key, layer_idx), member_id[m])
    
    # We'll compute this by iterating over rank
    # For each rank r:
    #   1. Generate A[:, r] and B[:, r] for all members in block
    #   2. Compute contribution to output
    
    # Accumulator for perturbation
    pert_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Compute member keys: fold_in(fold_in(base_key, layer_idx), member_id)
    # For antithetic: effective_member = member_id // 2, sign = 1 if even else -1
    if antithetic:
        effective_member = member_ids // 2
        sign = tl.where(member_ids % 2 == 0, 1.0, -1.0)
    else:
        effective_member = member_ids
        sign = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    
    # Fold in layer index to base key
    # Using simple mixing with 32-bit constants that fit in Triton's int32
    # We mix layer_idx into both halves of the 64-bit key
    LAYER_MIX: tl.constexpr = 0x9E3779B9  # 32-bit golden ratio constant
    layer_key = base_key ^ (layer_idx.to(tl.int64) * LAYER_MIX)
    
    # Member keys: one per batch element
    # member_key[m] = layer_key ^ (effective_member[m] * another_prime)
    MEMBER_MIX: tl.constexpr = 0xBB67AE85  # 32-bit mixing constant
    member_keys = layer_key ^ (effective_member.to(tl.int64) * MEMBER_MIX)
    
    # Scale factor for low-rank (matches JAX: sigma / sqrt(rank))
    scale = sigma / tl.sqrt(rank.to(tl.float32)) * sign
    
    # For simplicity in this first version, we compute the full perturbation
    # by generating factors and doing the matmul explicitly
    # A more optimized version would interleave this with the base matmul
    
    for r in range(rank):
        # Generate B[:, r] for all members and compute x @ B[:, r]
        # B has shape [in_features] per member
        xB = tl.zeros((BLOCK_M,), dtype=tl.float32)
        
        for k_start in range(0, in_features, BLOCK_K):
            k_offs = k_start + offs_k
            mask_k = k_offs < in_features
            
            # Load x block
            x_ptrs = x_ptr + offs_m[:, None] * stride_x_batch + k_offs[None, :] * stride_x_in
            x_block = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            
            # Generate B values for this k range
            # B[m, k, r] = normal(member_keys[m], k + r * in_features)
            # We need to generate per-member, per-k values
            for ki in range(BLOCK_K):
                k_idx = k_start + ki
                if k_idx < in_features:
                    # RNG index for B: offset by out_features to separate from A
                    rng_idx_B = (out_features + k_idx + r * in_features).to(tl.int64)
                    
                    # Generate B values for all members in block
                    B_vals = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    for mi in range(BLOCK_M):
                        if mi < batch_size:
                            mk = tl.load(member_keys + mi)
                            c0, c1, c2, c3 = philox_4x32_10(mk, rng_idx_B)
                            B_vals = tl.where(tl.arange(0, BLOCK_M) == mi, 
                                            uint32_to_normal(c0), B_vals)
                    
                    # Accumulate x[:, k] * B[:, k]
                    xB += x_block[:, ki] * B_vals
        
        # Now compute (xB) @ A[:, r].T = xB[:, None] * A[None, :]
        # Generate A[:, r] and multiply
        for n_start in range(0, out_features, BLOCK_N):
            n_offs_local = n_start + tl.arange(0, BLOCK_N)
            mask_n_local = n_offs_local < out_features
            
            # Only process our block
            if n_start == pid_n * BLOCK_N:
                for ni in range(BLOCK_N):
                    n_idx = pid_n * BLOCK_N + ni
                    if n_idx < out_features:
                        # RNG index for A
                        rng_idx_A = (n_idx + r * out_features).to(tl.int64)
                        
                        # Generate A values for all members
                        A_vals = tl.zeros((BLOCK_M,), dtype=tl.float32)
                        for mi in range(BLOCK_M):
                            if mi < batch_size:
                                mk = tl.load(member_keys + mi)
                                c0, c1, c2, c3 = philox_4x32_10(mk, rng_idx_A)
                                A_vals = tl.where(tl.arange(0, BLOCK_M) == mi,
                                                uint32_to_normal(c0), A_vals)
                        
                        # Accumulate: pert_acc[:, ni] += xB * A_vals * scale
                        pert_acc = tl.where(
                            tl.arange(0, BLOCK_N)[None, :] == ni,
                            pert_acc + (xB * A_vals * scale)[:, None],
                            pert_acc
                        )
    
    # Combine base and perturbation
    out = acc + pert_acc
    
    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_batch + offs_n[None, :] * stride_out_out
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


# =============================================================================
# Python Wrapper
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
    
    Equivalent to EggRoll.do_mm but in a single fused kernel.
    """
    assert x.is_cuda and weight.is_cuda
    assert x.ndim == 2 and weight.ndim == 2
    
    batch_size, in_features = x.shape
    out_features, in_features_w = weight.shape
    assert in_features == in_features_w
    
    # Ensure member_ids is on same device
    if not member_ids.is_cuda:
        member_ids = member_ids.to(x.device)
    
    # Output tensor
    out = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    
    # Block sizes (tuned for typical GPU)
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 64
    BLOCK_R = max(rank, 4)
    
    # Grid
    grid = (triton.cdiv(batch_size, BLOCK_M), triton.cdiv(out_features, BLOCK_N))
    
    # Launch kernel
    fused_perturbed_linear_kernel[grid](
        x, weight, bias, out,
        base_key, member_ids, layer_idx,
        sigma, rank, antithetic,
        batch_size, in_features, out_features,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_R=BLOCK_R,
        HAS_BIAS=bias is not None,
    )
    
    return out


# =============================================================================
# Simpler V2: Generate factors first, then fused matmul
# This is easier to verify correct and still much faster than naive
# =============================================================================

@triton.jit
def generate_lowrank_factors_kernel(
    A_ptr,           # [batch, out_features, rank]
    B_ptr,           # [batch, in_features, rank]
    base_key,
    member_ids_ptr,  # [batch]
    layer_idx,
    sigma,
    rank,
    antithetic,
    batch_size,
    out_features,
    in_features,
    BLOCK_B: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """Generate low-rank factors A, B for all batch members."""
    pid_b = tl.program_id(0)  # Batch
    pid_f = tl.program_id(1)  # Feature (covers both A and B)
    
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
    
    mask_b = offs_b < batch_size
    
    # Load member IDs
    member_ids = tl.load(member_ids_ptr + offs_b, mask=mask_b, other=0)
    
    # Compute member keys
    if antithetic:
        effective_member = member_ids // 2
        sign = tl.where(member_ids % 2 == 0, 1.0, -1.0)
    else:
        effective_member = member_ids
        sign = tl.full((BLOCK_B,), 1.0, dtype=tl.float32)
    
    layer_key = base_key ^ (layer_idx.to(tl.int64) * 0x9E3779B9)
    member_keys = layer_key ^ (effective_member.to(tl.int64) * 0xBB67AE85)
    
    scale = sigma / tl.sqrt(rank.to(tl.float32))
    
    # Generate factors for each rank
    for r in range(rank):
        for fi in range(BLOCK_F):
            f_idx = pid_f * BLOCK_F + fi
            
            # Generate A factors (if f_idx < out_features)
            if f_idx < out_features:
                rng_idx = (f_idx + r * out_features).to(tl.int64)
                for bi in range(BLOCK_B):
                    if offs_b[bi] < batch_size:
                        mk = tl.load(member_keys + bi)
                        c0, _, _, _ = philox_4x32_10(mk, rng_idx)
                        val = uint32_to_normal(c0) * scale * tl.load(sign + bi)
                        A_idx = offs_b[bi] * out_features * rank + f_idx * rank + r
                        tl.store(A_ptr + A_idx, val)
            
            # Generate B factors (if f_idx < in_features)
            if f_idx < in_features:
                rng_idx = (out_features + f_idx + r * in_features).to(tl.int64)
                for bi in range(BLOCK_B):
                    if offs_b[bi] < batch_size:
                        mk = tl.load(member_keys + bi)
                        c0, _, _, _ = philox_4x32_10(mk, rng_idx)
                        val = uint32_to_normal(c0)  # B doesn't get scaled by sigma
                        B_idx = offs_b[bi] * in_features * rank + f_idx * rank + r
                        tl.store(B_ptr + B_idx, val)
