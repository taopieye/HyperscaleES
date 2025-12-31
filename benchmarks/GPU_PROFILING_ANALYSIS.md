# PyTorch EGGROLL - GPU Profiling Analysis

## Executive Summary

The PyTorch EGGROLL implementation is **NOT using Triton kernels** despite the filename `triton_kernels.py`. It uses PyTorch native ops which are already well-optimized, but there are significant optimization opportunities.

## Performance Breakdown

### Full Forward Pass: ~1.5ms per batch (pop_size=2048)

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| **Noise Generation** (3 layers) | 0.77 | 51% |
| **Perturbed Linear** (3 layers) | 0.74 | 49% |
| Hook Overhead | 0.27 | 18% |
| Activations | 0.02 | 1% |

### Top CUDA Kernels (per 10 forward passes)

| Kernel | Time (μs) | % | Operation |
|--------|-----------|---|-----------|
| `gemvx` (GEMV) | 392 | 16.6% | Matrix-vector multiply (F.linear) |
| `ampere_sgemm_32x128` | 368 | 15.5% | Batched matmul (bmm) |
| `vectorized_gather` | 244 | 10.3% | Antithetic expansion (repeat_interleave) |
| `elementwise_kernel` (mul) | 233 | 9.9% | Sigma scaling |
| `distribution_normal` | 170 | 7.2% | Random number generation |
| `elementwise_kernel` (add) | 145 | 6.2% | Addition ops |

## Key Findings

### 1. Noise Generation is the Biggest Bottleneck (51%)
- 3 separate calls to `generate_lowrank_factors_torch` per forward
- Each call does: randn → repeat_interleave → slice → scale → sign flip
- **5+ kernel launches per layer** just for noise

### 2. Hook Overhead is Significant (18%)
- The `batched_forward` uses Python hooks to intercept Linear layers
- Direct execution is 0.27ms faster (1.19ms vs 1.46ms)

### 3. Many Small Kernels
- `vectorized_gather` (10.3%) - from `repeat_interleave`
- Multiple `elementwise_kernel` calls for simple math

## Optimization Opportunities

### Tier 1: High Impact (Expected 30-50% speedup)

#### 1.1 Fused Noise Generation Kernel (Triton)
```python
# Current: 5+ kernel launches per layer
noise = torch.randn(...)           # 1 kernel
expanded = noise.repeat_interleave(2)  # 1 kernel
B = expanded[:, :in_features]      # 1 kernel  
A = expanded[:, in_features:] * sigma  # 1 kernel
A = A * sign                       # 1 kernel

# Proposed: Single Triton kernel
@triton.jit
def fused_lowrank_factors_kernel(
    out_A, out_B, seed, epoch, param_id, sigma, rank,
    in_features, out_features, pop_size,
    BLOCK_SIZE: tl.constexpr
):
    # Generate noise directly in factored form with antithetic signs
    member_id = tl.program_id(0)
    true_member = member_id // 2
    sign = tl.where(member_id % 2 == 0, 1.0, -1.0)
    
    # Use Philox RNG with deterministic seeding
    seed_value = seed + epoch * 1009 + param_id * 10007 + true_member
    for i in range(rank):
        for j in range(in_features):
            B[member_id, j, i] = tl.random.normal(seed_value + i * in_features + j)
        for j in range(out_features):
            A[member_id, j, i] = tl.random.normal(...) * sigma * sign
```

**Expected impact**: Reduce noise generation from 0.77ms to ~0.3ms

#### 1.2 Eliminate Hook Overhead
```python
# Current: Uses Python hooks via _apply_perturbed_forward_with_hooks
# Proposed: Direct Sequential traversal (already exists for nn.Sequential)
# Just ensure the fast path is always taken
```

**Expected impact**: Save 0.27ms (18%)

### Tier 2: Medium Impact (Expected 10-20% speedup)

#### 2.1 Fused Perturbed Linear (CUDA C++ or Triton)
```cpp
// Proposed CUDA kernel
__global__ void perturbed_linear_kernel(
    float* output,           // [pop_size, out_features]
    const float* input,      // [pop_size, in_features]
    const float* weight,     // [out_features, in_features]
    const float* bias,       // [out_features]
    const float* A,          // [pop_size, out_features, rank]
    const float* B,          // [pop_size, in_features, rank]
    int pop_size, int in_features, int out_features, int rank
) {
    // Compute: out = x @ W.T + x @ B @ A.T + bias in single kernel
    // Use shared memory for W and intermediate results
}
```

**Expected impact**: Reduce perturbed linear from 0.74ms to ~0.5ms

#### 2.2 Memory Layout Optimization
- Current: A and B are `(pop_size, features, rank)` 
- Better for bmm: `(pop_size, rank, features)` to avoid transpose

### Tier 3: Lower Impact but Nice-to-Have

#### 3.1 Pre-allocate Noise Buffers
- Reuse tensor allocations across epochs
- Avoid allocation overhead

#### 3.2 Use TF32 for Matrix Operations
```python
torch.set_float32_matmul_precision('high')  # Enable TF32 on Ampere+
```

## Comparison with JAX EGGROLL

| Aspect | PyTorch | JAX |
|--------|---------|-----|
| GPU Utilization | 29% | 33% |
| Memory Used | 1.4 GB | 19.7 GB |
| Throughput | 3.6x faster | baseline |
| Compilation | None | XLA JIT |

The PyTorch version is faster despite lower GPU utilization because:
1. **Lower memory pressure** - 14x less memory
2. **No JIT overhead** - Eager execution
3. **Better CUDA integration** - Native cuBLAS

## Relationship to Sample Efficiency Gap

The sample efficiency gap (Torch needs ~10% more steps to solve) is likely **NOT** related to GPU optimization. Possible causes:

1. **Numerical differences in RNG**
   - PyTorch uses Mersenne Twister / cuRAND
   - JAX uses Threefry / Philox
   - Different random sequences → different exploration

2. **Floating point accumulation order**
   - bmm vs einsum may accumulate differently
   - Could affect gradient estimates

3. **Antithetic sampling implementation**
   - Subtle differences in how signs are applied
   - Check if the noise pairing is exactly equivalent

To investigate the sample efficiency gap, we should:
1. Compare gradient estimates on synthetic fitness functions
2. Verify noise statistics match (mean, variance, covariance)
3. Check if the update rule produces identical parameter changes given identical fitnesses

## Recommended Next Steps

1. **Implement Fused Noise Generation Triton Kernel** (Highest ROI)
2. **Add torch.compile to critical path** (Easy win)
3. **Profile the sample efficiency gap separately** (Different issue)
4. **Benchmark with TF32 enabled** (Free performance on Ampere+)
