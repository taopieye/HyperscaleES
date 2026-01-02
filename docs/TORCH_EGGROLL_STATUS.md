# PyTorch EGGROLL Status Document

**Date:** December 31, 2025  
**Purpose:** Handoff document for continued optimization work on PyTorch EGGROLL  
**Goal:** Match or exceed JAX EGGROLL's claim of **~91% inference time** (pre-gen noise) or **~69% inference time** (on-the-fly noise)

---

## Executive Summary

The PyTorch port of EGGROLL is **functionally complete and passes all unit tests**. On the CartPole-v1 benchmark (500M steps, 10 seeds), Torch EGGROLL achieves:

| Metric | Torch | JAX | Delta |
|--------|-------|-----|-------|
| Wall time | 3213s ± 24s | 3414s ± 42s | **6% faster** ✅ |
| Final return | 499.9 ± 0.2 | 500.0 ± 0.0 | Equivalent ✅ |
| Steps to solve | 17.2M ± 2.0M | 15.5M ± 0.8M | **~10% more steps** ⚠️ |

**Key concern:** The "triton_kernels.py" file is **misnamed** - it uses pure PyTorch ops (`torch.randn`, `torch.bmm`), not Triton kernels at all.

**Primary target:** The EGGROLL paper claims forward pass overhead of only ~9% (pre-gen) or ~31% (on-the-fly) vs standard inference. We need to validate and match this.

---

## Experiment-Driven Optimization Strategy

Instead of guessing at optimizations, we set up rigorous experiments, establish baselines, then profile and optimize against them.

### Experiment A: Pure Throughput (No Env Overhead)

**Goal:** Measure raw forward pass performance - EGGROLL vs standard inference.

**Paper claim:** ~91% of inference time (pre-gen noise) or ~69% (on-the-fly noise)

**Setup:**
```python
# Baseline: Standard batched inference
for _ in range(N):
    out = model(x_batch)  # (pop_size, input_dim) -> (pop_size, output_dim)

# EGGROLL: Perturbed forward pass
for _ in range(N):
    with strategy.perturb(pop_size, epoch) as ctx:
        out = ctx.batched_forward(model, x_batch)
```

**Metrics:**
- [ ] Samples/second (throughput)
- [ ] Time per forward pass (ms)
- [ ] Overhead vs inference: `(eggroll_time - inference_time) / inference_time`
- [ ] GPU utilization during forward pass
- [ ] Memory bandwidth utilization

**Variables to sweep:**
- Population size: 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
- Model size: small (4→256→2), medium (64→512→64), large (256→2048→256)
- Rank: 1, 2, 4, 8, 16

**Script:** `benchmarks/experiment_a_throughput.py`

---

### Experiment B: GPU-Native Vectorized Environment

**Goal:** Measure end-to-end RL training with a fast GPU env (eliminates CPU env bottleneck).

**Setup:** Use Brax or gymnax for fully GPU-resident environment stepping.

```python
# All on GPU: env step + forward pass + ES update
env = gymnax.make("CartPole-v1")  # or Brax
for epoch in range(N):
    obs = env.reset(keys)  # (pop_size, obs_dim)
    for _ in range(episode_len):
        with strategy.perturb(pop_size, epoch) as ctx:
            actions = ctx.batched_forward(policy, obs)
        obs, rewards, dones, _ = env.step(actions)
    strategy.step(fitnesses)
```

**Metrics:**
- [ ] Steps/second (full loop)
- [ ] Breakdown: env time vs forward time vs ES update time
- [ ] Scaling with population size

**Comparison:**
- JAX EGGROLL + gymnax/Brax
- Torch EGGROLL + gymnax (via jax2torch) or Brax PyTorch port
- Standard inference + OpenES

**Script:** `benchmarks/experiment_b_gpu_env.py`

---

### Experiment C: Supervised Learning (Dataloader Integration)

**Goal:** Validate EGGROLL on a simple SL task with standard PyTorch dataloader patterns.

**Why:** ES can be used for non-RL tasks. This tests integration with PyTorch ecosystem.

**Setup:**
```python
# MNIST or CIFAR-10 classification via ES
dataloader = DataLoader(dataset, batch_size=pop_size, shuffle=True)

for epoch in range(N):
    for x_batch, y_batch in dataloader:
        with strategy.perturb(pop_size, epoch) as ctx:
            logits = ctx.batched_forward(model, x_batch)
            fitnesses = -cross_entropy(logits, y_batch, reduction='none')
        strategy.step(fitnesses)
```

**Metrics:**
- [ ] Time per batch (ms)
- [ ] Samples/second
- [ ] Overhead vs standard inference
- [ ] Final accuracy (sanity check)

**Comparison:**
- EGGROLL
- Standard SGD (for reference, not apples-to-apples)
- OpenES

**Script:** `benchmarks/experiment_c_supervised.py`

---

## Baselines to Establish

For each experiment, measure these baselines **before** any optimization:

| Baseline | Description | Script |
|----------|-------------|--------|
| `inference_torch` | Standard PyTorch forward pass | All experiments |
| `inference_jax` | Standard JAX forward pass (jit+vmap) | All experiments |
| `eggroll_jax` | JAX EGGROLL (reference implementation) | All experiments |
| `eggroll_torch_current` | Current PyTorch EGGROLL | All experiments |
| `openes_jax` | Full-rank ES baseline | Experiment A, B |

---

## Profiling Methodology

Once baselines are established, profile to find bottlenecks:

### 1. Macro Profiling
```bash
# Overall timing breakdown
uv run python benchmarks/profile_head_to_head.py --pop-sizes 32 64 128 256 512 1024 2048 4096 8192
```

### 2. Micro Profiling (PyTorch)
```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    # Run forward pass
    ...
prof.export_chrome_trace("trace.json")  # View in chrome://tracing
```

### 3. NSight Systems (Deep GPU Analysis)
```bash
nsys profile --stats=true uv run python benchmarks/experiment_a_throughput.py
```

---

## Optimization Targets (Informed by Profiling)

**Do not implement these until profiling data confirms they're the bottleneck.**

| Target | Expected Impact | Effort | Prerequisite |
|--------|-----------------|--------|--------------|
| `torch.compile` on hot path | 1.5-2x | Low | Profile shows Python overhead |
| Fused noise+matmul kernel | 1.2-1.5x | High | Profile shows kernel launch overhead |
| CUDA C++ extension | 1.5-3x | Very High | Triton insufficient for memory-bound ops |
| Memory layout optimization | 1.1-1.3x | Medium | Profile shows cache misses |
| Async noise generation | 1.1-1.2x | Medium | Profile shows noise gen on critical path |

---

## Current Implementation Details

### Repository Structure

```
src/hyperscalees/
├── torch/                      # PyTorch implementation
│   ├── __init__.py             # Exports EggrollStrategy, etc.
│   ├── strategy.py             # EggrollStrategy (914 lines) - main API
│   ├── perturbation.py         # PerturbationContext, Perturbation (504 lines)
│   ├── triton_kernels.py       # ⚠️ MISNAMED: Pure PyTorch ops (243 lines)
│   └── module.py               # ESModule wrapper
│
├── noiser/                     # JAX implementation (reference)
│   ├── eggroll.py              # EggRoll class (132 lines)
│   ├── open_es.py              # OpenES baseline
│   └── base_noiser.py          # Noiser base class
│
└── models/                     # Shared model definitions
```

---

## Architecture Comparison

### JAX EGGROLL (`noiser/eggroll.py`)

```python
# Core forward pass (do_mm)
def do_mm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
    base_ans = x @ param.T
    if iterinfo is None:
        return base_ans
    A, B = get_lora_update_params(frozen_noiser_params, noiser_params["sigma"] / jnp.sqrt(rank), iterinfo, param, base_key)
    return base_ans + x @ B @ A.T
```

**Key characteristics:**
- Uses `jax.random.fold_in(fold_in(key, epoch), member_id)` for determinism
- XLA compiles the entire computation graph
- `jax.vmap` handles population batching automatically
- Low-rank factors generated on-the-fly per forward call

### PyTorch EGGROLL (`torch/triton_kernels.py`)

```python
def generate_lowrank_factors_torch(...):
    # Generate noise for all unique members at once
    gen = torch.Generator(device=device)
    gen.manual_seed(gen_seed)
    
    noise = torch.randn(num_unique, in_features + out_features, rank, generator=gen, ...)
    noise_expanded = noise.repeat_interleave(2, dim=0)  # Antithetic
    
    B = gathered_noise[:, :in_features, :]
    A = gathered_noise[:, in_features:, :] * scaled_sigma * signs
    return A, B

def batched_perturbed_linear_torch(x, weight, bias, A, B, member_ids):
    base = F.linear(x, weight, bias)
    xB = torch.bmm(x.unsqueeze(1), B).squeeze(1)  # (batch, rank)
    perturbation = torch.bmm(xB.unsqueeze(1), A.transpose(1, 2)).squeeze(1)
    return base + perturbation
```

**Key characteristics:**
- Uses `torch.Generator` for seeded RNG (not Philox PRNG directly)
- Two `torch.bmm` calls for the low-rank perturbation
- Pre-generates all factors, then indexes for each batch

---

## Profiling Data (Existing)

### GPU Utilization Profile (`gpu_profile_20251230_180254.json`)

| Metric | Torch | JAX |
|--------|-------|-----|
| Steps/second | 1,295,086 | 357,748 |
| GPU Utilization (mean) | 29.3% | 33.1% |
| Memory Used | 1,366 MB | 19,748 MB |
| Power | 76W | 22W |
| Triton used | ❌ | N/A |
| torch.compile used | ❌ | N/A |

**Critical observation:** Torch is **3.6x faster** in pure forward throughput but only 1.06x faster in end-to-end CartPole. The gap is in the **environment step loop**, not the neural network forward pass.

### Anatomy Profile (`anatomy_profile_20251230_181509.json`)

For pop_size=2048, layer_size=256, 3 layers:

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| Full Forward Pass | 1.41 ± 0.39 | 100% |
| Low-Rank Perturbation (bmm) | 0.10 ± 0.04 | 7.0% |
| Base Linear (F.linear) | 0.07 ± 0.02 | 5.1% |
| Antithetic Expansion | 0.06 ± 0.02 | 3.9% |
| Activation (Tanh) | 0.03 ± 0.03 | 2.2% |
| Noise Generation | 0.02 ± 0.02 | 1.7% |
| Factor Split + Scale | 0.03 ± 0.00 | 1.9% |

**Key insight:** The bottleneck is NOT noise generation or low-rank ops - it's likely in the Python-level loop overhead and memory movement patterns.

### Alternative Implementations Tested

| Implementation | Mean Time (ms) | Speedup |
|----------------|----------------|---------|
| Current (2x bmm) | 0.092 | baseline |
| torch.compile | 0.085 | 1.08x |
| einsum | 0.161 | 0.57x (slower) |
| bmm w/ contiguous | 0.044 | 2.1x |

---

## Open Questions

### 1. Sample Efficiency Gap (~10%)

Torch takes ~17.2M steps to solve vs JAX's ~15.5M. Possible causes:

- **Different RNG streams:** PyTorch Generator vs JAX's fold_in produce different noise sequences
- **Numerical differences:** Float32 operations may differ between cuBLAS and XLA
- **Seed folding formula:** `seed + epoch * PRIME_EPOCH + param_id * PRIME_PARAM` vs `fold_in(fold_in(key, epoch), member_id)`

**Investigation needed:**
```python
# Compare noise distributions
torch_noise = generate_lowrank_factors_torch(...)
jax_noise = get_lora_update_params(...)
# Are they identically distributed? Same correlation structure?
```

### 2. Triton Kernel Strategy

The file `triton_kernels.py` is **entirely PyTorch native ops**. No `@triton.jit` decorators. Two paths forward:

**Option A: True Triton Kernels**
```python
@triton.jit
def perturbed_linear_kernel(x_ptr, w_ptr, a_ptr, b_ptr, out_ptr, ...):
    # Fuse: base_linear + noise_gen + low_rank_perturbation
    # Benefits: Single kernel launch, better memory locality
    # Drawbacks: Complex, needs careful tuning for different shapes
```

**Option B: CUDA C++ Extensions**
```cpp
// torch/extension.h
torch::Tensor fused_perturbed_linear(torch::Tensor x, torch::Tensor w, ...) {
    // Custom CUDA kernel with:
    // - Philox PRNG inline
    // - Warp-level primitives for reduction
    // - Shared memory for factor reuse
}
```

**Recommendation:** Start with `torch.compile(mode="max-autotune")` first (easy win), then profile to see if custom kernels are justified.

### 3. Memory Bandwidth Analysis

The anatomy profile shows GPU utilization at only 29-33%. This suggests we're memory-bound, not compute-bound. Investigation needed:

```python
# Profile memory access patterns
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # Run forward pass
    ...
```

---

## Known Issues

### 1. Print Statement in JAX Code
```python
# noiser/eggroll.py:52
print("LORA UPDATE", A.shape, B.shape)  # Should be removed or guarded
```

### 2. Inconsistent Sigma Scaling
```python
# JAX (eggroll.py):
A * sigma  # sigma already includes 1/sqrt(rank)

# Torch (triton_kernels.py):
scaled_sigma = sigma / (rank ** 0.5)  # Explicit scaling
```

Both are mathematically correct but could cause confusion.

### 3. Missing torch.compile Integration

The current implementation doesn't use `torch.compile` anywhere. Adding it to the hot path could provide 1.5-2x speedup based on anatomy profile results.

---

## Test Coverage

All tests pass as of the last benchmark run:

| Test File | Description | Status |
|-----------|-------------|--------|
| `test_forward_equivalence.py` | do_mm matches explicit computation | ✅ |
| `test_noise_determinism.py` | Same seed → same noise | ✅ |
| `test_antithetic_sampling.py` | Pairs share noise with opposite signs | ✅ |
| `test_low_rank_structure.py` | Low-rank factorization correctness | ✅ |
| `test_update_mechanics.py` | ES gradient estimation | ✅ |
| `test_fitness_normalization.py` | Z-score, rank transforms | ✅ |
| `test_noiser_api.py` | API compatibility | ✅ |
| `test_high_rank_accumulation.py` | Numerical stability | ✅ |

---

## Benchmark Results (Full Paper Replication)

From `cartpole_results_20251230_172002.md`:

### Learning Curves
All methods solve CartPole (≥475 return):
- PPO: 1.6M steps (fastest)
- OpenES: 4.7M steps
- JAX EGGROLL: 15.5M steps
- Torch EGGROLL: 17.2M steps

### Wall Time (500M steps)
- PPO (Rejax): 2,468s ± 73s
- Torch EGGROLL: 3,213s ± 24s
- JAX EGGROLL: 3,414s ± 42s
- OpenES: 7,250s ± 27s

### Speedup vs OpenES
- PPO: 2.94x
- Torch EGGROLL: 2.26x
- JAX EGGROLL: 2.12x

---

## Next Steps

### Step 1: Create Experiment Scripts

- [ ] `benchmarks/experiment_a_throughput.py` - Pure forward pass comparison
- [ ] `benchmarks/experiment_b_gpu_env.py` - GPU-native env (gymnax/Brax)
- [ ] `benchmarks/experiment_c_supervised.py` - Dataloader SL task

### Step 2: Run Baselines

```bash
# Experiment A: Pure throughput
uv run python benchmarks/experiment_a_throughput.py --all-baselines

# Experiment B: GPU env
uv run python benchmarks/experiment_b_gpu_env.py --all-baselines

# Experiment C: Supervised
uv run python benchmarks/experiment_c_supervised.py --all-baselines
```

### Step 3: Analyze Results

Target: EGGROLL forward pass should be ≤ **1.1x inference time** (pre-gen) or ≤ **1.45x** (on-the-fly)

If we're above those targets, profile to find why and optimize.

### Step 4: Profile Bottlenecks

Only after baselines reveal a gap vs the paper's claims.

### Step 5: Optimize

Data-driven, targeted optimizations based on profiling results.

---

## Files to Read

For new agent picking up this work:

1. **Implementation:**
   - [strategy.py](../src/hyperscalees/torch/strategy.py) - Main API
   - [triton_kernels.py](../src/hyperscalees/torch/triton_kernels.py) - Core computation
   - [perturbation.py](../src/hyperscalees/torch/perturbation.py) - Context manager

2. **Reference (JAX):**
   - [noiser/eggroll.py](../src/hyperscalees/noiser/eggroll.py) - JAX reference

3. **Tests:**
   - [test_forward_equivalence.py](../tests/test_forward_equivalence.py) - Core correctness
   - [test_noise_determinism.py](../tests/test_noise_determinism.py) - RNG behavior

4. **Benchmarks:**
   - [benchmark_cartpole.py](../benchmarks/benchmark_cartpole.py) - Full comparison
   - [anatomy_profile_*.json](../benchmarks/) - Profiling data

---

## Contact / Context

This work is part of porting the EGGROLL paper to PyTorch for broader accessibility. The JAX implementation is the reference. Key paper claims to validate:

1. ✅ Low-rank perturbations are memory-efficient
2. ✅ O(r(m+n)) compute vs O(mn) for full-rank
3. ⚠️ "Faster than OpenES" - true in wall time, investigating sample efficiency
4. ❓ Triton/CUDA optimization potential unexplored

**Goal:** Match or exceed JAX performance while providing a cleaner, more Pythonic API. THIS DOES NOT MEAN WE SHOULD IGNORE THE JAX IMPLEMENTATION--IT HAS THE ANSWERS WE SEEK! We just need to translate into pytorch.
