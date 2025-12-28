"""
Optimized RNG Benchmark: Testing various optimization strategies.

This benchmark compares:
1. Original vmap approach
2. torch.compile on vmap
3. Batched operations (no vmap)
4. torch.compile on batched operations
5. JAX reference

Goal: Match or exceed JAX performance.
"""

import torch
import torch.nn as nn
import time
import math
from torch.func import vmap
from typing import Tuple

# Try to import JAX for comparison
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("JAX not installed - JAX benchmarks will be skipped")


# ==============================================================================
# ORIGINAL IMPLEMENTATIONS (for comparison)
# ==============================================================================

def _fold_in_original(key: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """Original JAX-style fold_in."""
    mixed = (key.to(torch.int64) + data.to(torch.int64) * 2654435761) & 0xFFFFFFFF
    mixed = (mixed ^ (mixed >> 15)) & 0xFFFFFFFF
    mixed = (mixed * 2246822519) & 0xFFFFFFFF
    mixed = (mixed ^ (mixed >> 13)) & 0xFFFFFFFF
    mixed = (mixed * 3266489917) & 0xFFFFFFFF
    mixed = (mixed ^ (mixed >> 16)) & 0xFFFFFFFF
    return mixed


def _random_normal_original(key: torch.Tensor, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Original random normal generation."""
    numel = 1
    for s in shape:
        numel *= s
    numel_even = numel + (numel % 2)
    
    counters = torch.arange(numel_even, device=device, dtype=torch.int64)
    seeds = (key.to(torch.int64) + counters) & 0xFFFFFFFF
    
    seeds = ((seeds ^ (seeds >> 17)) * 0xed5ad4bb) & 0xFFFFFFFF
    seeds = ((seeds ^ (seeds >> 11)) * 0xac4c1b51) & 0xFFFFFFFF
    seeds = ((seeds ^ (seeds >> 15)) * 0x31848bab) & 0xFFFFFFFF
    seeds = (seeds ^ (seeds >> 14)) & 0xFFFFFFFF
    
    uniform = seeds.float() / (2**32)
    uniform = uniform.view(-1, 2)
    u1 = uniform[:, 0].clamp(min=1e-10, max=1.0 - 1e-10)
    u2 = uniform[:, 1]
    r = torch.sqrt(-2.0 * torch.log(u1))
    theta = 2.0 * math.pi * u2
    z0 = r * torch.cos(theta)
    z1 = r * torch.sin(theta)
    normal = torch.stack([z0, z1], dim=-1).flatten()[:numel]
    return normal.view(shape).to(dtype)


# ==============================================================================
# OPTIMIZED IMPLEMENTATIONS
# ==============================================================================

def _fold_in_batched(keys: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """
    Batched fold_in - operates on entire tensors without vmap.
    
    Args:
        keys: Shape (N,) or scalar - base keys
        data: Shape (N,) or scalar - data to fold in
    
    Returns:
        Mixed keys of shape broadcast(keys, data)
    """
    keys_i64 = keys.to(torch.int64)
    data_i64 = data.to(torch.int64)
    
    mixed = (keys_i64 + data_i64 * 2654435761) & 0xFFFFFFFF
    mixed = (mixed ^ (mixed >> 15)) & 0xFFFFFFFF
    mixed = (mixed * 2246822519) & 0xFFFFFFFF
    mixed = (mixed ^ (mixed >> 13)) & 0xFFFFFFFF
    mixed = (mixed * 3266489917) & 0xFFFFFFFF
    mixed = (mixed ^ (mixed >> 16)) & 0xFFFFFFFF
    return mixed


# Compiled version
_fold_in_batched_compiled = torch.compile(_fold_in_batched, mode="reduce-overhead")


def _random_normal_batched(
    keys: torch.Tensor,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device
) -> torch.Tensor:
    """
    Batched random normal - generate (N, *shape) from N keys without vmap.
    
    Uses Philox-style counter-based RNG with full parallelization.
    
    Args:
        keys: Shape (N,) - one key per batch element
        shape: Shape of output per key (e.g., (128, 64))
        dtype: Output dtype
        device: Output device
    
    Returns:
        Tensor of shape (N, *shape) with normal random values
    """
    N = keys.shape[0]
    numel = 1
    for s in shape:
        numel *= s
    numel_even = numel + (numel % 2)
    
    # Create counter grid: (N, numel_even)
    counters = torch.arange(numel_even, device=device, dtype=torch.int64)
    counters = counters.unsqueeze(0).expand(N, -1)  # (N, numel_even)
    
    # Expand keys: (N, 1) for broadcasting
    keys_expanded = keys.unsqueeze(1).to(torch.int64)  # (N, 1)
    
    # Seed mixing: keys + counters
    seeds = (keys_expanded + counters) & 0xFFFFFFFF
    
    # Splitmix32 mixing - all vectorized
    seeds = ((seeds ^ (seeds >> 17)) * 0xed5ad4bb) & 0xFFFFFFFF
    seeds = ((seeds ^ (seeds >> 11)) * 0xac4c1b51) & 0xFFFFFFFF
    seeds = ((seeds ^ (seeds >> 15)) * 0x31848bab) & 0xFFFFFFFF
    seeds = (seeds ^ (seeds >> 14)) & 0xFFFFFFFF
    
    # To uniform [0,1)
    uniform = seeds.float() / (2**32)  # (N, numel_even)
    
    # Box-Muller transform
    uniform = uniform.view(N, -1, 2)  # (N, numel_even//2, 2)
    u1 = uniform[..., 0].clamp(min=1e-10, max=1.0 - 1e-10)
    u2 = uniform[..., 1]
    
    r = torch.sqrt(-2.0 * torch.log(u1))
    theta = 2.0 * math.pi * u2
    z0 = r * torch.cos(theta)
    z1 = r * torch.sin(theta)
    
    # Interleave and reshape
    normal = torch.stack([z0, z1], dim=-1).flatten(1)  # (N, numel_even)
    normal = normal[:, :numel]  # Trim to exact size
    
    return normal.view(N, *shape).to(dtype)


# Compiled version
_random_normal_batched_compiled = torch.compile(_random_normal_batched, mode="reduce-overhead")


def _generate_lowrank_factors_batched(
    base_key: torch.Tensor,
    member_ids: torch.Tensor,
    param_key: int,
    m: int,
    n: int,
    rank: int,
    sigma: float,
    dtype: torch.dtype,
    device: torch.device,
    antithetic: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate low-rank factors for entire population at once.
    
    Returns:
        A: (N, m, r) - left factors with sigma scaling and antithetic sign
        B: (N, n, r) - right factors
    """
    N = member_ids.shape[0]
    r = min(rank, m, n)
    sigma_scaled = sigma / math.sqrt(r)
    
    # Compute effective member IDs for noise reuse
    if antithetic:
        effective_ids = member_ids // 2
        signs = (1.0 - 2.0 * (member_ids % 2).float()).view(N, 1, 1)  # (N, 1, 1)
    else:
        effective_ids = member_ids
        signs = torch.ones(N, 1, 1, device=device, dtype=dtype)
    
    # Generate member keys: base_key folded with effective_id
    member_keys = _fold_in_batched(base_key, effective_ids)  # (N,)
    
    # Generate layer keys: member_key folded with param_key
    param_key_tensor = torch.tensor(param_key, dtype=torch.int64, device=device)
    layer_keys = _fold_in_batched(member_keys, param_key_tensor)  # (N,)
    
    # Generate all factors at once: (N, m+n, r)
    factors = _random_normal_batched(layer_keys, (m + n, r), dtype, device)
    
    # Split into A and B
    A = factors[:, :m, :] * (sigma_scaled * signs)  # (N, m, r) with sign
    B = factors[:, m:, :]  # (N, n, r)
    
    return A, B


# Compiled version 
_generate_lowrank_factors_batched_compiled = torch.compile(
    _generate_lowrank_factors_batched, mode="reduce-overhead"
)


def batched_forward_optimized(
    model: nn.Module,
    x: torch.Tensor,
    member_ids: torch.Tensor,
    base_key: torch.Tensor,
    rank: int,
    sigma: float,
    param_keys: dict,
    should_evolve_param,
    antithetic: bool = True,
) -> torch.Tensor:
    """
    Optimized batched forward using pre-batched RNG operations.
    
    Instead of vmap over single_member_forward, we:
    1. Pre-generate ALL perturbation factors for ALL layers at once
    2. Use batched matmul (bmm) for the forward pass
    """
    N = member_ids.shape[0]
    device = x.device
    dtype = x.dtype
    
    current = x  # (N, input_dim)
    
    param_names_map = {id(p): n for n, p in model.named_parameters()}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight
            bias = module.bias
            param_name = param_names_map.get(id(W), f"{name}.weight" if name else "weight")
            evolve = should_evolve_param(param_name, W)
            param_key = param_keys.get(param_name, hash(param_name) % 10000)
            
            m, n_in = W.shape
            
            # Base computation (same for all members)
            base_out = current @ W.T  # (N, m)
            if bias is not None:
                base_out = base_out + bias
            
            if evolve:
                # Generate perturbation factors for all members
                A, B = _generate_lowrank_factors_batched(
                    base_key, member_ids, param_key, m, n_in, rank, sigma, dtype, device, antithetic
                )
                # A: (N, m, r), B: (N, n_in, r)
                
                # Batched perturbation: x @ B @ A.T
                # current: (N, n_in) -> (N, 1, n_in)
                xB = torch.bmm(current.unsqueeze(1), B)  # (N, 1, r)
                pert = torch.bmm(xB, A.transpose(-1, -2)).squeeze(1)  # (N, m)
                
                current = base_out + pert
            else:
                current = base_out
                
        elif isinstance(module, nn.ReLU):
            current = torch.relu(current)
        elif isinstance(module, nn.Tanh):
            current = torch.tanh(current)
        elif isinstance(module, nn.Sigmoid):
            current = torch.sigmoid(current)
        elif isinstance(module, nn.GELU):
            current = torch.nn.functional.gelu(current)
    
    return current


# ==============================================================================
# BENCHMARK FUNCTIONS
# ==============================================================================

def benchmark_fold_in_variants(device, num_iterations=1000):
    """Compare fold_in implementations."""
    key = torch.tensor(42, dtype=torch.int64, device=device)
    data = torch.arange(1024, dtype=torch.int64, device=device)
    
    results = {}
    
    # 1. Original with vmap
    for _ in range(10):
        _ = vmap(lambda d: _fold_in_original(key, d))(data)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = vmap(lambda d: _fold_in_original(key, d))(data)
    torch.cuda.synchronize()
    results['vmap_original'] = (time.perf_counter() - start) / num_iterations * 1000
    
    # 2. Batched (no vmap)
    for _ in range(10):
        _ = _fold_in_batched(key, data)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = _fold_in_batched(key, data)
    torch.cuda.synchronize()
    results['batched'] = (time.perf_counter() - start) / num_iterations * 1000
    
    # 3. Batched + compiled (warmup more for compilation)
    for _ in range(20):
        _ = _fold_in_batched_compiled(key, data)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = _fold_in_batched_compiled(key, data)
    torch.cuda.synchronize()
    results['batched_compiled'] = (time.perf_counter() - start) / num_iterations * 1000
    
    return results


def benchmark_random_normal_variants(device, shape=(128, 64), pop_size=256, num_iterations=100):
    """Compare random normal implementations."""
    keys = torch.arange(pop_size, dtype=torch.int64, device=device)
    
    results = {}
    
    # 1. Original with vmap
    for _ in range(5):
        _ = vmap(lambda k: _random_normal_original(k, shape, torch.float32, device))(keys)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = vmap(lambda k: _random_normal_original(k, shape, torch.float32, device))(keys)
    torch.cuda.synchronize()
    results['vmap_original'] = (time.perf_counter() - start) / num_iterations * 1000
    
    # 2. Batched
    for _ in range(5):
        _ = _random_normal_batched(keys, shape, torch.float32, device)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = _random_normal_batched(keys, shape, torch.float32, device)
    torch.cuda.synchronize()
    results['batched'] = (time.perf_counter() - start) / num_iterations * 1000
    
    # 3. Batched + compiled
    for _ in range(20):
        _ = _random_normal_batched_compiled(keys, shape, torch.float32, device)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = _random_normal_batched_compiled(keys, shape, torch.float32, device)
    torch.cuda.synchronize()
    results['batched_compiled'] = (time.perf_counter() - start) / num_iterations * 1000
    
    return results


def benchmark_full_forward_variants(device, hidden_dim=64, pop_size=256, num_iterations=50):
    """Compare full forward implementations."""
    
    # Simple 2-layer MLP
    model = nn.Sequential(
        nn.Linear(32, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 16),
    ).to(device)
    
    x = torch.randn(pop_size, 32, device=device)
    member_ids = torch.arange(pop_size, dtype=torch.int64, device=device)
    base_key = torch.tensor(42, dtype=torch.int64, device=device)
    
    param_keys = {n: i for i, (n, _) in enumerate(model.named_parameters())}
    def should_evolve(name, param):
        return param.requires_grad and 'bias' not in name
    
    results = {}
    
    # 1. Original vmap approach (simulated)
    rank = 4
    sigma = 0.1
    
    def single_forward_original(member_key, sign, xi):
        current = xi
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight
                bias = module.bias
                m, n_in = W.shape
                r = min(rank, m, n_in)
                
                base_out = current @ W.T
                if bias is not None:
                    base_out = base_out + bias
                
                param_name = f"{name}.weight" if name else "weight"
                if should_evolve(param_name, W):
                    param_key_val = param_keys.get(param_name, 0)
                    layer_key = _fold_in_original(member_key, torch.tensor(param_key_val, dtype=torch.int64, device=device))
                    factors = _random_normal_original(layer_key, (m + n_in, r), xi.dtype, device)
                    A = factors[:m] * (sigma / math.sqrt(r) * sign)
                    B = factors[m:]
                    xB = current @ B
                    pert = xB @ A.T
                    current = base_out + pert
                else:
                    current = base_out
            elif isinstance(module, nn.ReLU):
                current = torch.relu(current)
        return current
    
    effective_ids = member_ids // 2
    signs = 1.0 - 2.0 * (member_ids % 2).float()
    member_keys = vmap(lambda mid: _fold_in_original(base_key, mid))(effective_ids)
    
    # Warmup
    for _ in range(5):
        _ = vmap(single_forward_original)(member_keys, signs, x)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = vmap(single_forward_original)(member_keys, signs, x)
    torch.cuda.synchronize()
    results['vmap_original'] = (time.perf_counter() - start) / num_iterations * 1000
    
    # 2. Optimized batched forward
    for _ in range(5):
        _ = batched_forward_optimized(model, x, member_ids, base_key, rank, sigma, param_keys, should_evolve)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = batched_forward_optimized(model, x, member_ids, base_key, rank, sigma, param_keys, should_evolve)
    torch.cuda.synchronize()
    results['batched_optimized'] = (time.perf_counter() - start) / num_iterations * 1000
    
    # 3. Compiled batched forward  
    compiled_forward = torch.compile(batched_forward_optimized, mode="reduce-overhead")
    for _ in range(20):
        _ = compiled_forward(model, x, member_ids, base_key, rank, sigma, param_keys, should_evolve)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = compiled_forward(model, x, member_ids, base_key, rank, sigma, param_keys, should_evolve)
    torch.cuda.synchronize()
    results['batched_compiled'] = (time.perf_counter() - start) / num_iterations * 1000
    
    return results


def benchmark_jax_fold_in(num_iterations=1000):
    if not HAS_JAX:
        return None
    
    key = jax.random.PRNGKey(42)
    data = jnp.arange(1024)
    
    @jax.jit
    @jax.vmap
    def fold_in_vmapped(d):
        return jax.random.fold_in(key, d)
    
    for _ in range(10):
        _ = fold_in_vmapped(data).block_until_ready()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = fold_in_vmapped(data).block_until_ready()
    
    return (time.perf_counter() - start) / num_iterations * 1000


def benchmark_jax_random_normal(shape=(128, 64), pop_size=256, num_iterations=100):
    if not HAS_JAX:
        return None
    
    base_key = jax.random.PRNGKey(42)
    keys = jax.vmap(lambda i: jax.random.fold_in(base_key, i))(jnp.arange(pop_size))
    
    @jax.jit
    @jax.vmap
    def generate_normal(key):
        return jax.random.normal(key, shape)
    
    for _ in range(5):
        _ = generate_normal(keys).block_until_ready()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = generate_normal(keys).block_until_ready()
    
    return (time.perf_counter() - start) / num_iterations * 1000


def benchmark_jax_full_forward(hidden_dim=64, pop_size=256, num_iterations=50):
    if not HAS_JAX:
        return None
    
    W1 = jax.random.normal(jax.random.PRNGKey(0), (hidden_dim, 32))
    W2 = jax.random.normal(jax.random.PRNGKey(1), (16, hidden_dim))
    b1 = jnp.zeros(hidden_dim)
    b2 = jnp.zeros(16)
    x = jax.random.normal(jax.random.PRNGKey(2), (pop_size, 32))
    
    rank = 4
    sigma = 0.1
    base_key = jax.random.PRNGKey(42)
    
    @jax.jit
    def forward_all():
        def single_forward(member_idx, xi):
            member_key = jax.random.fold_in(base_key, member_idx // 2)
            sign = jnp.where(member_idx % 2 == 0, 1.0, -1.0)
            current = xi
            
            # Layer 1
            m1, n1 = W1.shape
            r1 = min(rank, m1, n1)
            layer_key1 = jax.random.fold_in(member_key, 0)
            factors1 = jax.random.normal(layer_key1, (m1 + n1, r1))
            A1 = factors1[:m1] * (sigma / jnp.sqrt(r1) * sign)
            B1 = factors1[m1:]
            current = current @ W1.T + b1 + (current @ B1) @ A1.T
            current = jax.nn.relu(current)
            
            # Layer 2
            m2, n2 = W2.shape
            r2 = min(rank, m2, n2)
            layer_key2 = jax.random.fold_in(member_key, 1)
            factors2 = jax.random.normal(layer_key2, (m2 + n2, r2))
            A2 = factors2[:m2] * (sigma / jnp.sqrt(r2) * sign)
            B2 = factors2[m2:]
            current = current @ W2.T + b2 + (current @ B2) @ A2.T
            
            return current
        
        return jax.vmap(single_forward)(jnp.arange(pop_size), x)
    
    for _ in range(5):
        _ = forward_all().block_until_ready()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = forward_all().block_until_ready()
    
    return (time.perf_counter() - start) / num_iterations * 1000


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return
    
    device = torch.device('cuda')
    
    print("=" * 80)
    print("EGGROLL Optimized RNG Benchmark: Finding the fastest approach")
    print("=" * 80)
    print(f"JAX available: {HAS_JAX}")
    if HAS_JAX:
        print(f"JAX backend: {jax.default_backend()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    # 1. fold_in benchmark
    print("-" * 80)
    print("1. fold_in operation (1024 keys)")
    print("-" * 80)
    fold_in_results = benchmark_fold_in_variants(device)
    for name, time_ms in fold_in_results.items():
        print(f"   {name:25s}: {time_ms:.4f} ms")
    if HAS_JAX:
        jax_time = benchmark_jax_fold_in()
        print(f"   {'JAX vmap(fold_in)':25s}: {jax_time:.4f} ms")
        best_pytorch = min(fold_in_results.values())
        ratio = best_pytorch / jax_time
        print(f"   Best PyTorch/JAX ratio: {ratio:.2f}x {'(PyTorch faster!)' if ratio < 1 else '(JAX faster)'}")
    print()
    
    # 2. random_normal benchmark
    print("-" * 80)
    print("2. Random normal generation (256 members × 128×64 matrix)")
    print("-" * 80)
    rng_results = benchmark_random_normal_variants(device)
    for name, time_ms in rng_results.items():
        print(f"   {name:25s}: {time_ms:.4f} ms")
    if HAS_JAX:
        jax_time = benchmark_jax_random_normal()
        print(f"   {'JAX vmap(random.normal)':25s}: {jax_time:.4f} ms")
        best_pytorch = min(rng_results.values())
        ratio = best_pytorch / jax_time
        print(f"   Best PyTorch/JAX ratio: {ratio:.2f}x {'(PyTorch faster!)' if ratio < 1 else '(JAX faster)'}")
    print()
    
    # 3. Full forward benchmark (small)
    print("-" * 80)
    print("3. Full forward pass (256 population, 2-layer MLP, hidden=64)")
    print("-" * 80)
    forward_results = benchmark_full_forward_variants(device, hidden_dim=64, pop_size=256)
    for name, time_ms in forward_results.items():
        print(f"   {name:25s}: {time_ms:.4f} ms")
    if HAS_JAX:
        jax_time = benchmark_jax_full_forward(hidden_dim=64, pop_size=256)
        print(f"   {'JAX vmap + on-the-fly':25s}: {jax_time:.4f} ms")
        best_pytorch = min(forward_results.values())
        ratio = best_pytorch / jax_time
        print(f"   Best PyTorch/JAX ratio: {ratio:.2f}x {'(PyTorch faster!)' if ratio < 1 else '(JAX faster)'}")
    print()
    
    # 4. Full forward benchmark (large)
    print("-" * 80)
    print("4. Larger scale (2048 population, hidden=256)")
    print("-" * 80)
    forward_results_large = benchmark_full_forward_variants(device, hidden_dim=256, pop_size=2048, num_iterations=20)
    for name, time_ms in forward_results_large.items():
        print(f"   {name:25s}: {time_ms:.4f} ms")
    if HAS_JAX:
        jax_time = benchmark_jax_full_forward(hidden_dim=256, pop_size=2048, num_iterations=20)
        print(f"   {'JAX vmap + on-the-fly':25s}: {jax_time:.4f} ms")
        best_pytorch = min(forward_results_large.values())
        ratio = best_pytorch / jax_time
        print(f"   Best PyTorch/JAX ratio: {ratio:.2f}x {'(PyTorch faster!)' if ratio < 1 else '(JAX faster)'}")
    print()
    
    print("=" * 80)
    print("Summary of Optimizations:")
    print("=" * 80)
    print("1. Replaced vmap(_fold_in) with direct batched tensor operations")
    print("2. Replaced vmap(_random_normal) with batched version generating all at once")
    print("3. Added torch.compile for kernel fusion (similar to JAX's XLA)")
    print("4. Restructured forward pass to use bmm instead of vmap over single samples")
    print("=" * 80)


if __name__ == "__main__":
    main()
