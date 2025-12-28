"""
Benchmark: Compare RNG approaches for EGGROLL perturbation generation.

Compares:
- PyTorch custom _fold_in + _random_normal (pure tensor ops)
- JAX jax.random.fold_in + jax.random.normal (reference implementation)
- torch.Generator loop (baseline)

Also benchmarks the core operations:
- fold_in performance
- _random_normal performance  
- Full batched forward with vmap
"""

import torch
import torch.nn as nn
import time
import math
from torch.func import vmap

# Try to import JAX for comparison
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("JAX not installed - JAX benchmarks will be skipped")

# ============================================================================
# RNG Functions (copy from strategy.py for isolated benchmark)
# ============================================================================

def _fold_in(key: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """JAX-style fold_in: deterministically derive a new key from key + data."""
    mixed = (key.to(torch.int64) + data.to(torch.int64) * 2654435761) & 0xFFFFFFFF
    mixed = (mixed ^ (mixed >> 15)) & 0xFFFFFFFF
    mixed = (mixed * 2246822519) & 0xFFFFFFFF
    mixed = (mixed ^ (mixed >> 13)) & 0xFFFFFFFF
    mixed = (mixed * 3266489917) & 0xFFFFFFFF
    mixed = (mixed ^ (mixed >> 16)) & 0xFFFFFFFF
    return mixed


def _random_normal(key: torch.Tensor, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Generate normal random numbers from a key. Pure tensor ops - works inside vmap."""
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


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_fold_in(device, num_iterations=1000):
    """Benchmark fold_in operation."""
    key = torch.tensor(42, dtype=torch.int64, device=device)
    data = torch.arange(1024, dtype=torch.int64, device=device)
    
    # Warmup
    for _ in range(10):
        _ = vmap(lambda d: _fold_in(key, d))(data)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = vmap(lambda d: _fold_in(key, d))(data)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / num_iterations * 1000  # ms


def benchmark_jax_fold_in(num_iterations=1000):
    """Benchmark JAX fold_in for comparison."""
    if not HAS_JAX:
        return None
    
    key = jax.random.PRNGKey(42)
    data = jnp.arange(1024)
    
    @jax.jit
    @jax.vmap
    def fold_in_vmapped(d):
        return jax.random.fold_in(key, d)
    
    # Warmup
    for _ in range(10):
        _ = fold_in_vmapped(data).block_until_ready()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = fold_in_vmapped(data).block_until_ready()
    elapsed = time.perf_counter() - start
    
    return elapsed / num_iterations * 1000  # ms


def benchmark_random_normal(device, shape=(128, 64), num_iterations=100):
    """Benchmark _random_normal generation."""
    keys = torch.arange(256, dtype=torch.int64, device=device)
    
    # Warmup
    for _ in range(5):
        _ = vmap(lambda k: _random_normal(k, shape, torch.float32, device))(keys)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = vmap(lambda k: _random_normal(k, shape, torch.float32, device))(keys)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / num_iterations * 1000  # ms


def benchmark_jax_random_normal(shape=(128, 64), num_iterations=100):
    """Benchmark JAX random.normal for comparison."""
    if not HAS_JAX:
        return None
    
    base_key = jax.random.PRNGKey(42)
    keys = jax.vmap(lambda i: jax.random.fold_in(base_key, i))(jnp.arange(256))
    
    @jax.jit
    @jax.vmap
    def generate_normal(key):
        return jax.random.normal(key, shape)
    
    # Warmup
    for _ in range(5):
        _ = generate_normal(keys).block_until_ready()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = generate_normal(keys).block_until_ready()
    elapsed = time.perf_counter() - start
    
    return elapsed / num_iterations * 1000  # ms


def benchmark_torch_randn_generator(device, shape=(128, 64), pop_size=256, num_iterations=100):
    """Benchmark torch.randn with Generator (CPU) - for comparison."""
    # Warmup
    for _ in range(5):
        all_noise = torch.empty(pop_size, *shape, device=device)
        for i in range(pop_size):
            gen = torch.Generator().manual_seed(42 + i)
            all_noise[i] = torch.randn(shape, generator=gen).to(device)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        all_noise = torch.empty(pop_size, *shape, device=device)
        for i in range(pop_size):
            gen = torch.Generator().manual_seed(42 + i)
            all_noise[i] = torch.randn(shape, generator=gen).to(device)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / num_iterations * 1000  # ms


def benchmark_full_forward_vmap(device, hidden_dim=64, pop_size=256, num_iterations=50):
    """Benchmark full forward pass with vmap and on-the-fly RNG."""
    
    # Simple 2-layer network
    W1 = torch.randn(hidden_dim, 32, device=device)
    W2 = torch.randn(16, hidden_dim, device=device)
    x = torch.randn(pop_size, 32, device=device)
    
    rank = 4
    sigma = 0.1
    base_key = torch.tensor(42, dtype=torch.int64, device=device)
    member_ids = torch.arange(pop_size, dtype=torch.int64, device=device)
    
    def single_forward(member_key, xi):
        current = xi
        
        # Layer 1
        m1, n1 = W1.shape
        r1 = min(rank, m1, n1)
        layer_key1 = _fold_in(member_key, torch.tensor(0, dtype=torch.int64, device=device))
        factors1 = _random_normal(layer_key1, (m1 + n1, r1), xi.dtype, device)
        A1 = factors1[:m1] * (sigma / math.sqrt(r1))
        B1 = factors1[m1:]
        current = current @ W1.T + (current @ B1) @ A1.T
        current = torch.relu(current)
        
        # Layer 2
        m2, n2 = W2.shape
        r2 = min(rank, m2, n2)
        layer_key2 = _fold_in(member_key, torch.tensor(1, dtype=torch.int64, device=device))
        factors2 = _random_normal(layer_key2, (m2 + n2, r2), xi.dtype, device)
        A2 = factors2[:m2] * (sigma / math.sqrt(r2))
        B2 = factors2[m2:]
        current = current @ W2.T + (current @ B2) @ A2.T
        
        return current
    
    member_keys = vmap(lambda mid: _fold_in(base_key, mid))(member_ids)
    
    # Warmup
    for _ in range(5):
        _ = vmap(single_forward)(member_keys, x)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = vmap(single_forward)(member_keys, x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / num_iterations * 1000  # ms


def benchmark_full_forward_pregenerate(device, hidden_dim=64, pop_size=256, num_iterations=50):
    """Benchmark forward with pre-generated perturbations (no vmap on RNG)."""
    
    W1 = torch.randn(hidden_dim, 32, device=device)
    W2 = torch.randn(16, hidden_dim, device=device)
    x = torch.randn(pop_size, 32, device=device)
    
    rank = 4
    sigma = 0.1
    
    # Pre-generate all perturbations
    m1, n1 = W1.shape
    r1 = min(rank, m1, n1)
    A1_all = torch.randn(pop_size, m1, r1, device=device) * (sigma / math.sqrt(r1))
    B1_all = torch.randn(pop_size, n1, r1, device=device)
    
    m2, n2 = W2.shape
    r2 = min(rank, m2, n2)
    A2_all = torch.randn(pop_size, m2, r2, device=device) * (sigma / math.sqrt(r2))
    B2_all = torch.randn(pop_size, n2, r2, device=device)
    
    def forward_with_pregen():
        current = x
        
        # Layer 1: batched
        base1 = current @ W1.T
        xB1 = torch.bmm(current.unsqueeze(1), B1_all).squeeze(1)
        pert1 = torch.bmm(xB1.unsqueeze(1), A1_all.transpose(-1, -2)).squeeze(1)
        current = torch.relu(base1 + pert1)
        
        # Layer 2: batched
        base2 = current @ W2.T
        xB2 = torch.bmm(current.unsqueeze(1), B2_all).squeeze(1)
        pert2 = torch.bmm(xB2.unsqueeze(1), A2_all.transpose(-1, -2)).squeeze(1)
        current = base2 + pert2
        
        return current
    
    # Warmup
    for _ in range(5):
        _ = forward_with_pregen()
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = forward_with_pregen()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / num_iterations * 1000  # ms


def benchmark_jax_full_forward(hidden_dim=64, pop_size=256, num_iterations=50):
    """Benchmark JAX EGGROLL forward for comparison."""
    if not HAS_JAX:
        return None
    
    W1 = jax.random.normal(jax.random.PRNGKey(0), (hidden_dim, 32))
    W2 = jax.random.normal(jax.random.PRNGKey(1), (16, hidden_dim))
    x = jax.random.normal(jax.random.PRNGKey(2), (pop_size, 32))
    
    rank = 4
    sigma = 0.1
    base_key = jax.random.PRNGKey(42)
    
    @jax.jit
    def forward_all():
        def single_forward(member_idx, xi):
            member_key = jax.random.fold_in(base_key, member_idx)
            current = xi
            
            # Layer 1
            m1, n1 = W1.shape
            r1 = min(rank, m1, n1)
            layer_key1 = jax.random.fold_in(member_key, 0)
            factors1 = jax.random.normal(layer_key1, (m1 + n1, r1))
            A1 = factors1[:m1] * (sigma / jnp.sqrt(r1))
            B1 = factors1[m1:]
            current = current @ W1.T + (current @ B1) @ A1.T
            current = jax.nn.relu(current)
            
            # Layer 2
            m2, n2 = W2.shape
            r2 = min(rank, m2, n2)
            layer_key2 = jax.random.fold_in(member_key, 1)
            factors2 = jax.random.normal(layer_key2, (m2 + n2, r2))
            A2 = factors2[:m2] * (sigma / jnp.sqrt(r2))
            B2 = factors2[m2:]
            current = current @ W2.T + (current @ B2) @ A2.T
            
            return current
        
        return jax.vmap(single_forward)(jnp.arange(pop_size), x)
    
    # Warmup
    for _ in range(5):
        _ = forward_all().block_until_ready()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = forward_all().block_until_ready()
    elapsed = time.perf_counter() - start
    
    return elapsed / num_iterations * 1000  # ms


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return
    
    device = torch.device('cuda')
    print("=" * 70)
    print("EGGROLL RNG Approach Benchmark: PyTorch vs JAX")
    print("=" * 70)
    print(f"JAX available: {HAS_JAX}")
    if HAS_JAX:
        print(f"JAX backend: {jax.default_backend()}")
    print()
    
    # fold_in benchmark
    print("1. fold_in operation (1024 keys):")
    fold_in_time = benchmark_fold_in(device)
    print(f"   PyTorch vmap(_fold_in):   {fold_in_time:.4f} ms")
    if HAS_JAX:
        jax_fold_in_time = benchmark_jax_fold_in()
        print(f"   JAX vmap(fold_in):        {jax_fold_in_time:.4f} ms")
        ratio = fold_in_time / jax_fold_in_time
        print(f"   PyTorch/JAX ratio:        {ratio:.2f}x {'(PyTorch faster)' if ratio < 1 else '(JAX faster)'}")
    print()
    
    # random_normal benchmark
    print("2. Random normal generation (256 members × 128×64 matrix):")
    rng_vmap_time = benchmark_random_normal(device, shape=(128, 64))
    rng_generator_time = benchmark_torch_randn_generator(device, shape=(128, 64), pop_size=256)
    print(f"   PyTorch vmap(_random_normal): {rng_vmap_time:.4f} ms")
    print(f"   torch.Generator loop:         {rng_generator_time:.4f} ms")
    if HAS_JAX:
        jax_rng_time = benchmark_jax_random_normal(shape=(128, 64))
        print(f"   JAX vmap(random.normal):      {jax_rng_time:.4f} ms")
        ratio = rng_vmap_time / jax_rng_time
        print(f"   PyTorch/JAX ratio:            {ratio:.2f}x {'(PyTorch faster)' if ratio < 1 else '(JAX faster)'}")
    print(f"   Speedup vs Generator loop:    {rng_generator_time / rng_vmap_time:.1f}x")
    print()
    
    # Full forward benchmark
    print("3. Full forward pass (256 population, 2-layer MLP):")
    vmap_forward_time = benchmark_full_forward_vmap(device, hidden_dim=64, pop_size=256)
    pregen_forward_time = benchmark_full_forward_pregenerate(device, hidden_dim=64, pop_size=256)
    print(f"   PyTorch vmap + on-the-fly:    {vmap_forward_time:.4f} ms")
    print(f"   PyTorch pre-gen + bmm:        {pregen_forward_time:.4f} ms")
    if HAS_JAX:
        jax_forward_time = benchmark_jax_full_forward(hidden_dim=64, pop_size=256)
        print(f"   JAX vmap + on-the-fly:        {jax_forward_time:.4f} ms")
        ratio = vmap_forward_time / jax_forward_time
        print(f"   PyTorch/JAX ratio:            {ratio:.2f}x {'(PyTorch faster)' if ratio < 1 else '(JAX faster)'}")
    print()
    
    # Larger scale test
    print("4. Larger scale (2048 population, hidden=256):")
    vmap_large = benchmark_full_forward_vmap(device, hidden_dim=256, pop_size=2048, num_iterations=20)
    pregen_large = benchmark_full_forward_pregenerate(device, hidden_dim=256, pop_size=2048, num_iterations=20)
    print(f"   PyTorch vmap + on-the-fly:    {vmap_large:.4f} ms")
    print(f"   PyTorch pre-gen + bmm:        {pregen_large:.4f} ms")
    if HAS_JAX:
        jax_large = benchmark_jax_full_forward(hidden_dim=256, pop_size=2048, num_iterations=20)
        print(f"   JAX vmap + on-the-fly:        {jax_large:.4f} ms")
        ratio = vmap_large / jax_large
        print(f"   PyTorch/JAX ratio:            {ratio:.2f}x {'(PyTorch faster)' if ratio < 1 else '(JAX faster)'}")
    print()
    
    print("=" * 70)
    print("Summary:")
    print("- PyTorch vmap + on-the-fly RNG: Memory efficient, no pre-allocation")
    print("- Pre-generated: May be faster for small models, but uses more memory")
    if HAS_JAX:
        print("- JAX comparison: Reference implementation performance baseline")
    print("=" * 70)


if __name__ == "__main__":
    main()
