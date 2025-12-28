"""
Benchmark: EGGROLL PyTorch vs JAX RNG and Forward Pass Performance.

Compares the optimized PyTorch implementation against JAX baseline.
All PyTorch RNG functions are imported from strategy.py (single source of truth).
"""

import torch
import torch.nn as nn
import time
import math
from typing import Tuple, Dict, Any
from pathlib import Path

# Import RNG functions from the actual implementation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from hyperscalees.torch.strategy import (
    _fold_in,
    _random_normal,
    _random_normal_batched,
    _generate_lowrank_factors_batched,
)

# Try to import JAX for comparison
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("JAX not installed - JAX benchmarks will be skipped")


# ==============================================================================
# BENCHMARK FUNCTIONS
# ==============================================================================

def benchmark_fold_in(device, num_iterations=1000) -> Dict[str, float]:
    """Benchmark fold_in: batched PyTorch vs JAX."""
    key = torch.tensor(42, dtype=torch.int64, device=device)
    data = torch.arange(1024, dtype=torch.int64, device=device)
    
    results = {}
    
    # PyTorch batched (the optimized version - fold_in already supports batching)
    for _ in range(10):
        _ = _fold_in(key, data)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = _fold_in(key, data)
    torch.cuda.synchronize()
    results['pytorch_batched'] = (time.perf_counter() - start) / num_iterations * 1000
    
    # PyTorch compiled
    fold_in_compiled = torch.compile(_fold_in, mode="reduce-overhead")
    for _ in range(20):
        _ = fold_in_compiled(key, data)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = fold_in_compiled(key, data)
    torch.cuda.synchronize()
    results['pytorch_compiled'] = (time.perf_counter() - start) / num_iterations * 1000
    
    return results


def benchmark_random_normal(device, shape=(128, 64), pop_size=256, num_iterations=100) -> Dict[str, float]:
    """Benchmark random normal generation: batched PyTorch vs JAX."""
    keys = torch.arange(pop_size, dtype=torch.int64, device=device)
    
    results = {}
    
    # PyTorch batched
    for _ in range(5):
        _ = _random_normal_batched(keys, shape, torch.float32, device)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = _random_normal_batched(keys, shape, torch.float32, device)
    torch.cuda.synchronize()
    results['pytorch_batched'] = (time.perf_counter() - start) / num_iterations * 1000
    
    # PyTorch compiled
    rng_compiled = torch.compile(_random_normal_batched, mode="reduce-overhead")
    for _ in range(20):
        _ = rng_compiled(keys, shape, torch.float32, device)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = rng_compiled(keys, shape, torch.float32, device)
    torch.cuda.synchronize()
    results['pytorch_compiled'] = (time.perf_counter() - start) / num_iterations * 1000
    
    return results


def benchmark_full_forward(device, hidden_dim=64, pop_size=256, num_iterations=50) -> Dict[str, float]:
    """Benchmark full forward pass with perturbations."""
    
    # Simple 2-layer MLP
    model = nn.Sequential(
        nn.Linear(32, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 16),
    ).to(device)
    model.eval()
    
    x = torch.randn(pop_size, 32, device=device)
    member_ids = torch.arange(pop_size, dtype=torch.int64, device=device)
    base_key = torch.tensor(42, dtype=torch.int64, device=device)
    
    rank = 4
    sigma = 0.1
    antithetic = True
    
    # Get layer info
    layers = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    
    def forward_batched(x, member_ids, base_key):
        """Forward pass using batched RNG."""
        current = x
        for i, (name, module) in enumerate(layers):
            W = module.weight
            bias = module.bias
            m, n_in = W.shape
            
            # Base computation
            out = current @ W.T
            if bias is not None:
                out = out + bias
            
            # Generate and apply perturbation
            A, B = _generate_lowrank_factors_batched(
                base_key, member_ids, i, m, n_in, rank, sigma,
                current.dtype, current.device, antithetic
            )
            
            # Perturbation: x @ B @ A.T using einsum for potential fusion
            out = out + torch.einsum('bi,bir,bmr->bm', current, B, A)
            
            current = out
            
            # Apply ReLU after first layer
            if i == 0:
                current = torch.relu(current)
        
        return current
    
    results = {}
    
    # PyTorch batched
    for _ in range(5):
        _ = forward_batched(x, member_ids, base_key)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = forward_batched(x, member_ids, base_key)
    torch.cuda.synchronize()
    results['pytorch_batched'] = (time.perf_counter() - start) / num_iterations * 1000
    
    # PyTorch compiled (reduce-overhead)
    forward_compiled = torch.compile(forward_batched, mode="reduce-overhead")
    for _ in range(20):
        _ = forward_compiled(x, member_ids, base_key)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = forward_compiled(x, member_ids, base_key)
    torch.cuda.synchronize()
    results['pytorch_compiled'] = (time.perf_counter() - start) / num_iterations * 1000
    
    # PyTorch max-autotune (more aggressive optimization)
    forward_autotune = torch.compile(forward_batched, mode="max-autotune")
    for _ in range(30):
        _ = forward_autotune(x, member_ids, base_key)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = forward_autotune(x, member_ids, base_key)
    torch.cuda.synchronize()
    results['pytorch_max_autotune'] = (time.perf_counter() - start) / num_iterations * 1000
    
    return results


# ==============================================================================
# JAX BENCHMARKS
# ==============================================================================

def benchmark_jax_fold_in(num_iterations=1000):
    if not HAS_JAX:
        return None
    
    key = jax.random.PRNGKey(42)
    data = jnp.arange(1024)
    
    @jax.jit
    def fold_in_batched(data):
        return jax.vmap(lambda d: jax.random.fold_in(key, d))(data)
    
    for _ in range(10):
        _ = fold_in_batched(data).block_until_ready()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = fold_in_batched(data).block_until_ready()
    
    return (time.perf_counter() - start) / num_iterations * 1000


def benchmark_jax_random_normal(shape=(128, 64), pop_size=256, num_iterations=100):
    if not HAS_JAX:
        return None
    
    base_key = jax.random.PRNGKey(42)
    indices = jnp.arange(pop_size)
    
    @jax.jit
    def generate_all(indices):
        keys = jax.vmap(lambda i: jax.random.fold_in(base_key, i))(indices)
        return jax.vmap(lambda k: jax.random.normal(k, shape))(keys)
    
    for _ in range(5):
        _ = generate_all(indices).block_until_ready()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = generate_all(indices).block_until_ready()
    
    return (time.perf_counter() - start) / num_iterations * 1000


def benchmark_jax_full_forward(hidden_dim=64, pop_size=256, num_iterations=50):
    if not HAS_JAX:
        return None
    
    # Initialize weights
    W1 = jax.random.normal(jax.random.PRNGKey(0), (hidden_dim, 32))
    W2 = jax.random.normal(jax.random.PRNGKey(1), (16, hidden_dim))
    b1 = jnp.zeros(hidden_dim)
    b2 = jnp.zeros(16)
    x = jax.random.normal(jax.random.PRNGKey(2), (pop_size, 32))
    
    rank = 4
    sigma = 0.1
    base_key = jax.random.PRNGKey(42)
    
    @jax.jit
    def forward_all(x):
        def single_forward(member_idx, xi):
            effective_idx = member_idx // 2
            sign = jnp.where(member_idx % 2 == 0, 1.0, -1.0)
            member_key = jax.random.fold_in(base_key, effective_idx)
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
        _ = forward_all(x).block_until_ready()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = forward_all(x).block_until_ready()
    
    return (time.perf_counter() - start) / num_iterations * 1000


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return
    
    device = torch.device('cuda')
    
    print("=" * 70)
    print("EGGROLL RNG Benchmark: PyTorch vs JAX")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"JAX available: {HAS_JAX}")
    if HAS_JAX:
        print(f"JAX backend: {jax.default_backend()}")
    print()
    
    # 1. fold_in benchmark
    print("-" * 70)
    print("1. fold_in operation (1024 keys)")
    print("-" * 70)
    pt_results = benchmark_fold_in(device)
    for name, ms in pt_results.items():
        print(f"   {name:25s}: {ms:.4f} ms")
    if HAS_JAX:
        jax_ms = benchmark_jax_fold_in()
        print(f"   {'jax_jit_vmap':25s}: {jax_ms:.4f} ms")
        best = min(pt_results.values())
        print(f"   → Best PyTorch/JAX: {best/jax_ms:.2f}x")
    print()
    
    # 2. random_normal benchmark
    print("-" * 70)
    print("2. Random normal (256 members × 128×64)")
    print("-" * 70)
    pt_results = benchmark_random_normal(device)
    for name, ms in pt_results.items():
        print(f"   {name:25s}: {ms:.4f} ms")
    if HAS_JAX:
        jax_ms = benchmark_jax_random_normal()
        print(f"   {'jax_jit_vmap':25s}: {jax_ms:.4f} ms")
        best = min(pt_results.values())
        print(f"   → Best PyTorch/JAX: {best/jax_ms:.2f}x")
    print()
    
    # 3. Full forward (small)
    print("-" * 70)
    print("3. Full forward (256 pop, hidden=64)")
    print("-" * 70)
    pt_results = benchmark_full_forward(device, hidden_dim=64, pop_size=256)
    for name, ms in pt_results.items():
        print(f"   {name:25s}: {ms:.4f} ms")
    if HAS_JAX:
        jax_ms = benchmark_jax_full_forward(hidden_dim=64, pop_size=256)
        print(f"   {'jax_jit_vmap':25s}: {jax_ms:.4f} ms")
        best = min(pt_results.values())
        print(f"   → Best PyTorch/JAX: {best/jax_ms:.2f}x")
    print()
    
    # 4. Full forward (large)
    print("-" * 70)
    print("4. Full forward (2048 pop, hidden=256)")
    print("-" * 70)
    pt_results = benchmark_full_forward(device, hidden_dim=256, pop_size=2048, num_iterations=20)
    for name, ms in pt_results.items():
        print(f"   {name:25s}: {ms:.4f} ms")
    if HAS_JAX:
        jax_ms = benchmark_jax_full_forward(hidden_dim=256, pop_size=2048, num_iterations=20)
        print(f"   {'jax_jit_vmap':25s}: {jax_ms:.4f} ms")
        best = min(pt_results.values())
        print(f"   → Best PyTorch/JAX: {best/jax_ms:.2f}x")
    print()
    
    print("=" * 70)
    print("Notes:")
    print("- pytorch_batched: Uses _random_normal_batched (no vmap)")
    print("- pytorch_compiled: + torch.compile(mode='reduce-overhead')")
    print("- pytorch_max_autotune: + torch.compile(mode='max-autotune')")
    print("- jax_jit_vmap: JAX with @jax.jit + vmap (XLA compiled)")
    print("=" * 70)


if __name__ == "__main__":
    main()
