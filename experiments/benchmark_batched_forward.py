#!/usr/bin/env python3
"""
Benchmark script for batched forward pass performance.

Tests batched_forward throughput at different population sizes.
Compares PyTorch EGGROLL vs JAX EGGROLL implementation.

Run with: python experiments/benchmark_batched_forward.py
"""

import torch
import torch.nn as nn
import time
import sys
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyperscalees.torch import EggrollStrategy, EggrollConfig

# Try importing JAX for comparison
try:
    import jax
    import jax.numpy as jnp
    from jax import random, vmap, jit
    HAS_JAX = True
    JAX_BACKEND = jax.default_backend()
except ImportError:
    HAS_JAX = False
    JAX_BACKEND = None


def create_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, device: str):
    """Create a simple MLP for benchmarking."""
    layers = []
    in_d = input_dim
    for i in range(num_layers - 1):
        layers.append(nn.Linear(in_d, hidden_dim))
        layers.append(nn.ReLU())
        in_d = hidden_dim
    layers.append(nn.Linear(in_d, output_dim))
    return nn.Sequential(*layers).to(device)


# =============================================================================
# JAX EGGROLL Implementation (for comparison)
# =============================================================================

def create_jax_mlp_params(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, key):
    """Create JAX MLP parameters matching PyTorch structure."""
    params = []
    in_d = input_dim
    for i in range(num_layers - 1):
        key, k1, k2 = random.split(key, 3)
        W = random.normal(k1, (in_d, hidden_dim)) * 0.01
        b = jnp.zeros(hidden_dim)
        params.append({'W': W, 'b': b})
        in_d = hidden_dim
    key, k1, k2 = random.split(key, 3)
    W = random.normal(k1, (in_d, output_dim)) * 0.01
    b = jnp.zeros(output_dim)
    params.append({'W': W, 'b': b})
    return params


def jax_mlp_forward(params, x):
    """Forward pass through JAX MLP."""
    for i, layer in enumerate(params[:-1]):
        x = x @ layer['W'] + layer['b']
        x = jax.nn.relu(x)
    x = x @ params[-1]['W'] + params[-1]['b']
    return x


def jax_generate_lowrank_factors(key, shape, rank):
    """Generate low-rank factors for a parameter."""
    k1, k2 = random.split(key)
    # A: (rank, num_params), B: (rank,)
    A = random.normal(k1, (rank, shape[0])) / jnp.sqrt(rank)
    B = random.normal(k2, (rank,))
    return A, B


def jax_perturb_params_single(params, key, sigma, rank):
    """Perturb parameters for a single population member."""
    perturbed = []
    for i, layer in enumerate(params):
        layer_perturbed = {}
        for name, param in layer.items():
            key, subkey = random.split(key)
            flat = param.flatten()
            A, B = jax_generate_lowrank_factors(subkey, (flat.shape[0],), rank)
            # Low-rank perturbation: sigma * B @ A
            delta = sigma * jnp.dot(B, A)
            layer_perturbed[name] = param + delta.reshape(param.shape)
        perturbed.append(layer_perturbed)
    return perturbed


def jax_forward_single(params, x, key, sigma, rank):
    """Forward pass with perturbed parameters for one member."""
    perturbed = jax_perturb_params_single(params, key, sigma, rank)
    return jax_mlp_forward(perturbed, x)


# =============================================================================
# PyTorch Benchmark Functions
# =============================================================================

def benchmark_pytorch_batched_forward(
    model: nn.Module,
    strategy: EggrollStrategy,
    pop_size: int,
    input_dim: int,
    device: str,
    num_iterations: int = 100,
    warmup: int = 10
):
    """Benchmark the batched_forward method."""
    x = torch.randn(pop_size, input_dim, device=device)
    
    # Warmup
    for i in range(warmup):
        with strategy.perturb(pop_size, epoch=i) as pop:
            _ = pop.batched_forward(model, x)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for i in range(num_iterations):
        with strategy.perturb(pop_size, epoch=i) as pop:
            outputs = pop.batched_forward(model, x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / num_iterations, outputs.shape


def benchmark_pytorch_with_breakdown(
    model: nn.Module,
    strategy: EggrollStrategy,
    pop_size: int,
    input_dim: int,
    device: str,
    num_iterations: int = 100,
    warmup: int = 10
):
    """Benchmark with timing breakdown to identify bottlenecks."""
    x = torch.randn(pop_size, input_dim, device=device)
    
    timings = defaultdict(float)
    
    # Warmup
    for i in range(warmup):
        with strategy.perturb(pop_size, epoch=i) as pop:
            _ = pop.batched_forward(model, x)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Detailed timing
    for i in range(num_iterations):
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Time context manager entry (key generation)
        t0 = time.perf_counter()
        ctx = strategy.perturb(pop_size, epoch=i)
        pop = ctx.__enter__()
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings['context_enter'] += t1 - t0
        
        # Time batched_forward
        t0 = time.perf_counter()
        outputs = pop.batched_forward(model, x)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings['batched_forward'] += t1 - t0
        
        # Time context exit
        t0 = time.perf_counter()
        ctx.__exit__(None, None, None)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings['context_exit'] += t1 - t0
    
    # Average
    return {k: v / num_iterations * 1000 for k, v in timings.items()}


# =============================================================================
# JAX Benchmark Functions  
# =============================================================================

def benchmark_jax_batched_forward(
    params,
    input_dim: int,
    pop_size: int,
    sigma: float,
    rank: int,
    num_iterations: int = 100,
    warmup: int = 10
):
    """Benchmark JAX EGGROLL-equivalent batched forward."""
    if not HAS_JAX:
        return None, None
    
    key = random.PRNGKey(42)
    x = random.normal(key, (input_dim,))  # Single input, vmapped over pop
    
    # Create forward function with sigma/rank captured in closure
    def make_batched_forward(sigma_val, rank_val):
        def forward_one(params, x, k):
            return jax_forward_single(params, x, k, sigma_val, rank_val)
        
        @jit
        def batched_forward(params, x, keys):
            return vmap(lambda k: forward_one(params, x, k))(keys)
        return batched_forward
    
    batched_forward = make_batched_forward(sigma, rank)
    
    # Warmup
    for i in range(warmup):
        key, subkey = random.split(key)
        keys = random.split(subkey, pop_size)
        _ = batched_forward(params, x, keys)
        _.block_until_ready()
    
    # Benchmark
    start = time.perf_counter()
    for i in range(num_iterations):
        key, subkey = random.split(key)
        keys = random.split(subkey, pop_size)
        outputs = batched_forward(params, x, keys)
        outputs.block_until_ready()
    elapsed = time.perf_counter() - start
    
    return elapsed / num_iterations, outputs.shape


def benchmark_jax_with_breakdown(
    params,
    input_dim: int,
    pop_size: int,
    sigma: float,
    rank: int,
    num_iterations: int = 100,
    warmup: int = 10
):
    """JAX benchmark with timing breakdown."""
    if not HAS_JAX:
        return None
    
    key = random.PRNGKey(42)
    x = random.normal(key, (input_dim,))
    
    @jit
    def split_keys(key, n):
        return random.split(key, n + 1)
    
    # Create forward function with sigma/rank captured in closure
    def make_batched_forward(sigma_val, rank_val):
        def forward_one(params, x, k):
            return jax_forward_single(params, x, k, sigma_val, rank_val)
        
        @jit
        def batched_forward(params, x, keys):
            return vmap(lambda k: forward_one(params, x, k))(keys)
        return batched_forward
    
    batched_forward = make_batched_forward(sigma, rank)
    
    timings = defaultdict(float)
    
    # Warmup
    for i in range(warmup):
        key, subkey = random.split(key)
        keys = random.split(subkey, pop_size)
        _ = batched_forward(params, x, keys)
        _.block_until_ready()
    
    # Detailed timing
    for i in range(num_iterations):
        # Time key generation
        t0 = time.perf_counter()
        split_result = split_keys(key, pop_size)
        split_result.block_until_ready()
        key = split_result[0]
        keys = split_result[1:]
        t1 = time.perf_counter()
        timings['key_generation'] += t1 - t0
        
        # Time forward pass
        t0 = time.perf_counter()
        outputs = batched_forward(params, x, keys)
        outputs.block_until_ready()
        t1 = time.perf_counter()
        timings['batched_forward'] += t1 - t0
    
    return {k: v / num_iterations * 1000 for k, v in timings.items()}


# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    print("=" * 70)
    print("EGGROLL Batched Forward Benchmark: PyTorch vs JAX")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nPyTorch Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    print(f"JAX Available: {HAS_JAX}")
    if HAS_JAX:
        print(f"JAX Backend: {JAX_BACKEND}")
    
    # Configuration
    configs = [
        {"pop_size": 256, "input_dim": 64, "hidden_dim": 64, "output_dim": 32, "num_layers": 2, "rank": 4},
        {"pop_size": 512, "input_dim": 128, "hidden_dim": 128, "output_dim": 64, "num_layers": 3, "rank": 4},
        {"pop_size": 2048, "input_dim": 128, "hidden_dim": 256, "output_dim": 64, "num_layers": 3, "rank": 4},
    ]
    
    sigma = 0.1
    num_iterations = 100
    
    print(f"\nRunning {num_iterations} iterations per benchmark")
    
    for cfg in configs:
        pop_size = cfg["pop_size"]
        input_dim = cfg["input_dim"]
        hidden_dim = cfg["hidden_dim"]
        output_dim = cfg["output_dim"]
        num_layers = cfg["num_layers"]
        rank = cfg["rank"]
        
        print("\n" + "-" * 70)
        print(f"Config: pop={pop_size}, input={input_dim}, hidden={hidden_dim}, "
              f"output={output_dim}, layers={num_layers}, rank={rank}")
        print("-" * 70)
        
        # Create PyTorch model and strategy
        model = create_mlp(input_dim, hidden_dim, output_dim, num_layers, device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
        
        config = EggrollConfig(sigma=sigma, lr=0.01, rank=rank, antithetic=True, seed=42)
        strategy = EggrollStrategy.from_config(config)
        strategy.setup(model)
        
        # PyTorch benchmark
        pt_time, pt_shape = benchmark_pytorch_batched_forward(
            model, strategy, pop_size, input_dim, device, num_iterations
        )
        print(f"\n  PyTorch batched_forward: {pt_time*1000:.3f} ms")
        print(f"    Throughput: {pop_size / pt_time:,.0f} evals/sec")
        
        # PyTorch timing breakdown
        pt_breakdown = benchmark_pytorch_with_breakdown(
            model, strategy, pop_size, input_dim, device, num_iterations
        )
        print(f"    Breakdown:")
        for name, ms in pt_breakdown.items():
            print(f"      {name}: {ms:.3f} ms")
        
        # JAX benchmark
        if HAS_JAX:
            key = random.PRNGKey(42)
            jax_params = create_jax_mlp_params(input_dim, hidden_dim, output_dim, num_layers, key)
            
            jax_time, jax_shape = benchmark_jax_batched_forward(
                jax_params, input_dim, pop_size, sigma, rank, num_iterations
            )
            print(f"\n  JAX batched_forward: {jax_time*1000:.3f} ms")
            print(f"    Throughput: {pop_size / jax_time:,.0f} evals/sec")
            
            # JAX timing breakdown
            jax_breakdown = benchmark_jax_with_breakdown(
                jax_params, input_dim, pop_size, sigma, rank, num_iterations
            )
            print(f"    Breakdown:")
            for name, ms in jax_breakdown.items():
                print(f"      {name}: {ms:.3f} ms")
            
            # Comparison
            ratio = pt_time / jax_time
            print(f"\n  → PyTorch/JAX ratio: {ratio:.2f}x")
            
            # Identify the bottleneck
            pt_forward = pt_breakdown['batched_forward']
            pt_overhead = pt_breakdown['context_enter'] + pt_breakdown['context_exit']
            jax_forward = jax_breakdown['batched_forward']
            jax_overhead = jax_breakdown['key_generation']
            
            print(f"\n  Analysis:")
            print(f"    PyTorch overhead (ctx enter+exit): {pt_overhead:.3f} ms")
            print(f"    JAX overhead (key generation):     {jax_overhead:.3f} ms")
            print(f"    PyTorch forward only:              {pt_forward:.3f} ms")
            print(f"    JAX forward only:                  {jax_forward:.3f} ms")
            print(f"    Forward-only ratio:                {pt_forward/jax_forward:.2f}x")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("=" * 70)
    print("""
- 'context_enter' includes key derivation (fold_in) for all members
- 'batched_forward' includes: noise generation + perturbation + forward
- JAX fuses the entire forward (perturbation + matmul) into one XLA kernel
- PyTorch launches separate kernels for: noise gen, perturbation, each matmul

If 'batched_forward' dominates, the bottleneck is kernel launch overhead.
If 'context_enter' dominates, the bottleneck is key generation.

Solutions for kernel launch overhead:
1. torch.compile with fullgraph=True (if model is static)
2. CUDA graphs to capture and replay kernel sequences
3. Custom Triton kernel to fuse perturbation + matmul
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
