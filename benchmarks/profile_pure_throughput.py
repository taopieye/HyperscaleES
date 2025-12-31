#!/usr/bin/env python3
"""
Pure GPU Throughput Profiler - No Environment Bottleneck

This isolates just the neural network forward pass (with perturbations)
to measure true GPU throughput without CPU environment overhead.
"""
import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict
from pathlib import Path

import torch
import numpy as np


@dataclass 
class ThroughputResult:
    """Results from pure GPU throughput test."""
    method: str
    batch_size: int
    pop_size: int
    num_iterations: int
    total_forward_passes: int
    wall_time_s: float
    forward_passes_per_second: float
    samples_per_second: float  # forward_passes * pop_size
    mean_latency_ms: float
    memory_used_mb: float
    notes: List[str]


def profile_torch_pure_throughput(
    num_iterations: int = 1000,
    pop_size: int = 2048,
    batch_size: int = 1,  # Single observation per forward
    obs_dim: int = 4,
    act_dim: int = 2,
    layer_size: int = 256,
    n_layers: int = 3,
    rank: int = 4,
) -> ThroughputResult:
    """Profile pure forward pass throughput for PyTorch EGGROLL."""
    print("\n" + "="*60)
    print("PyTorch EGGROLL - Pure Forward Pass Throughput")
    print("="*60)
    
    from hyperscalees.torch import EggrollStrategy
    
    notes = []
    
    # Build network
    layers = []
    in_dim = obs_dim
    for _ in range(n_layers - 1):
        layers.append(torch.nn.Linear(in_dim, layer_size))
        layers.append(torch.nn.Tanh())
        in_dim = layer_size
    layers.append(torch.nn.Linear(in_dim, act_dim))
    
    model = torch.nn.Sequential(*layers).cuda()
    
    # Create strategy
    strategy = EggrollStrategy(
        sigma=0.2,
        rank=rank,
        lr=0.1,
        seed=42,
        antithetic=True,
    )
    strategy.setup(model)
    
    # Create fixed input (no CPU-GPU transfer overhead in loop)
    obs_batch = torch.randn(pop_size, obs_dim, device='cuda', dtype=torch.float32)
    
    # Warmup
    print("Warming up...")
    for warmup_epoch in range(10):
        with strategy.perturb(population_size=pop_size, epoch=warmup_epoch) as pop:
            with torch.no_grad():
                _ = pop.batched_forward(model, obs_batch)
    torch.cuda.synchronize()
    
    # Measure memory
    torch.cuda.reset_peak_memory_stats()
    
    # Time the forward passes
    print(f"Running {num_iterations} forward passes with pop_size={pop_size}...")
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for epoch in range(num_iterations):
        with strategy.perturb(population_size=pop_size, epoch=epoch) as pop:
            with torch.no_grad():
                logits = pop.batched_forward(model, obs_batch)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    wall_time = end_time - start_time
    memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    return ThroughputResult(
        method="EGGROLL (Torch)",
        batch_size=batch_size,
        pop_size=pop_size,
        num_iterations=num_iterations,
        total_forward_passes=num_iterations,
        wall_time_s=wall_time,
        forward_passes_per_second=num_iterations / wall_time,
        samples_per_second=(num_iterations * pop_size) / wall_time,
        mean_latency_ms=(wall_time / num_iterations) * 1000,
        memory_used_mb=memory_used,
        notes=notes,
    )


def profile_jax_pure_throughput(
    num_iterations: int = 1000,
    pop_size: int = 2048,
    batch_size: int = 1,
    obs_dim: int = 4,
    act_dim: int = 2,
    layer_size: int = 256,
    n_layers: int = 3,
    rank: int = 4,
) -> ThroughputResult:
    """Profile pure forward pass throughput for JAX EGGROLL."""
    print("\n" + "="*60)
    print("JAX EGGROLL - Pure Forward Pass Throughput")
    print("="*60)
    
    import jax
    import jax.numpy as jnp
    import optax
    from hyperscalees.noiser.eggroll import EggRoll
    from hyperscalees.models.common import MLP, simple_es_tree_key
    
    notes = []
    notes.append(f"JAX version: {jax.__version__}")
    notes.append(f"XLA backend: {jax.devices()[0].platform}")
    
    # Initialize model
    key = jax.random.key(42)
    model_key = jax.random.fold_in(key, 0)
    es_key = jax.random.fold_in(key, 1)
    
    frozen_params, params, scan_map, es_map = MLP.rand_init(
        model_key,
        in_dim=obs_dim,
        out_dim=act_dim,
        hidden_dims=[layer_size] * (n_layers - 1),
        use_bias=True,
        activation="pqn",
        dtype="float32",
    )
    
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)
    
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params,
        sigma=0.2,
        lr=0.1,
        solver=optax.sgd,
        rank=rank,
        noise_reuse=0,
    )
    
    # JIT-compile forward function
    def forward_noisy(noiser_params, params, iterinfo, obs):
        return MLP.forward(
            EggRoll, frozen_noiser_params, noiser_params, frozen_params,
            params, es_tree_key, iterinfo, obs
        )
    
    jit_forward = jax.jit(jax.vmap(
        lambda n, p, i, x: forward_noisy(n, p, i, x),
        in_axes=(None, None, 0, 0)
    ))
    
    # Create fixed input
    obs_batch = jnp.ones((pop_size, obs_dim), dtype=jnp.float32)
    
    def make_iterinfos(epoch):
        return jnp.stack([
            jnp.array([epoch, member_id], dtype=jnp.int32)
            for member_id in range(pop_size)
        ])
    
    # Warmup
    print("Warming up (JIT compilation)...")
    for warmup_epoch in range(10):
        iterinfos = make_iterinfos(warmup_epoch)
        _ = jit_forward(noiser_params, params, iterinfos, obs_batch)
    _ = jit_forward(noiser_params, params, iterinfos, obs_batch).block_until_ready()
    
    # Time the forward passes
    print(f"Running {num_iterations} forward passes with pop_size={pop_size}...")
    
    start_time = time.perf_counter()
    
    for epoch in range(num_iterations):
        iterinfos = make_iterinfos(epoch)
        logits = jit_forward(noiser_params, params, iterinfos, obs_batch)
    
    logits.block_until_ready()
    end_time = time.perf_counter()
    
    wall_time = end_time - start_time
    
    # Get memory (approximate via device memory stats)
    try:
        memory_stats = jax.devices()[0].memory_stats()
        memory_used = memory_stats.get('peak_bytes_in_use', 0) / (1024 * 1024)
    except:
        memory_used = 0
    
    return ThroughputResult(
        method="EGGROLL (JAX)",
        batch_size=batch_size,
        pop_size=pop_size,
        num_iterations=num_iterations,
        total_forward_passes=num_iterations,
        wall_time_s=wall_time,
        forward_passes_per_second=num_iterations / wall_time,
        samples_per_second=(num_iterations * pop_size) / wall_time,
        mean_latency_ms=(wall_time / num_iterations) * 1000,
        memory_used_mb=memory_used,
        notes=notes,
    )


def print_comparison(torch_result: ThroughputResult, jax_result: ThroughputResult):
    """Print side-by-side comparison."""
    print("\n" + "="*80)
    print("PURE GPU THROUGHPUT COMPARISON (No Environment Overhead)")
    print("="*80)
    
    print(f"\n{'Metric':<35} {'PyTorch':>15} {'JAX':>15} {'Ratio':>12}")
    print("-"*80)
    print(f"{'Forward passes/second':<35} {torch_result.forward_passes_per_second:>15.0f} {jax_result.forward_passes_per_second:>15.0f} {torch_result.forward_passes_per_second/jax_result.forward_passes_per_second:>11.2f}x")
    print(f"{'Samples/second (pop×forward)':<35} {torch_result.samples_per_second:>15,.0f} {jax_result.samples_per_second:>15,.0f} {torch_result.samples_per_second/jax_result.samples_per_second:>11.2f}x")
    print(f"{'Mean latency (ms)':<35} {torch_result.mean_latency_ms:>15.3f} {jax_result.mean_latency_ms:>15.3f} {jax_result.mean_latency_ms/torch_result.mean_latency_ms:>11.2f}x")
    print(f"{'Wall time (s)':<35} {torch_result.wall_time_s:>15.3f} {jax_result.wall_time_s:>15.3f}")
    print(f"{'Peak memory (MB)':<35} {torch_result.memory_used_mb:>15.0f} {jax_result.memory_used_mb:>15.0f}")
    
    print("\n### Analysis ###")
    ratio = torch_result.forward_passes_per_second / jax_result.forward_passes_per_second
    if ratio > 1.1:
        print(f"✅ PyTorch is {ratio:.1f}x faster than JAX for pure forward passes")
    elif ratio < 0.9:
        print(f"❌ JAX is {1/ratio:.1f}x faster than PyTorch for pure forward passes")
    else:
        print(f"≈ PyTorch and JAX have similar throughput (ratio: {ratio:.2f}x)")
    
    if torch_result.memory_used_mb < jax_result.memory_used_mb * 0.5:
        print(f"✅ PyTorch uses {jax_result.memory_used_mb/torch_result.memory_used_mb:.1f}x less memory")


def main():
    parser = argparse.ArgumentParser(description="Pure GPU throughput profiler")
    parser.add_argument("--num-iterations", type=int, default=1000, help="Number of forward passes")
    parser.add_argument("--pop-size", type=int, default=2048, help="Population size")
    parser.add_argument("--layer-size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--rank", type=int, default=4, help="Low-rank dimension")
    parser.add_argument("--output-dir", type=str, default="benchmarks", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("EGGROLL Pure GPU Throughput Profiler")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Iterations: {args.num_iterations}")
    print(f"  Population size: {args.pop_size}")
    print(f"  Layer size: {args.layer_size}")
    print(f"  N layers: {args.n_layers}")
    print(f"  Rank: {args.rank}")
    print(f"  obs_dim=4, act_dim=2 (CartPole)")
    
    # Profile PyTorch
    torch_result = profile_torch_pure_throughput(
        num_iterations=args.num_iterations,
        pop_size=args.pop_size,
        layer_size=args.layer_size,
        n_layers=args.n_layers,
        rank=args.rank,
    )
    
    # Clear GPU
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(1)
    
    # Profile JAX
    jax_result = profile_jax_pure_throughput(
        num_iterations=args.num_iterations,
        pop_size=args.pop_size,
        layer_size=args.layer_size,
        n_layers=args.n_layers,
        rank=args.rank,
    )
    
    # Print comparison
    print_comparison(torch_result, jax_result)
    
    # Save results
    results = {
        "torch": asdict(torch_result),
        "jax": asdict(jax_result),
        "config": vars(args),
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"pure_throughput_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")


if __name__ == "__main__":
    main()
