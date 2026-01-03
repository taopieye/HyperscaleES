#!/usr/bin/env python3
"""
Head-to-Head Empirical Profiling: JAX vs PyTorch EGGROLL

This script provides rigorous empirical comparison to inform optimization decisions.
No guessing - just data.

Profiles:
1. Forward pass throughput at various population sizes
2. Maximum population size (memory limit)
3. Per-operation breakdown (noise gen, matmul, perturbation, etc.)
4. Memory usage patterns
5. GPU utilization and power consumption
6. Scaling behavior (how does perf change with pop_size, model_size, rank?)

Output: JSON + plots for easy analysis
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# =============================================================================
# GPU Monitoring
# =============================================================================

def get_gpu_stats() -> Dict[str, float]:
    """Get current GPU stats via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return {
                "gpu_util_pct": float(parts[0]),
                "memory_used_mb": float(parts[1]),
                "memory_total_mb": float(parts[2]),
                "power_w": float(parts[3]),
                "temp_c": float(parts[4]),
            }
    except Exception:
        pass
    return {}


def get_gpu_name() -> str:
    """Get GPU model name."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return "Unknown"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class OperationProfile:
    """Profile for a single operation."""
    name: str
    total_time_ms: float
    num_calls: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    
    
@dataclass 
class PopulationProfile:
    """Profile at a specific population size."""
    pop_size: int
    forward_time_ms: float
    forward_std_ms: float
    backward_time_ms: float  # ES gradient estimation
    backward_std_ms: float
    total_epoch_time_ms: float
    total_epoch_std_ms: float
    throughput_samples_per_sec: float
    memory_used_mb: float
    memory_peak_mb: float
    gpu_util_pct: float
    operations: Dict[str, OperationProfile] = field(default_factory=dict)


@dataclass
class ScalingProfile:
    """How performance scales with a parameter."""
    parameter_name: str
    parameter_values: List[int]
    throughputs: List[float]
    memory_used: List[float]
    times_ms: List[float]


@dataclass
class ImplementationProfile:
    """Full profile for one implementation."""
    name: str
    framework: str
    gpu_name: str
    timestamp: str
    
    # Config
    layer_size: int
    n_layers: int
    rank: int
    input_dim: int
    
    # Max capacity (set after finding max)
    max_population: int = 0
    max_population_memory_mb: float = 0.0
    
    # Per-population profiles
    population_profiles: List[PopulationProfile] = field(default_factory=list)
    
    # Scaling studies
    scaling_pop_size: Optional[ScalingProfile] = None
    scaling_layer_size: Optional[ScalingProfile] = None
    scaling_rank: Optional[ScalingProfile] = None
    
    # Operation breakdown at reference config
    operation_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Raw timing data for statistical analysis
    raw_forward_times: Dict[int, List[float]] = field(default_factory=dict)


# =============================================================================
# JAX Profiling
# =============================================================================

def profile_jax_eggroll(
    pop_sizes: List[int],
    layer_size: int = 256,
    n_layers: int = 3,
    rank: int = 4,
    input_dim: int = 4,
    output_dim: int = 2,
    warmup_iters: int = 10,
    bench_iters: int = 50,
    verbose: bool = True,
) -> ImplementationProfile:
    """Profile JAX EGGROLL implementation."""
    
    import jax
    import jax.numpy as jnp
    import optax
    from functools import partial
    from hyperscalees.noiser.eggroll import EggRoll
    from hyperscalees.models.common import MLP, simple_es_tree_key
    
    if verbose:
        print(f"\n{'='*60}")
        print("Profiling JAX EGGROLL")
        print(f"{'='*60}")
        print(f"JAX version: {jax.__version__}")
        print(f"Devices: {jax.devices()}")
    
    profile = ImplementationProfile(
        name="EGGROLL (JAX)",
        framework=f"JAX {jax.__version__}",
        gpu_name=get_gpu_name(),
        timestamp=datetime.now().isoformat(),
        layer_size=layer_size,
        n_layers=n_layers,
        rank=rank,
        input_dim=input_dim,
    )
    
    # Initialize model using correct API
    key = jax.random.PRNGKey(42)
    model_key, es_key = jax.random.split(key)
    
    # Build hidden_dims for n_layers: [layer_size] * (n_layers - 1) for hidden layers
    hidden_dims = [layer_size] * (n_layers - 1) if n_layers > 1 else []
    
    frozen_params, params, scan_map, es_map = MLP.rand_init(
        model_key,
        in_dim=input_dim,
        out_dim=output_dim,
        hidden_dims=hidden_dims,
        use_bias=True,
        activation="relu",
        dtype="float32"
    )
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)
    
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, sigma=0.2, lr=0.1, solver=optax.sgd, rank=rank
    )
    
    # Count parameters
    total_params = sum(p.size for p in jax.tree.leaves(params))
    if verbose:
        print(f"Total parameters: {total_params:,}")
    
    # Find max population size
    if verbose:
        print("\nFinding maximum population size...")
    
    max_pop = 32
    for test_pop in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        try:
            gc.collect()
            # Clear JAX caches
            jax.clear_caches()
            
            x = jnp.ones((test_pop, input_dim))
            iterinfos = (jnp.zeros(test_pop, dtype=jnp.int32), jnp.arange(test_pop, dtype=jnp.int32))
            
            # JIT compile forward
            @jax.jit
            def test_forward(noiser_params, params, iterinfos, x):
                return jax.vmap(
                    lambda i, xi: MLP.forward(EggRoll, frozen_noiser_params, noiser_params, 
                                              frozen_params, params, es_tree_key, i, xi),
                    in_axes=(0, 0)
                )(iterinfos, x)
            
            # Try to run
            out = test_forward(noiser_params, params, iterinfos, x)
            out.block_until_ready()
            
            max_pop = test_pop
            if verbose:
                print(f"  pop_size={test_pop}: OK")
        except Exception as e:
            if verbose:
                print(f"  pop_size={test_pop}: FAILED ({type(e).__name__})")
            break
    
    profile.max_population = max_pop
    stats = get_gpu_stats()
    profile.max_population_memory_mb = stats.get("memory_used_mb", 0)
    
    if verbose:
        print(f"Maximum population: {max_pop}")
    
    # Profile each population size
    for pop_size in pop_sizes:
        if pop_size > max_pop:
            if verbose:
                print(f"\nSkipping pop_size={pop_size} (exceeds max)")
            continue
            
        if verbose:
            print(f"\nProfiling pop_size={pop_size}...")
        
        gc.collect()
        jax.clear_caches()
        
        x = jnp.ones((pop_size, input_dim))
        iterinfos = (jnp.zeros(pop_size, dtype=jnp.int32), jnp.arange(pop_size, dtype=jnp.int32))
        fitnesses = jnp.ones(pop_size)
        
        # JIT compile
        @jax.jit
        def forward_fn(noiser_params, params, iterinfos, x):
            return jax.vmap(
                lambda i, xi: MLP.forward(EggRoll, frozen_noiser_params, noiser_params,
                                          frozen_params, params, es_tree_key, i, xi),
                in_axes=(0, 0)
            )(iterinfos, x)
        
        @jax.jit
        def update_fn(noiser_params, params, fitnesses, iterinfos):
            return EggRoll.do_updates(frozen_noiser_params, noiser_params, params, 
                                      es_tree_key, fitnesses, iterinfos, es_map)
        
        # Warmup
        for _ in range(warmup_iters):
            out = forward_fn(noiser_params, params, iterinfos, x)
            out.block_until_ready()
            new_noiser, new_params = update_fn(noiser_params, params, fitnesses, iterinfos)
            jax.tree.map(lambda x: x.block_until_ready(), new_params)
        
        # Benchmark forward
        forward_times = []
        for _ in range(bench_iters):
            start = time.perf_counter()
            out = forward_fn(noiser_params, params, iterinfos, x)
            out.block_until_ready()
            forward_times.append((time.perf_counter() - start) * 1000)
        
        # Benchmark backward (ES update)
        backward_times = []
        for _ in range(bench_iters):
            start = time.perf_counter()
            new_noiser, new_params = update_fn(noiser_params, params, fitnesses, iterinfos)
            jax.tree.map(lambda x: x.block_until_ready(), new_params)
            backward_times.append((time.perf_counter() - start) * 1000)
        
        # Benchmark full epoch
        epoch_times = []
        for _ in range(bench_iters):
            start = time.perf_counter()
            out = forward_fn(noiser_params, params, iterinfos, x)
            out.block_until_ready()
            new_noiser, new_params = update_fn(noiser_params, params, fitnesses, iterinfos)
            jax.tree.map(lambda x: x.block_until_ready(), new_params)
            epoch_times.append((time.perf_counter() - start) * 1000)
        
        stats = get_gpu_stats()
        
        pop_profile = PopulationProfile(
            pop_size=pop_size,
            forward_time_ms=np.mean(forward_times),
            forward_std_ms=np.std(forward_times),
            backward_time_ms=np.mean(backward_times),
            backward_std_ms=np.std(backward_times),
            total_epoch_time_ms=np.mean(epoch_times),
            total_epoch_std_ms=np.std(epoch_times),
            throughput_samples_per_sec=pop_size / (np.mean(epoch_times) / 1000),
            memory_used_mb=stats.get("memory_used_mb", 0),
            memory_peak_mb=stats.get("memory_used_mb", 0),
            gpu_util_pct=stats.get("gpu_util_pct", 0),
        )
        
        profile.population_profiles.append(pop_profile)
        profile.raw_forward_times[pop_size] = forward_times
        
        if verbose:
            print(f"  Forward: {pop_profile.forward_time_ms:.3f} ± {pop_profile.forward_std_ms:.3f} ms")
            print(f"  Backward: {pop_profile.backward_time_ms:.3f} ± {pop_profile.backward_std_ms:.3f} ms")
            print(f"  Throughput: {pop_profile.throughput_samples_per_sec:,.0f} samples/sec")
            print(f"  Memory: {pop_profile.memory_used_mb:.0f} MB")
    
    return profile


# =============================================================================
# PyTorch Profiling  
# =============================================================================

def profile_torch_eggroll(
    pop_sizes: List[int],
    layer_size: int = 256,
    n_layers: int = 3,
    rank: int = 4,
    input_dim: int = 4,
    output_dim: int = 2,
    warmup_iters: int = 10,
    bench_iters: int = 50,
    verbose: bool = True,
) -> ImplementationProfile:
    """Profile PyTorch EGGROLL implementation."""
    
    import torch
    import torch.nn as nn
    from hyperscalees.torch import EggrollStrategy
    
    if verbose:
        print(f"\n{'='*60}")
        print("Profiling PyTorch EGGROLL")
        print(f"{'='*60}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    profile = ImplementationProfile(
        name="EGGROLL (Torch)",
        framework=f"PyTorch {torch.__version__}",
        gpu_name=get_gpu_name(),
        timestamp=datetime.now().isoformat(),
        layer_size=layer_size,
        n_layers=n_layers,
        rank=rank,
        input_dim=input_dim,
    )
    
    # Create model
    layers = []
    dims = [input_dim] + [layer_size] * (n_layers - 1) + [output_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    model = nn.Sequential(*layers).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Total parameters: {total_params:,}")
    
    # Find max population size
    if verbose:
        print("\nFinding maximum population size...")
    
    max_pop = 32
    for test_pop in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            strategy = EggrollStrategy(sigma=0.2, lr=0.1, rank=rank, seed=42)
            strategy.setup(model)
            
            x = torch.randn(test_pop, input_dim, device=device)
            
            with strategy.perturb(population_size=test_pop, epoch=0) as ctx:
                out = ctx.batched_forward(model, x)
            
            torch.cuda.synchronize()
            max_pop = test_pop
            if verbose:
                print(f"  pop_size={test_pop}: OK")
        except Exception as e:
            if verbose:
                print(f"  pop_size={test_pop}: FAILED ({type(e).__name__})")
            break
    
    profile.max_population = max_pop
    stats = get_gpu_stats()
    profile.max_population_memory_mb = stats.get("memory_used_mb", 0)
    
    if verbose:
        print(f"Maximum population: {max_pop}")
    
    # Profile each population size
    for pop_size in pop_sizes:
        if pop_size > max_pop:
            if verbose:
                print(f"\nSkipping pop_size={pop_size} (exceeds max)")
            continue
            
        if verbose:
            print(f"\nProfiling pop_size={pop_size}...")
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Fresh strategy for each pop_size
        strategy = EggrollStrategy(sigma=0.2, lr=0.1, rank=rank, seed=42)
        strategy.setup(model)
        
        x = torch.randn(pop_size, input_dim, device=device)
        
        # Warmup
        for i in range(warmup_iters):
            with strategy.perturb(population_size=pop_size, epoch=i) as ctx:
                out = ctx.batched_forward(model, x)
                fitnesses = out.mean(dim=-1)
            strategy.step(fitnesses)
        torch.cuda.synchronize()
        
        # Benchmark forward
        forward_times = []
        for i in range(bench_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with strategy.perturb(population_size=pop_size, epoch=warmup_iters + i) as ctx:
                out = ctx.batched_forward(model, x)
            torch.cuda.synchronize()
            forward_times.append((time.perf_counter() - start) * 1000)
        
        # Benchmark backward (ES update)
        backward_times = []
        for i in range(bench_iters):
            with strategy.perturb(population_size=pop_size, epoch=warmup_iters + bench_iters + i) as ctx:
                out = ctx.batched_forward(model, x)
                fitnesses = out.mean(dim=-1)
            torch.cuda.synchronize()
            start = time.perf_counter()
            strategy.step(fitnesses)
            torch.cuda.synchronize()
            backward_times.append((time.perf_counter() - start) * 1000)
        
        # Benchmark full epoch
        epoch_times = []
        for i in range(bench_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with strategy.perturb(population_size=pop_size, epoch=warmup_iters + 2*bench_iters + i) as ctx:
                out = ctx.batched_forward(model, x)
                fitnesses = out.mean(dim=-1)
            strategy.step(fitnesses)
            torch.cuda.synchronize()
            epoch_times.append((time.perf_counter() - start) * 1000)
        
        stats = get_gpu_stats()
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        pop_profile = PopulationProfile(
            pop_size=pop_size,
            forward_time_ms=np.mean(forward_times),
            forward_std_ms=np.std(forward_times),
            backward_time_ms=np.mean(backward_times),
            backward_std_ms=np.std(backward_times),
            total_epoch_time_ms=np.mean(epoch_times),
            total_epoch_std_ms=np.std(epoch_times),
            throughput_samples_per_sec=pop_size / (np.mean(epoch_times) / 1000),
            memory_used_mb=stats.get("memory_used_mb", 0),
            memory_peak_mb=peak_memory,
            gpu_util_pct=stats.get("gpu_util_pct", 0),
        )
        
        profile.population_profiles.append(pop_profile)
        profile.raw_forward_times[pop_size] = forward_times
        
        if verbose:
            print(f"  Forward: {pop_profile.forward_time_ms:.3f} ± {pop_profile.forward_std_ms:.3f} ms")
            print(f"  Backward: {pop_profile.backward_time_ms:.3f} ± {pop_profile.backward_std_ms:.3f} ms")
            print(f"  Throughput: {pop_profile.throughput_samples_per_sec:,.0f} samples/sec")
            print(f"  Memory: {pop_profile.memory_used_mb:.0f} MB (peak: {peak_memory:.0f} MB)")
    
    # Detailed operation breakdown at reference config
    if verbose:
        print("\nProfiling operation breakdown...")
    
    profile.operation_breakdown = profile_torch_operations(
        model, strategy, pop_size=2048, input_dim=input_dim, 
        device=device, num_iters=50, verbose=verbose
    )
    
    return profile


def profile_torch_operations(
    model: "nn.Module",
    strategy: "EggrollStrategy", 
    pop_size: int,
    input_dim: int,
    device: "torch.device",
    num_iters: int = 50,
    verbose: bool = True,
) -> Dict[str, float]:
    """Detailed breakdown of PyTorch operations."""
    import torch
    
    x = torch.randn(pop_size, input_dim, device=device)
    
    operations = {}
    
    # Profile noise generation
    times = []
    for i in range(num_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Simulate noise generation
        gen = torch.Generator(device=device).manual_seed(42 + i)
        noise = torch.randn(pop_size // 2, 256 + 256, 4, generator=gen, device=device)
        
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    operations["noise_generation_ms"] = np.mean(times)
    
    # Profile antithetic expansion
    times = []
    noise = torch.randn(pop_size // 2, 256, 4, device=device)
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        expanded = noise.repeat_interleave(2, dim=0)
        
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    operations["antithetic_expansion_ms"] = np.mean(times)
    
    # Profile bmm (low-rank perturbation)
    times = []
    A = torch.randn(pop_size, 256, 4, device=device)
    B = torch.randn(pop_size, 256, 4, device=device)
    x_batch = torch.randn(pop_size, 256, device=device)
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        xB = torch.bmm(x_batch.unsqueeze(1), B).squeeze(1)
        pert = torch.bmm(xB.unsqueeze(1), A.transpose(1, 2)).squeeze(1)
        
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    operations["lowrank_bmm_ms"] = np.mean(times)
    
    # Profile base linear
    times = []
    weight = torch.randn(256, 256, device=device)
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        out = torch.nn.functional.linear(x_batch, weight)
        
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    operations["base_linear_ms"] = np.mean(times)
    
    # Profile activation
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        out = torch.tanh(x_batch)
        
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    operations["activation_ms"] = np.mean(times)
    
    if verbose:
        print("  Operation breakdown:")
        for name, time_ms in operations.items():
            print(f"    {name}: {time_ms:.3f} ms")
    
    return operations


# =============================================================================
# Scaling Studies
# =============================================================================

def run_scaling_study(
    profile_fn,
    param_name: str,
    param_values: List[int],
    base_config: Dict,
    warmup_iters: int = 5,
    bench_iters: int = 20,
    verbose: bool = True,
) -> ScalingProfile:
    """Run scaling study varying one parameter."""
    
    throughputs = []
    memory_used = []
    times_ms = []
    
    for val in param_values:
        config = base_config.copy()
        config[param_name] = val
        
        if verbose:
            print(f"\n  {param_name}={val}...")
        
        try:
            profile = profile_fn(
                pop_sizes=[config.get("pop_size", 2048)],
                layer_size=config.get("layer_size", 256),
                n_layers=config.get("n_layers", 3),
                rank=config.get("rank", 4),
                warmup_iters=warmup_iters,
                bench_iters=bench_iters,
                verbose=False,
            )
            
            if profile.population_profiles:
                pp = profile.population_profiles[0]
                throughputs.append(pp.throughput_samples_per_sec)
                memory_used.append(pp.memory_used_mb)
                times_ms.append(pp.total_epoch_time_ms)
                
                if verbose:
                    print(f"    Throughput: {pp.throughput_samples_per_sec:,.0f} samples/sec")
                    print(f"    Memory: {pp.memory_used_mb:.0f} MB")
            else:
                throughputs.append(0)
                memory_used.append(0)
                times_ms.append(float('inf'))
        except Exception as e:
            if verbose:
                print(f"    FAILED: {e}")
            throughputs.append(0)
            memory_used.append(0)
            times_ms.append(float('inf'))
    
    return ScalingProfile(
        parameter_name=param_name,
        parameter_values=param_values,
        throughputs=throughputs,
        memory_used=memory_used,
        times_ms=times_ms,
    )


# =============================================================================
# Visualization
# =============================================================================

def generate_comparison_plots(
    jax_profile: ImplementationProfile,
    torch_profile: ImplementationProfile,
    output_dir: str,
):
    """Generate comparison plots."""
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    jax_pops = [p.pop_size for p in jax_profile.population_profiles]
    jax_throughput = [p.throughput_samples_per_sec for p in jax_profile.population_profiles]
    jax_memory = [p.memory_used_mb for p in jax_profile.population_profiles]
    jax_forward = [p.forward_time_ms for p in jax_profile.population_profiles]
    jax_backward = [p.backward_time_ms for p in jax_profile.population_profiles]
    
    torch_pops = [p.pop_size for p in torch_profile.population_profiles]
    torch_throughput = [p.throughput_samples_per_sec for p in torch_profile.population_profiles]
    torch_memory = [p.memory_used_mb for p in torch_profile.population_profiles]
    torch_forward = [p.forward_time_ms for p in torch_profile.population_profiles]
    torch_backward = [p.backward_time_ms for p in torch_profile.population_profiles]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Throughput vs Population Size
    ax = axes[0, 0]
    ax.plot(jax_pops, jax_throughput, 'o-', label='JAX', linewidth=2, markersize=8)
    ax.plot(torch_pops, torch_throughput, 's-', label='PyTorch', linewidth=2, markersize=8)
    ax.set_xlabel('Population Size')
    ax.set_ylabel('Throughput (samples/sec)')
    ax.set_title('Throughput vs Population Size')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Memory Usage vs Population Size
    ax = axes[0, 1]
    ax.plot(jax_pops, jax_memory, 'o-', label='JAX', linewidth=2, markersize=8)
    ax.plot(torch_pops, torch_memory, 's-', label='PyTorch', linewidth=2, markersize=8)
    ax.set_xlabel('Population Size')
    ax.set_ylabel('Memory Used (MB)')
    ax.set_title('Memory Usage vs Population Size')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Forward Time vs Population Size
    ax = axes[1, 0]
    ax.plot(jax_pops, jax_forward, 'o-', label='JAX', linewidth=2, markersize=8)
    ax.plot(torch_pops, torch_forward, 's-', label='PyTorch', linewidth=2, markersize=8)
    ax.set_xlabel('Population Size')
    ax.set_ylabel('Forward Time (ms)')
    ax.set_title('Forward Pass Time vs Population Size')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Backward (ES Update) Time vs Population Size
    ax = axes[1, 1]
    ax.plot(jax_pops, jax_backward, 'o-', label='JAX', linewidth=2, markersize=8)
    ax.plot(torch_pops, torch_backward, 's-', label='PyTorch', linewidth=2, markersize=8)
    ax.set_xlabel('Population Size')
    ax.set_ylabel('ES Update Time (ms)')
    ax.set_title('ES Gradient Update Time vs Population Size')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_overview.png'), dpi=150)
    plt.close()
    
    # Speedup plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Find common pop sizes
    common_pops = sorted(set(jax_pops) & set(torch_pops))
    jax_times = {p.pop_size: p.total_epoch_time_ms for p in jax_profile.population_profiles}
    torch_times = {p.pop_size: p.total_epoch_time_ms for p in torch_profile.population_profiles}
    
    speedups = [jax_times[p] / torch_times[p] for p in common_pops]
    
    bars = ax.bar(range(len(common_pops)), speedups, color=['green' if s > 1 else 'red' for s in speedups])
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(common_pops)))
    ax.set_xticklabels([str(p) for p in common_pops])
    ax.set_xlabel('Population Size')
    ax.set_ylabel('Speedup (JAX time / PyTorch time)')
    ax.set_title('PyTorch Speedup vs JAX (>1 means PyTorch faster)')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_comparison.png'), dpi=150)
    plt.close()
    
    # Operation breakdown (if available)
    if torch_profile.operation_breakdown:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ops = list(torch_profile.operation_breakdown.keys())
        times = list(torch_profile.operation_breakdown.values())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(ops)))
        bars = ax.barh(ops, times, color=colors)
        ax.set_xlabel('Time (ms)')
        ax.set_title('PyTorch EGGROLL Operation Breakdown (pop_size=2048)')
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar, t in zip(bars, times):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{t:.3f}ms', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'operation_breakdown.png'), dpi=150)
        plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    jax_profile: ImplementationProfile,
    torch_profile: ImplementationProfile,
    output_path: str,
):
    """Generate markdown report."""
    
    lines = [
        "# Head-to-Head Profiling: JAX vs PyTorch EGGROLL",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**GPU:** {jax_profile.gpu_name}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "| Metric | JAX | PyTorch | Winner |",
        "|--------|-----|---------|--------|",
    ]
    
    # Max population
    jax_max = jax_profile.max_population
    torch_max = torch_profile.max_population
    winner = "PyTorch" if torch_max > jax_max else ("JAX" if jax_max > torch_max else "Tie")
    lines.append(f"| Max Population | {jax_max:,} | {torch_max:,} | {winner} |")
    
    # Find common pop sizes for comparison
    jax_by_pop = {p.pop_size: p for p in jax_profile.population_profiles}
    torch_by_pop = {p.pop_size: p for p in torch_profile.population_profiles}
    common_pops = sorted(set(jax_by_pop.keys()) & set(torch_by_pop.keys()))
    
    if common_pops:
        ref_pop = 2048 if 2048 in common_pops else common_pops[-1]
        jax_p = jax_by_pop[ref_pop]
        torch_p = torch_by_pop[ref_pop]
        
        # Throughput
        winner = "PyTorch" if torch_p.throughput_samples_per_sec > jax_p.throughput_samples_per_sec else "JAX"
        lines.append(f"| Throughput (pop={ref_pop}) | {jax_p.throughput_samples_per_sec:,.0f}/s | {torch_p.throughput_samples_per_sec:,.0f}/s | {winner} |")
        
        # Forward time
        winner = "PyTorch" if torch_p.forward_time_ms < jax_p.forward_time_ms else "JAX"
        lines.append(f"| Forward Time (pop={ref_pop}) | {jax_p.forward_time_ms:.2f}ms | {torch_p.forward_time_ms:.2f}ms | {winner} |")
        
        # Memory
        winner = "PyTorch" if torch_p.memory_used_mb < jax_p.memory_used_mb else "JAX"
        lines.append(f"| Memory (pop={ref_pop}) | {jax_p.memory_used_mb:.0f}MB | {torch_p.memory_used_mb:.0f}MB | {winner} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Configuration",
        "",
        f"- Layer size: {jax_profile.layer_size}",
        f"- Number of layers: {jax_profile.n_layers}",
        f"- Rank: {jax_profile.rank}",
        f"- Input dim: {jax_profile.input_dim}",
        "",
        "---",
        "",
        "## Detailed Results by Population Size",
        "",
        "### JAX EGGROLL",
        "",
        "| Pop Size | Forward (ms) | Backward (ms) | Throughput | Memory (MB) |",
        "|----------|--------------|---------------|------------|-------------|",
    ])
    
    for p in jax_profile.population_profiles:
        lines.append(f"| {p.pop_size:,} | {p.forward_time_ms:.2f} ± {p.forward_std_ms:.2f} | {p.backward_time_ms:.2f} ± {p.backward_std_ms:.2f} | {p.throughput_samples_per_sec:,.0f}/s | {p.memory_used_mb:.0f} |")
    
    lines.extend([
        "",
        "### PyTorch EGGROLL",
        "",
        "| Pop Size | Forward (ms) | Backward (ms) | Throughput | Memory (MB) | Peak Memory |",
        "|----------|--------------|---------------|------------|-------------|-------------|",
    ])
    
    for p in torch_profile.population_profiles:
        lines.append(f"| {p.pop_size:,} | {p.forward_time_ms:.2f} ± {p.forward_std_ms:.2f} | {p.backward_time_ms:.2f} ± {p.backward_std_ms:.2f} | {p.throughput_samples_per_sec:,.0f}/s | {p.memory_used_mb:.0f} | {p.memory_peak_mb:.0f} |")
    
    # Operation breakdown
    if torch_profile.operation_breakdown:
        lines.extend([
            "",
            "---",
            "",
            "## PyTorch Operation Breakdown (pop_size=2048)",
            "",
            "| Operation | Time (ms) | % of Total |",
            "|-----------|-----------|------------|",
        ])
        
        total = sum(torch_profile.operation_breakdown.values())
        for name, time_ms in sorted(torch_profile.operation_breakdown.items(), key=lambda x: -x[1]):
            pct = 100 * time_ms / total if total > 0 else 0
            lines.append(f"| {name} | {time_ms:.3f} | {pct:.1f}% |")
    
    # Speedup analysis
    lines.extend([
        "",
        "---",
        "",
        "## Speedup Analysis",
        "",
        "| Pop Size | JAX Time (ms) | PyTorch Time (ms) | Speedup |",
        "|----------|---------------|-------------------|---------|",
    ])
    
    for pop in common_pops:
        jax_t = jax_by_pop[pop].total_epoch_time_ms
        torch_t = torch_by_pop[pop].total_epoch_time_ms
        speedup = jax_t / torch_t
        emoji = "🟢" if speedup > 1.1 else ("🔴" if speedup < 0.9 else "🟡")
        lines.append(f"| {pop:,} | {jax_t:.2f} | {torch_t:.2f} | {emoji} {speedup:.2f}x |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Plots",
        "",
        "![Comparison Overview](comparison_overview.png)",
        "",
        "![Speedup](speedup_comparison.png)",
        "",
    ])
    
    if torch_profile.operation_breakdown:
        lines.append("![Operation Breakdown](operation_breakdown.png)")
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## Raw Data",
        "",
        "See `profile_results.json` for complete raw data.",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nReport saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Head-to-head profiling of JAX vs PyTorch EGGROLL")
    parser.add_argument("--pop-sizes", type=int, nargs="+", 
                        default=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
                        help="Population sizes to benchmark")
    parser.add_argument("--layer-size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--rank", type=int, default=4, help="Low-rank perturbation rank")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--bench-iters", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--output-dir", type=str, default="benchmarks", help="Output directory")
    parser.add_argument("--skip-jax", action="store_true", help="Skip JAX profiling")
    parser.add_argument("--skip-torch", action="store_true", help="Skip PyTorch profiling")
    parser.add_argument("--scaling-study", action="store_true", help="Run scaling studies")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("HEAD-TO-HEAD PROFILING: JAX vs PyTorch EGGROLL")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Population sizes: {args.pop_sizes}")
    print(f"  Layer size: {args.layer_size}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Rank: {args.rank}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Benchmark iterations: {args.bench_iters}")
    print(f"  GPU: {get_gpu_name()}")
    
    jax_profile = None
    torch_profile = None
    
    # Profile JAX
    if not args.skip_jax:
        jax_profile = profile_jax_eggroll(
            pop_sizes=args.pop_sizes,
            layer_size=args.layer_size,
            n_layers=args.n_layers,
            rank=args.rank,
            warmup_iters=args.warmup,
            bench_iters=args.bench_iters,
        )
    
    # Profile PyTorch
    if not args.skip_torch:
        torch_profile = profile_torch_eggroll(
            pop_sizes=args.pop_sizes,
            layer_size=args.layer_size,
            n_layers=args.n_layers,
            rank=args.rank,
            warmup_iters=args.warmup,
            bench_iters=args.bench_iters,
        )
    
    # Save raw results
    results = {
        "timestamp": timestamp,
        "config": {
            "pop_sizes": args.pop_sizes,
            "layer_size": args.layer_size,
            "n_layers": args.n_layers,
            "rank": args.rank,
        },
    }
    
    if jax_profile:
        results["jax"] = asdict(jax_profile)
    if torch_profile:
        results["torch"] = asdict(torch_profile)
    
    results_path = os.path.join(output_dir, f"profile_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to {results_path}")
    
    # Generate plots and report
    if jax_profile and torch_profile:
        generate_comparison_plots(jax_profile, torch_profile, output_dir)
        report_path = os.path.join(output_dir, f"profile_report_{timestamp}.md")
        generate_report(jax_profile, torch_profile, report_path)
    
    # Quick summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if jax_profile:
        print(f"\nJAX EGGROLL:")
        print(f"  Max population: {jax_profile.max_population:,}")
        if jax_profile.population_profiles:
            best = max(jax_profile.population_profiles, key=lambda x: x.throughput_samples_per_sec)
            print(f"  Best throughput: {best.throughput_samples_per_sec:,.0f} samples/sec @ pop={best.pop_size}")
    
    if torch_profile:
        print(f"\nPyTorch EGGROLL:")
        print(f"  Max population: {torch_profile.max_population:,}")
        if torch_profile.population_profiles:
            best = max(torch_profile.population_profiles, key=lambda x: x.throughput_samples_per_sec)
            print(f"  Best throughput: {best.throughput_samples_per_sec:,.0f} samples/sec @ pop={best.pop_size}")
    
    if jax_profile and torch_profile:
        # Compare at largest common pop size
        jax_by_pop = {p.pop_size: p for p in jax_profile.population_profiles}
        torch_by_pop = {p.pop_size: p for p in torch_profile.population_profiles}
        common = sorted(set(jax_by_pop.keys()) & set(torch_by_pop.keys()))
        
        if common:
            ref = common[-1]  # Largest common
            speedup = jax_by_pop[ref].total_epoch_time_ms / torch_by_pop[ref].total_epoch_time_ms
            print(f"\n  Speedup @ pop={ref}: {speedup:.2f}x {'(PyTorch faster)' if speedup > 1 else '(JAX faster)'}")


if __name__ == "__main__":
    main()
