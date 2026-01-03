"""
EGGROLL Benchmarks and Profiling

This module contains all benchmarking and profiling functions for EGGROLL.
Separated from the main eggroll_fncl.py for clarity.

Benchmarks:
- benchmark_inference_jax: JAX-only inference benchmark
- benchmark_inference_torch: Torch inference benchmark with torch.compile
- benchmark_inference_ratio: Compare EGGROLL training vs pure inference speed

Profiling:
- profile_torch_breakdown: Detailed timing of each EGGROLL component
- profile_torch_optimizations: Compare optimization strategies (einsum vs bmm)
- profile_head_to_head: JAX vs Torch head-to-head comparison

Hyperscale:
- main_hyperscale_torch: Find max population size before OOM (Torch)
- main_hyperscale_jax: Find max population size before OOM (JAX)

Usage:
    python -m hyperscalees.torch.fncl.eggroll_benchmarks --benchmark inference
    python -m hyperscalees.torch.fncl.eggroll_benchmarks --benchmark head_to_head
    python -m hyperscalees.torch.fncl.eggroll_benchmarks --benchmark profile
    python -m hyperscalees.torch.fncl.eggroll_benchmarks --benchmark hyperscale
    python -m hyperscalees.torch.fncl.eggroll_benchmarks --benchmark all
"""

import torch
import time
import math
from rich.console import Console
from rich.table import Table

# Import core EGGROLL functions from main module
from hyperscalees.torch.fncl.eggroll_fncl import (
    EggrollConfig,
    get_gpu_stats,
    get_gpu_stats_jax,
    get_gpu_stats_nvidia,
    print_gpu_stats,
    print_gpu_stats_jax,
    reset_gpu_stats,
    # Dict-based API
    get_weight_shapes,
    generate_perturbations,
    compute_gradients,
    update_params,
    perturbed_forward,
    # Raw primitives
    generate_lowrank_perturbations,
    perturbed_linear,
    apply_lowrank_perturbation,
    compute_es_gradient,
    normalize_fitnesses,
)

console = Console()


# =============================================================================
# Compiled Functions for Benchmarking
# =============================================================================

@torch.compile
def compiled_batched_inference(x, W1, b1, W2, b2):
    """Compiled pure inference (no perturbations)."""
    h = torch.relu(x @ W1.T + b1)
    return h @ W2.T + b2


@torch.compile  
def compiled_eggroll_step(
    x, W1, b1, W2, b2, 
    A1_scaled, B1, 
    A2_scaled, B2,
    sqrt_N
):
    """
    Complete compiled EGGROLL step: forward + gradient computation.
    This is the main optimization - fusing all operations into compiled graph.
    """
    # Forward with perturbations
    base1 = x @ W1.T + b1
    pert1 = torch.einsum('pbi,pir,pjr->pbj', x, B1, A1_scaled)
    h = torch.relu(base1 + pert1)
    
    base2 = h @ W2.T + b2
    pert2 = torch.einsum('pbi,pir,pjr->pbj', h, B2, A2_scaled)
    logits = base2 + pert2
    
    # Compute fitness
    fitnesses = logits.mean(dim=(1, 2))
    fitnesses = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
    
    # ES gradient - use A_scaled for proper magnitude
    f = fitnesses[:, None, None]
    grad1 = torch.einsum('nir,njr->ij', f * A1_scaled, B1) / sqrt_N
    grad2 = torch.einsum('nir,njr->ij', f * A2_scaled, B2) / sqrt_N
    
    return logits, grad1, grad2


def warmup_compiled_functions(pop_size, batch_size, in_dim, hidden_dim, out_dim, rank, dtype=torch.float32):
    """
    Warm up compiled functions to avoid compilation overhead during benchmarking.
    """
    console.print("  [dim]Warming up torch.compile...[/dim]")
    
    W1 = torch.randn(hidden_dim, in_dim, device="cuda", dtype=dtype)
    W2 = torch.randn(out_dim, hidden_dim, device="cuda", dtype=dtype)
    b1 = torch.zeros(hidden_dim, device="cuda", dtype=dtype)
    b2 = torch.zeros(out_dim, device="cuda", dtype=dtype)
    x = torch.randn(pop_size, batch_size, in_dim, device="cuda", dtype=dtype)
    
    gen = torch.Generator(device="cuda").manual_seed(0)
    A1_scaled, A1, B1 = generate_lowrank_perturbations(pop_size, hidden_dim, in_dim, rank, 0.1, gen, dtype)
    A2_scaled, A2, B2 = generate_lowrank_perturbations(pop_size, out_dim, hidden_dim, rank, 0.1, gen, dtype)
    sqrt_N = math.sqrt(pop_size)
    
    # Trigger compilation
    for _ in range(3):
        _ = compiled_batched_inference(x, W1, b1, W2, b2)
        _ = compiled_eggroll_step(x, W1, b1, W2, b2, A1_scaled, B1, A2_scaled, B2, sqrt_N)
    torch.cuda.synchronize()
    
    del W1, W2, b1, b2, x, A1_scaled, A1, B1, A2_scaled, A2, B2
    torch.cuda.empty_cache()


# =============================================================================
# JAX Inference Benchmark
# =============================================================================

def benchmark_inference_jax():
    """
    JAX-only inference benchmark. Run in a separate process from Torch.
    
    NOTE: JAX EggRoll uses on-the-fly noise generation via seeded PRNG.
    The noise is regenerated deterministically each forward pass using
    jax.random.fold_in(key, epoch, thread_idx).
    """
    import jax
    import jax.numpy as jnp
    import optax
    from hyperscalees.noiser.eggroll import EggRoll
    from hyperscalees.models.common import MLP, simple_es_tree_key
    
    console.print(f"\n[bold blue]═══ JAX Inference Benchmark ═══[/bold blue]")
    console.print("JAX EggRoll uses on-the-fly noise (seeded PRNG regeneration)")
    console.print("Paper target for on-the-fly: ~69% of pure inference speed")
    console.print()
    
    configs = [("MLP 784->256->10", 784, 256, 10, 128)]
    population_sizes = [1024, 2048, 4096, 8192]
    rank = 4
    sigma = 0.1
    num_warmup = 5
    num_iters = 20
    
    results = []
    
    for name, in_dim, hidden_dim, out_dim, batch_size in configs:
        console.print(f"[cyan]{name}[/cyan]")
        
        for pop_size in population_sizes:
            try:
                key = jax.random.key(42)
                model_key, es_key = jax.random.split(key)
                
                frozen_params, params, scan_map, es_map = MLP.rand_init(
                    model_key, in_dim=in_dim, out_dim=out_dim,
                    hidden_dims=[hidden_dim], use_bias=False,
                    activation="relu", dtype="float32",
                )
                
                es_tree_key = simple_es_tree_key(params, es_key, scan_map)
                frozen_noiser_params, noiser_params = EggRoll.init_noiser(
                    params, sigma=sigma, lr=0.1, solver=optax.sgd, rank=rank,
                )
                
                x_jax = jnp.ones((batch_size, in_dim))
                x_batch_jax = jnp.ones((pop_size, batch_size, in_dim))
                
                @jax.jit
                def batched_inference_jax(params, x):
                    def single_forward(x):
                        h = jax.nn.relu(x @ params['0']['weight'].T)
                        return h @ params['1']['weight'].T
                    return jax.vmap(single_forward)(x)
                
                def forward_noisy(noiser_params, params, iterinfo, x):
                    return MLP.forward(
                        EggRoll, frozen_noiser_params, noiser_params, frozen_params,
                        params, es_tree_key, iterinfo, x
                    )
                
                jit_forward = jax.jit(jax.vmap(
                    lambda n, p, i, x: forward_noisy(n, p, i, x),
                    in_axes=(None, None, 0, 0)
                ))
                
                @jax.jit
                def do_update(noiser_params, params, fitnesses, iterinfos):
                    return EggRoll.do_updates(
                        frozen_noiser_params, noiser_params, params,
                        es_tree_key, fitnesses, iterinfos, es_map
                    )
                
                def run_eggroll_jax():
                    x_pop = jnp.broadcast_to(x_jax, (pop_size, batch_size, in_dim))
                    iterinfo = (jnp.zeros(pop_size, dtype=jnp.int32), jnp.arange(pop_size))
                    logits = jit_forward(noiser_params, params, iterinfo, x_pop)
                    fitnesses = logits.mean(axis=(1, 2))
                    fitnesses = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
                    return do_update(noiser_params, params, fitnesses, iterinfo)
                
                # Warmup
                _ = batched_inference_jax(params, x_batch_jax).block_until_ready()
                _ = run_eggroll_jax()
                for _ in range(num_warmup):
                    _ = batched_inference_jax(params, x_batch_jax).block_until_ready()
                    _ = run_eggroll_jax()
                
                # Time inference
                t0 = time.perf_counter()
                for _ in range(num_iters):
                    _ = batched_inference_jax(params, x_batch_jax).block_until_ready()
                inference_time = (time.perf_counter() - t0) / num_iters
                
                # Time EGGROLL
                t0 = time.perf_counter()
                for _ in range(num_iters):
                    _ = run_eggroll_jax()
                eggroll_time = (time.perf_counter() - t0) / num_iters
                
                efficiency = (inference_time / eggroll_time) * 100
                
                console.print(
                    f"  pop={pop_size:>5}: "
                    f"inference={inference_time*1000:.2f}ms, "
                    f"eggroll={eggroll_time*1000:.2f}ms, "
                    f"[bold]efficiency={efficiency:.1f}%[/bold]"
                )
                
                results.append({
                    "config": name, "pop_size": pop_size, "backend": "jax",
                    "inference_ms": inference_time * 1000,
                    "eggroll_ms": eggroll_time * 1000,
                    "efficiency": efficiency,
                })
                
                jax.clear_caches()
                
            except Exception as e:
                console.print(f"  pop={pop_size:>5}: [red]{e}[/red]")
                break
    
    return results


# =============================================================================
# Torch Inference Benchmark
# =============================================================================

def benchmark_inference_torch():
    """
    Torch-only inference benchmark using torch.compile for JAX-parity performance.
    """
    torch.set_float32_matmul_precision('high')
    
    console.print(f"\n[bold green]═══ Torch Inference Benchmark (Compiled) ═══[/bold green]")
    console.print("Using torch.compile for JAX-parity performance")
    console.print("Measuring training speed as % of pure inference speed")
    console.print()
    
    dtype = torch.float32
    
    configs = [("MLP 784->256->10", 784, 256, 10, 128)]
    population_sizes = [1024, 2048, 4096]
    rank = 4
    sigma = 0.1
    num_warmup = 20
    num_iters = 100
    
    results = []
    
    for name, in_dim, hidden_dim, out_dim, batch_size in configs:
        console.print(f"[cyan]{name}[/cyan]")
        
        # Warmup torch.compile once
        console.print("  [dim]Warming up torch.compile (one-time cost)...[/dim]")
        warmup_compiled_functions(population_sizes[0], batch_size, in_dim, hidden_dim, out_dim, rank, dtype)
        
        for pop_size in population_sizes:
            try:
                reset_gpu_stats()
                torch.cuda.empty_cache()
                
                W1 = torch.randn(hidden_dim, in_dim, device="cuda", dtype=dtype)
                W2 = torch.randn(out_dim, hidden_dim, device="cuda", dtype=dtype)
                b1 = torch.zeros(hidden_dim, device="cuda", dtype=dtype)
                b2 = torch.zeros(out_dim, device="cuda", dtype=dtype)
                x_batch = torch.randn(pop_size, batch_size, in_dim, device="cuda", dtype=dtype)
                
                gen = torch.Generator(device="cuda").manual_seed(42)
                A1_scaled, A1, B1 = generate_lowrank_perturbations(
                    pop_size, hidden_dim, in_dim, rank, sigma, gen, dtype
                )
                A2_scaled, A2, B2 = generate_lowrank_perturbations(
                    pop_size, out_dim, hidden_dim, rank, sigma, gen, dtype
                )
                sqrt_N = math.sqrt(pop_size)
                
                # Warmup compiled functions for this pop_size
                for _ in range(num_warmup):
                    _ = compiled_batched_inference(x_batch, W1, b1, W2, b2)
                    _ = compiled_eggroll_step(x_batch, W1, b1, W2, b2, A1_scaled, B1, A2_scaled, B2, sqrt_N)
                torch.cuda.synchronize()
                
                # Time compiled inference
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(num_iters):
                    _ = compiled_batched_inference(x_batch, W1, b1, W2, b2)
                torch.cuda.synchronize()
                inference_time = (time.perf_counter() - t0) / num_iters
                
                # Time compiled EGGROLL step
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(num_iters):
                    _ = compiled_eggroll_step(x_batch, W1, b1, W2, b2, A1_scaled, B1, A2_scaled, B2, sqrt_N)
                torch.cuda.synchronize()
                eggroll_time = (time.perf_counter() - t0) / num_iters
                
                efficiency = (inference_time / eggroll_time) * 100
                
                console.print(
                    f"  pop={pop_size:>5}: "
                    f"inf={inference_time*1000:.2f}ms | "
                    f"eggroll={eggroll_time*1000:.2f}ms | "
                    f"[bold]efficiency={efficiency:.1f}%[/bold]"
                )
                
                results.append({
                    "config": name, "pop_size": pop_size, "backend": "torch_compiled",
                    "inference_ms": inference_time * 1000,
                    "eggroll_ms": eggroll_time * 1000,
                    "efficiency": efficiency,
                })
                
                del W1, W2, b1, b2, x_batch
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                console.print(f"  pop={pop_size:>5}: [red]OOM[/red]")
                break
            except Exception as e:
                console.print(f"  pop={pop_size:>5}: [red]{e}[/red]")
                import traceback
                traceback.print_exc()
                break
    
    return results


def benchmark_inference_ratio():
    """
    Benchmark EGGROLL training time vs pure inference time.
    
    Key metric from EGGROLL paper (Figure 8):
    - EGGROLL achieves 91% (pre-gen noise) or 69% (on-the-fly noise) of pure inference throughput
    - PPO: 34%
    - OpenES: 0.41% (0.054% with noise regen)
    """
    console.print("[bold]Torch EGGROLL Inference Benchmark (with torch.compile)[/bold]")
    console.print()
    
    torch_results = benchmark_inference_torch()
    
    # Summary
    console.print()
    console.print("[bold]Summary - Training Efficiency (% of pure inference speed)[/bold]")
    table = Table()
    table.add_column("Population", justify="right")
    table.add_column("Inference", justify="right")
    table.add_column("EGGROLL", justify="right")
    table.add_column("Efficiency", justify="right")
    
    for r in torch_results:
        table.add_row(
            f"{r['pop_size']:,}",
            f"{r['inference_ms']:.2f}ms",
            f"{r['eggroll_ms']:.2f}ms",
            f"[bold]{r['efficiency']:.1f}%[/bold]"
        )
    
    console.print(table)
    return torch_results


# =============================================================================
# Hyperscale Tests - Find max population size
# =============================================================================

def main_hyperscale_torch():
    """Find maximum population size before OOM for Torch EGGROLL."""
    
    dtype = torch.float32
    
    console.print(f"\n[bold]Hyperscale Test - Torch EGGROLL[/bold]")
    console.print("Finding maximum population size before OOM...")
    console.print()
    
    configs = [
        ("CartPole MLP", 4, 256, 2),
        ("MNIST MLP", 784, 256, 10),
        ("MNIST CNN FC", 1568, 10, 0),
    ]
    
    results = []
    
    for name, in_dim, hidden_dim, out_dim in configs:
        if out_dim == 0:
            out_dim = hidden_dim
            hidden_dim = 0
        
        console.print(f"[cyan]Testing {name}...[/cyan]")
        
        pop_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        max_pop = 0
        rank = 4
        
        for pop_size in pop_sizes:
            try:
                reset_gpu_stats()
                torch.cuda.empty_cache()
                
                if hidden_dim > 0:
                    W1 = torch.randn(hidden_dim, in_dim, device="cuda", dtype=dtype)
                    W2 = torch.randn(out_dim, hidden_dim, device="cuda", dtype=dtype)
                    
                    A1 = torch.randn(pop_size, hidden_dim, rank, device="cuda", dtype=dtype)
                    B1 = torch.randn(pop_size, in_dim, rank, device="cuda", dtype=dtype)
                    A2 = torch.randn(pop_size, out_dim, rank, device="cuda", dtype=dtype)
                    B2 = torch.randn(pop_size, hidden_dim, rank, device="cuda", dtype=dtype)
                    
                    batch_size = 256 if in_dim < 100 else 64
                    x = torch.randn(pop_size, batch_size, in_dim, device="cuda", dtype=dtype)
                    h = x @ W1.T
                    h = h + torch.einsum('pbi,pir,pjr->pbj', x, B1, A1)
                    h = torch.relu(h)
                    out = h @ W2.T
                    out = out + torch.einsum('pbi,pir,pjr->pbj', h, B2, A2)
                else:
                    W = torch.randn(out_dim, in_dim, device="cuda", dtype=dtype)
                    A = torch.randn(pop_size, out_dim, rank, device="cuda", dtype=dtype)
                    B = torch.randn(pop_size, in_dim, rank, device="cuda", dtype=dtype)
                    
                    batch_size = 64
                    x = torch.randn(pop_size, batch_size, in_dim, device="cuda", dtype=dtype)
                    out = x @ W.T
                    out = out + torch.einsum('pbi,pir,pjr->pbj', x, B, A)
                
                torch.cuda.synchronize()
                
                stats = get_gpu_stats()
                max_pop = pop_size
                console.print(f"  pop={pop_size:,}: ✓ ({stats['allocated_gb']:.2f}GB)")
                
                del x, out
                if hidden_dim > 0:
                    del W1, W2, A1, B1, A2, B2, h
                else:
                    del W, A, B
                
            except torch.cuda.OutOfMemoryError:
                console.print(f"  pop={pop_size:,}: ✗ OOM")
                break
            except Exception as e:
                console.print(f"  pop={pop_size:,}: ✗ {e}")
                break
        
        results.append((name, max_pop))
        torch.cuda.empty_cache()
    
    console.print()
    console.print("[bold]Results:[/bold]")
    table = Table()
    table.add_column("Config")
    table.add_column("Max Population", justify="right")
    
    for name, max_pop in results:
        table.add_row(name, f"{max_pop:,}")
    
    console.print(table)
    print_gpu_stats("Final ")


def main_hyperscale_jax():
    """Find maximum population size before OOM for JAX EGGROLL."""
    import jax
    import jax.numpy as jnp
    
    console.print(f"\n[bold]Hyperscale Test - JAX EGGROLL[/bold]")
    console.print("Finding maximum population size before OOM...")
    console.print()
    
    in_dim = 784
    hidden_dim = 256
    out_dim = 10
    rank = 4
    
    pop_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    max_pop = 0
    
    for pop_size in pop_sizes:
        try:
            jax.clear_caches()
            
            W1 = jnp.ones((hidden_dim, in_dim))
            W2 = jnp.ones((out_dim, hidden_dim))
            
            A1 = jnp.ones((pop_size, hidden_dim, rank))
            B1 = jnp.ones((pop_size, in_dim, rank))
            A2 = jnp.ones((pop_size, out_dim, rank))
            B2 = jnp.ones((pop_size, hidden_dim, rank))
            
            batch_size = 64
            x = jnp.ones((pop_size, batch_size, in_dim))
            
            h = x @ W1.T + jnp.einsum('pbi,pir,pjr->pbj', x, B1, A1)
            h = jax.nn.relu(h)
            out = h @ W2.T + jnp.einsum('pbi,pir,pjr->pbj', h, B2, A2)
            
            _ = float(out.mean())
            
            max_pop = pop_size
            console.print(f"  pop={pop_size:,}: ✓")
            
        except Exception as e:
            console.print(f"  pop={pop_size:,}: ✗ {type(e).__name__}")
            break
    
    console.print()
    console.print(f"[bold]Max JAX population: {max_pop:,}[/bold]")


# =============================================================================
# Detailed Torch Profiling
# =============================================================================

def profile_torch_breakdown():
    """
    Profile where time is spent in the Torch EGGROLL forward pass.
    
    Breaks down:
    1. Noise generation
    2. Low-rank perturbation computation
    3. Base matmul
    4. Activation
    5. Fitness normalization
    6. ES gradient computation
    """
    console.print(f"\n[bold magenta]═══ Torch EGGROLL Detailed Profiling ═══[/bold magenta]")
    console.print("Breaking down where time is spent in the forward pass")
    console.print()
    
    dtype = torch.float32
    
    in_dim, hidden_dim, out_dim = 784, 256, 10
    batch_size = 128
    rank = 4
    sigma = 0.1
    num_warmup = 10
    num_iters = 50
    
    population_sizes = [1024, 2048, 4096, 8192]
    
    for pop_size in population_sizes:
        console.print(f"[cyan]Population: {pop_size:,}[/cyan]")
        
        try:
            reset_gpu_stats()
            torch.cuda.empty_cache()
            
            # Dict-based params
            params = {
                'layer1.weight': torch.randn(hidden_dim, in_dim, device="cuda", dtype=dtype),
                'layer1.bias': torch.zeros(hidden_dim, device="cuda", dtype=dtype),
                'layer2.weight': torch.randn(out_dim, hidden_dim, device="cuda", dtype=dtype),
                'layer2.bias': torch.zeros(out_dim, device="cuda", dtype=dtype),
            }
            shapes = get_weight_shapes(params)
            
            x = torch.randn(batch_size, in_dim, device="cuda", dtype=dtype)
            x_pop = x.unsqueeze(0).expand(pop_size, -1, -1)
            
            gen = torch.Generator(device="cuda").manual_seed(42)
            perts = generate_perturbations(shapes, pop_size, rank, sigma, gen, dtype)
            
            def time_noise_gen():
                gen = torch.Generator(device="cuda").manual_seed(42)
                return generate_perturbations(shapes, pop_size, rank, sigma, gen, dtype)
            
            def time_base_matmul():
                base1 = x_pop @ params['layer1.weight'].T + params['layer1.bias']
                h = torch.relu(base1)
                base2 = h @ params['layer2.weight'].T + params['layer2.bias']
                return base2
            
            def time_perturbation():
                A1, B1 = perts['layer1.weight']
                A2, B2 = perts['layer2.weight']
                pert1 = apply_lowrank_perturbation(x_pop, B1, A1)
                h = torch.relu(x_pop @ params['layer1.weight'].T + params['layer1.bias'] + pert1)
                pert2 = apply_lowrank_perturbation(h, B2, A2)
                return pert2
            
            def time_full_forward():
                h = torch.relu(perturbed_forward(x_pop, params['layer1.weight'], 
                                                  params['layer1.bias'], perts, 'layer1.weight'))
                return perturbed_forward(h, params['layer2.weight'], 
                                        params['layer2.bias'], perts, 'layer2.weight')
            
            def time_fitness_norm():
                h = torch.relu(perturbed_forward(x_pop, params['layer1.weight'], 
                                                  params['layer1.bias'], perts, 'layer1.weight'))
                logits = perturbed_forward(h, params['layer2.weight'], 
                                          params['layer2.bias'], perts, 'layer2.weight')
                fitnesses = normalize_fitnesses(logits.mean(dim=(1, 2)))
                return fitnesses
            
            def time_es_gradient():
                h = torch.relu(perturbed_forward(x_pop, params['layer1.weight'], 
                                                  params['layer1.bias'], perts, 'layer1.weight'))
                logits = perturbed_forward(h, params['layer2.weight'], 
                                          params['layer2.bias'], perts, 'layer2.weight')
                fitnesses = normalize_fitnesses(logits.mean(dim=(1, 2)))
                grads = compute_gradients(fitnesses, perts, pop_size)
                return grads
            
            def time_full_step():
                gen = torch.Generator(device="cuda").manual_seed(42)
                perts_local = generate_perturbations(shapes, pop_size, rank, sigma, gen, dtype)
                h = torch.relu(perturbed_forward(x_pop, params['layer1.weight'], 
                                                  params['layer1.bias'], perts_local, 'layer1.weight'))
                logits = perturbed_forward(h, params['layer2.weight'], 
                                          params['layer2.bias'], perts_local, 'layer2.weight')
                fitnesses = normalize_fitnesses(logits.mean(dim=(1, 2)))
                grads = compute_gradients(fitnesses, perts_local, pop_size)
                return grads
            
            # Warmup
            for _ in range(num_warmup):
                _ = time_noise_gen()
                _ = time_base_matmul()
                _ = time_perturbation()
                _ = time_full_forward()
                _ = time_fitness_norm()
                _ = time_es_gradient()
                _ = time_full_step()
            torch.cuda.synchronize()
            
            timings = {}
            
            def measure(name, fn, iters=num_iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    _ = fn()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) / iters * 1000
                timings[name] = elapsed
            
            measure("1_noise_gen", time_noise_gen)
            measure("2_base_matmul", time_base_matmul)
            measure("3_perturbation", time_perturbation)
            measure("4_full_forward", time_full_forward)
            measure("5_fitness_norm", time_fitness_norm)
            measure("6_es_gradient", time_es_gradient)
            measure("7_full_step", time_full_step)
            
            base_inference = timings["2_base_matmul"]
            overhead_pert = timings["4_full_forward"] - base_inference
            overhead_noise = timings["1_noise_gen"]
            
            efficiency_pregen = (base_inference / timings["6_es_gradient"]) * 100
            efficiency_onthefly = (base_inference / timings["7_full_step"]) * 100
            
            console.print(f"  {'Component':<20} {'Time (ms)':>10} {'% of full step':>15}")
            console.print(f"  {'-'*20} {'-'*10} {'-'*15}")
            full = timings["7_full_step"]
            for name, t in sorted(timings.items()):
                pct = (t / full) * 100
                console.print(f"  {name[2:]:<20} {t:>10.3f} {pct:>14.1f}%")
            
            console.print()
            console.print(f"  [bold]Analysis:[/bold]")
            console.print(f"    Base inference:       {base_inference:.3f}ms")
            console.print(f"    Perturbation overhead:{overhead_pert:.3f}ms ({(overhead_pert/base_inference)*100:.1f}% of base)")
            console.print(f"    Noise gen overhead:   {overhead_noise:.3f}ms ({(overhead_noise/base_inference)*100:.1f}% of base)")
            console.print()
            console.print(f"    [green]Pre-gen efficiency:    {efficiency_pregen:.1f}%[/green] (paper target: 91%)")
            console.print(f"    [green]On-the-fly efficiency: {efficiency_onthefly:.1f}%[/green] (paper target: 69%)")
            console.print()
            
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()


def profile_torch_optimizations():
    """
    Compare different optimization strategies for the perturbation operation.
    
    Tests:
    1. einsum (current)
    2. bmm (two separate matmuls)
    3. torch.compile on einsum
    4. torch.compile on fused matmul+pert
    """
    console.print(f"\n[bold magenta]═══ Torch Optimization Comparison ═══[/bold magenta]")
    console.print("Comparing different strategies for low-rank perturbation")
    console.print()
    
    dtype = torch.float32
    
    in_dim, hidden_dim, out_dim = 784, 256, 10
    batch_size = 128
    rank = 4
    sigma = 0.1
    num_warmup = 20
    num_iters = 100
    
    population_sizes = [1024, 2048, 4096, 8192]
    
    for pop_size in population_sizes:
        console.print(f"[cyan]Population: {pop_size:,}[/cyan]")
        
        try:
            reset_gpu_stats()
            torch.cuda.empty_cache()
            
            W = torch.randn(hidden_dim, in_dim, device="cuda", dtype=dtype)
            x = torch.randn(pop_size, batch_size, in_dim, device="cuda", dtype=dtype)
            
            gen = torch.Generator(device="cuda").manual_seed(42)
            A_scaled, A, B = generate_lowrank_perturbations(
                pop_size, hidden_dim, in_dim, rank, sigma, gen, dtype
            )
            
            def impl_einsum():
                return apply_lowrank_perturbation(x, B, A_scaled)
            
            def impl_bmm():
                return apply_lowrank_perturbation_bmm(x, B, A_scaled)
            
            def impl_fused():
                return fused_matmul_with_lowrank(x, W, B, A_scaled)
            
            def impl_base_only():
                return x @ W.T
            
            compiled_einsum = torch.compile(impl_einsum, mode="reduce-overhead")
            compiled_bmm = torch.compile(impl_bmm, mode="reduce-overhead")
            compiled_fused = torch.compile(impl_fused, mode="reduce-overhead")
            compiled_base = torch.compile(impl_base_only, mode="reduce-overhead")
            
            # Warmup all
            for _ in range(num_warmup):
                _ = impl_einsum()
                _ = impl_bmm()
                _ = impl_fused()
                _ = impl_base_only()
            torch.cuda.synchronize()
            
            # Warmup compiled
            for _ in range(3):
                _ = compiled_einsum()
                _ = compiled_bmm()
                _ = compiled_fused()
                _ = compiled_base()
            torch.cuda.synchronize()
            
            for _ in range(num_warmup):
                _ = compiled_einsum()
                _ = compiled_bmm()
                _ = compiled_fused()
                _ = compiled_base()
            torch.cuda.synchronize()
            
            timings = {}
            
            def measure(name, fn, iters=num_iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    _ = fn()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) / iters * 1000
                timings[name] = elapsed
            
            measure("base_only", impl_base_only)
            measure("einsum", impl_einsum)
            measure("bmm", impl_bmm)
            measure("fused", impl_fused)
            measure("compiled_base", compiled_base)
            measure("compiled_einsum", compiled_einsum)
            measure("compiled_bmm", compiled_bmm)
            measure("compiled_fused", compiled_fused)
            
            base = timings["base_only"]
            compiled_base_t = timings["compiled_base"]
            
            console.print(f"  {'Method':<20} {'Time (ms)':>10} {'Efficiency':>12}")
            console.print(f"  {'-'*20} {'-'*10} {'-'*12}")
            
            console.print(f"  {'base_only':<20} {base:>10.3f} {'100.0%':>12}")
            for name in ["einsum", "bmm", "fused"]:
                t = timings[name]
                eff = (base / t) * 100
                console.print(f"  {name:<20} {t:>10.3f} {eff:>11.1f}%")
            
            console.print(f"  {'compiled_base':<20} {compiled_base_t:>10.3f} {'-':>12}")
            for name in ["compiled_einsum", "compiled_bmm", "compiled_fused"]:
                t = timings[name]
                eff = (compiled_base_t / t) * 100
                console.print(f"  {name:<20} {t:>10.3f} {eff:>11.1f}%")
            
            console.print()
            
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()


# =============================================================================
# Head-to-Head: JAX vs Torch
# =============================================================================

def profile_head_to_head():
    """
    Head-to-head comparison of JAX vs Torch EGGROLL.
    
    Measures identical operations on both backends:
    1. Pure inference (batched forward without perturbation)
    2. EGGROLL forward (with perturbation, pre-gen noise)
    3. Full EGGROLL step (forward + ES gradient computation)
    """
    import jax
    import jax.numpy as jnp
    import optax
    from hyperscalees.noiser.eggroll import EggRoll
    from hyperscalees.models.common import MLP, simple_es_tree_key
    
    console.print(f"\n[bold cyan]═══ JAX vs Torch Head-to-Head Benchmark ═══[/bold cyan]")
    console.print("Comparing Torch EGGROLL against JAX EGGROLL (source of truth)")
    console.print()
    
    in_dim, hidden_dim, out_dim = 784, 256, 10
    batch_size = 128
    rank = 4
    sigma = 0.1
    num_warmup = 10
    num_iters = 50
    seed = 42
    
    population_sizes = [1024, 2048, 4096, 8192]
    
    results = []
    
    for pop_size in population_sizes:
        console.print(f"[bold]Population: {pop_size:,}[/bold]")
        
        # ===== JAX Setup =====
        try:
            jax.clear_caches()
            
            key = jax.random.key(seed)
            model_key = jax.random.fold_in(key, 0)
            es_key = jax.random.fold_in(key, 1)
            
            frozen_params, params, scan_map, es_map = MLP.rand_init(
                model_key,
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dims=[hidden_dim],
                use_bias=True,
                activation="relu",
                dtype="float32",
            )
            
            es_tree_key = simple_es_tree_key(params, es_key, scan_map)
            
            frozen_noiser_params, noiser_params = EggRoll.init_noiser(
                params,
                sigma=sigma,
                lr=0.1,
                solver=optax.sgd,
                rank=rank,
            )
            
            x_jax = jnp.ones((batch_size, in_dim))
            x_pop_jax = jnp.broadcast_to(x_jax, (pop_size, batch_size, in_dim))
            
            @jax.jit
            def jax_inference(params, x):
                def single_forward(x):
                    h = jax.nn.relu(x @ params['0']['weight'].T + params['0']['bias'])
                    return h @ params['1']['weight'].T + params['1']['bias']
                return jax.vmap(single_forward)(x)
            
            def forward_noisy(noiser_params, params, iterinfo, x):
                return MLP.forward(
                    EggRoll, frozen_noiser_params, noiser_params, frozen_params,
                    params, es_tree_key, iterinfo, x
                )
            
            jit_forward_jax = jax.jit(jax.vmap(
                lambda n, p, i, x: forward_noisy(n, p, i, x),
                in_axes=(None, None, 0, 0)
            ))
            
            @jax.jit
            def jax_do_update(noiser_params, params, fitnesses, iterinfos):
                return EggRoll.do_updates(
                    frozen_noiser_params, noiser_params, params,
                    es_tree_key, fitnesses, iterinfos, es_map
                )
            
            def jax_eggroll_forward():
                iterinfo = (jnp.zeros(pop_size, dtype=jnp.int32), jnp.arange(pop_size))
                return jit_forward_jax(noiser_params, params, iterinfo, x_pop_jax)
            
            def jax_full_step():
                iterinfo = (jnp.zeros(pop_size, dtype=jnp.int32), jnp.arange(pop_size))
                logits = jit_forward_jax(noiser_params, params, iterinfo, x_pop_jax)
                fitnesses = logits.mean(axis=(1, 2))
                fitnesses = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
                return jax_do_update(noiser_params, params, fitnesses, iterinfo)
            
            # Warmup JAX
            _ = jax_inference(params, x_pop_jax).block_until_ready()
            _ = jax_eggroll_forward().block_until_ready()
            _ = jax_full_step()
            
            for _ in range(num_warmup):
                _ = jax_inference(params, x_pop_jax).block_until_ready()
                _ = jax_eggroll_forward().block_until_ready()
                _ = jax_full_step()
            
            # Time JAX
            t0 = time.perf_counter()
            for _ in range(num_iters):
                _ = jax_inference(params, x_pop_jax).block_until_ready()
            jax_inference_time = (time.perf_counter() - t0) / num_iters * 1000
            
            t0 = time.perf_counter()
            for _ in range(num_iters):
                _ = jax_eggroll_forward().block_until_ready()
            jax_eggroll_time = (time.perf_counter() - t0) / num_iters * 1000
            
            t0 = time.perf_counter()
            for _ in range(num_iters):
                _ = jax_full_step()
            jax_full_time = (time.perf_counter() - t0) / num_iters * 1000
            
            jax_efficiency = (jax_inference_time / jax_full_time) * 100
            
        except Exception as e:
            console.print(f"  [red]JAX Error: {e}[/red]")
            jax_inference_time = jax_eggroll_time = jax_full_time = jax_efficiency = float('nan')
        
        # ===== Torch Setup =====
        try:
            torch.cuda.empty_cache()
            reset_gpu_stats()
            torch.set_float32_matmul_precision('high')
            
            dtype = torch.float32
            
            torch.manual_seed(seed)
            
            # Dict-based params
            params = {
                'layer1.weight': torch.randn(hidden_dim, in_dim, device="cuda", dtype=dtype),
                'layer1.bias': torch.zeros(hidden_dim, device="cuda", dtype=dtype),
                'layer2.weight': torch.randn(out_dim, hidden_dim, device="cuda", dtype=dtype),
                'layer2.bias': torch.zeros(out_dim, device="cuda", dtype=dtype),
            }
            shapes = get_weight_shapes(params)
            
            x_torch = torch.ones(batch_size, in_dim, device="cuda", dtype=dtype)
            x_pop_torch = x_torch.unsqueeze(0).expand(pop_size, -1, -1).contiguous()
            
            gen = torch.Generator(device="cuda").manual_seed(seed)
            perts = generate_perturbations(shapes, pop_size, rank, sigma, gen, dtype)
            
            def torch_inference():
                h = torch.relu(x_pop_torch @ params['layer1.weight'].T + params['layer1.bias'])
                return h @ params['layer2.weight'].T + params['layer2.bias']
            
            def torch_eggroll_forward():
                h = torch.relu(perturbed_forward(x_pop_torch, params['layer1.weight'], 
                                                  params['layer1.bias'], perts, 'layer1.weight'))
                return perturbed_forward(h, params['layer2.weight'], 
                                        params['layer2.bias'], perts, 'layer2.weight')
            
            def torch_full_step():
                h = torch.relu(perturbed_forward(x_pop_torch, params['layer1.weight'], 
                                                  params['layer1.bias'], perts, 'layer1.weight'))
                logits = perturbed_forward(h, params['layer2.weight'], 
                                          params['layer2.bias'], perts, 'layer2.weight')
                fitnesses = normalize_fitnesses(logits.mean(dim=(1, 2)))
                grads = compute_gradients(fitnesses, perts, pop_size)
                return grads
            
            compiled_inference = torch.compile(torch_inference)
            compiled_forward = torch.compile(torch_eggroll_forward)
            compiled_full_step = torch.compile(torch_full_step)
            
            # Warmup eager
            for _ in range(num_warmup):
                _ = torch_inference()
                _ = torch_eggroll_forward()
                _ = torch_full_step()
            torch.cuda.synchronize()
            
            # Warmup compiled
            for _ in range(5):
                _ = compiled_inference()
                _ = compiled_forward()
                _ = compiled_full_step()
            torch.cuda.synchronize()
            
            for _ in range(num_warmup):
                _ = compiled_inference()
                _ = compiled_forward()
                _ = compiled_full_step()
            torch.cuda.synchronize()
            
            # Time Torch eager
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_iters):
                _ = torch_inference()
            torch.cuda.synchronize()
            torch_inference_time = (time.perf_counter() - t0) / num_iters * 1000
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_iters):
                _ = torch_full_step()
            torch.cuda.synchronize()
            torch_full_time = (time.perf_counter() - t0) / num_iters * 1000
            
            torch_efficiency = (torch_inference_time / torch_full_time) * 100
            
            # Time Torch compiled
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_iters):
                _ = compiled_inference()
            torch.cuda.synchronize()
            torch_compiled_inference_time = (time.perf_counter() - t0) / num_iters * 1000
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_iters):
                _ = compiled_full_step()
            torch.cuda.synchronize()
            torch_compiled_full_time = (time.perf_counter() - t0) / num_iters * 1000
            
            torch_compiled_efficiency = (torch_compiled_inference_time / torch_compiled_full_time) * 100
            
        except Exception as e:
            console.print(f"  [red]Torch Error: {e}[/red]")
            import traceback
            traceback.print_exc()
            torch_inference_time = torch_full_time = torch_efficiency = float('nan')
            torch_compiled_inference_time = torch_compiled_full_time = torch_compiled_efficiency = float('nan')
        
        # Print results
        console.print(f"  {'Metric':<30} {'JAX':>10} {'Torch':>10} {'Compiled':>10}")
        console.print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
        
        console.print(f"  {'Pure inference (ms)':<30} {jax_inference_time:>10.2f} {torch_inference_time:>10.2f} {torch_compiled_inference_time:>10.2f}")
        console.print(f"  {'Full EGGROLL step (ms)':<30} {jax_full_time:>10.2f} {torch_full_time:>10.2f} {torch_compiled_full_time:>10.2f}")
        console.print(f"  {'Efficiency (inf/full)':<30} {jax_efficiency:>9.1f}% {torch_efficiency:>9.1f}% {torch_compiled_efficiency:>9.1f}%")
        console.print(f"  {'vs JAX speed':<30} {'-':>10} {jax_full_time/torch_full_time:>9.2f}x {jax_full_time/torch_compiled_full_time:>9.2f}x")
        console.print()
        
        results.append({
            "pop_size": pop_size,
            "jax_inference_ms": jax_inference_time,
            "jax_full_ms": jax_full_time,
            "jax_efficiency": jax_efficiency,
            "torch_inference_ms": torch_inference_time,
            "torch_full_ms": torch_full_time,
            "torch_efficiency": torch_efficiency,
            "torch_compiled_inference_ms": torch_compiled_inference_time,
            "torch_compiled_full_ms": torch_compiled_full_time,
            "torch_compiled_efficiency": torch_compiled_efficiency,
        })
        
        jax.clear_caches()
    
    console.print("[bold]Summary:[/bold]")
    console.print("  Efficiency = pure_inference_time / full_step_time")
    console.print("  Higher efficiency = EGGROLL overhead is lower relative to inference")
    console.print("  vs JAX speed > 1.0 means Torch is faster than JAX")
    console.print()
    console.print("  [green]Target: Match or exceed JAX efficiency and speed[/green]")
    
    return results


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EGGROLL Benchmarks and Profiling")
    parser.add_argument(
        "--benchmark", 
        choices=["inference", "head_to_head", "profile", "optimizations", "hyperscale", "all"], 
        default="inference",
        help="Which benchmark to run"
    )
    parser.add_argument(
        "--backend", 
        choices=["torch", "jax", "both"], 
        default="both",
        help="Backend to benchmark (for hyperscale)"
    )
    args = parser.parse_args()
    
    if args.benchmark == "inference":
        benchmark_inference_ratio()
    
    elif args.benchmark == "head_to_head":
        profile_head_to_head()
    
    elif args.benchmark == "profile":
        profile_torch_breakdown()
    
    elif args.benchmark == "optimizations":
        profile_torch_optimizations()
    
    elif args.benchmark == "hyperscale":
        if args.backend in ("torch", "both"):
            main_hyperscale_torch()
        if args.backend in ("jax", "both"):
            main_hyperscale_jax()
    
    elif args.benchmark == "all":
        console.print("[bold cyan]═══ Inference Benchmark ═══[/bold cyan]")
        benchmark_inference_ratio()
        
        console.print("\n[bold cyan]═══ Head-to-Head ═══[/bold cyan]")
        profile_head_to_head()
        
        console.print("\n[bold cyan]═══ Profile Breakdown ═══[/bold cyan]")
        profile_torch_breakdown()
        
        console.print("\n[bold cyan]═══ Optimizations ═══[/bold cyan]")
        profile_torch_optimizations()
        
        console.print("\n[bold cyan]═══ Hyperscale ═══[/bold cyan]")
        if args.backend in ("torch", "both"):
            main_hyperscale_torch()
        if args.backend in ("jax", "both"):
            main_hyperscale_jax()
