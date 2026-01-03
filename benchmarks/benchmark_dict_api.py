#!/usr/bin/env python3
"""
Benchmark: Compare Dict-Based API vs Raw Primitives API

This verifies the new dict-based API has zero overhead compared to
raw functional primitives.
"""

import torch
import time
import math
from rich.console import Console
from rich.table import Table

console = Console()

# Import both APIs
from hyperscalees.torch.fncl.eggroll_fncl import (
    # Raw Primitives API
    generate_lowrank_perturbations,
    perturbed_linear,
    compute_es_gradient,
    normalize_fitnesses,
    mlp_forward_perturbed,
    mlp_forward_clean,
    init_mlp_params,
    # Dict-Based API
    get_weight_shapes,
    generate_perturbations,
    compute_gradients,
    update_params,
    perturbed_forward,
)


@torch.compile
def forward_raw_api(x, W1, b1, W2, b2, A1, B1, A2, B2):
    """Forward pass using raw primitives API."""
    h = torch.tanh(perturbed_linear(x, W1, b1, A1, B1))
    return perturbed_linear(h, W2, b2, A2, B2)


@torch.compile
def forward_dict_api(x, params, perts):
    """Forward pass using dict-based API."""
    h = torch.tanh(perturbed_forward(x, params['layer1.weight'], params['layer1.bias'], 
                                      perts, 'layer1.weight'))
    return perturbed_forward(h, params['layer2.weight'], params['layer2.bias'],
                             perts, 'layer2.weight')


def benchmark_api_comparison():
    """Compare performance of raw primitives vs dict-based API."""
    torch.set_float32_matmul_precision('high')
    
    console.print(f"\n[bold cyan]═══ API Overhead Benchmark ═══[/bold cyan]")
    console.print("Comparing Raw Primitives API vs Dict-Based API")
    console.print()
    
    # Config
    dtype = torch.float32
    in_dim = 4
    hidden_dim = 256
    out_dim = 2
    pop_size = 2048
    rank = 4
    sigma = 0.1
    lr = 0.1
    num_warmup = 100
    num_iters = 1000
    
    # ==========================================================================
    # Setup: Raw Primitives API
    # ==========================================================================
    W1, b1, W2, b2 = init_mlp_params(in_dim, hidden_dim, out_dim, dtype)
    gen = torch.Generator(device="cuda").manual_seed(42)
    
    # ==========================================================================
    # Setup: Dict-Based API
    # ==========================================================================
    params = {
        'layer1.weight': W1.clone(),
        'layer1.bias': b1.clone(),
        'layer2.weight': W2.clone(),
        'layer2.bias': b2.clone(),
    }
    shapes = get_weight_shapes(params)
    
    console.print(f"Config: {in_dim} -> {hidden_dim} -> {out_dim}")
    console.print(f"Population: {pop_size}, Rank: {rank}")
    console.print(f"Shapes: {shapes}")
    console.print()
    
    # ==========================================================================
    # Test Input
    # ==========================================================================
    x = torch.randn(pop_size, in_dim, device="cuda", dtype=dtype)
    
    # ==========================================================================
    # Warmup both APIs (more iterations to fully compile)
    # ==========================================================================
    console.print("[dim]Warming up torch.compile...[/dim]")
    
    # Warmup raw API first
    for _ in range(num_warmup):
        gen.manual_seed(42)
        A1, _, B1 = generate_lowrank_perturbations(pop_size, hidden_dim, in_dim, rank, sigma, gen, dtype)
        A2, _, B2 = generate_lowrank_perturbations(pop_size, out_dim, hidden_dim, rank, sigma, gen, dtype)
        out1 = forward_raw_api(x, W1, b1, W2, b2, A1, B1, A2, B2)
        fitnesses = out1.mean(dim=-1)
        f = normalize_fitnesses(fitnesses)
        g1 = compute_es_gradient(f, A1, B1, pop_size)
        g2 = compute_es_gradient(f, A2, B2, pop_size)
    torch.cuda.synchronize()
    
    # Warmup dict API
    for _ in range(num_warmup):
        gen.manual_seed(42)
        perts = generate_perturbations(shapes, pop_size, rank, sigma, gen, dtype)
        out2 = forward_dict_api(x, params, perts)
        fitnesses = out2.mean(dim=-1)
        f = normalize_fitnesses(fitnesses)
        grads = compute_gradients(f, perts, pop_size)
    torch.cuda.synchronize()
    
    # ==========================================================================
    # Verify correctness
    # ==========================================================================
    gen.manual_seed(123)
    A1, _, B1 = generate_lowrank_perturbations(pop_size, hidden_dim, in_dim, rank, sigma, gen, dtype)
    A2, _, B2 = generate_lowrank_perturbations(pop_size, out_dim, hidden_dim, rank, sigma, gen, dtype)
    out_raw = forward_raw_api(x, W1, b1, W2, b2, A1, B1, A2, B2)
    
    gen.manual_seed(123)
    perts = generate_perturbations(shapes, pop_size, rank, sigma, gen, dtype)
    out_dict = forward_dict_api(x, params, perts)
    
    diff = (out_raw - out_dict).abs().max().item()
    console.print(f"[green]Max difference: {diff:.2e}[/green] (should be ~0)")
    console.print()
    
    # ==========================================================================
    # Benchmark: Full training step
    # ==========================================================================
    console.print(f"[cyan]Benchmarking full training step ({num_iters} iterations)...[/cyan]")
    
    # Raw API timing
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(num_iters):
        gen.manual_seed(i)
        A1, _, B1 = generate_lowrank_perturbations(pop_size, hidden_dim, in_dim, rank, sigma, gen, dtype)
        A2, _, B2 = generate_lowrank_perturbations(pop_size, out_dim, hidden_dim, rank, sigma, gen, dtype)
        out = forward_raw_api(x, W1, b1, W2, b2, A1, B1, A2, B2)
        fitnesses = out.mean(dim=-1)
        f = normalize_fitnesses(fitnesses)
        g1 = compute_es_gradient(f, A1, B1, pop_size)
        g2 = compute_es_gradient(f, A2, B2, pop_size)
        W1 = W1 + lr * g1
        W2 = W2 + lr * g2
    torch.cuda.synchronize()
    raw_time = (time.perf_counter() - t0) / num_iters * 1000
    raw_steps_per_sec = pop_size / (raw_time / 1000)
    
    console.print(f"  Raw API:  {raw_time:.3f} ms/step  ({raw_steps_per_sec/1e6:.2f}M steps/s)")
    
    # Reset params
    W1, b1, W2, b2 = init_mlp_params(in_dim, hidden_dim, out_dim, dtype)
    params = {
        'layer1.weight': W1.clone(),
        'layer1.bias': b1.clone(),
        'layer2.weight': W2.clone(),
        'layer2.bias': b2.clone(),
    }
    
    # Dict API timing
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(num_iters):
        gen.manual_seed(i)
        perts = generate_perturbations(shapes, pop_size, rank, sigma, gen, dtype)
        out = forward_dict_api(x, params, perts)
        fitnesses = out.mean(dim=-1)
        f = normalize_fitnesses(fitnesses)
        grads = compute_gradients(f, perts, pop_size)
        update_params(params, grads, lr)
    torch.cuda.synchronize()
    dict_time = (time.perf_counter() - t0) / num_iters * 1000
    dict_steps_per_sec = pop_size / (dict_time / 1000)
    
    console.print(f"  Dict API: {dict_time:.3f} ms/step  ({dict_steps_per_sec/1e6:.2f}M steps/s)")
    
    # ==========================================================================
    # Results
    # ==========================================================================
    overhead = (dict_time - raw_time) / raw_time * 100
    
    console.print()
    if abs(overhead) < 5:
        console.print(f"[bold green]✓ Dict API overhead: {overhead:+.1f}% (within 5% tolerance)[/bold green]")
    else:
        console.print(f"[bold red]✗ Dict API overhead: {overhead:+.1f}% (exceeds 5% tolerance)[/bold red]")
    
    console.print()
    console.print("[dim]Note: Small variations are normal due to Python dict overhead.[/dim]")
    console.print("[dim]The dict API enables cleaner code with negligible performance cost.[/dim]")


if __name__ == "__main__":
    benchmark_api_comparison()
