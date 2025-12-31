#!/usr/bin/env python3
"""
Computational Anatomy Profiler for EGGROLL

Breaks down the forward pass into its component operations to identify
optimization opportunities for Triton/CUDA C++.

Operations profiled:
1. Noise generation (torch.randn vs JAX random.normal)
2. Low-rank factor computation (antithetic handling, sigma scaling)
3. Base linear (x @ W.T + bias)
4. Low-rank perturbation (x @ B @ A.T) 
5. Memory transfers (if any)
"""
import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Callable
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class OpProfile:
    """Profile for a single operation."""
    name: str
    total_time_ms: float
    num_calls: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    percentage: float = 0.0


class CUDATimer:
    """High-precision CUDA timer using events."""
    
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    @contextmanager
    def time(self):
        """Context manager that returns elapsed time in ms."""
        self.start_event.record()
        yield
        self.end_event.record()
        torch.cuda.synchronize()
    
    def elapsed_ms(self) -> float:
        return self.start_event.elapsed_time(self.end_event)


def profile_operation(
    name: str,
    op: Callable,
    num_iterations: int = 100,
    warmup: int = 10,
) -> OpProfile:
    """Profile a single operation."""
    timer = CUDATimer()
    times = []
    
    # Warmup
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()
    
    # Measure
    for _ in range(num_iterations):
        with timer.time():
            op()
        times.append(timer.elapsed_ms())
    
    times = np.array(times)
    return OpProfile(
        name=name,
        total_time_ms=times.sum(),
        num_calls=num_iterations,
        mean_time_ms=times.mean(),
        std_time_ms=times.std(),
        min_time_ms=times.min(),
        max_time_ms=times.max(),
    )


def profile_torch_anatomy(
    pop_size: int = 2048,
    obs_dim: int = 4,
    act_dim: int = 2,
    layer_size: int = 256,
    n_layers: int = 3,
    rank: int = 4,
    num_iterations: int = 100,
) -> Dict[str, OpProfile]:
    """Profile individual operations in the PyTorch EGGROLL implementation."""
    print("\n" + "="*70)
    print("PyTorch EGGROLL - Computational Anatomy")
    print("="*70)
    
    device = torch.device('cuda')
    dtype = torch.float32
    
    # Setup
    # Layer dimensions for a 3-layer MLP: 4 -> 256 -> 256 -> 2
    layer_dims = [(obs_dim, layer_size)]  # First hidden
    for _ in range(n_layers - 2):
        layer_dims.append((layer_size, layer_size))
    layer_dims.append((layer_size, act_dim))  # Output
    
    print(f"\nLayer dimensions: {layer_dims}")
    print(f"Population size: {pop_size}")
    print(f"Rank: {rank}")
    
    # Pre-allocate tensors
    x = torch.randn(pop_size, obs_dim, device=device, dtype=dtype)
    member_ids = torch.arange(pop_size, device=device)
    
    profiles = {}
    
    # === Profile noise generation ===
    print("\n--- Profiling: Noise Generation ---")
    
    # For largest layer
    in_features, out_features = layer_dims[0][0], layer_dims[0][1]
    total_features = in_features + out_features
    num_unique = pop_size // 2  # Antithetic
    
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    
    def noise_gen_randn():
        return torch.randn(
            num_unique, total_features, rank,
            generator=gen, device=device, dtype=dtype
        )
    
    profiles["1_noise_generation"] = profile_operation(
        "Noise Generation (torch.randn)",
        noise_gen_randn,
        num_iterations=num_iterations,
    )
    
    # === Profile antithetic expansion ===
    print("--- Profiling: Antithetic Expansion ---")
    
    noise = torch.randn(num_unique, total_features, rank, device=device, dtype=dtype)
    
    def antithetic_expand():
        return noise.repeat_interleave(2, dim=0)
    
    profiles["2_antithetic_expand"] = profile_operation(
        "Antithetic Expansion (repeat_interleave)",
        antithetic_expand,
        num_iterations=num_iterations,
    )
    
    # === Profile factor splitting and sigma scaling ===
    print("--- Profiling: Factor Split + Sigma Scaling ---")
    
    noise_expanded = noise.repeat_interleave(2, dim=0)
    scaled_sigma = 0.2 / (rank ** 0.5)
    signs = torch.where(
        (member_ids % 2 == 0).view(-1, 1, 1),
        torch.ones(1, device=device, dtype=dtype),
        -torch.ones(1, device=device, dtype=dtype)
    )
    
    def factor_split_scale():
        B = noise_expanded[:, :in_features, :]
        A_base = noise_expanded[:, in_features:, :] * scaled_sigma
        A = A_base * signs
        return A, B
    
    profiles["3_factor_split_scale"] = profile_operation(
        "Factor Split + Sigma Scaling",
        factor_split_scale,
        num_iterations=num_iterations,
    )
    
    # === Profile base linear ===
    print("--- Profiling: Base Linear (F.linear) ---")
    
    weight = torch.randn(out_features, in_features, device=device, dtype=dtype)
    bias = torch.randn(out_features, device=device, dtype=dtype)
    x_layer = torch.randn(pop_size, in_features, device=device, dtype=dtype)
    
    def base_linear():
        return F.linear(x_layer, weight, bias)
    
    profiles["4_base_linear"] = profile_operation(
        "Base Linear (F.linear)",
        base_linear,
        num_iterations=num_iterations,
    )
    
    # === Profile low-rank perturbation ===
    print("--- Profiling: Low-Rank Perturbation (x @ B @ A.T) ---")
    
    A, B = factor_split_scale()
    
    def lowrank_perturbation():
        # x @ B: (pop, in) @ (pop, in, rank) -> need bmm
        xB = torch.bmm(x_layer.unsqueeze(1), B).squeeze(1)  # (pop, rank)
        # xB @ A.T: (pop, rank) @ (pop, rank, out) -> need bmm
        return torch.bmm(xB.unsqueeze(1), A.transpose(1, 2)).squeeze(1)  # (pop, out)
    
    profiles["5_lowrank_perturbation"] = profile_operation(
        "Low-Rank Perturbation (bmm)",
        lowrank_perturbation,
        num_iterations=num_iterations,
    )
    
    # === Profile combined perturbed linear ===
    print("--- Profiling: Combined Perturbed Linear ---")
    
    def perturbed_linear_combined():
        base = F.linear(x_layer, weight, bias)
        xB = torch.bmm(x_layer.unsqueeze(1), B).squeeze(1)
        pert = torch.bmm(xB.unsqueeze(1), A.transpose(1, 2)).squeeze(1)
        return base + pert
    
    profiles["6_perturbed_linear_combined"] = profile_operation(
        "Combined Perturbed Linear",
        perturbed_linear_combined,
        num_iterations=num_iterations,
    )
    
    # === Profile full forward pass (all layers) ===
    print("--- Profiling: Full Forward Pass (all layers) ---")
    
    from hyperscalees.torch import EggrollStrategy
    
    # Build model
    layers = []
    in_dim = obs_dim
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(in_dim, layer_size))
        layers.append(nn.Tanh())
        in_dim = layer_size
    layers.append(nn.Linear(in_dim, act_dim))
    model = nn.Sequential(*layers).cuda()
    
    strategy = EggrollStrategy(
        sigma=0.2,
        rank=rank,
        lr=0.1,
        seed=42,
        antithetic=True,
    )
    strategy.setup(model)
    
    obs_batch = torch.randn(pop_size, obs_dim, device=device, dtype=dtype)
    
    # Warmup
    for warmup_epoch in range(10):
        with strategy.perturb(population_size=pop_size, epoch=warmup_epoch) as pop:
            with torch.no_grad():
                _ = pop.batched_forward(model, obs_batch)
    torch.cuda.synchronize()
    
    def full_forward():
        with strategy.perturb(population_size=pop_size, epoch=0) as pop:
            with torch.no_grad():
                return pop.batched_forward(model, obs_batch)
    
    profiles["7_full_forward_pass"] = profile_operation(
        "Full Forward Pass (all layers)",
        full_forward,
        num_iterations=num_iterations,
    )
    
    # === Profile activation function ===
    print("--- Profiling: Activation (Tanh) ---")
    
    hidden = torch.randn(pop_size, layer_size, device=device, dtype=dtype)
    
    def activation_tanh():
        return torch.tanh(hidden)
    
    profiles["8_activation_tanh"] = profile_operation(
        "Activation (Tanh)",
        activation_tanh,
        num_iterations=num_iterations,
    )
    
    # Compute percentages
    total_time = profiles["7_full_forward_pass"].total_time_ms
    for name, prof in profiles.items():
        if name != "7_full_forward_pass":
            prof.percentage = (prof.total_time_ms / total_time) * 100
    profiles["7_full_forward_pass"].percentage = 100.0
    
    return profiles


def profile_potential_optimizations(
    pop_size: int = 2048,
    layer_size: int = 256,
    rank: int = 4,
    num_iterations: int = 100,
) -> Dict[str, OpProfile]:
    """Profile potential optimization opportunities."""
    print("\n" + "="*70)
    print("Potential Optimization Opportunities")
    print("="*70)
    
    device = torch.device('cuda')
    dtype = torch.float32
    
    profiles = {}
    
    in_features = layer_size
    out_features = layer_size
    
    # === Current: Two separate bmm calls ===
    print("\n--- Current: Two separate bmm ---")
    
    x = torch.randn(pop_size, in_features, device=device, dtype=dtype)
    A = torch.randn(pop_size, out_features, rank, device=device, dtype=dtype)
    B = torch.randn(pop_size, in_features, rank, device=device, dtype=dtype)
    
    def two_bmm():
        xB = torch.bmm(x.unsqueeze(1), B).squeeze(1)
        return torch.bmm(xB.unsqueeze(1), A.transpose(1, 2)).squeeze(1)
    
    profiles["current_two_bmm"] = profile_operation(
        "Current: Two bmm calls",
        two_bmm,
        num_iterations=num_iterations,
    )
    
    # === Alternative: einsum ===
    print("--- Alternative: einsum ---")
    
    def einsum_approach():
        return torch.einsum('bi,bir,bor->bo', x, B, A)
    
    profiles["alt_einsum"] = profile_operation(
        "Alternative: einsum",
        einsum_approach,
        num_iterations=num_iterations,
    )
    
    # === Alternative: Fused with torch.compile ===
    print("--- Alternative: torch.compile ---")
    
    @torch.compile(mode="reduce-overhead")
    def compiled_lowrank(x, B, A):
        xB = torch.bmm(x.unsqueeze(1), B).squeeze(1)
        return torch.bmm(xB.unsqueeze(1), A.transpose(1, 2)).squeeze(1)
    
    # Warmup torch.compile
    for _ in range(10):
        _ = compiled_lowrank(x, B, A)
    torch.cuda.synchronize()
    
    def compiled_approach():
        return compiled_lowrank(x, B, A)
    
    profiles["alt_torch_compile"] = profile_operation(
        "Alternative: torch.compile",
        compiled_approach,
        num_iterations=num_iterations,
    )
    
    # === Profile noise generation alternatives ===
    print("\n--- Noise Generation Alternatives ---")
    
    num_unique = pop_size // 2
    total_features = in_features + out_features
    
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    
    def noise_with_generator():
        return torch.randn(num_unique, total_features, rank, generator=gen, device=device, dtype=dtype)
    
    profiles["noise_with_generator"] = profile_operation(
        "Noise: torch.randn with Generator",
        noise_with_generator,
        num_iterations=num_iterations,
    )
    
    def noise_without_generator():
        return torch.randn(num_unique, total_features, rank, device=device, dtype=dtype)
    
    profiles["noise_without_generator"] = profile_operation(
        "Noise: torch.randn without Generator",
        noise_without_generator,
        num_iterations=num_iterations,
    )
    
    # === Profile memory layout impact ===
    print("\n--- Memory Layout Impact ---")
    
    # Contiguous vs non-contiguous
    B_contig = torch.randn(pop_size, in_features, rank, device=device, dtype=dtype)
    B_noncontig = B_contig.transpose(1, 2).transpose(1, 2)  # Force non-contiguous
    
    def bmm_contiguous():
        return torch.bmm(x.unsqueeze(1), B_contig).squeeze(1)
    
    def bmm_noncontiguous():
        return torch.bmm(x.unsqueeze(1), B_noncontig).squeeze(1)
    
    profiles["bmm_contiguous"] = profile_operation(
        "bmm with contiguous tensor",
        bmm_contiguous,
        num_iterations=num_iterations,
    )
    
    profiles["bmm_noncontiguous"] = profile_operation(
        "bmm with non-contiguous tensor",
        bmm_noncontiguous,
        num_iterations=num_iterations,
    )
    
    return profiles


def print_profiles(profiles: Dict[str, OpProfile], title: str):
    """Print profiles in a table."""
    print(f"\n{'='*80}")
    print(title)
    print('='*80)
    print(f"{'Operation':<45} {'Mean (ms)':>10} {'Std (ms)':>10} {'% of Total':>12}")
    print('-'*80)
    
    for name, prof in sorted(profiles.items()):
        print(f"{prof.name:<45} {prof.mean_time_ms:>10.3f} {prof.std_time_ms:>10.3f} {prof.percentage:>11.1f}%")


def generate_optimization_report(
    anatomy_profiles: Dict[str, OpProfile],
    opt_profiles: Dict[str, OpProfile],
) -> str:
    """Generate a report with optimization recommendations."""
    report = []
    report.append("\n" + "="*80)
    report.append("OPTIMIZATION RECOMMENDATIONS")
    report.append("="*80)
    
    # Identify bottlenecks
    sorted_ops = sorted(
        [(name, prof) for name, prof in anatomy_profiles.items() if name != "7_full_forward_pass"],
        key=lambda x: x[1].mean_time_ms,
        reverse=True
    )
    
    report.append("\n### Top Bottlenecks ###")
    for i, (name, prof) in enumerate(sorted_ops[:3], 1):
        report.append(f"{i}. {prof.name}: {prof.mean_time_ms:.3f}ms ({prof.percentage:.1f}%)")
    
    # Optimization opportunities
    report.append("\n### Optimization Opportunities ###")
    
    # Compare two_bmm vs alternatives
    if "current_two_bmm" in opt_profiles and "alt_torch_compile" in opt_profiles:
        current = opt_profiles["current_two_bmm"].mean_time_ms
        compiled = opt_profiles["alt_torch_compile"].mean_time_ms
        speedup = current / compiled
        if speedup > 1.1:
            report.append(f"✅ torch.compile provides {speedup:.1f}x speedup for low-rank ops")
        else:
            report.append(f"⚠️  torch.compile provides no significant speedup ({speedup:.2f}x)")
    
    # Compare einsum
    if "current_two_bmm" in opt_profiles and "alt_einsum" in opt_profiles:
        current = opt_profiles["current_two_bmm"].mean_time_ms
        einsum = opt_profiles["alt_einsum"].mean_time_ms
        speedup = current / einsum
        if speedup > 1.1:
            report.append(f"✅ einsum provides {speedup:.1f}x speedup for low-rank ops")
        elif speedup < 0.9:
            report.append(f"❌ einsum is {1/speedup:.1f}x slower than bmm")
    
    # Generator overhead
    if "noise_with_generator" in opt_profiles and "noise_without_generator" in opt_profiles:
        with_gen = opt_profiles["noise_with_generator"].mean_time_ms
        without_gen = opt_profiles["noise_without_generator"].mean_time_ms
        overhead = (with_gen - without_gen) / without_gen * 100
        if overhead > 10:
            report.append(f"⚠️  Generator adds {overhead:.0f}% overhead to noise generation")
        else:
            report.append(f"✅ Generator overhead is minimal ({overhead:.1f}%)")
    
    report.append("\n### Triton/CUDA C++ Opportunities ###")
    report.append("""
1. **Fused Noise + Factor Generation**
   - Currently: torch.randn → split → scale → sign flip (4 kernel launches)
   - Opportunity: Single Triton kernel that generates factors directly
   - Expected gain: ~30-50% reduction in noise generation time

2. **Fused Low-Rank Perturbation**
   - Currently: bmm(x, B) → bmm(result, A.T) (2 kernel launches)
   - Opportunity: Single kernel that computes x @ B @ A.T
   - Especially valuable for small rank (r=4)
   - Expected gain: ~20-40% reduction in perturbation time

3. **Fused Perturbed Linear**
   - Currently: F.linear + bmm + bmm + add (4 operations)
   - Opportunity: Single kernel: out = x @ W.T + x @ B @ A.T + bias
   - This is the biggest opportunity for CUDA C++
   - Expected gain: ~40-60% reduction in layer time

4. **Memory-Efficient Antithetic Sampling**
   - Currently: Generate half, then repeat_interleave
   - Opportunity: Generate with antithetic pattern directly in kernel
   - Saves memory bandwidth
""")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="EGGROLL Computational Anatomy Profiler")
    parser.add_argument("--pop-size", type=int, default=2048, help="Population size")
    parser.add_argument("--layer-size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--rank", type=int, default=4, help="Low-rank dimension")
    parser.add_argument("--num-iterations", type=int, default=100, help="Profiling iterations")
    parser.add_argument("--output-dir", type=str, default="benchmarks", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("="*70)
    print("EGGROLL Computational Anatomy Profiler")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Population size: {args.pop_size}")
    print(f"  Layer size: {args.layer_size}")
    print(f"  N layers: {args.n_layers}")
    print(f"  Rank: {args.rank}")
    print(f"  Iterations: {args.num_iterations}")
    
    # Profile anatomy
    anatomy_profiles = profile_torch_anatomy(
        pop_size=args.pop_size,
        layer_size=args.layer_size,
        n_layers=args.n_layers,
        rank=args.rank,
        num_iterations=args.num_iterations,
    )
    
    print_profiles(anatomy_profiles, "PyTorch EGGROLL - Operation Breakdown")
    
    # Profile optimization opportunities
    opt_profiles = profile_potential_optimizations(
        pop_size=args.pop_size,
        layer_size=args.layer_size,
        rank=args.rank,
        num_iterations=args.num_iterations,
    )
    
    print_profiles(opt_profiles, "Optimization Alternatives")
    
    # Generate report
    report = generate_optimization_report(anatomy_profiles, opt_profiles)
    print(report)
    
    # Save results
    results = {
        "config": vars(args),
        "anatomy": {name: asdict(prof) for name, prof in anatomy_profiles.items()},
        "optimizations": {name: asdict(prof) for name, prof in opt_profiles.items()},
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"anatomy_profile_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")


if __name__ == "__main__":
    main()
