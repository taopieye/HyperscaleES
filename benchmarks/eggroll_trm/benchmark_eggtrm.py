#!/usr/bin/env python3
"""
Benchmark: EGGROLL-Compatible TRM vs Standard TRM

Compares:
1. Standard TRM (with GELU activations, fp32)
2. EGG-TRM (no activations, fp16) - EGGROLL optimized

Key metrics:
- Max population size before OOM
- Throughput (evals/sec)
- Memory usage
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hyperscalees.torch.strategy import EggrollStrategy

# Import our TRM variants
from eggroll_compatible_trm import EGGTRM, EGGTRMDeep, create_eggtrm, count_parameters


# Standard TRM with activations for comparison
class StandardTRM(nn.Module):
    """Standard TRM with GELU activations (baseline)."""
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 5,
        L_cycles: int = 6,
        H_cycles: int = 3,
    ):
        super().__init__()
        self.L_cycles = L_cycles
        self.H_cycles = H_cycles
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.y_init = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        
        # Tiny network WITH activations
        self.tiny_up = nn.Linear(hidden_dim, hidden_dim * 4)
        self.tiny_down = nn.Linear(hidden_dim * 4, hidden_dim)
        self.tiny_norm = nn.LayerNorm(hidden_dim)
        
        self.y_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.y_transform = nn.Linear(hidden_dim, hidden_dim)
        
        self.head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        z = self.input_proj(x)
        y = self.y_init.expand(batch_size, -1)
        
        for _ in range(self.H_cycles):
            for _ in range(self.L_cycles):
                # WITH activation
                h = self.tiny_up(z)
                h = nn.functional.gelu(h)  # <-- Activation!
                h = self.tiny_down(h)
                z = self.tiny_norm(z + h)
            
            combined = torch.cat([z, y], dim=-1)
            h = self.y_combine(combined)
            h = nn.functional.gelu(h)  # <-- Activation!
            y = y + self.y_transform(h)
        
        return self.head(y)


def get_gpu_memory() -> Tuple[float, float]:
    """Get GPU memory in GB."""
    if torch.cuda.is_available():
        return (
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved() / 1e9,
        )
    return 0.0, 0.0


def find_max_population(
    model_fn,
    input_dim: int,
    rank: int = 8,
    device: str = "cuda",
) -> int:
    """Find max population size before OOM."""
    test_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    max_working = 128
    
    for size in test_sizes:
        torch.cuda.empty_cache()
        try:
            model = model_fn().to(device)
            strategy = EggrollStrategy(sigma=0.01, lr=0.01, rank=rank, antithetic=True)
            strategy.setup(model)
            
            x = torch.randn(size, input_dim, device=device, dtype=model.input_proj.weight.dtype)
            
            with strategy.perturb(population_size=size, epoch=0) as pop:
                outputs = pop.batched_forward(model, x)
                fitnesses = -outputs.pow(2).mean(dim=-1)
            strategy.step(fitnesses)
            torch.cuda.synchronize()
            
            max_working = size
            del model, strategy, x, outputs, fitnesses
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
            raise
    
    return max_working


def benchmark_eggroll_throughput(
    model_fn,
    input_dim: int,
    population_size: int,
    rank: int = 8,
    num_warmup: int = 5,
    num_iterations: int = 30,
    device: str = "cuda",
) -> Dict:
    """Benchmark EGGROLL throughput."""
    torch.cuda.empty_cache()
    
    try:
        model = model_fn().to(device)
        strategy = EggrollStrategy(sigma=0.01, lr=0.01, rank=rank, antithetic=True)
        strategy.setup(model)
        
        dtype = model.input_proj.weight.dtype
        x = torch.randn(population_size, input_dim, device=device, dtype=dtype)
        
        # Warmup
        for i in range(num_warmup):
            with strategy.perturb(population_size=population_size, epoch=i) as pop:
                outputs = pop.batched_forward(model, x)
                fitnesses = -outputs.pow(2).mean(dim=-1)
            strategy.step(fitnesses)
        
        torch.cuda.synchronize()
        mem_alloc, mem_res = get_gpu_memory()
        
        # Benchmark
        start = time.perf_counter()
        for i in range(num_iterations):
            with strategy.perturb(population_size=population_size, epoch=num_warmup + i) as pop:
                outputs = pop.batched_forward(model, x)
                fitnesses = -outputs.pow(2).mean(dim=-1)
            strategy.step(fitnesses)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        evals_per_sec = (population_size * num_iterations) / elapsed
        
        del model, strategy
        torch.cuda.empty_cache()
        
        return {
            "evals_per_sec": evals_per_sec,
            "time_per_gen_ms": (elapsed / num_iterations) * 1000,
            "memory_gb": mem_alloc,
            "oom": False,
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return {"oom": True}
        raise


def run_comparison():
    """Run full comparison benchmark."""
    print("=" * 70)
    print("EGGROLL-Compatible TRM vs Standard TRM Benchmark")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    input_dim = 64
    hidden_dim = 64
    output_dim = 5
    L_cycles = 4
    H_cycles = 2
    rank = 8
    
    results = {}
    
    # Define model variants
    variants = {
        "StandardTRM_fp32": lambda: StandardTRM(
            input_dim, hidden_dim, output_dim, L_cycles, H_cycles
        ).float(),
        
        "StandardTRM_fp16": lambda: StandardTRM(
            input_dim, hidden_dim, output_dim, L_cycles, H_cycles
        ).half(),
        
        "EGGTRM_fp32": lambda: EGGTRM(
            input_dim, hidden_dim, output_dim, L_cycles, H_cycles
        ).float(),
        
        "EGGTRM_fp16": lambda: EGGTRM(
            input_dim, hidden_dim, output_dim, L_cycles, H_cycles
        ).half(),
        
        "EGGTRMDeep_fp16": lambda: EGGTRMDeep(
            input_dim, hidden_dim, output_dim, L_cycles, H_cycles
        ).half(),
    }
    
    for name, model_fn in variants.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        
        # Count params
        model = model_fn()
        n_params = count_parameters(model)
        del model
        torch.cuda.empty_cache()
        
        print(f"Parameters: {n_params:,}")
        
        # Find max population
        print("\nFinding max population size...")
        max_pop = find_max_population(model_fn, input_dim, rank)
        print(f"Max population: {max_pop:,}")
        
        # Benchmark at various sizes
        variant_results = {
            "params": n_params,
            "max_population": max_pop,
            "throughput": {},
        }
        
        test_pops = [256, 1024, 4096, 16384]
        print("\nThroughput benchmarks:")
        
        for pop_size in test_pops:
            if pop_size > max_pop:
                print(f"  Pop {pop_size:>6}: SKIPPED (exceeds max)")
                continue
            
            result = benchmark_eggroll_throughput(
                model_fn, input_dim, pop_size, rank
            )
            
            if result.get("oom"):
                print(f"  Pop {pop_size:>6}: OOM")
                break
            
            variant_results["throughput"][pop_size] = result
            print(f"  Pop {pop_size:>6}: {result['evals_per_sec']:>10,.0f} evals/s, "
                  f"{result['time_per_gen_ms']:>7.1f} ms/gen, "
                  f"{result['memory_gb']:.2f} GB")
        
        results[name] = variant_results
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Params':>10} {'Max Pop':>10} {'Best Evals/s':>15}")
    print("-" * 60)
    
    for name, data in results.items():
        best = max(
            (v.get("evals_per_sec", 0) for v in data["throughput"].values()),
            default=0
        )
        print(f"{name:<20} {data['params']:>10,} {data['max_population']:>10,} {best:>15,.0f}")
    
    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    # Compare fp16 variants
    std_fp16 = results.get("StandardTRM_fp16", {})
    egg_fp16 = results.get("EGGTRM_fp16", {})
    
    if std_fp16 and egg_fp16:
        std_max = std_fp16.get("max_population", 0)
        egg_max = egg_fp16.get("max_population", 0)
        
        print(f"\n1. Population Capacity (fp16):")
        print(f"   StandardTRM: {std_max:,}")
        print(f"   EGGTRM:      {egg_max:,}")
        if std_max > 0:
            print(f"   Improvement: {egg_max / std_max:.1f}x")
        
        # Find common pop size for throughput comparison
        common_pops = (
            set(std_fp16.get("throughput", {}).keys()) &
            set(egg_fp16.get("throughput", {}).keys())
        )
        if common_pops:
            pop = max(common_pops)
            std_tp = std_fp16["throughput"][pop].get("evals_per_sec", 0)
            egg_tp = egg_fp16["throughput"][pop].get("evals_per_sec", 0)
            
            print(f"\n2. Throughput at pop={pop:,}:")
            print(f"   StandardTRM: {std_tp:,.0f} evals/s")
            print(f"   EGGTRM:      {egg_tp:,.0f} evals/s")
            if std_tp > 0:
                print(f"   Improvement: {egg_tp / std_tp:.2f}x")
    
    # Save results
    output_path = Path(__file__).parent / f"eggtrm_comparison_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    run_comparison()
