#!/usr/bin/env python3
"""
Benchmark: EGGROLL-Optimized TRM vs Standard TRM

This compares:
1. Original TRM with EggrollStrategy (uses hooks, high memory)
2. EGGROLL-optimized TRM (functional, batched, memory-efficient)

Key metrics:
- Maximum population size
- Throughput (evals/sec)
- Memory usage
"""

import gc
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import EGGROLL-optimized TRM
from eggroll_trm.core import init_trm_params, trm_forward, count_parameters, get_param_shapes
from eggroll_trm.batched_forward import forward_population_batched, forward_population_memory_efficient

# Import standard EggrollStrategy for comparison
from hyperscalees.torch.strategy import EggrollStrategy


# =============================================================================
# Standard TRM (nn.Module based)
# =============================================================================

class TinyNetwork(nn.Module):
    """Standard PyTorch tiny network."""
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class StandardTRM(nn.Module):
    """Standard PyTorch TRM for comparison."""
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
        self.output_head = nn.Linear(hidden_dim, output_dim)
        self.tiny_net = TinyNetwork(hidden_dim)
        
        self.y_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.y_init = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        z = self.input_proj(x)
        y = self.y_init.expand(batch_size, -1)
        
        for h in range(self.H_cycles):
            for l in range(self.L_cycles):
                z = self.tiny_net(z)
            combined = torch.cat([z, y], dim=-1)
            y = y + self.y_update(combined)
        
        return self.output_head(y)


# =============================================================================
# Benchmarks
# =============================================================================

def get_gpu_memory() -> Tuple[float, float]:
    """Get GPU memory in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved
    return 0.0, 0.0


def benchmark_standard_trm(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    L_cycles: int,
    H_cycles: int,
    batch_size: int,
    rank: int,
    test_pop_sizes: List[int],
    num_warmup: int = 3,
    num_iterations: int = 10,
) -> Dict:
    """Benchmark standard TRM with EggrollStrategy."""
    results = {"method": "standard_trm", "eggroll": {}}
    
    device = torch.device("cuda")
    max_pop = 0
    
    for pop_size in test_pop_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            model = StandardTRM(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                L_cycles=L_cycles,
                H_cycles=H_cycles,
            ).to(device)
            
            strategy = EggrollStrategy(sigma=0.01, lr=0.01, rank=rank, antithetic=True)
            strategy.setup(model)
            
            x = torch.randn(pop_size, input_dim, device=device)
            
            # Warmup
            for i in range(num_warmup):
                with strategy.perturb(population_size=pop_size, epoch=i) as pop:
                    outputs = pop.batched_forward(model, x)
                    fitnesses = -outputs.pow(2).mean(dim=-1)
                strategy.step(fitnesses)
            torch.cuda.synchronize()
            
            mem_alloc, mem_res = get_gpu_memory()
            
            # Benchmark
            start = time.perf_counter()
            for i in range(num_iterations):
                with strategy.perturb(population_size=pop_size, epoch=num_warmup + i) as pop:
                    outputs = pop.batched_forward(model, x)
                    fitnesses = -outputs.pow(2).mean(dim=-1)
                strategy.step(fitnesses)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            evals_per_sec = (pop_size * num_iterations) / elapsed
            
            results["eggroll"][pop_size] = {
                "evals_per_sec": evals_per_sec,
                "time_per_gen_ms": (elapsed / num_iterations) * 1000,
                "memory_gb": mem_alloc,
            }
            max_pop = pop_size
            
            del model, strategy
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
            raise
    
    results["max_population"] = max_pop
    return results


def benchmark_optimized_trm(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    L_cycles: int,
    H_cycles: int,
    batch_size: int,
    sigma: float,
    test_pop_sizes: List[int],
    num_warmup: int = 3,
    num_iterations: int = 10,
    use_memory_efficient: bool = False,
) -> Dict:
    """Benchmark EGGROLL-optimized TRM."""
    results = {"method": "optimized_trm", "eggroll": {}}
    
    device = torch.device("cuda")
    max_pop = 0
    
    # Initialize params once
    params = init_trm_params(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        device=device,
    )
    
    for pop_size in test_pop_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            x = torch.randn(batch_size, input_dim, device=device)
            
            forward_fn = forward_population_memory_efficient if use_memory_efficient else forward_population_batched
            
            # Warmup
            for i in range(num_warmup):
                outputs = forward_fn(
                    params, x, pop_size, epoch=i, sigma=sigma,
                    L_cycles=L_cycles, H_cycles=H_cycles,
                )
                fitnesses = -outputs.pow(2).mean(dim=(1, 2))  # (pop,)
            torch.cuda.synchronize()
            
            mem_alloc, mem_res = get_gpu_memory()
            
            # Benchmark
            start = time.perf_counter()
            for i in range(num_iterations):
                outputs = forward_fn(
                    params, x, pop_size, epoch=num_warmup + i, sigma=sigma,
                    L_cycles=L_cycles, H_cycles=H_cycles,
                )
                fitnesses = -outputs.pow(2).mean(dim=(1, 2))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            evals_per_sec = (pop_size * num_iterations) / elapsed
            
            results["eggroll"][pop_size] = {
                "evals_per_sec": evals_per_sec,
                "time_per_gen_ms": (elapsed / num_iterations) * 1000,
                "memory_gb": mem_alloc,
            }
            max_pop = pop_size
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
            raise
    
    results["max_population"] = max_pop
    return results


def run_comparison(
    L_cycles: int = 4,
    H_cycles: int = 2,
    hidden_dim: int = 64,
    batch_size: int = 64,
) -> Dict:
    """Run full comparison benchmark."""
    print("=" * 70)
    print("EGGROLL-Optimized TRM vs Standard TRM")
    print("=" * 70)
    
    config = {
        "input_dim": 64,
        "hidden_dim": hidden_dim,
        "output_dim": 5,
        "L_cycles": L_cycles,
        "H_cycles": H_cycles,
        "batch_size": batch_size,
        "rank": 8,
        "sigma": 0.01,
    }
    
    print(f"Config: L_cycles={L_cycles}, H_cycles={H_cycles}, hidden={hidden_dim}")
    print(f"Effective depth: {L_cycles * H_cycles}")
    print()
    
    # Test population sizes (powers of 2)
    test_pops = [256, 1024, 4096, 16384, 32768, 65536, 131072]
    
    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
    
    # Benchmark standard TRM
    print("=" * 50)
    print("Standard TRM (EggrollStrategy + hooks)")
    print("=" * 50)
    
    std_results = benchmark_standard_trm(
        **{k: v for k, v in config.items() if k != 'sigma'},
        test_pop_sizes=test_pops,
    )
    results["standard"] = std_results
    
    print(f"Max population: {std_results['max_population']:,}")
    for pop, data in std_results["eggroll"].items():
        print(f"  Pop {pop:>7}: {data['evals_per_sec']:>10,.0f} evals/sec, {data['memory_gb']:.2f} GB")
    
    # Benchmark optimized TRM
    print()
    print("=" * 50)
    print("EGGROLL-Optimized TRM (functional, batched)")
    print("=" * 50)
    
    opt_results = benchmark_optimized_trm(
        **{k: v for k, v in config.items() if k != 'rank'},
        test_pop_sizes=test_pops,
    )
    results["optimized"] = opt_results
    
    print(f"Max population: {opt_results['max_population']:,}")
    for pop, data in opt_results["eggroll"].items():
        print(f"  Pop {pop:>7}: {data['evals_per_sec']:>10,.0f} evals/sec, {data['memory_gb']:.2f} GB")
    
    # Summary comparison
    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    std_max = std_results["max_population"]
    opt_max = opt_results["max_population"]
    
    print(f"\nMax Population:")
    print(f"  Standard: {std_max:,}")
    print(f"  Optimized: {opt_max:,}")
    print(f"  Improvement: {opt_max / std_max:.1f}x" if std_max > 0 else "  N/A")
    
    # Compare throughput at common population sizes
    common_pops = set(std_results["eggroll"].keys()) & set(opt_results["eggroll"].keys())
    if common_pops:
        print(f"\nThroughput at common population sizes:")
        for pop in sorted(common_pops):
            std_tp = std_results["eggroll"][pop]["evals_per_sec"]
            opt_tp = opt_results["eggroll"][pop]["evals_per_sec"]
            print(f"  Pop {pop:>7}: Standard {std_tp:>10,.0f} vs Optimized {opt_tp:>10,.0f} ({opt_tp/std_tp:.1f}x)")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="EGGROLL TRM Comparison Benchmark")
    parser.add_argument("--L-cycles", type=int, default=4, help="Latent refinement steps")
    parser.add_argument("--H-cycles", type=int, default=2, help="Supervision steps")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    
    args = parser.parse_args()
    
    results = run_comparison(
        L_cycles=args.L_cycles,
        H_cycles=args.H_cycles,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
    )
    
    # Save results
    output_file = args.output or f"trm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = Path(__file__).parent / output_file
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
