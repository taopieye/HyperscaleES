#!/usr/bin/env python3
"""
Experiment: TRM (Tiny Recursive Model) Throughput with EGGROLL

This benchmark measures how EGGROLL throughput is affected when training
Tiny Recursive Models (TRMs) compared to standard MLPs.

Key questions:
1. How does recursive depth (L_cycles × H_cycles) affect throughput?
2. Does EGGROLL's overhead amortize well with recursive networks?
3. What population sizes are optimal for TRM training?

TRM Architecture (from paper):
- Single 2-layer "tiny" network applied recursively
- z = latent reasoning (refined L_cycles times)
- y = solution (updated once per H_cycle)
- Deep supervision: loss at each supervision step

Reference: "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)
"""

import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyperscalees.torch.strategy import EggrollStrategy


# =============================================================================
# TRM-STYLE MODELS
# =============================================================================

class TinyNetwork(nn.Module):
    """
    The "tiny network" from TRM - a simple 2-layer MLP.
    Applied recursively many times to build effective depth.
    """
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


class TRM(nn.Module):
    """
    Tiny Recursive Model for benchmarking.
    
    Simplified version without graph structure - just the recursive pattern:
    - z is refined L_cycles times per supervision step
    - y is updated once per supervision step
    - H_cycles supervision steps total
    
    Total forward passes through tiny network: L_cycles × H_cycles
    """
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        L_cycles: int = 6,   # Latent refinement steps (n in paper)
        H_cycles: int = 3,   # Supervision steps (T in paper)
    ):
        super().__init__()
        self.L_cycles = L_cycles
        self.H_cycles = H_cycles
        
        # Input/output projections
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, output_dim)
        
        # The tiny network (shared across all recursive calls)
        self.tiny_net = TinyNetwork(hidden_dim)
        
        # y update network
        self.y_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Learnable initial y
        self.y_init = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with TRM-style recursion.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Output tensor (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        
        # Initial embeddings
        z = self.input_proj(x)  # (batch_size, hidden_dim)
        y = self.y_init.expand(batch_size, -1)  # (batch_size, hidden_dim)
        
        # H_cycles supervision steps
        for h in range(self.H_cycles):
            # Refine z L_cycles times
            for l in range(self.L_cycles):
                z = self.tiny_net(z)
            
            # Update y
            combined = torch.cat([z, y], dim=-1)
            y = y + self.y_update(combined)
        
        # Final output
        return self.output_head(y)


class StandardMLP(nn.Module):
    """
    Standard MLP with equivalent depth for comparison.
    
    Matches TRM's effective depth: L_cycles × H_cycles layers.
    """
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# BENCHMARK UTILITIES
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gpu_memory() -> Tuple[float, float]:
    """Get GPU memory usage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved
    return 0.0, 0.0


def benchmark_forward_pass(
    model: nn.Module,
    input_dim: int,
    batch_size: int,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = "cuda",
) -> Dict:
    """Benchmark pure forward pass throughput."""
    model = model.to(device)
    model.eval()
    
    try:
        x = torch.randn(batch_size, input_dim, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(x)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        samples_per_sec = (batch_size * num_iterations) / elapsed
        
        result = {
            "samples_per_sec": samples_per_sec,
            "time_per_batch_ms": (elapsed / num_iterations) * 1000,
            "batch_size": batch_size,
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result = {"oom": True, "batch_size": batch_size}
        else:
            raise
    finally:
        del model
        torch.cuda.empty_cache()
    
    return result


def benchmark_eggroll(
    model: nn.Module,
    input_dim: int,
    batch_size: int,
    population_size: int,
    rank: int = 8,
    num_warmup: int = 3,
    num_iterations: int = 20,
    device: str = "cuda",
) -> Dict:
    """Benchmark EGGROLL forward pass throughput.
    
    Note: In EGGROLL, each sample in the batch gets a different perturbation.
    So the 'batch_size' here becomes the number of inputs per population member,
    and the actual batch is (population_size, input_dim) - each row gets a
    different perturbation applied.
    """
    model = model.to(device)
    
    # Initialize EggrollStrategy
    strategy = EggrollStrategy(
        sigma=0.01,
        lr=0.01,
        rank=rank,
        antithetic=True,
    )
    strategy.setup(model)
    
    # In EGGROLL, x shape is (population_size, input_dim)
    # Each row gets a different perturbation
    x = torch.randn(population_size, input_dim, device=device)
    
    # Warmup
    for i in range(num_warmup):
        try:
            with strategy.perturb(population_size=population_size, epoch=i) as pop:
                outputs = pop.batched_forward(model, x)  # (pop_size, output_dim)
                # Compute dummy fitness - one per population member
                fitnesses = -outputs.pow(2).mean(dim=-1)  # (pop_size,)
            strategy.step(fitnesses)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return {"oom": True, "population_size": population_size}
            raise
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # Get memory after warmup
    mem_allocated, mem_reserved = get_gpu_memory()
    
    # Benchmark
    start = time.perf_counter()
    for i in range(num_iterations):
        with strategy.perturb(population_size=population_size, epoch=num_warmup + i) as pop:
            outputs = pop.batched_forward(model, x)
            fitnesses = -outputs.pow(2).mean(dim=-1)
        strategy.step(fitnesses)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Throughput: population members evaluated per second
    evals_per_sec = (population_size * num_iterations) / elapsed
    
    return {
        "evals_per_sec": evals_per_sec,
        "time_per_generation_ms": (elapsed / num_iterations) * 1000,
        "population_size": population_size,
        "batch_size": batch_size,
        "memory_allocated_gb": mem_allocated,
        "memory_reserved_gb": mem_reserved,
        "oom": False,
    }


def find_max_population(
    model_fn,
    input_dim: int,
    batch_size: int,
    rank: int = 8,
    device: str = "cuda",
) -> int:
    """Binary search for maximum population size."""
    # Start smaller and test upward
    test_sizes = [256, 1024, 4096, 16384, 32768, 65536, 131072, 262144, 524288]
    max_working = 128
    
    for size in test_sizes:
        torch.cuda.empty_cache()
        try:
            model = model_fn()
            model = model.to(device)
            
            strategy = EggrollStrategy(sigma=0.01, lr=0.01, rank=rank, antithetic=True)
            strategy.setup(model)
            
            # x shape is (pop_size, input_dim) - each row gets different perturbation
            x = torch.randn(size, input_dim, device=device)
            
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


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    input_dim: int = 64
    hidden_dim: int = 64
    output_dim: int = 5
    batch_size: int = 64
    
    # TRM configs to test
    trm_configs: List[Tuple[int, int]] = None  # (L_cycles, H_cycles) pairs
    
    # Population sizes to test
    population_sizes: List[int] = None
    
    # EGGROLL
    rank: int = 8
    
    # Benchmark
    num_warmup: int = 5
    num_iterations: int = 30
    
    def __post_init__(self):
        if self.trm_configs is None:
            self.trm_configs = [
                (1, 1),   # Minimal recursion (effective depth 1)
                (2, 1),   # 2 refinements
                (4, 1),   # 4 refinements
                (6, 1),   # 6 refinements (TRM default n)
                (3, 2),   # 6 total (split)
                (4, 2),   # 8 total
                (6, 2),   # 12 total
                (4, 3),   # 12 total (TRM-like)
                (6, 3),   # 18 total (TRM default)
            ]
        
        if self.population_sizes is None:
            self.population_sizes = [256, 1024, 4096, 16384, 65536]


def run_benchmark(config: BenchmarkConfig, device: str = "cuda") -> Dict:
    """Run the full TRM throughput benchmark."""
    results = {
        "config": asdict(config),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }
    
    print("=" * 70)
    print("TRM THROUGHPUT BENCHMARK WITH EGGROLL")
    print("=" * 70)
    print(f"Device: {results['gpu_name']}")
    print(f"Input: {config.input_dim} → Hidden: {config.hidden_dim} → Output: {config.output_dim}")
    print(f"Batch size: {config.batch_size}, EGGROLL rank: {config.rank}")
    print()
    
    # Test each TRM configuration
    for L_cycles, H_cycles in config.trm_configs:
        effective_depth = L_cycles * H_cycles
        model_name = f"TRM_L{L_cycles}_H{H_cycles}"
        
        print(f"\n{'='*60}")
        print(f"Testing {model_name} (effective depth: {effective_depth})")
        print(f"{'='*60}")
        
        # Create model factory - capture loop vars explicitly
        def make_model_fn(l, h, cfg):
            def fn():
                return TRM(
                    input_dim=cfg.input_dim,
                    hidden_dim=cfg.hidden_dim,
                    output_dim=cfg.output_dim,
                    L_cycles=l,
                    H_cycles=h,
                )
            return fn
        
        model_fn = make_model_fn(L_cycles, H_cycles, config)
        
        model = model_fn()
        num_params = count_parameters(model)
        print(f"Parameters: {num_params:,}")
        
        model_results = {
            "L_cycles": L_cycles,
            "H_cycles": H_cycles,
            "effective_depth": effective_depth,
            "num_params": num_params,
            "forward_pass": {},
            "eggroll": {},
        }
        
        # Benchmark pure forward pass
        print("\n--- Pure Forward Pass ---")
        for batch_mult in [1, 4, 16]:
            bs = config.batch_size * batch_mult
            fp_result = benchmark_forward_pass(
                model_fn(),
                config.input_dim,
                bs,
                num_warmup=config.num_warmup,
                num_iterations=config.num_iterations * 2,
                device=device,
            )
            model_results["forward_pass"][bs] = fp_result
            if fp_result.get("oom"):
                print(f"  Batch {bs:>6}: OOM")
                break
            else:
                print(f"  Batch {bs:>6}: {fp_result['samples_per_sec']:>12,.0f} samples/sec")
        
        # Find max population
        print("\n--- Finding Max Population Size ---")
        max_pop = find_max_population(
            model_fn,
            config.input_dim,
            config.batch_size,
            config.rank,
            device,
        )
        model_results["max_population"] = max_pop
        print(f"  Max population: {max_pop:,}")
        
        # Benchmark EGGROLL at various population sizes
        print("\n--- EGGROLL Throughput ---")
        for pop_size in config.population_sizes:
            if pop_size > max_pop:
                print(f"  Pop {pop_size:>7}: SKIPPED (exceeds max)")
                continue
                
            eg_result = benchmark_eggroll(
                model_fn(),
                config.input_dim,
                config.batch_size,
                pop_size,
                config.rank,
                num_warmup=config.num_warmup,
                num_iterations=config.num_iterations,
                device=device,
            )
            
            if eg_result.get("oom"):
                print(f"  Pop {pop_size:>7}: OOM")
                break
            
            model_results["eggroll"][pop_size] = eg_result
            print(f"  Pop {pop_size:>7}: {eg_result['evals_per_sec']:>12,.0f} evals/sec, "
                  f"{eg_result['time_per_generation_ms']:>8.1f} ms/gen, "
                  f"{eg_result['memory_allocated_gb']:.2f} GB")
        
        results["models"][model_name] = model_results
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    # Also benchmark equivalent standard MLP for comparison
    print(f"\n{'='*60}")
    print("Baseline: Standard MLP (18 layers = TRM L6×H3 equivalent)")
    print(f"{'='*60}")
    
    mlp_fn = lambda: StandardMLP(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        num_layers=18,  # Equivalent to TRM L6×H3
    )
    
    mlp = mlp_fn()
    mlp_params = count_parameters(mlp)
    print(f"Parameters: {mlp_params:,}")
    
    mlp_results = {
        "num_layers": 18,
        "num_params": mlp_params,
        "forward_pass": {},
        "eggroll": {},
    }
    
    # Forward pass
    print("\n--- Pure Forward Pass ---")
    for batch_mult in [1, 4, 16]:
        bs = config.batch_size * batch_mult
        fp_result = benchmark_forward_pass(
            mlp_fn(),
            config.input_dim,
            bs,
            num_warmup=config.num_warmup,
            num_iterations=config.num_iterations * 2,
            device=device,
        )
        mlp_results["forward_pass"][bs] = fp_result
        print(f"  Batch {bs:>6}: {fp_result['samples_per_sec']:>12,.0f} samples/sec")
    
    # Max population
    print("\n--- Finding Max Population Size ---")
    max_pop_mlp = find_max_population(
        mlp_fn,
        config.input_dim,
        config.batch_size,
        config.rank,
        device,
    )
    mlp_results["max_population"] = max_pop_mlp
    print(f"  Max population: {max_pop_mlp:,}")
    
    # EGGROLL
    print("\n--- EGGROLL Throughput ---")
    for pop_size in config.population_sizes:
        if pop_size > max_pop_mlp:
            print(f"  Pop {pop_size:>7}: SKIPPED (exceeds max)")
            continue
            
        eg_result = benchmark_eggroll(
            mlp_fn(),
            config.input_dim,
            config.batch_size,
            pop_size,
            config.rank,
            num_warmup=config.num_warmup,
            num_iterations=config.num_iterations,
            device=device,
        )
        
        if eg_result.get("oom"):
            print(f"  Pop {pop_size:>7}: OOM")
            break
        
        mlp_results["eggroll"][pop_size] = eg_result
        print(f"  Pop {pop_size:>7}: {eg_result['evals_per_sec']:>12,.0f} evals/sec, "
              f"{eg_result['time_per_generation_ms']:>8.1f} ms/gen, "
              f"{eg_result['memory_allocated_gb']:.2f} GB")
    
    results["models"]["StandardMLP_18L"] = mlp_results
    
    return results


def print_summary(results: Dict):
    """Print a summary comparison table."""
    print("\n" + "=" * 80)
    print("SUMMARY: TRM vs MLP Throughput with EGGROLL")
    print("=" * 80)
    
    # Header
    print(f"\n{'Model':<20} {'Params':>10} {'Max Pop':>10} {'Best Evals/s':>15} {'Effective Depth':>15}")
    print("-" * 75)
    
    for model_name, model_data in results["models"].items():
        num_params = model_data["num_params"]
        max_pop = model_data.get("max_population", "N/A")
        
        # Find best throughput
        eggroll_data = model_data.get("eggroll", {})
        best_throughput = 0
        for pop_data in eggroll_data.values():
            if isinstance(pop_data, dict) and not pop_data.get("oom"):
                best_throughput = max(best_throughput, pop_data.get("evals_per_sec", 0))
        
        # Effective depth
        if "TRM" in model_name:
            eff_depth = model_data.get("effective_depth", "N/A")
        else:
            eff_depth = model_data.get("num_layers", "N/A")
        
        print(f"{model_name:<20} {num_params:>10,} {max_pop:>10,} {best_throughput:>15,.0f} {eff_depth:>15}")
    
    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    # Compare TRM L6×H3 vs Standard 18L MLP
    trm_data = results["models"].get("TRM_L6_H3", {})
    mlp_data = results["models"].get("StandardMLP_18L", {})
    
    if trm_data and mlp_data:
        trm_params = trm_data.get("num_params", 0)
        mlp_params = mlp_data.get("num_params", 0)
        trm_max = trm_data.get("max_population", 0)
        mlp_max = mlp_data.get("max_population", 0)
        
        print(f"\n1. Parameter Efficiency:")
        print(f"   TRM (L6×H3): {trm_params:,} params")
        print(f"   MLP (18L):   {mlp_params:,} params")
        print(f"   Ratio: TRM uses {trm_params/mlp_params:.1%} of MLP params")
        
        print(f"\n2. Population Capacity:")
        print(f"   TRM max population: {trm_max:,}")
        print(f"   MLP max population: {mlp_max:,}")
        if mlp_max > 0:
            print(f"   TRM allows {trm_max/mlp_max:.1f}x larger populations")
        
        # Compare throughput at common population size
        common_pops = set(trm_data.get("eggroll", {}).keys()) & set(mlp_data.get("eggroll", {}).keys())
        if common_pops:
            pop = max(common_pops)
            trm_throughput = trm_data["eggroll"][pop].get("evals_per_sec", 0)
            mlp_throughput = mlp_data["eggroll"][pop].get("evals_per_sec", 0)
            
            print(f"\n3. Throughput at population {pop:,}:")
            print(f"   TRM: {trm_throughput:,.0f} evals/sec")
            print(f"   MLP: {mlp_throughput:,.0f} evals/sec")


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TRM Throughput Benchmark with EGGROLL")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for fitness evaluation")
    parser.add_argument("--rank", type=int, default=8, help="EGGROLL rank")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer configs")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    
    args = parser.parse_args()
    
    # Configure
    if args.quick:
        config = BenchmarkConfig(
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            rank=args.rank,
            trm_configs=[(1, 1), (6, 1), (6, 3)],
            population_sizes=[256, 4096, 32768],
            num_iterations=10,
        )
    else:
        config = BenchmarkConfig(
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            rank=args.rank,
        )
    
    # Run benchmark
    results = run_benchmark(config)
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_file = args.output or f"trm_throughput_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = Path(__file__).parent / output_file
    
    results = convert_numpy_types(results)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
