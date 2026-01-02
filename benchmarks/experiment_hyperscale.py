#!/usr/bin/env python3
"""
Experiment: Hyperscale Population Size Limit

Goal: Find the maximum population size that fits in GPU memory for EGGROLL
and measure throughput at extreme scales.

This tests the memory efficiency claims of EGGROLL's low-rank perturbations.

Usage:
    uv run python benchmarks/experiment_hyperscale.py
    uv run python benchmarks/experiment_hyperscale.py --model-size large
    uv run python benchmarks/experiment_hyperscale.py --find-limit
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


MODEL_CONFIGS = {
    "small": {"input_dim": 4, "hidden_dim": 256, "output_dim": 2, "n_layers": 3},
    "medium": {"input_dim": 64, "hidden_dim": 512, "output_dim": 64, "n_layers": 3},
    "large": {"input_dim": 256, "hidden_dim": 2048, "output_dim": 256, "n_layers": 3},
}


def round_to_even(n: int) -> int:
    """Round to nearest even number (required for antithetic sampling)."""
    return 2 * round(n / 2)


def get_gpu_stats() -> Dict[str, float]:
    """Get current GPU stats via nvidia-smi."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
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
            }
    except Exception:
        pass
    return {}


def get_gpu_name() -> str:
    """Get GPU model name."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return "Unknown"


def test_population_size(pop_size: int, model_cfg: dict, rank: int = 4, sigma: float = 0.2) -> Dict[str, Any]:
    """Test if a given population size works and measure performance."""
    import torch
    import torch.nn as nn
    from hyperscalees.torch import EggrollStrategy
    
    device = torch.device("cuda")
    
    # Ensure even population size
    pop_size = round_to_even(pop_size)
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        # Create model
        layers = []
        dims = [model_cfg["input_dim"]] + [model_cfg["hidden_dim"]] * (model_cfg["n_layers"] - 1) + [model_cfg["output_dim"]]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        model = nn.Sequential(*layers).to(device)
        
        # Setup strategy
        strategy = EggrollStrategy(sigma=sigma, lr=0.1, rank=rank, seed=42)
        strategy.setup(model)
        
        # Create input
        x = torch.randn(pop_size, model_cfg["input_dim"], device=device)
        
        # Warmup
        for i in range(3):
            with strategy.perturb(population_size=pop_size, epoch=i) as ctx:
                out = ctx.batched_forward(model, x)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for i in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with strategy.perturb(population_size=pop_size, epoch=10 + i) as ctx:
                out = ctx.batched_forward(model, x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        stats = get_gpu_stats()
        
        return {
            "success": True,
            "pop_size": pop_size,
            "mean_time_ms": np.mean(times),
            "std_time_ms": np.std(times),
            "throughput": pop_size / (np.mean(times) / 1000),
            "peak_memory_mb": peak_memory,
            "current_memory_mb": stats.get("memory_used_mb", 0),
            "total_memory_mb": stats.get("memory_total_mb", 0),
        }
        
    except torch.cuda.OutOfMemoryError as e:
        return {
            "success": False,
            "pop_size": pop_size,
            "error": "OOM",
            "message": str(e),
        }
    except Exception as e:
        return {
            "success": False,
            "pop_size": pop_size,
            "error": type(e).__name__,
            "message": str(e),
        }


def find_max_population(model_cfg: dict, rank: int = 4, verbose: bool = True) -> int:
    """Binary search to find maximum population size."""
    
    if verbose:
        print(f"\n{'='*60}")
        print("Finding maximum population size...")
        print(f"{'='*60}")
    
    # First, find an upper bound by doubling
    low = 1024
    high = low
    
    if verbose:
        print("\nPhase 1: Finding upper bound by doubling...")
    
    while True:
        high = round_to_even(high)
        result = test_population_size(high, model_cfg, rank)
        if result["success"]:
            if verbose:
                print(f"  pop_size={high:,}: ✓ OK ({result['peak_memory_mb']:.0f} MB)")
            low = high
            high *= 2
        else:
            if verbose:
                print(f"  pop_size={high:,}: ✗ {result['error']}")
            break
        
        # Safety limit
        if high > 100_000_000:
            if verbose:
                print("  Reached safety limit (100M)")
            break
    
    if verbose:
        print(f"\nPhase 2: Binary search between {low:,} and {high:,}...")
    
    # Binary search with even numbers only
    best_working = low
    iterations = 0
    max_iterations = 30  # Safety limit
    
    while high - low > 2 and iterations < max_iterations:
        iterations += 1
        mid = round_to_even((low + high) // 2)
        
        # Avoid testing the same value
        if mid == low or mid == high:
            break
            
        result = test_population_size(mid, model_cfg, rank)
        
        if result["success"]:
            if verbose:
                print(f"  pop_size={mid:,}: ✓ OK ({result['peak_memory_mb']:.0f} MB)")
            best_working = mid
            low = mid
        else:
            if verbose:
                print(f"  pop_size={mid:,}: ✗ {result['error']}")
            high = mid
    
    return best_working


def benchmark_hyperscale(
    model_size: str = "small",
    pop_sizes: Optional[List[int]] = None,
    rank: int = 4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run hyperscale benchmark."""
    import torch
    
    model_cfg = MODEL_CONFIGS[model_size]
    gpu_name = get_gpu_name()
    
    print(f"\n{'#'*60}")
    print("# Hyperscale Population Size Benchmark")
    print(f"{'#'*60}")
    print(f"GPU: {gpu_name}")
    print(f"Model: {model_size}")
    print(f"Config: {model_cfg}")
    print(f"Rank: {rank}")
    
    # Find maximum population size
    max_pop = find_max_population(model_cfg, rank, verbose)
    print(f"\n{'='*60}")
    print(f"MAXIMUM POPULATION SIZE: {max_pop:,}")
    print(f"{'='*60}")
    
    # Test at various large scales
    if pop_sizes is None:
        # Test at powers of 2 up to max, plus the max itself
        pop_sizes = []
        p = 1024
        while p <= max_pop:
            pop_sizes.append(round_to_even(p))
            p *= 2
        if max_pop not in pop_sizes:
            pop_sizes.append(max_pop)
    else:
        pop_sizes = [round_to_even(p) for p in pop_sizes]
    
    print(f"\nTesting population sizes: {pop_sizes}")
    
    results = []
    for pop_size in pop_sizes:
        result = test_population_size(pop_size, model_cfg, rank)
        results.append(result)
        
        if result["success"]:
            print(f"\n  pop_size={pop_size:,}:")
            print(f"    Time: {result['mean_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms")
            print(f"    Throughput: {result['throughput']:,.0f} samples/sec")
            print(f"    Peak memory: {result['peak_memory_mb']:.0f} MB")
        else:
            print(f"\n  pop_size={pop_size:,}: FAILED ({result['error']})")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"GPU: {gpu_name}")
    print(f"Model: {model_size} ({model_cfg})")
    print(f"Maximum population: {max_pop:,}")
    
    working_results = [r for r in results if r["success"]]
    if working_results:
        best = max(working_results, key=lambda r: r["throughput"])
        print(f"Best throughput: {best['throughput']:,.0f} samples/sec at pop_size={best['pop_size']:,}")
        
        largest = max(working_results, key=lambda r: r["pop_size"])
        print(f"At max population ({largest['pop_size']:,}): {largest['throughput']:,.0f} samples/sec")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "gpu_name": gpu_name,
        "model_size": model_size,
        "model_config": model_cfg,
        "rank": rank,
        "max_population": max_pop,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperscale Population Benchmark")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--pop-sizes", type=int, nargs="+", default=None,
                        help="Specific population sizes to test")
    parser.add_argument("--find-limit", action="store_true",
                        help="Only find the maximum population size")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    results = benchmark_hyperscale(
        model_size=args.model_size,
        pop_sizes=args.pop_sizes,
        rank=args.rank,
        verbose=not args.quiet,
    )
    
    # Save results
    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / f"hyperscale_results_{timestamp}.json"
    else:
        output_path = Path(output_path)
    
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.generic, np.ndarray)):
            return obj.item() if np.ndim(obj) == 0 else obj.tolist()
        return obj
    
    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
