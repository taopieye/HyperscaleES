#!/usr/bin/env python3
"""
Benchmark script for batched forward pass performance.

Compares:
1. batched_forward (the optimized path using einsum)
2. Sequential iteration (for comparison)

Run with: python experiments/benchmark_batched_forward.py
"""

import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyperscalees.torch import EggrollStrategy, EggrollConfig


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


def benchmark_batched_forward(
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


def benchmark_sequential(
    model: nn.Module,
    strategy: EggrollStrategy,
    pop_size: int,
    input_dim: int,
    device: str,
    num_iterations: int = 100,
    warmup: int = 10
):
    """Benchmark sequential iteration (for comparison)."""
    x = torch.randn(1, input_dim, device=device)
    
    # Warmup
    for i in range(warmup):
        with strategy.perturb(pop_size, epoch=i) as pop:
            for member_id in range(min(pop_size, 10)):  # Only warmup on a few
                _ = pop.evaluate_member(member_id, model, x)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark (use smaller pop for sequential since it's slow)
    seq_pop_size = min(pop_size, 64)
    start = time.perf_counter()
    for i in range(num_iterations):
        with strategy.perturb(seq_pop_size, epoch=i) as pop:
            outputs = []
            for member_id in range(seq_pop_size):
                out = pop.evaluate_member(member_id, model, x)
                outputs.append(out)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Extrapolate to full population size
    extrapolated = (elapsed / num_iterations) * (pop_size / seq_pop_size)
    
    return extrapolated, seq_pop_size


def main():
    print("=" * 70)
    print("EGGROLL Batched Forward Benchmark")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Configuration
    configs = [
        {"pop_size": 128, "input_dim": 64, "hidden_dim": 128, "output_dim": 32, "num_layers": 3},
        {"pop_size": 512, "input_dim": 128, "hidden_dim": 256, "output_dim": 64, "num_layers": 3},
        {"pop_size": 2048, "input_dim": 128, "hidden_dim": 256, "output_dim": 64, "num_layers": 4},
    ]
    
    if device == "cpu":
        # Reduce sizes for CPU testing
        configs = [
            {"pop_size": 32, "input_dim": 32, "hidden_dim": 64, "output_dim": 16, "num_layers": 2},
            {"pop_size": 64, "input_dim": 64, "hidden_dim": 128, "output_dim": 32, "num_layers": 3},
        ]
        num_iterations = 10
    else:
        num_iterations = 100
    
    print(f"\nRunning {num_iterations} iterations per benchmark")
    print("-" * 70)
    
    for cfg in configs:
        pop_size = cfg["pop_size"]
        input_dim = cfg["input_dim"]
        hidden_dim = cfg["hidden_dim"]
        output_dim = cfg["output_dim"]
        num_layers = cfg["num_layers"]
        
        print(f"\nConfig: pop={pop_size}, input={input_dim}, hidden={hidden_dim}, "
              f"output={output_dim}, layers={num_layers}")
        
        # Create model and strategy
        model = create_mlp(input_dim, hidden_dim, output_dim, num_layers, device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {num_params:,}")
        
        config = EggrollConfig(sigma=0.1, lr=0.01, rank=4, antithetic=True, seed=42)
        strategy = EggrollStrategy.from_config(config)
        strategy.setup(model)
        
        # Benchmark batched forward
        batched_time, output_shape = benchmark_batched_forward(
            model, strategy, pop_size, input_dim, device, num_iterations
        )
        print(f"  batched_forward: {batched_time*1000:.2f} ms")
        print(f"    Output shape: {output_shape}")
        print(f"    Throughput: {pop_size / batched_time:,.0f} evals/sec")
        
        # Benchmark sequential (for comparison)
        if pop_size <= 512:  # Skip for very large populations
            seq_time, actual_seq_pop = benchmark_sequential(
                model, strategy, pop_size, input_dim, device, num_iterations
            )
            print(f"  sequential (extrapolated from {actual_seq_pop}): {seq_time*1000:.2f} ms")
            print(f"    Speedup: {seq_time / batched_time:.1f}x")
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
