"""
Benchmark: Compare batched forward pass between JAX and PyTorch EGGROLL implementations.

This script benchmarks the core computational primitive of EGGROLL - the batched forward
pass that evaluates all population members simultaneously. This is the key operation
that determines ES training throughput.

Benchmarks:
1. JAX implementation using vmap over do_mm
2. PyTorch implementation using batched_forward

Tested population sizes: 32, 64, 128, 256, 512, 1024, 2048
"""

import argparse
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

import numpy as np

# Check for available frameworks
HAS_JAX = False
HAS_TORCH = False

try:
    import jax
    import jax.numpy as jnp
    from functools import partial
    HAS_JAX = True
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    pass

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""
    population_sizes: List[int]
    input_dim: int = 512
    hidden_dim: int = 2048
    output_dim: int = 512
    batch_size: int = 1  # Input batch size per population member
    rank: int = 4
    sigma: float = 0.1
    seed: int = 42
    warmup_iters: int = 10
    bench_iters: int = 50
    num_layers: int = 3  # Number of linear layers in the model


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    framework: str
    population_size: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_samples_per_sec: float
    total_params: int
    config: Dict


def create_jax_model(config: BenchmarkConfig, dtype=jnp.float32):
    """Create a simple JAX MLP model for benchmarking."""
    key = jax.random.PRNGKey(config.seed)
    
    params = {}
    dims = [config.input_dim] + [config.hidden_dim] * (config.num_layers - 1) + [config.output_dim]
    
    for i in range(len(dims) - 1):
        key, subkey = jax.random.split(key)
        in_dim, out_dim = dims[i], dims[i + 1]
        scale = 1.0 / np.sqrt(in_dim)
        params[f"layer_{i}"] = (jax.random.normal(subkey, (out_dim, in_dim)) * scale).astype(dtype)
    
    return params


def create_torch_model(config: BenchmarkConfig, device: str = "cuda"):
    """Create a simple PyTorch MLP model for benchmarking."""
    dims = [config.input_dim] + [config.hidden_dim] * (config.num_layers - 1) + [config.output_dim]
    
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
        if i < len(dims) - 2:  # No activation after last layer
            layers.append(nn.ReLU())
    
    model = nn.Sequential(*layers)
    model = model.to(device)
    return model


def benchmark_jax(config: BenchmarkConfig, population_size: int) -> BenchmarkResult:
    """Benchmark JAX EGGROLL batched forward."""
    from hyperscalees.noiser.eggroll import EggRoll, get_lora_update_params
    
    dtype = jnp.float32
    params = create_jax_model(config, dtype)
    
    # Initialize the noiser
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, 
        sigma=config.sigma, 
        lr=0.01,
        rank=config.rank,
        noise_reuse=0
    )
    
    # Create random keys for each parameter
    base_key = jax.random.PRNGKey(config.seed)
    es_tree_keys = jax.tree.map(
        lambda p, k: k,
        params,
        {name: jax.random.fold_in(base_key, i) for i, name in enumerate(params.keys())}
    )
    
    # Create input data: (population_size, batch_size, input_dim)
    x = jax.random.normal(
        jax.random.PRNGKey(config.seed + 1),
        (population_size, config.batch_size, config.input_dim),
        dtype=dtype
    )
    
    # Create iterinfo: (epoch, thread_ids)
    epoch = 0
    thread_ids = jnp.arange(population_size)
    
    def single_forward(x_single, thread_id):
        """Forward pass for a single population member."""
        iterinfo = (epoch, thread_id)
        out = x_single
        
        for name, param in params.items():
            # Use the noiser's do_mm for perturbed matmul
            out = EggRoll.do_mm(
                frozen_noiser_params, 
                noiser_params, 
                param, 
                es_tree_keys[name], 
                iterinfo, 
                out
            )
            # ReLU except for last layer
            if name != f"layer_{config.num_layers - 1}":
                out = jax.nn.relu(out)
        
        return out
    
    # Vmap over population
    batched_forward = jax.jit(jax.vmap(single_forward, in_axes=(0, 0)))
    
    # Warmup
    for _ in range(config.warmup_iters):
        output = batched_forward(x, thread_ids)
        output.block_until_ready()
    
    # Benchmark
    times = []
    for _ in range(config.bench_iters):
        start = time.perf_counter()
        output = batched_forward(x, thread_ids)
        output.block_until_ready()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    total_params = sum(p.size for p in params.values())
    
    return BenchmarkResult(
        framework="JAX",
        population_size=population_size,
        mean_time_ms=float(np.mean(times)),
        std_time_ms=float(np.std(times)),
        min_time_ms=float(np.min(times)),
        max_time_ms=float(np.max(times)),
        throughput_samples_per_sec=population_size * config.batch_size / (np.mean(times) / 1000),
        total_params=total_params,
        config=asdict(config),
    )


def benchmark_torch(config: BenchmarkConfig, population_size: int) -> BenchmarkResult:
    """Benchmark PyTorch EGGROLL batched forward."""
    from hyperscalees.torch import EggrollStrategy
    
    device = "cuda"
    model = create_torch_model(config, device)
    
    # Set up the strategy
    strategy = EggrollStrategy(
        sigma=config.sigma,
        lr=0.01,
        rank=config.rank,
        seed=config.seed,
        antithetic=True,
    )
    strategy.setup(model)
    
    # Create input data: (population_size * batch_size, input_dim)
    # For batched_forward, we need one sample per population member
    torch.manual_seed(config.seed + 1)
    x = torch.randn(population_size, config.input_dim, device=device)
    
    # Warmup
    for _ in range(config.warmup_iters):
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            output = pop.batched_forward(model, x)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(config.bench_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            output = pop.batched_forward(model, x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    total_params = sum(p.numel() for p in model.parameters())
    
    return BenchmarkResult(
        framework="PyTorch",
        population_size=population_size,
        mean_time_ms=float(np.mean(times)),
        std_time_ms=float(np.std(times)),
        min_time_ms=float(np.min(times)),
        max_time_ms=float(np.max(times)),
        throughput_samples_per_sec=population_size / (np.mean(times) / 1000),
        total_params=total_params,
        config=asdict(config),
    )


def run_benchmarks(config: BenchmarkConfig, frameworks: List[str]) -> List[BenchmarkResult]:
    """Run all benchmarks.
    
    Note: We run all pop sizes for each framework separately to avoid
    JAX/PyTorch GPU context switching overhead which can cause measurement noise.
    """
    results = []
    
    # Run all JAX benchmarks first (to avoid JAX/PyTorch interference)
    if "jax" in frameworks and HAS_JAX:
        print("\n" + "="*60)
        print("Running JAX benchmarks...")
        print("="*60)
        for pop_size in config.population_sizes:
            print(f"\n  Population Size: {pop_size}")
            try:
                result = benchmark_jax(config, pop_size)
                results.append(result)
                print(f"    Mean: {result.mean_time_ms:.3f} ms (±{result.std_time_ms:.3f})")
                print(f"    Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")
            except Exception as e:
                print(f"    JAX benchmark failed: {e}")
    
    # Then run all PyTorch benchmarks
    if "torch" in frameworks and HAS_TORCH:
        print("\n" + "="*60)
        print("Running PyTorch benchmarks...")
        print("="*60)
        # Clear any JAX memory before PyTorch runs
        if HAS_JAX:
            import gc
            gc.collect()
        for pop_size in config.population_sizes:
            print(f"\n  Population Size: {pop_size}")
            try:
                result = benchmark_torch(config, pop_size)
                results.append(result)
                print(f"    Mean: {result.mean_time_ms:.3f} ms (±{result.std_time_ms:.3f})")
                print(f"    Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")
            except Exception as e:
                print(f"    PyTorch benchmark failed: {e}")
    
    return results


def print_summary(results: List[BenchmarkResult]):
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Group by population size
    pop_sizes = sorted(set(r.population_size for r in results))
    frameworks = sorted(set(r.framework for r in results))
    
    # Print header
    header = f"{'Pop Size':>10}"
    for fw in frameworks:
        header += f" | {fw + ' (ms)':>15} | {fw + ' (samp/s)':>15}"
    print(header)
    print("-" * len(header))
    
    # Print results
    for pop_size in pop_sizes:
        row = f"{pop_size:>10}"
        for fw in frameworks:
            matching = [r for r in results if r.population_size == pop_size and r.framework == fw]
            if matching:
                r = matching[0]
                row += f" | {r.mean_time_ms:>12.3f} ms | {r.throughput_samples_per_sec:>12.1f}/s"
            else:
                row += f" | {'N/A':>15} | {'N/A':>15}"
        print(row)
    
    # Compute speedups if both frameworks present
    if "JAX" in frameworks and "PyTorch" in frameworks:
        print("\n" + "-"*40)
        print("SPEEDUP (JAX time / PyTorch time)")
        print("-"*40)
        for pop_size in pop_sizes:
            jax_result = next((r for r in results if r.population_size == pop_size and r.framework == "JAX"), None)
            torch_result = next((r for r in results if r.population_size == pop_size and r.framework == "PyTorch"), None)
            if jax_result and torch_result:
                speedup = jax_result.mean_time_ms / torch_result.mean_time_ms
                faster = "PyTorch" if speedup > 1 else "JAX"
                ratio = speedup if speedup > 1 else 1/speedup
                print(f"  Pop {pop_size:>5}: {faster} is {ratio:.2f}x faster")


def save_results(results: List[BenchmarkResult], output_path: Path):
    """Save results to JSON file."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [asdict(r) for r in results],
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark EGGROLL batched forward")
    parser.add_argument("--frameworks", nargs="+", default=["jax", "torch"],
                        choices=["jax", "torch"], help="Frameworks to benchmark")
    parser.add_argument("--input-dim", type=int, default=512, help="Input dimension")
    parser.add_argument("--hidden-dim", type=int, default=2048, help="Hidden dimension")
    parser.add_argument("--output-dim", type=int, default=512, help="Output dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--rank", type=int, default=4, help="Low-rank perturbation rank")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise scale")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--pop-sizes", nargs="+", type=int, 
                        default=[32, 64, 128, 256, 512, 1024, 2048],
                        help="Population sizes to test")
    args = parser.parse_args()
    
    # Check available frameworks
    print("="*60)
    print("EGGROLL Batched Forward Benchmark")
    print("="*60)
    print(f"\nFramework availability:")
    print(f"  JAX: {'Available' if HAS_JAX else 'Not available'}")
    if HAS_JAX:
        print(f"       Devices: {jax.devices()}")
    print(f"  PyTorch: {'Available (CUDA)' if HAS_TORCH else 'Not available (no CUDA)'}")
    if HAS_TORCH:
        print(f"           Device: {torch.cuda.get_device_name(0)}")
    
    # Filter frameworks
    available_frameworks = []
    for fw in args.frameworks:
        if fw == "jax" and HAS_JAX:
            available_frameworks.append(fw)
        elif fw == "torch" and HAS_TORCH:
            available_frameworks.append(fw)
        else:
            print(f"\nWarning: {fw} not available, skipping")
    
    if not available_frameworks:
        print("\nError: No frameworks available for benchmarking!")
        sys.exit(1)
    
    # Create config
    config = BenchmarkConfig(
        population_sizes=args.pop_sizes,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        rank=args.rank,
        sigma=args.sigma,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
    )
    
    print(f"\nBenchmark Configuration:")
    print(f"  Model: {config.num_layers}-layer MLP")
    print(f"  Dims: {config.input_dim} -> {config.hidden_dim} -> {config.output_dim}")
    print(f"  Rank: {config.rank}")
    print(f"  Sigma: {config.sigma}")
    print(f"  Population sizes: {config.population_sizes}")
    print(f"  Warmup iters: {config.warmup_iters}")
    print(f"  Bench iters: {config.bench_iters}")
    
    # Run benchmarks
    results = run_benchmarks(config, available_frameworks)
    
    # Print summary
    print_summary(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    save_results(results, output_path)


if __name__ == "__main__":
    main()
