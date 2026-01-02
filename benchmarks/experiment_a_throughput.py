#!/usr/bin/env python3
"""
Experiment A: Pure Forward Pass Throughput Benchmark

Goal: Measure raw forward pass performance - EGGROLL vs standard inference.

Paper claim: ~91% of inference time (pre-gen noise) or ~69% (on-the-fly noise)

This experiment isolates the forward pass from environment overhead to measure
the true computational cost of EGGROLL's low-rank perturbations.

Baselines:
- inference_torch: Standard PyTorch batched forward pass
- inference_jax: Standard JAX forward pass (jit+vmap)
- eggroll_jax: JAX EGGROLL (reference implementation)
- eggroll_torch: PyTorch EGGROLL

Variables swept:
- Population size: 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
- Model size: small (4→256→2), medium (64→512→64), large (256→2048→256)
- Rank: 1, 2, 4, 8, 16

Metrics:
- Samples/second (throughput)
- Time per forward pass (ms)
- Overhead vs inference: (eggroll_time - inference_time) / inference_time
- Memory usage

Usage:
    uv run python benchmarks/experiment_a_throughput.py
    uv run python benchmarks/experiment_a_throughput.py --pop-sizes 256 512 1024
    uv run python benchmarks/experiment_a_throughput.py --model-size medium
    uv run python benchmarks/experiment_a_throughput.py --all-baselines
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Configuration
# =============================================================================

MODEL_CONFIGS = {
    "small": {"input_dim": 4, "hidden_dim": 256, "output_dim": 2, "n_layers": 3},
    "medium": {"input_dim": 64, "hidden_dim": 512, "output_dim": 64, "n_layers": 3},
    "large": {"input_dim": 256, "hidden_dim": 2048, "output_dim": 256, "n_layers": 3},
}

DEFAULT_POP_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_RANKS = [1, 2, 4, 8, 16]


@dataclass
class ExperimentConfig:
    """Configuration for Experiment A."""
    model_size: str = "small"
    pop_sizes: List[int] = field(default_factory=lambda: DEFAULT_POP_SIZES.copy())
    ranks: List[int] = field(default_factory=lambda: [4])  # Default to rank=4
    sigma: float = 0.2
    warmup_iters: int = 20
    bench_iters: int = 100
    seed: int = 42


@dataclass
class ThroughputResult:
    """Results for a single configuration."""
    method: str
    model_size: str
    pop_size: int
    rank: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_samples_per_sec: float
    memory_mb: float
    overhead_vs_inference: Optional[float] = None


@dataclass
class ExperimentResults:
    """Full experiment results."""
    timestamp: str
    config: Dict[str, Any]
    gpu_name: str
    results: List[ThroughputResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Utility Functions
# =============================================================================

def get_gpu_stats() -> Dict[str, float]:
    """Get current GPU stats via nvidia-smi."""
    import subprocess
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
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return "Unknown"


# =============================================================================
# PyTorch Benchmarks
# =============================================================================

def benchmark_torch_inference(
    config: ExperimentConfig,
    pop_sizes: List[int],
    verbose: bool = True,
) -> List[ThroughputResult]:
    """Benchmark standard PyTorch batched inference (baseline)."""
    import torch
    import torch.nn as nn
    
    if verbose:
        print(f"\n{'='*60}")
        print("Benchmarking: PyTorch Standard Inference (baseline)")
        print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = MODEL_CONFIGS[config.model_size]
    
    # Create model
    layers = []
    dims = [model_cfg["input_dim"]] + [model_cfg["hidden_dim"]] * (model_cfg["n_layers"] - 1) + [model_cfg["output_dim"]]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    model = nn.Sequential(*layers).to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Model: {config.model_size} ({total_params:,} params)")
        print(f"Architecture: {model_cfg['input_dim']}→{model_cfg['hidden_dim']}→{model_cfg['output_dim']}")
    
    results = []
    
    for pop_size in pop_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        
        x = torch.randn(pop_size, model_cfg["input_dim"], device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(config.warmup_iters):
                _ = model(x)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(config.bench_iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(x)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        
        stats = get_gpu_stats()
        
        result = ThroughputResult(
            method="inference_torch",
            model_size=config.model_size,
            pop_size=pop_size,
            rank=0,  # N/A for inference
            mean_time_ms=np.mean(times),
            std_time_ms=np.std(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times),
            throughput_samples_per_sec=pop_size / (np.mean(times) / 1000),
            memory_mb=stats.get("memory_used_mb", 0),
        )
        results.append(result)
        
        if verbose:
            print(f"  pop_size={pop_size:5d}: {result.mean_time_ms:.3f} ± {result.std_time_ms:.3f} ms, "
                  f"{result.throughput_samples_per_sec:,.0f} samples/sec")
    
    return results


def benchmark_torch_eggroll(
    config: ExperimentConfig,
    pop_sizes: List[int],
    ranks: List[int],
    verbose: bool = True,
) -> List[ThroughputResult]:
    """Benchmark PyTorch EGGROLL batched forward."""
    import torch
    import torch.nn as nn
    from hyperscalees.torch import EggrollStrategy
    
    if verbose:
        print(f"\n{'='*60}")
        print("Benchmarking: PyTorch EGGROLL")
        print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = MODEL_CONFIGS[config.model_size]
    
    # Create model
    layers = []
    dims = [model_cfg["input_dim"]] + [model_cfg["hidden_dim"]] * (model_cfg["n_layers"] - 1) + [model_cfg["output_dim"]]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    model = nn.Sequential(*layers).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Model: {config.model_size} ({total_params:,} params)")
    
    results = []
    
    for rank in ranks:
        if verbose:
            print(f"\n  Rank={rank}")
        
        for pop_size in pop_sizes:
            gc.collect()
            torch.cuda.empty_cache()
            
            strategy = EggrollStrategy(
                sigma=config.sigma,
                lr=0.1,
                rank=rank,
                seed=config.seed,
            )
            strategy.setup(model)
            
            x = torch.randn(pop_size, model_cfg["input_dim"], device=device)
            
            # Warmup
            for i in range(config.warmup_iters):
                with strategy.perturb(population_size=pop_size, epoch=i) as ctx:
                    _ = ctx.batched_forward(model, x)
            torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for i in range(config.bench_iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with strategy.perturb(population_size=pop_size, epoch=config.warmup_iters + i) as ctx:
                    _ = ctx.batched_forward(model, x)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            
            stats = get_gpu_stats()
            
            result = ThroughputResult(
                method="eggroll_torch",
                model_size=config.model_size,
                pop_size=pop_size,
                rank=rank,
                mean_time_ms=np.mean(times),
                std_time_ms=np.std(times),
                min_time_ms=np.min(times),
                max_time_ms=np.max(times),
                throughput_samples_per_sec=pop_size / (np.mean(times) / 1000),
                memory_mb=stats.get("memory_used_mb", 0),
            )
            results.append(result)
            
            if verbose:
                print(f"    pop_size={pop_size:5d}: {result.mean_time_ms:.3f} ± {result.std_time_ms:.3f} ms, "
                      f"{result.throughput_samples_per_sec:,.0f} samples/sec")
    
    return results


def benchmark_torch_eggroll_compiled(
    config: ExperimentConfig,
    pop_sizes: List[int],
    ranks: List[int],
    verbose: bool = True,
) -> List[ThroughputResult]:
    """Benchmark PyTorch EGGROLL with torch.compile."""
    import torch
    import torch.nn as nn
    from hyperscalees.torch import EggrollStrategy
    
    if verbose:
        print(f"\n{'='*60}")
        print("Benchmarking: PyTorch EGGROLL (torch.compile)")
        print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = MODEL_CONFIGS[config.model_size]
    
    # Create model
    layers = []
    dims = [model_cfg["input_dim"]] + [model_cfg["hidden_dim"]] * (model_cfg["n_layers"] - 1) + [model_cfg["output_dim"]]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    model = nn.Sequential(*layers).to(device)
    
    # Compile the model
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        if verbose:
            print(f"Warning: torch.compile failed: {e}")
            print("Falling back to eager mode")
    
    results = []
    
    for rank in ranks:
        if verbose:
            print(f"\n  Rank={rank}")
        
        for pop_size in pop_sizes:
            gc.collect()
            torch.cuda.empty_cache()
            
            strategy = EggrollStrategy(
                sigma=config.sigma,
                lr=0.1,
                rank=rank,
                seed=config.seed,
            )
            strategy.setup(model)
            
            x = torch.randn(pop_size, model_cfg["input_dim"], device=device)
            
            # Warmup (more iterations for compilation)
            for i in range(config.warmup_iters * 2):
                with strategy.perturb(population_size=pop_size, epoch=i) as ctx:
                    _ = ctx.batched_forward(model, x)
            torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for i in range(config.bench_iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with strategy.perturb(population_size=pop_size, epoch=config.warmup_iters * 2 + i) as ctx:
                    _ = ctx.batched_forward(model, x)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            
            stats = get_gpu_stats()
            
            result = ThroughputResult(
                method="eggroll_torch_compiled",
                model_size=config.model_size,
                pop_size=pop_size,
                rank=rank,
                mean_time_ms=np.mean(times),
                std_time_ms=np.std(times),
                min_time_ms=np.min(times),
                max_time_ms=np.max(times),
                throughput_samples_per_sec=pop_size / (np.mean(times) / 1000),
                memory_mb=stats.get("memory_used_mb", 0),
            )
            results.append(result)
            
            if verbose:
                print(f"    pop_size={pop_size:5d}: {result.mean_time_ms:.3f} ± {result.std_time_ms:.3f} ms, "
                      f"{result.throughput_samples_per_sec:,.0f} samples/sec")
    
    return results


# =============================================================================
# JAX Benchmarks
# =============================================================================

def benchmark_jax_inference(
    config: ExperimentConfig,
    pop_sizes: List[int],
    verbose: bool = True,
) -> List[ThroughputResult]:
    """Benchmark standard JAX batched inference (baseline)."""
    try:
        import jax
        import jax.numpy as jnp
        from functools import partial
    except ImportError:
        if verbose:
            print("JAX not available, skipping JAX inference benchmark")
        return []
    
    if verbose:
        print(f"\n{'='*60}")
        print("Benchmarking: JAX Standard Inference (baseline)")
        print(f"{'='*60}")
        print(f"JAX version: {jax.__version__}")
        print(f"Devices: {jax.devices()}")
    
    model_cfg = MODEL_CONFIGS[config.model_size]
    
    # Create parameters
    key = jax.random.PRNGKey(config.seed)
    params = {}
    dims = [model_cfg["input_dim"]] + [model_cfg["hidden_dim"]] * (model_cfg["n_layers"] - 1) + [model_cfg["output_dim"]]
    
    for i in range(len(dims) - 1):
        key, wkey, bkey = jax.random.split(key, 3)
        scale = 1.0 / np.sqrt(dims[i])
        params[f"w{i}"] = jax.random.normal(wkey, (dims[i+1], dims[i])) * scale
        params[f"b{i}"] = jnp.zeros(dims[i+1])
    
    n_layers = model_cfg["n_layers"]
    
    @jax.jit
    def forward(params, x):
        for i in range(n_layers - 1):
            x = x @ params[f"w{i}"].T + params[f"b{i}"]
            if i < n_layers - 2:
                x = jnp.tanh(x)
        return x
    
    # Batched forward
    @jax.jit
    def batched_forward(params, x):
        return jax.vmap(lambda xi: forward(params, xi))(x)
    
    total_params = sum(p.size for p in jax.tree.leaves(params))
    if verbose:
        print(f"Model: {config.model_size} ({total_params:,} params)")
    
    results = []
    
    for pop_size in pop_sizes:
        gc.collect()
        jax.clear_caches()
        
        x = jax.random.normal(jax.random.PRNGKey(config.seed + 1), (pop_size, model_cfg["input_dim"]))
        
        # Warmup
        for _ in range(config.warmup_iters):
            out = batched_forward(params, x)
            out.block_until_ready()
        
        # Benchmark
        times = []
        for _ in range(config.bench_iters):
            start = time.perf_counter()
            out = batched_forward(params, x)
            out.block_until_ready()
            times.append((time.perf_counter() - start) * 1000)
        
        stats = get_gpu_stats()
        
        result = ThroughputResult(
            method="inference_jax",
            model_size=config.model_size,
            pop_size=pop_size,
            rank=0,
            mean_time_ms=np.mean(times),
            std_time_ms=np.std(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times),
            throughput_samples_per_sec=pop_size / (np.mean(times) / 1000),
            memory_mb=stats.get("memory_used_mb", 0),
        )
        results.append(result)
        
        if verbose:
            print(f"  pop_size={pop_size:5d}: {result.mean_time_ms:.3f} ± {result.std_time_ms:.3f} ms, "
                  f"{result.throughput_samples_per_sec:,.0f} samples/sec")
    
    return results


def benchmark_jax_eggroll(
    config: ExperimentConfig,
    pop_sizes: List[int],
    ranks: List[int],
    verbose: bool = True,
) -> List[ThroughputResult]:
    """Benchmark JAX EGGROLL (reference implementation)."""
    try:
        import jax
        import jax.numpy as jnp
        from functools import partial
        from hyperscalees.noiser.eggroll import EggRoll
        from hyperscalees.models.common import MLP, simple_es_tree_key, PARAM, MM_PARAM, EMB_PARAM
    except ImportError as e:
        if verbose:
            print(f"JAX EGGROLL not available: {e}")
        return []
    
    if verbose:
        print(f"\n{'='*60}")
        print("Benchmarking: JAX EGGROLL (reference)")
        print(f"{'='*60}")
    
    model_cfg = MODEL_CONFIGS[config.model_size]
    
    results = []
    
    for rank in ranks:
        if verbose:
            print(f"\n  Rank={rank}")
        
        # Initialize model
        key = jax.random.PRNGKey(config.seed)
        init_result = MLP.rand_init(
            key,
            model_cfg["input_dim"],
            model_cfg["output_dim"],
            [model_cfg["hidden_dim"]] * (model_cfg["n_layers"] - 1),
            use_bias=True,
            activation="pqn",
            dtype=jnp.float32,
        )
        
        params = init_result.params
        frozen_params = init_result.frozen_params
        es_map = init_result.es_map
        scan_map = init_result.scan_map
        
        # Create ES tree key
        es_tree_key = simple_es_tree_key(params, key, scan_map)
        
        # Initialize noiser
        frozen_noiser_params, noiser_params = EggRoll.init_noiser(
            params, sigma=config.sigma, lr=0.1, rank=rank
        )
        
        for pop_size in pop_sizes:
            gc.collect()
            jax.clear_caches()
            
            x = jnp.ones((pop_size, model_cfg["input_dim"]))
            iterinfos = (jnp.zeros(pop_size, dtype=jnp.int32), jnp.arange(pop_size, dtype=jnp.int32))
            
            # Create CommonParams for forward
            from hyperscalees.models.base_model import CommonParams
            
            @jax.jit
            def forward_fn(noiser_params, params, iterinfos, x):
                def single_forward(iterinfo, xi):
                    common_params = CommonParams(
                        noiser=EggRoll,
                        frozen_noiser_params=frozen_noiser_params,
                        noiser_params=noiser_params,
                        frozen_params=frozen_params,
                        params=params,
                        es_tree_key=es_tree_key,
                        iterinfo=iterinfo,
                    )
                    return MLP._forward(common_params, xi)
                
                return jax.vmap(single_forward)(iterinfos, x)
            
            # Warmup
            for _ in range(config.warmup_iters):
                out = forward_fn(noiser_params, params, iterinfos, x)
                out.block_until_ready()
            
            # Benchmark
            times = []
            for _ in range(config.bench_iters):
                start = time.perf_counter()
                out = forward_fn(noiser_params, params, iterinfos, x)
                out.block_until_ready()
                times.append((time.perf_counter() - start) * 1000)
            
            stats = get_gpu_stats()
            
            result = ThroughputResult(
                method="eggroll_jax",
                model_size=config.model_size,
                pop_size=pop_size,
                rank=rank,
                mean_time_ms=np.mean(times),
                std_time_ms=np.std(times),
                min_time_ms=np.min(times),
                max_time_ms=np.max(times),
                throughput_samples_per_sec=pop_size / (np.mean(times) / 1000),
                memory_mb=stats.get("memory_used_mb", 0),
            )
            results.append(result)
            
            if verbose:
                print(f"    pop_size={pop_size:5d}: {result.mean_time_ms:.3f} ± {result.std_time_ms:.3f} ms, "
                      f"{result.throughput_samples_per_sec:,.0f} samples/sec")
    
    return results


# =============================================================================
# Analysis & Reporting
# =============================================================================

def compute_overhead(results: List[ThroughputResult]) -> List[ThroughputResult]:
    """Compute overhead vs inference baseline for each result."""
    # Build lookup of inference baselines by (model_size, pop_size)
    inference_baselines = {}
    for r in results:
        if r.method in ("inference_torch", "inference_jax"):
            key = (r.method.split("_")[1], r.model_size, r.pop_size)  # (framework, model_size, pop_size)
            inference_baselines[key] = r.mean_time_ms
    
    # Compute overhead
    for r in results:
        if "eggroll" in r.method:
            framework = "torch" if "torch" in r.method else "jax"
            key = (framework, r.model_size, r.pop_size)
            if key in inference_baselines:
                baseline = inference_baselines[key]
                r.overhead_vs_inference = (r.mean_time_ms - baseline) / baseline
    
    return results


def generate_summary(results: List[ThroughputResult]) -> Dict[str, Any]:
    """Generate summary statistics."""
    summary = {
        "by_method": {},
        "by_pop_size": {},
        "paper_comparison": {},
    }
    
    # Group by method
    methods = set(r.method for r in results)
    for method in methods:
        method_results = [r for r in results if r.method == method]
        summary["by_method"][method] = {
            "avg_throughput": np.mean([r.throughput_samples_per_sec for r in method_results]),
            "avg_time_ms": np.mean([r.mean_time_ms for r in method_results]),
            "count": len(method_results),
        }
    
    # Paper comparison: EGGROLL overhead should be ~9% (pre-gen) or ~31% (on-the-fly)
    eggroll_torch = [r for r in results if r.method == "eggroll_torch" and r.overhead_vs_inference is not None]
    eggroll_jax = [r for r in results if r.method == "eggroll_jax" and r.overhead_vs_inference is not None]
    
    if eggroll_torch:
        avg_overhead = np.mean([r.overhead_vs_inference for r in eggroll_torch])
        summary["paper_comparison"]["torch_overhead_pct"] = avg_overhead * 100
        summary["paper_comparison"]["torch_meets_91_target"] = avg_overhead <= 0.10  # ~9% overhead
        summary["paper_comparison"]["torch_meets_69_target"] = avg_overhead <= 0.45  # ~31% overhead -> 1.45x
    
    if eggroll_jax:
        avg_overhead = np.mean([r.overhead_vs_inference for r in eggroll_jax])
        summary["paper_comparison"]["jax_overhead_pct"] = avg_overhead * 100
        summary["paper_comparison"]["jax_meets_91_target"] = avg_overhead <= 0.10
        summary["paper_comparison"]["jax_meets_69_target"] = avg_overhead <= 0.45
    
    return summary


def print_results_table(results: List[ThroughputResult], verbose: bool = True):
    """Print results in a formatted table."""
    if not results:
        return
    
    print(f"\n{'='*100}")
    print("RESULTS SUMMARY")
    print(f"{'='*100}")
    
    # Table header
    print(f"{'Method':<25} {'Model':<8} {'PopSize':>8} {'Rank':>5} {'Time(ms)':>12} {'Throughput':>15} {'Overhead':>10}")
    print("-" * 100)
    
    for r in sorted(results, key=lambda x: (x.method, x.model_size, x.pop_size, x.rank)):
        overhead_str = f"{r.overhead_vs_inference*100:+.1f}%" if r.overhead_vs_inference is not None else "N/A"
        print(f"{r.method:<25} {r.model_size:<8} {r.pop_size:>8} {r.rank:>5} "
              f"{r.mean_time_ms:>8.3f}±{r.std_time_ms:.3f} {r.throughput_samples_per_sec:>12,.0f}/s {overhead_str:>10}")


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.bool_, np.generic)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_results(experiment_results: ExperimentResults, output_path: Path):
    """Save results to JSON file."""
    # Convert dataclasses to dicts
    data = {
        "timestamp": experiment_results.timestamp,
        "config": experiment_results.config,
        "gpu_name": experiment_results.gpu_name,
        "results": [asdict(r) for r in experiment_results.results],
        "summary": experiment_results.summary,
    }
    
    # Convert numpy types to native Python types
    data = convert_numpy_types(data)
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment A: Pure Forward Pass Throughput")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="small",
                        help="Model size configuration")
    parser.add_argument("--pop-sizes", type=int, nargs="+", default=None,
                        help="Population sizes to test")
    parser.add_argument("--ranks", type=int, nargs="+", default=[4],
                        help="Ranks to test")
    parser.add_argument("--all-baselines", action="store_true",
                        help="Run all baselines (inference + EGGROLL for both Torch and JAX)")
    parser.add_argument("--torch-only", action="store_true",
                        help="Only run PyTorch benchmarks")
    parser.add_argument("--jax-only", action="store_true",
                        help="Only run JAX benchmarks")
    parser.add_argument("--include-compiled", action="store_true",
                        help="Include torch.compile benchmark")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--quiet", action="store_true",
                        help="Less verbose output")
    args = parser.parse_args()
    
    # Configure experiment
    config = ExperimentConfig(
        model_size=args.model_size,
        pop_sizes=args.pop_sizes or DEFAULT_POP_SIZES,
        ranks=args.ranks,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
    )
    
    print(f"\n{'#'*60}")
    print("# Experiment A: Pure Forward Pass Throughput Benchmark")
    print(f"{'#'*60}")
    print(f"Model size: {config.model_size}")
    print(f"Model config: {MODEL_CONFIGS[config.model_size]}")
    print(f"Population sizes: {config.pop_sizes}")
    print(f"Ranks: {config.ranks}")
    print(f"Warmup: {config.warmup_iters}, Benchmark: {config.bench_iters} iterations")
    
    all_results = []
    verbose = not args.quiet
    
    # Run PyTorch benchmarks
    if not args.jax_only:
        # Inference baseline
        all_results.extend(benchmark_torch_inference(config, config.pop_sizes, verbose=verbose))
        
        # EGGROLL
        all_results.extend(benchmark_torch_eggroll(config, config.pop_sizes, config.ranks, verbose=verbose))
        
        # Compiled EGGROLL (optional)
        if args.include_compiled:
            all_results.extend(benchmark_torch_eggroll_compiled(config, config.pop_sizes, config.ranks, verbose=verbose))
    
    # Run JAX benchmarks
    if not args.torch_only and (args.all_baselines or args.jax_only):
        all_results.extend(benchmark_jax_inference(config, config.pop_sizes, verbose=verbose))
        all_results.extend(benchmark_jax_eggroll(config, config.pop_sizes, config.ranks, verbose=verbose))
    
    # Compute overhead
    all_results = compute_overhead(all_results)
    
    # Generate summary
    summary = generate_summary(all_results)
    
    # Print results
    print_results_table(all_results, verbose=verbose)
    
    # Print paper comparison
    print(f"\n{'='*60}")
    print("PAPER COMPARISON")
    print(f"{'='*60}")
    print("Paper claims EGGROLL forward pass should be:")
    print("  - ~91% of inference time (pre-gen noise) → max ~10% overhead")
    print("  - ~69% of inference time (on-the-fly noise) → max ~45% overhead")
    print()
    
    if "torch_overhead_pct" in summary.get("paper_comparison", {}):
        overhead = summary["paper_comparison"]["torch_overhead_pct"]
        meets_91 = summary["paper_comparison"]["torch_meets_91_target"]
        meets_69 = summary["paper_comparison"]["torch_meets_69_target"]
        status_91 = "✅" if meets_91 else "❌"
        status_69 = "✅" if meets_69 else "❌"
        print(f"PyTorch EGGROLL: {overhead:.1f}% overhead vs inference")
        print(f"  {status_91} Meets 91% target (≤10% overhead): {meets_91}")
        print(f"  {status_69} Meets 69% target (≤45% overhead): {meets_69}")
    
    if "jax_overhead_pct" in summary.get("paper_comparison", {}):
        overhead = summary["paper_comparison"]["jax_overhead_pct"]
        meets_91 = summary["paper_comparison"]["jax_meets_91_target"]
        meets_69 = summary["paper_comparison"]["jax_meets_69_target"]
        status_91 = "✅" if meets_91 else "❌"
        status_69 = "✅" if meets_69 else "❌"
        print(f"\nJAX EGGROLL: {overhead:.1f}% overhead vs inference")
        print(f"  {status_91} Meets 91% target (≤10% overhead): {meets_91}")
        print(f"  {status_69} Meets 69% target (≤45% overhead): {meets_69}")
    
    # Create experiment results
    experiment_results = ExperimentResults(
        timestamp=datetime.now().isoformat(),
        config=asdict(config),
        gpu_name=get_gpu_name(),
        results=all_results,
        summary=summary,
    )
    
    # Save results
    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / f"experiment_a_results_{timestamp}.json"
    else:
        output_path = Path(output_path)
    
    save_results(experiment_results, output_path)
    
    return experiment_results


if __name__ == "__main__":
    main()
