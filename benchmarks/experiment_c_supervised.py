#!/usr/bin/env python3
"""
Experiment C: Supervised Learning Benchmark (DataLoader Integration)

Goal: Validate EGGROLL on a simple supervised learning task with standard
PyTorch DataLoader patterns.

Why: ES can be used for non-RL tasks. This tests integration with the
PyTorch ecosystem and measures overhead in a controlled setting.

Setup:
    # MNIST or CIFAR-10 classification via ES
    dataloader = DataLoader(dataset, batch_size=pop_size, shuffle=True)

    for epoch in range(N):
        for x_batch, y_batch in dataloader:
            with strategy.perturb(pop_size, epoch) as ctx:
                logits = ctx.batched_forward(model, x_batch)
                fitnesses = -cross_entropy(logits, y_batch, reduction='none')
            strategy.step(fitnesses)

Metrics:
- Time per batch (ms)
- Samples/second
- Overhead vs standard inference
- Final accuracy (sanity check)

Comparison:
- EGGROLL
- Standard inference (baseline)
- SGD (for reference, not apples-to-apples)

Usage:
    uv run python benchmarks/experiment_c_supervised.py
    uv run python benchmarks/experiment_c_supervised.py --dataset mnist
    uv run python benchmarks/experiment_c_supervised.py --pop-sizes 128 256 512
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_POP_SIZES = [64, 128, 256, 512, 1024]


@dataclass
class ExperimentConfig:
    """Configuration for Experiment C."""
    dataset: str = "mnist"  # mnist or cifar10
    pop_sizes: List[int] = field(default_factory=lambda: DEFAULT_POP_SIZES.copy())
    rank: int = 4
    sigma: float = 0.1
    lr: float = 0.05
    num_epochs: int = 3
    warmup_batches: int = 5
    bench_batches: int = 20
    hidden_size: int = 256
    n_layers: int = 3
    seed: int = 42


@dataclass
class SupervisedResult:
    """Results for a single configuration."""
    method: str
    dataset: str
    pop_size: int
    rank: int
    mean_batch_time_ms: float
    std_batch_time_ms: float
    samples_per_sec: float
    overhead_vs_inference: Optional[float]
    final_accuracy: float
    memory_mb: float


@dataclass
class ExperimentResults:
    """Full experiment results."""
    timestamp: str
    config: Dict[str, Any]
    gpu_name: str
    results: List[SupervisedResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Utility Functions
# =============================================================================

def get_gpu_stats() -> Dict[str, float]:
    """Get current GPU stats via nvidia-smi."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return {
                "gpu_util_pct": float(parts[0]),
                "memory_used_mb": float(parts[1]),
                "memory_total_mb": float(parts[2]),
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


def get_dataset(name: str, train: bool = True):
    """Load dataset."""
    import torch
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms
    
    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = torchvision.datasets.MNIST(
            root='./data', train=train, download=True, transform=transform
        )
        input_dim = 28 * 28
        num_classes = 10
    elif name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transform
        )
        input_dim = 32 * 32 * 3
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return dataset, input_dim, num_classes


# =============================================================================
# PyTorch Benchmarks
# =============================================================================

def benchmark_torch_inference(
    config: ExperimentConfig,
    pop_sizes: List[int],
    verbose: bool = True,
) -> List[SupervisedResult]:
    """Benchmark standard PyTorch inference (baseline)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    
    if verbose:
        print(f"\n{'='*60}")
        print("Benchmarking: PyTorch Standard Inference (baseline)")
        print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset, input_dim, num_classes = get_dataset(config.dataset, train=True)
    
    results = []
    
    for pop_size in pop_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        
        if verbose:
            print(f"\n  Batch size: {pop_size}")
        
        # Create model
        torch.manual_seed(config.seed)
        layers = []
        dims = [input_dim] + [config.hidden_size] * (config.n_layers - 1) + [num_classes]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        model = nn.Sequential(*layers).to(device)
        model.eval()
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=pop_size, shuffle=True, num_workers=0)
        
        # Warmup
        batch_iter = iter(dataloader)
        with torch.no_grad():
            for _ in range(config.warmup_batches):
                try:
                    x, y = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(dataloader)
                    x, y = next(batch_iter)
                x = x.view(x.size(0), -1).to(device)
                _ = model(x)
        torch.cuda.synchronize()
        
        # Benchmark
        batch_times = []
        correct = 0
        total = 0
        
        batch_iter = iter(dataloader)
        with torch.no_grad():
            for _ in range(config.bench_batches):
                try:
                    x, y = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(dataloader)
                    x, y = next(batch_iter)
                
                x = x.view(x.size(0), -1).to(device)
                y = y.to(device)
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                logits = model(x)
                torch.cuda.synchronize()
                batch_times.append((time.perf_counter() - start) * 1000)
                
                # Accuracy
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        stats = get_gpu_stats()
        
        result = SupervisedResult(
            method="inference_torch",
            dataset=config.dataset,
            pop_size=pop_size,
            rank=0,
            mean_batch_time_ms=np.mean(batch_times),
            std_batch_time_ms=np.std(batch_times),
            samples_per_sec=pop_size / (np.mean(batch_times) / 1000),
            overhead_vs_inference=None,
            final_accuracy=accuracy,
            memory_mb=stats.get("memory_used_mb", 0),
        )
        results.append(result)
        
        if verbose:
            print(f"    Time: {result.mean_batch_time_ms:.3f} ± {result.std_batch_time_ms:.3f} ms")
            print(f"    Throughput: {result.samples_per_sec:,.0f} samples/sec")
            print(f"    Accuracy: {result.final_accuracy*100:.1f}%")
    
    return results


def benchmark_torch_eggroll(
    config: ExperimentConfig,
    pop_sizes: List[int],
    verbose: bool = True,
) -> List[SupervisedResult]:
    """Benchmark PyTorch EGGROLL on supervised learning."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from hyperscalees.torch import EggrollStrategy
    
    if verbose:
        print(f"\n{'='*60}")
        print("Benchmarking: PyTorch EGGROLL (Supervised Learning)")
        print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset, input_dim, num_classes = get_dataset(config.dataset, train=True)
    test_dataset, _, _ = get_dataset(config.dataset, train=False)
    
    results = []
    
    for pop_size in pop_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        
        if verbose:
            print(f"\n  Population size: {pop_size}")
        
        # Create model
        torch.manual_seed(config.seed)
        layers = []
        dims = [input_dim] + [config.hidden_size] * (config.n_layers - 1) + [num_classes]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        model = nn.Sequential(*layers).to(device)
        
        # Setup strategy
        strategy = EggrollStrategy(
            sigma=config.sigma,
            lr=config.lr,
            rank=config.rank,
            seed=config.seed,
        )
        strategy.setup(model)
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=pop_size, shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        # Warmup
        batch_iter = iter(dataloader)
        for i in range(config.warmup_batches):
            try:
                x, y = next(batch_iter)
            except StopIteration:
                batch_iter = iter(dataloader)
                x, y = next(batch_iter)
            
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            
            with strategy.perturb(population_size=pop_size, epoch=i) as ctx:
                logits = ctx.batched_forward(model, x)
                # Fitness = negative cross entropy (higher is better)
                # For ES, we compute per-sample fitness
                fitnesses = -F.cross_entropy(logits, y, reduction='none')
            strategy.step(fitnesses)
        torch.cuda.synchronize()
        
        # Benchmark: run multiple epochs
        batch_times = []
        batch_iter = iter(dataloader)
        
        for epoch in range(config.num_epochs):
            for batch_idx in range(config.bench_batches):
                try:
                    x, y = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(dataloader)
                    x, y = next(batch_iter)
                
                x = x.view(x.size(0), -1).to(device)
                y = y.to(device)
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                with strategy.perturb(population_size=pop_size, epoch=config.warmup_batches + epoch * config.bench_batches + batch_idx) as ctx:
                    logits = ctx.batched_forward(model, x)
                    fitnesses = -F.cross_entropy(logits, y, reduction='none')
                strategy.step(fitnesses)
                
                torch.cuda.synchronize()
                batch_times.append((time.perf_counter() - start) * 1000)
        
        # Evaluate accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.view(x.size(0), -1).to(device)
                y = y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        accuracy = correct / total if total > 0 else 0.0
        model.train()
        
        stats = get_gpu_stats()
        
        result = SupervisedResult(
            method="eggroll_torch",
            dataset=config.dataset,
            pop_size=pop_size,
            rank=config.rank,
            mean_batch_time_ms=np.mean(batch_times),
            std_batch_time_ms=np.std(batch_times),
            samples_per_sec=pop_size / (np.mean(batch_times) / 1000),
            overhead_vs_inference=None,
            final_accuracy=accuracy,
            memory_mb=stats.get("memory_used_mb", 0),
        )
        results.append(result)
        
        if verbose:
            print(f"    Time: {result.mean_batch_time_ms:.3f} ± {result.std_batch_time_ms:.3f} ms")
            print(f"    Throughput: {result.samples_per_sec:,.0f} samples/sec")
            print(f"    Accuracy: {result.final_accuracy*100:.1f}%")
    
    return results


def benchmark_torch_sgd(
    config: ExperimentConfig,
    pop_sizes: List[int],
    verbose: bool = True,
) -> List[SupervisedResult]:
    """Benchmark standard SGD training (reference, not apples-to-apples)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    
    if verbose:
        print(f"\n{'='*60}")
        print("Benchmarking: PyTorch SGD (reference)")
        print(f"{'='*60}")
        print("Note: SGD is not directly comparable to ES (uses gradients)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset, input_dim, num_classes = get_dataset(config.dataset, train=True)
    test_dataset, _, _ = get_dataset(config.dataset, train=False)
    
    results = []
    
    for batch_size in pop_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        
        if verbose:
            print(f"\n  Batch size: {batch_size}")
        
        # Create model
        torch.manual_seed(config.seed)
        layers = []
        dims = [input_dim] + [config.hidden_size] * (config.n_layers - 1) + [num_classes]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        model = nn.Sequential(*layers).to(device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        # Warmup
        batch_iter = iter(dataloader)
        for _ in range(config.warmup_batches):
            try:
                x, y = next(batch_iter)
            except StopIteration:
                batch_iter = iter(dataloader)
                x, y = next(batch_iter)
            
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        
        # Benchmark
        batch_times = []
        batch_iter = iter(dataloader)
        
        for epoch in range(config.num_epochs):
            for batch_idx in range(config.bench_batches):
                try:
                    x, y = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(dataloader)
                    x, y = next(batch_iter)
                
                x = x.view(x.size(0), -1).to(device)
                y = y.to(device)
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                optimizer.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                
                torch.cuda.synchronize()
                batch_times.append((time.perf_counter() - start) * 1000)
        
        # Evaluate accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.view(x.size(0), -1).to(device)
                y = y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        accuracy = correct / total if total > 0 else 0.0
        
        stats = get_gpu_stats()
        
        result = SupervisedResult(
            method="sgd_torch",
            dataset=config.dataset,
            pop_size=batch_size,
            rank=0,
            mean_batch_time_ms=np.mean(batch_times),
            std_batch_time_ms=np.std(batch_times),
            samples_per_sec=batch_size / (np.mean(batch_times) / 1000),
            overhead_vs_inference=None,
            final_accuracy=accuracy,
            memory_mb=stats.get("memory_used_mb", 0),
        )
        results.append(result)
        
        if verbose:
            print(f"    Time: {result.mean_batch_time_ms:.3f} ± {result.std_batch_time_ms:.3f} ms")
            print(f"    Throughput: {result.samples_per_sec:,.0f} samples/sec")
            print(f"    Accuracy: {result.final_accuracy*100:.1f}%")
    
    return results


# =============================================================================
# Analysis & Reporting
# =============================================================================

def compute_overhead(results: List[SupervisedResult]) -> List[SupervisedResult]:
    """Compute overhead vs inference baseline."""
    # Build lookup by (dataset, pop_size)
    inference_baselines = {}
    for r in results:
        if r.method == "inference_torch":
            key = (r.dataset, r.pop_size)
            inference_baselines[key] = r.mean_batch_time_ms
    
    # Compute overhead
    for r in results:
        key = (r.dataset, r.pop_size)
        if key in inference_baselines and r.method != "inference_torch":
            baseline = inference_baselines[key]
            r.overhead_vs_inference = (r.mean_batch_time_ms - baseline) / baseline
    
    return results


def generate_summary(results: List[SupervisedResult]) -> Dict[str, Any]:
    """Generate summary statistics."""
    summary = {
        "by_method": {},
        "overhead_analysis": {},
    }
    
    # Group by method
    methods = set(r.method for r in results)
    for method in methods:
        method_results = [r for r in results if r.method == method]
        summary["by_method"][method] = {
            "avg_throughput": np.mean([r.samples_per_sec for r in method_results]),
            "avg_batch_time_ms": np.mean([r.mean_batch_time_ms for r in method_results]),
            "avg_accuracy": np.mean([r.final_accuracy for r in method_results]),
        }
    
    # EGGROLL overhead
    eggroll_results = [r for r in results if r.method == "eggroll_torch" and r.overhead_vs_inference is not None]
    if eggroll_results:
        avg_overhead = np.mean([r.overhead_vs_inference for r in eggroll_results])
        summary["overhead_analysis"]["eggroll_avg_overhead_pct"] = avg_overhead * 100
        summary["overhead_analysis"]["meets_paper_target"] = avg_overhead <= 0.45  # ≤45% overhead
    
    # SGD comparison
    sgd_results = [r for r in results if r.method == "sgd_torch" and r.overhead_vs_inference is not None]
    if sgd_results:
        avg_overhead = np.mean([r.overhead_vs_inference for r in sgd_results])
        summary["overhead_analysis"]["sgd_avg_overhead_pct"] = avg_overhead * 100
    
    return summary


def print_results_table(results: List[SupervisedResult], verbose: bool = True):
    """Print results in a formatted table."""
    if not results:
        return
    
    print(f"\n{'='*110}")
    print("RESULTS SUMMARY")
    print(f"{'='*110}")
    
    print(f"{'Method':<20} {'Dataset':<10} {'BatchSize':>10} {'Rank':>5} {'Time(ms)':>15} "
          f"{'Samples/s':>12} {'Overhead':>10} {'Accuracy':>10}")
    print("-" * 110)
    
    for r in sorted(results, key=lambda x: (x.method, x.dataset, x.pop_size)):
        overhead_str = f"{r.overhead_vs_inference*100:+.1f}%" if r.overhead_vs_inference is not None else "N/A"
        print(f"{r.method:<20} {r.dataset:<10} {r.pop_size:>10} {r.rank:>5} "
              f"{r.mean_batch_time_ms:>8.3f}±{r.std_batch_time_ms:.3f} "
              f"{r.samples_per_sec:>12,.0f} {overhead_str:>10} {r.final_accuracy*100:>9.1f}%")


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
    parser = argparse.ArgumentParser(description="Experiment C: Supervised Learning Benchmark")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist",
                        help="Dataset to use")
    parser.add_argument("--pop-sizes", type=int, nargs="+", default=None,
                        help="Population/batch sizes to test")
    parser.add_argument("--rank", type=int, default=4,
                        help="EGGROLL rank")
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--bench-batches", type=int, default=20,
                        help="Number of batches to benchmark")
    parser.add_argument("--include-sgd", action="store_true",
                        help="Include SGD benchmark (for reference)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--quiet", action="store_true",
                        help="Less verbose output")
    args = parser.parse_args()
    
    config = ExperimentConfig(
        dataset=args.dataset,
        pop_sizes=args.pop_sizes or DEFAULT_POP_SIZES,
        rank=args.rank,
        num_epochs=args.num_epochs,
        bench_batches=args.bench_batches,
    )
    
    print(f"\n{'#'*60}")
    print("# Experiment C: Supervised Learning Benchmark")
    print(f"{'#'*60}")
    print(f"Dataset: {config.dataset}")
    print(f"Population sizes: {config.pop_sizes}")
    print(f"Rank: {config.rank}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Benchmark batches: {config.bench_batches}")
    
    all_results = []
    verbose = not args.quiet
    
    # Run benchmarks
    all_results.extend(benchmark_torch_inference(config, config.pop_sizes, verbose=verbose))
    all_results.extend(benchmark_torch_eggroll(config, config.pop_sizes, verbose=verbose))
    
    if args.include_sgd:
        all_results.extend(benchmark_torch_sgd(config, config.pop_sizes, verbose=verbose))
    
    # Compute overhead
    all_results = compute_overhead(all_results)
    
    # Generate summary
    summary = generate_summary(all_results)
    
    # Print results
    print_results_table(all_results, verbose=verbose)
    
    # Print analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    if "overhead_analysis" in summary:
        if "eggroll_avg_overhead_pct" in summary["overhead_analysis"]:
            overhead = summary["overhead_analysis"]["eggroll_avg_overhead_pct"]
            meets = summary["overhead_analysis"].get("meets_paper_target", False)
            status = "✅" if meets else "❌"
            print(f"EGGROLL average overhead: {overhead:.1f}%")
            print(f"  {status} Meets paper target (≤45% overhead): {meets}")
        
        if "sgd_avg_overhead_pct" in summary["overhead_analysis"]:
            overhead = summary["overhead_analysis"]["sgd_avg_overhead_pct"]
            print(f"SGD average overhead: {overhead:.1f}% (includes backward pass)")
    
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
        output_path = Path(__file__).parent / f"experiment_c_results_{timestamp}.json"
    else:
        output_path = Path(output_path)
    
    save_results(experiment_results, output_path)
    
    return experiment_results


if __name__ == "__main__":
    main()
