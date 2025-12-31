#!/usr/bin/env python3
"""
GPU Utilization Profiler for EGGROLL implementations.

Profiles both PyTorch and JAX EGGROLL to measure:
1. GPU utilization (SM occupancy)
2. Memory bandwidth utilization
3. Throughput (steps/second, samples/second)
4. Whether Triton is actually being used

This script does NOT modify any implementations - it's purely observational.
"""
import argparse
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
import numpy as np
import gymnasium as gym


@dataclass
class GPUMetrics:
    """Container for GPU metrics collected during profiling."""
    timestamp: float
    gpu_utilization: float  # SM utilization %
    memory_used_mb: float
    memory_total_mb: float
    power_draw_w: float
    temperature_c: float
    sm_clock_mhz: float
    mem_clock_mhz: float


@dataclass
class ProfileResult:
    """Results from profiling a single implementation."""
    method: str
    total_steps: int
    total_samples: int  # steps * population_size
    wall_time_s: float
    steps_per_second: float
    samples_per_second: float
    mean_gpu_utilization: float
    max_gpu_utilization: float
    min_gpu_utilization: float
    std_gpu_utilization: float
    mean_memory_used_mb: float
    max_memory_used_mb: float
    mean_power_w: float
    gpu_metrics: List[Dict] = field(default_factory=list)
    triton_used: bool = False
    torch_compile_used: bool = False
    notes: List[str] = field(default_factory=list)


class GPUMonitor:
    """Background thread that polls nvidia-smi for GPU metrics."""
    
    def __init__(self, poll_interval: float = 0.1, gpu_id: int = 0):
        self.poll_interval = poll_interval
        self.gpu_id = gpu_id
        self.metrics: List[GPUMetrics] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the monitoring thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> List[GPUMetrics]:
        """Stop monitoring and return collected metrics."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.metrics
    
    def _poll_loop(self):
        """Poll nvidia-smi in a loop."""
        while not self._stop_event.is_set():
            try:
                metrics = self._query_nvidia_smi()
                if metrics:
                    self.metrics.append(metrics)
            except Exception as e:
                pass  # Silently ignore polling errors
            time.sleep(self.poll_interval)
    
    def _query_nvidia_smi(self) -> Optional[GPUMetrics]:
        """Query nvidia-smi for current GPU stats."""
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    f'--id={self.gpu_id}',
                    '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                timeout=1.0
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                if len(parts) >= 7:
                    return GPUMetrics(
                        timestamp=time.time(),
                        gpu_utilization=float(parts[0]),
                        memory_used_mb=float(parts[1]),
                        memory_total_mb=float(parts[2]),
                        power_draw_w=float(parts[3]) if parts[3] != '[N/A]' else 0.0,
                        temperature_c=float(parts[4]),
                        sm_clock_mhz=float(parts[5]),
                        mem_clock_mhz=float(parts[6]),
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return None


def check_triton_usage():
    """Check if Triton is available and being used."""
    notes = []
    triton_available = False
    
    try:
        import triton
        triton_available = True
        notes.append(f"Triton version: {triton.__version__}")
    except ImportError:
        notes.append("Triton not installed")
    
    # Check if torch.compile is using triton backend
    try:
        if hasattr(torch, '_dynamo'):
            notes.append("torch.compile (dynamo) available")
    except:
        pass
    
    return triton_available, notes


def check_torch_eggroll_implementation():
    """Analyze what the Torch EGGROLL implementation actually uses."""
    notes = []
    triton_used = False
    torch_compile_used = False
    
    try:
        from hyperscalees.torch.triton_kernels import (
            generate_lowrank_factors_torch,
            batched_perturbed_linear_torch,
        )
        
        # Check the source code for actual triton usage
        import inspect
        source = inspect.getsource(generate_lowrank_factors_torch)
        
        if 'triton' in source.lower() and '@triton.jit' in source:
            triton_used = True
            notes.append("triton_kernels.py uses actual Triton kernels")
        else:
            notes.append("triton_kernels.py uses PyTorch native ops (torch.randn, torch.bmm)")
        
        if 'torch.compile' in source:
            torch_compile_used = True
            notes.append("Uses torch.compile")
        
        # Check for cuda graphs
        if 'cuda.graph' in source.lower():
            notes.append("Uses CUDA graphs")
            
    except Exception as e:
        notes.append(f"Could not inspect torch implementation: {e}")
    
    return triton_used, torch_compile_used, notes


def profile_torch_eggroll(
    num_epochs: int = 100,
    pop_size: int = 2048,
    layer_size: int = 256,
    n_layers: int = 3,
    rank: int = 4,
) -> ProfileResult:
    """Profile the PyTorch EGGROLL implementation."""
    print("\n" + "="*60)
    print("Profiling PyTorch EGGROLL")
    print("="*60)
    
    from hyperscalees.torch import EggrollStrategy
    
    # Check implementation details
    triton_used, torch_compile_used, impl_notes = check_torch_eggroll_implementation()
    
    # Create environment
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # Build network
    layers = []
    in_dim = obs_dim
    for _ in range(n_layers - 1):
        layers.append(torch.nn.Linear(in_dim, layer_size))
        layers.append(torch.nn.Tanh())
        in_dim = layer_size
    layers.append(torch.nn.Linear(in_dim, act_dim))
    
    model = torch.nn.Sequential(*layers).cuda()
    
    # Create strategy (API: setup() is called separately)
    strategy = EggrollStrategy(
        sigma=0.2,
        rank=rank,
        lr=0.1,
        seed=42,
        antithetic=True,
    )
    strategy.setup(model)
    
    # Warmup
    print("Warming up...")
    for warmup_epoch in range(3):
        obs, _ = env.reset()
        obs_batch = torch.tensor(obs, dtype=torch.float32).cuda().unsqueeze(0).expand(pop_size, -1)
        with strategy.perturb(population_size=pop_size, epoch=warmup_epoch) as pop:
            with torch.no_grad():
                _ = pop.batched_forward(model, obs_batch)
    torch.cuda.synchronize()
    
    # Start monitoring
    monitor = GPUMonitor(poll_interval=0.05)
    monitor.start()
    
    # Run profiling
    print(f"Running {num_epochs} epochs with pop_size={pop_size}...")
    total_steps = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        obs, _ = env.reset()
        obs_batch = torch.tensor(obs, dtype=torch.float32).cuda().unsqueeze(0).expand(pop_size, -1)
        
        with strategy.perturb(population_size=pop_size, epoch=epoch) as pop:
            # Run episode (truncated for profiling)
            for step in range(200):
                with torch.no_grad():
                    logits = pop.batched_forward(model, obs_batch)
                    actions = logits.argmax(dim=-1).cpu().numpy()
                
                # Simulate environment step (just for throughput measurement)
                obs, reward, terminated, truncated, _ = env.step(actions[0])
                obs_batch = torch.tensor(obs, dtype=torch.float32).cuda().unsqueeze(0).expand(pop_size, -1)
                total_steps += pop_size
                
                if terminated or truncated:
                    obs, _ = env.reset()
                    obs_batch = torch.tensor(obs, dtype=torch.float32).cuda().unsqueeze(0).expand(pop_size, -1)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Stop monitoring
    metrics = monitor.stop()
    
    wall_time = end_time - start_time
    
    # Compute statistics
    if metrics:
        gpu_utils = [m.gpu_utilization for m in metrics]
        mem_used = [m.memory_used_mb for m in metrics]
        power = [m.power_draw_w for m in metrics]
        
        result = ProfileResult(
            method="EGGROLL (Torch)",
            total_steps=total_steps,
            total_samples=total_steps,  # Each step processes pop_size samples
            wall_time_s=wall_time,
            steps_per_second=total_steps / wall_time,
            samples_per_second=total_steps / wall_time,
            mean_gpu_utilization=np.mean(gpu_utils),
            max_gpu_utilization=np.max(gpu_utils),
            min_gpu_utilization=np.min(gpu_utils),
            std_gpu_utilization=np.std(gpu_utils),
            mean_memory_used_mb=np.mean(mem_used),
            max_memory_used_mb=np.max(mem_used),
            mean_power_w=np.mean([p for p in power if p > 0]) if any(p > 0 for p in power) else 0,
            gpu_metrics=[asdict(m) for m in metrics],
            triton_used=triton_used,
            torch_compile_used=torch_compile_used,
            notes=impl_notes,
        )
    else:
        result = ProfileResult(
            method="EGGROLL (Torch)",
            total_steps=total_steps,
            total_samples=total_steps,
            wall_time_s=wall_time,
            steps_per_second=total_steps / wall_time,
            samples_per_second=total_steps / wall_time,
            mean_gpu_utilization=0,
            max_gpu_utilization=0,
            min_gpu_utilization=0,
            std_gpu_utilization=0,
            mean_memory_used_mb=0,
            max_memory_used_mb=0,
            mean_power_w=0,
            gpu_metrics=[],
            triton_used=triton_used,
            torch_compile_used=torch_compile_used,
            notes=impl_notes + ["WARNING: No GPU metrics collected"],
        )
    
    env.close()
    return result


def profile_jax_eggroll(
    num_epochs: int = 100,
    pop_size: int = 2048,
    layer_size: int = 256,
    n_layers: int = 3,
    rank: int = 4,
) -> ProfileResult:
    """Profile the JAX EGGROLL implementation."""
    print("\n" + "="*60)
    print("Profiling JAX EGGROLL")
    print("="*60)
    
    import jax
    import jax.numpy as jnp
    import optax
    from hyperscalees.noiser.eggroll import EggRoll
    from hyperscalees.models.common import MLP, simple_es_tree_key
    
    notes = []
    notes.append(f"JAX version: {jax.__version__}")
    notes.append(f"JAX devices: {jax.devices()}")
    notes.append("Uses XLA for GPU compilation")
    
    # Create environment for dimensions
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # Initialize model (same as benchmark_cartpole.py)
    key = jax.random.key(42)
    model_key = jax.random.fold_in(key, 0)
    es_key = jax.random.fold_in(key, 1)
    
    frozen_params, params, scan_map, es_map = MLP.rand_init(
        model_key,
        in_dim=obs_dim,
        out_dim=act_dim,
        hidden_dims=[layer_size] * (n_layers - 1),
        use_bias=True,
        activation="pqn",
        dtype="float32",
    )
    
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)
    
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params,
        sigma=0.2,
        lr=0.1,
        solver=optax.sgd,
        rank=rank,
        noise_reuse=0,
    )
    
    # JIT-compile forward function
    def forward_noisy(noiser_params, params, iterinfo, obs):
        """Forward with noise."""
        return MLP.forward(
            EggRoll, frozen_noiser_params, noiser_params, frozen_params,
            params, es_tree_key, iterinfo, obs
        )
    
    jit_forward = jax.jit(jax.vmap(
        lambda n, p, i, x: forward_noisy(n, p, i, x),
        in_axes=(None, None, 0, 0)
    ))
    
    # Warmup - JIT compile
    print("Warming up (JIT compilation)...")
    obs_batch = jnp.zeros((pop_size, obs_dim))
    
    # Create iterinfos for population
    def make_iterinfos(epoch):
        return jnp.stack([
            jnp.array([epoch, member_id], dtype=jnp.int32)
            for member_id in range(pop_size)
        ])
    
    iterinfos = make_iterinfos(0)
    
    # Warmup calls
    for warmup_epoch in range(3):
        iterinfos = make_iterinfos(warmup_epoch)
        _ = jit_forward(noiser_params, params, iterinfos, obs_batch)
    _ = jit_forward(noiser_params, params, iterinfos, obs_batch).block_until_ready()
    
    # Start monitoring
    monitor = GPUMonitor(poll_interval=0.05)
    monitor.start()
    
    # Run profiling
    print(f"Running {num_epochs} epochs with pop_size={pop_size}...")
    total_steps = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        obs, _ = env.reset()
        obs_batch = jnp.tile(jnp.array(obs), (pop_size, 1))
        iterinfos = make_iterinfos(epoch)
        
        # Run episode (truncated for profiling)
        for step in range(200):
            logits = jit_forward(noiser_params, params, iterinfos, obs_batch)
            actions = jnp.argmax(logits, axis=-1)
            
            # Simulate environment step
            obs, reward, terminated, truncated, _ = env.step(int(actions[0]))
            obs_batch = jnp.tile(jnp.array(obs), (pop_size, 1))
            total_steps += pop_size
            
            if terminated or truncated:
                obs, _ = env.reset()
                obs_batch = jnp.tile(jnp.array(obs), (pop_size, 1))
    
    # Ensure all JAX ops are complete
    jax.block_until_ready(logits)
    end_time = time.time()
    
    # Stop monitoring
    metrics = monitor.stop()
    
    wall_time = end_time - start_time
    
    # Compute statistics
    if metrics:
        gpu_utils = [m.gpu_utilization for m in metrics]
        mem_used = [m.memory_used_mb for m in metrics]
        power = [m.power_draw_w for m in metrics]
        
        result = ProfileResult(
            method="EGGROLL (JAX)",
            total_steps=total_steps,
            total_samples=total_steps,
            wall_time_s=wall_time,
            steps_per_second=total_steps / wall_time,
            samples_per_second=total_steps / wall_time,
            mean_gpu_utilization=np.mean(gpu_utils),
            max_gpu_utilization=np.max(gpu_utils),
            min_gpu_utilization=np.min(gpu_utils),
            std_gpu_utilization=np.std(gpu_utils),
            mean_memory_used_mb=np.mean(mem_used),
            max_memory_used_mb=np.max(mem_used),
            mean_power_w=np.mean([p for p in power if p > 0]) if any(p > 0 for p in power) else 0,
            gpu_metrics=[asdict(m) for m in metrics],
            triton_used=False,  # JAX uses XLA, not Triton
            torch_compile_used=False,
            notes=notes,
        )
    else:
        result = ProfileResult(
            method="EGGROLL (JAX)",
            total_steps=total_steps,
            total_samples=total_steps,
            wall_time_s=wall_time,
            steps_per_second=total_steps / wall_time,
            samples_per_second=total_steps / wall_time,
            mean_gpu_utilization=0,
            max_gpu_utilization=0,
            min_gpu_utilization=0,
            std_gpu_utilization=0,
            mean_memory_used_mb=0,
            max_memory_used_mb=0,
            mean_power_w=0,
            gpu_metrics=[],
            triton_used=False,
            torch_compile_used=False,
            notes=notes + ["WARNING: No GPU metrics collected"],
        )
    
    env.close()
    return result


def print_comparison(torch_result: ProfileResult, jax_result: ProfileResult):
    """Print a side-by-side comparison of results."""
    print("\n" + "="*80)
    print("GPU UTILIZATION COMPARISON")
    print("="*80)
    
    print("\n### Implementation Analysis ###")
    print(f"\nPyTorch EGGROLL:")
    print(f"  - Triton kernels used: {torch_result.triton_used}")
    print(f"  - torch.compile used: {torch_result.torch_compile_used}")
    for note in torch_result.notes:
        print(f"  - {note}")
    
    print(f"\nJAX EGGROLL:")
    for note in jax_result.notes:
        print(f"  - {note}")
    
    print("\n### Throughput ###")
    print(f"{'Metric':<30} {'PyTorch':>15} {'JAX':>15} {'Ratio (Torch/JAX)':>20}")
    print("-"*80)
    print(f"{'Total steps':<30} {torch_result.total_steps:>15,} {jax_result.total_steps:>15,}")
    print(f"{'Wall time (s)':<30} {torch_result.wall_time_s:>15.2f} {jax_result.wall_time_s:>15.2f} {torch_result.wall_time_s/jax_result.wall_time_s:>19.2f}x")
    print(f"{'Steps/second':<30} {torch_result.steps_per_second:>15,.0f} {jax_result.steps_per_second:>15,.0f} {torch_result.steps_per_second/jax_result.steps_per_second:>19.2f}x")
    
    print("\n### GPU Utilization ###")
    print(f"{'Metric':<30} {'PyTorch':>15} {'JAX':>15}")
    print("-"*60)
    print(f"{'Mean GPU util (%)':<30} {torch_result.mean_gpu_utilization:>15.1f} {jax_result.mean_gpu_utilization:>15.1f}")
    print(f"{'Max GPU util (%)':<30} {torch_result.max_gpu_utilization:>15.1f} {jax_result.max_gpu_utilization:>15.1f}")
    print(f"{'Min GPU util (%)':<30} {torch_result.min_gpu_utilization:>15.1f} {jax_result.min_gpu_utilization:>15.1f}")
    print(f"{'Std GPU util (%)':<30} {torch_result.std_gpu_utilization:>15.1f} {jax_result.std_gpu_utilization:>15.1f}")
    
    print("\n### Memory & Power ###")
    print(f"{'Mean memory used (MB)':<30} {torch_result.mean_memory_used_mb:>15.0f} {jax_result.mean_memory_used_mb:>15.0f}")
    print(f"{'Max memory used (MB)':<30} {torch_result.max_memory_used_mb:>15.0f} {jax_result.max_memory_used_mb:>15.0f}")
    print(f"{'Mean power draw (W)':<30} {torch_result.mean_power_w:>15.1f} {jax_result.mean_power_w:>15.1f}")
    
    # Efficiency analysis
    print("\n### Efficiency Analysis ###")
    if torch_result.mean_gpu_utilization > 0 and jax_result.mean_gpu_utilization > 0:
        torch_efficiency = torch_result.steps_per_second / torch_result.mean_gpu_utilization
        jax_efficiency = jax_result.steps_per_second / jax_result.mean_gpu_utilization
        print(f"{'Steps per GPU% (efficiency)':<30} {torch_efficiency:>15.1f} {jax_efficiency:>15.1f}")
    
    # Diagnosis
    print("\n### Diagnosis ###")
    if torch_result.mean_gpu_utilization < 50:
        print("⚠️  PyTorch GPU utilization is LOW (<50%)")
        print("   Possible causes:")
        print("   - CPU-bound operations (environment stepping)")
        print("   - Small batch sizes not saturating GPU")
        print("   - Memory transfers between CPU/GPU")
        print("   - Kernel launch overhead")
    
    if not torch_result.triton_used:
        print("\n⚠️  PyTorch implementation is NOT using Triton kernels")
        print("   The 'triton_kernels.py' file uses PyTorch native ops:")
        print("   - torch.randn for noise generation")
        print("   - torch.bmm for batched matrix multiply")
        print("   - F.linear for base linear layer")
        print("\n   This may be fine if these ops are already optimized,")
        print("   but actual Triton kernels could potentially fuse operations.")


def generate_utilization_plot(torch_result: ProfileResult, jax_result: ProfileResult, output_dir: Path):
    """Generate GPU utilization over time plot."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # GPU Utilization over time
        ax = axes[0, 0]
        if torch_result.gpu_metrics:
            times = [m['timestamp'] - torch_result.gpu_metrics[0]['timestamp'] for m in torch_result.gpu_metrics]
            utils = [m['gpu_utilization'] for m in torch_result.gpu_metrics]
            ax.plot(times, utils, label='PyTorch', alpha=0.7)
        if jax_result.gpu_metrics:
            times = [m['timestamp'] - jax_result.gpu_metrics[0]['timestamp'] for m in jax_result.gpu_metrics]
            utils = [m['gpu_utilization'] for m in jax_result.gpu_metrics]
            ax.plot(times, utils, label='JAX', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('GPU Utilization (%)')
        ax.set_title('GPU Utilization Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Memory usage over time
        ax = axes[0, 1]
        if torch_result.gpu_metrics:
            times = [m['timestamp'] - torch_result.gpu_metrics[0]['timestamp'] for m in torch_result.gpu_metrics]
            mem = [m['memory_used_mb'] for m in torch_result.gpu_metrics]
            ax.plot(times, mem, label='PyTorch', alpha=0.7)
        if jax_result.gpu_metrics:
            times = [m['timestamp'] - jax_result.gpu_metrics[0]['timestamp'] for m in jax_result.gpu_metrics]
            mem = [m['memory_used_mb'] for m in jax_result.gpu_metrics]
            ax.plot(times, mem, label='JAX', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Memory Used (MB)')
        ax.set_title('GPU Memory Usage Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Utilization histogram
        ax = axes[1, 0]
        if torch_result.gpu_metrics:
            utils = [m['gpu_utilization'] for m in torch_result.gpu_metrics]
            ax.hist(utils, bins=20, alpha=0.5, label=f'PyTorch (mean={np.mean(utils):.1f}%)')
        if jax_result.gpu_metrics:
            utils = [m['gpu_utilization'] for m in jax_result.gpu_metrics]
            ax.hist(utils, bins=20, alpha=0.5, label=f'JAX (mean={np.mean(utils):.1f}%)')
        ax.set_xlabel('GPU Utilization (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('GPU Utilization Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Throughput comparison
        ax = axes[1, 1]
        methods = ['PyTorch', 'JAX']
        throughputs = [torch_result.steps_per_second, jax_result.steps_per_second]
        colors = ['#1f77b4', '#ff7f0e']
        bars = ax.bar(methods, throughputs, color=colors)
        ax.set_ylabel('Steps per Second')
        ax.set_title('Throughput Comparison')
        for bar, val in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                   f'{val:,.0f}', ha='center', va='bottom')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gpu_utilization_profile.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved plot to {output_dir / 'gpu_utilization_profile.png'}")
        
    except ImportError:
        print("matplotlib not available, skipping plot generation")


def main():
    parser = argparse.ArgumentParser(description="Profile GPU utilization for EGGROLL implementations")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs to run")
    parser.add_argument("--pop-size", type=int, default=2048, help="Population size")
    parser.add_argument("--layer-size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--rank", type=int, default=4, help="Low-rank dimension")
    parser.add_argument("--output-dir", type=str, default="benchmarks", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("EGGROLL GPU Utilization Profiler")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Population size: {args.pop_size}")
    print(f"  Layer size: {args.layer_size}")
    print(f"  N layers: {args.n_layers}")
    print(f"  Rank: {args.rank}")
    
    # Check Triton availability
    triton_available, triton_notes = check_triton_usage()
    print(f"\nTriton status:")
    for note in triton_notes:
        print(f"  {note}")
    
    # Profile both implementations
    torch_result = profile_torch_eggroll(
        num_epochs=args.num_epochs,
        pop_size=args.pop_size,
        layer_size=args.layer_size,
        n_layers=args.n_layers,
        rank=args.rank,
    )
    
    # Clear CUDA cache before JAX
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(1)  # Give GPU a moment to reset
    
    jax_result = profile_jax_eggroll(
        num_epochs=args.num_epochs,
        pop_size=args.pop_size,
        layer_size=args.layer_size,
        n_layers=args.n_layers,
        rank=args.rank,
    )
    
    # Print comparison
    print_comparison(torch_result, jax_result)
    
    # Generate plots
    generate_utilization_plot(torch_result, jax_result, output_dir)
    
    # Save results
    results = {
        "torch": {k: v for k, v in asdict(torch_result).items() if k != 'gpu_metrics'},
        "jax": {k: v for k, v in asdict(jax_result).items() if k != 'gpu_metrics'},
        "config": vars(args),
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"gpu_profile_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")


if __name__ == "__main__":
    main()
