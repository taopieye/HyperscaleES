"""
Functional EGGROLL - Single file implementation.

Experiments:
1. CartPole-v1 (RL) - Torch and JAX
2. MNIST MLP (Supervised) - Torch and JAX
3. MNIST CNN (Supervised) - Torch and JAX
4. Hyperscale test - find max population size before OOM

Optimization:
- Uses torch.compile by default for JAX-parity performance (~0.99x JAX speed)
- TensorFloat32 precision can be enabled in benchmarks for faster matmul

Usage:
    python -m hyperscalees.torch.fncl.eggroll_fncl
    python -m hyperscalees.torch.fncl.eggroll_fncl --experiment mnist
    python -m hyperscalees.torch.fncl.eggroll_fncl --experiment mnist_cnn
    python -m hyperscalees.torch.fncl.eggroll_fncl --experiment hyperscale
"""

import torch
import gymnasium as gym
import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from rich.console import Console
from rich.table import Table

console = Console()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EggrollConfig:
    """Configuration for EGGROLL training.
    
    This is a general-purpose config for the EGGROLL framework.
    Experiments should instantiate with appropriate values for their domain.
    
    NOTE: Torch EGGROLL is GPU-only. All tensors are created on CUDA.
    """
    # ES hyperparameters
    population_size: int = 2048
    rank: int = 4
    sigma: float = 0.1
    lr: float = 0.1
    lr_decay: float = 1.0        # 1.0 = no decay
    sigma_decay: float = 0.999
    max_epochs: int = 100
    
    # Training
    batch_size: int = 256
    seed: int = 42
    
    # Hardware (GPU-only, dtype configurable)
    dtype: torch.dtype = field(default=torch.float32)

# NOTE: We do NOT set TF32 at module level because it can cause numerical
# instability in RL training (chaotic dynamics + early stopping).
# Enable TF32 only in benchmark functions where speed matters more than precision.


# =============================================================================
# GPU Monitoring Utilities
# =============================================================================

def get_gpu_stats():
    """Get current GPU memory and utilization stats (PyTorch)."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    torch.cuda.synchronize()
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return {
        "available": True,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
        "total_gb": total,
        "utilization_pct": (allocated / total) * 100,
    }


def get_gpu_stats_jax():
    """Get current GPU memory stats for JAX."""
    try:
        import jax
        devices = jax.local_devices()
        if not devices or devices[0].platform != 'gpu':
            return {"available": False}
        
        device = devices[0]
        # JAX memory stats
        stats = device.memory_stats()
        
        # Get total memory via nvidia-smi as backup
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            total_mb, used_mb = map(float, result.stdout.strip().split(','))
            total_gb = total_mb / 1024
            used_gb = used_mb / 1024
        else:
            total_gb = 0
            used_gb = 0
        
        # JAX stats are in bytes
        bytes_in_use = stats.get('bytes_in_use', 0) / 1e9
        peak_bytes = stats.get('peak_bytes_in_use', 0) / 1e9
        
        return {
            "available": True,
            "jax_in_use_gb": bytes_in_use,
            "jax_peak_gb": peak_bytes,
            "nvidia_used_gb": used_gb,
            "nvidia_total_gb": total_gb,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def get_gpu_stats_nvidia():
    """Get GPU memory stats via nvidia-smi (works for both JAX and PyTorch)."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            total_mb, used_mb, free_mb = map(float, result.stdout.strip().split(','))
            return {
                "available": True,
                "total_gb": total_mb / 1024,
                "used_gb": used_mb / 1024,
                "free_gb": free_mb / 1024,
                "utilization_pct": (used_mb / total_mb) * 100,
            }
    except Exception as e:
        pass
    return {"available": False}


def print_gpu_stats(prefix=""):
    """Print current GPU stats."""
    stats = get_gpu_stats()
    if not stats["available"]:
        console.print(f"{prefix}[yellow]GPU not available[/yellow]")
        return
    
    console.print(
        f"{prefix}[cyan]GPU:[/cyan] "
        f"{stats['allocated_gb']:.2f}/{stats['total_gb']:.1f}GB allocated "
        f"({stats['utilization_pct']:.1f}%), "
        f"peak={stats['max_allocated_gb']:.2f}GB"
    )


def print_gpu_stats_jax(prefix=""):
    """Print current GPU stats for JAX (uses nvidia-smi)."""
    stats = get_gpu_stats_nvidia()
    if not stats["available"]:
        console.print(f"{prefix}[yellow]GPU stats not available[/yellow]")
        return
    
    console.print(
        f"{prefix}[cyan]GPU (nvidia-smi):[/cyan] "
        f"{stats['used_gb']:.2f}/{stats['total_gb']:.1f}GB used "
        f"({stats['utilization_pct']:.1f}%)"
    )


def reset_gpu_stats():
    """Reset GPU peak memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_mnist_flat(dtype: torch.dtype = torch.float32):
    """Load MNIST dataset with flattened images for MLP.
    
    Returns:
        train_imgs: (60000, 784) - flattened training images on CUDA
        train_labels: (60000,) - training labels on CUDA
        test_imgs: (10000, 784) - flattened test images on CUDA
        test_labels: (10000,) - test labels on CUDA
    """
    from torchvision import datasets, transforms
    
    transform = transforms.ToTensor()
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    train_imgs = torch.stack([train_data[i][0].flatten() for i in range(len(train_data))])
    train_labels = torch.tensor([train_data[i][1] for i in range(len(train_data))])
    test_imgs = torch.stack([test_data[i][0].flatten() for i in range(len(test_data))])
    test_labels = torch.tensor([test_data[i][1] for i in range(len(test_data))])
    
    return (
        train_imgs.to("cuda", dtype),
        train_labels.to("cuda"),
        test_imgs.to("cuda", dtype),
        test_labels.to("cuda"),
    )


def load_mnist_2d(dtype: torch.dtype = torch.float32):
    """Load MNIST dataset with 2D images for CNN.
    
    Returns:
        train_imgs: (60000, 1, 28, 28) - training images on CUDA
        train_labels: (60000,) - training labels on CUDA
        test_imgs: (10000, 1, 28, 28) - test images on CUDA
        test_labels: (10000,) - test labels on CUDA
    """
    from torchvision import datasets, transforms
    
    transform = transforms.ToTensor()
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    train_imgs = torch.stack([train_data[i][0] for i in range(len(train_data))])
    train_labels = torch.tensor([train_data[i][1] for i in range(len(train_data))])
    test_imgs = torch.stack([test_data[i][0] for i in range(len(test_data))])
    test_labels = torch.tensor([test_data[i][1] for i in range(len(test_data))])
    
    return (
        train_imgs.to("cuda", dtype),
        train_labels.to("cuda"),
        test_imgs.to("cuda", dtype),
        test_labels.to("cuda"),
    )


# =============================================================================
# Parameter Initialization Utilities
# =============================================================================

def init_mlp_params(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dtype: torch.dtype = torch.float32,
    scale: float = None,
):
    """Initialize 2-layer MLP parameters on CUDA.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        dtype: Data type for tensors
        scale: Initialization scale (default: Xavier-like 1/sqrt(fan_in))
        
    Returns:
        W1, b1, W2, b2: Layer weights and biases on CUDA
    """
    W1 = torch.randn(hidden_dim, input_dim, device="cuda", dtype=dtype)
    W1 = W1 / math.sqrt(input_dim) if scale is None else W1 * scale
    b1 = torch.zeros(hidden_dim, device="cuda", dtype=dtype)
    
    W2 = torch.randn(output_dim, hidden_dim, device="cuda", dtype=dtype)
    W2 = W2 / math.sqrt(hidden_dim) if scale is None else W2 * scale
    b2 = torch.zeros(output_dim, device="cuda", dtype=dtype)
    
    return W1, b1, W2, b2


def count_params(*tensors) -> int:
    """Count total number of parameters in tensors."""
    return sum(t.numel() for t in tensors)


# =============================================================================
# EGGROLL Core Utilities (reusable across all experiments)
# =============================================================================

def generate_lowrank_perturbations(
    population_size: int,
    out_dim: int,
    in_dim: int,
    rank: int,
    sigma: float,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
):
    """
    Generate antithetic low-rank perturbations for EGGROLL on CUDA.
    
    Returns:
        A_scaled: (population_size, out_dim, rank) - scaled A factors
        A: (population_size, out_dim, rank) - unscaled A factors
        B: (population_size, in_dim, rank) - B factors (unscaled)
        
    The perturbation to weight W is: A_scaled @ B.T (low-rank approximation)
    Antithetic: first half and second half are negatives of each other.
    """
    half_pop = population_size // 2
    scale = sigma / (rank ** 0.5)
    
    # Generate positive half
    A_pos = torch.randn(half_pop, out_dim, rank, device="cuda", dtype=dtype, generator=generator)
    B_pos = torch.randn(half_pop, in_dim, rank, device="cuda", dtype=dtype, generator=generator)
    
    # Antithetic pairs: negate A, keep B
    A = torch.cat([A_pos, -A_pos], dim=0)
    B = torch.cat([B_pos, B_pos], dim=0)
    
    # Scale A by sigma/sqrt(rank)
    A_scaled = A * scale
    
    return A_scaled, A, B


def apply_lowrank_perturbation(x, B, A_scaled):
    """
    Apply low-rank perturbation to batched input.
    
    Args:
        x: (population, batch, in_dim) - input tensor
        B: (population, in_dim, rank) - B factors
        A_scaled: (population, out_dim, rank) - scaled A factors
        
    Returns:
        perturbation: (population, batch, out_dim)
    """
    return torch.einsum('pbi,pir,pjr->pbj', x, B, A_scaled)


def apply_lowrank_perturbation_bmm(x, B, A_scaled):
    """
    Apply low-rank perturbation using bmm (potentially faster than einsum).
    
    Args:
        x: (population, batch, in_dim) - input tensor
        B: (population, in_dim, rank) - B factors
        A_scaled: (population, out_dim, rank) - scaled A factors
        
    Returns:
        perturbation: (population, batch, out_dim)
    """
    # x @ B: (pop, batch, in_dim) @ (pop, in_dim, rank) -> (pop, batch, rank)
    xB = torch.bmm(x, B)
    # xB @ A.T: (pop, batch, rank) @ (pop, rank, out_dim) -> (pop, batch, out_dim)
    return torch.bmm(xB, A_scaled.transpose(-1, -2))


def fused_matmul_with_lowrank(x, W, B, A_scaled):
    """
    Fused base matmul + low-rank perturbation.
    
    Computes: x @ W.T + x @ B @ A.T
    
    This can potentially be optimized by torch.compile to use a single kernel.
    """
    base = x @ W.T
    pert = torch.einsum('pbi,pir,pjr->pbj', x, B, A_scaled)
    return base + pert


# =============================================================================
# Compiled Core Functions (torch.compile for JAX-parity performance)
# =============================================================================

@torch.compile
@torch.compile
def compiled_batched_inference(x, W1, b1, W2, b2):
    """
    Compiled pure inference (no perturbations).
    """
    h = torch.relu(x @ W1.T + b1)
    return h @ W2.T + b2


@torch.compile  
def compiled_eggroll_step(
    x, W1, b1, W2, b2, 
    A1_scaled, B1, 
    A2_scaled, B2,
    sqrt_N
):
    """
    Complete compiled EGGROLL step: forward + gradient computation.
    This is the main optimization - fusing all operations into compiled graph.
    
    NOTE: Uses A_scaled (not unscaled A) for gradient computation to match JAX.
    """
    # Forward with perturbations
    base1 = x @ W1.T + b1
    pert1 = torch.einsum('pbi,pir,pjr->pbj', x, B1, A1_scaled)
    h = torch.relu(base1 + pert1)
    
    base2 = h @ W2.T + b2
    pert2 = torch.einsum('pbi,pir,pjr->pbj', h, B2, A2_scaled)
    logits = base2 + pert2
    
    # Compute fitness
    fitnesses = logits.mean(dim=(1, 2))
    fitnesses = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
    
    # ES gradient - use A_scaled (includes sigma/sqrt(rank)) for proper magnitude
    f = fitnesses[:, None, None]
    grad1 = torch.einsum('nir,njr->ij', f * A1_scaled, B1) / sqrt_N
    grad2 = torch.einsum('nir,njr->ij', f * A2_scaled, B2) / sqrt_N
    
    return logits, grad1, grad2


def warmup_compiled_functions(pop_size, batch_size, in_dim, hidden_dim, out_dim, rank, dtype=torch.float32):
    """
    Warm up compiled functions to avoid compilation overhead during benchmarking.
    Call this once before timing to ensure fair comparison.
    """
    console.print("  [dim]Warming up torch.compile...[/dim]")
    
    W1 = torch.randn(hidden_dim, in_dim, device="cuda", dtype=dtype)
    W2 = torch.randn(out_dim, hidden_dim, device="cuda", dtype=dtype)
    b1 = torch.zeros(hidden_dim, device="cuda", dtype=dtype)
    b2 = torch.zeros(out_dim, device="cuda", dtype=dtype)
    x = torch.randn(pop_size, batch_size, in_dim, device="cuda", dtype=dtype)
    
    gen = torch.Generator(device="cuda").manual_seed(0)
    A1_scaled, A1, B1 = generate_lowrank_perturbations(pop_size, hidden_dim, in_dim, rank, 0.1, gen, dtype)
    A2_scaled, A2, B2 = generate_lowrank_perturbations(pop_size, out_dim, hidden_dim, rank, 0.1, gen, dtype)
    sqrt_N = math.sqrt(pop_size)
    
    # Trigger compilation (first few calls compile the graph)
    for _ in range(3):
        _ = compiled_batched_inference(x, W1, b1, W2, b2)
        _ = compiled_eggroll_step(x, W1, b1, W2, b2, A1_scaled, B1, A2_scaled, B2, sqrt_N)
    torch.cuda.synchronize()
    
    del W1, W2, b1, b2, x, A1_scaled, A1, B1, A2_scaled, A2, B2
    torch.cuda.empty_cache()


def compute_es_gradient(fitnesses, A_scaled, B, population_size):
    """
    Compute ES gradient from fitnesses and perturbation factors.
    
    Args:
        fitnesses: (population,) - normalized fitness scores
        A_scaled: (population, out_dim, rank) - scaled A factors (includes sigma/sqrt(rank))
        B: (population, in_dim, rank) - B factors
        population_size: int
        
    Returns:
        gradient: (out_dim, in_dim) - gradient estimate for weight matrix
    """
    sqrt_N = math.sqrt(population_size)
    f = fitnesses[:, None, None]
    return torch.einsum('nir,njr->ij', f * A_scaled, B) / sqrt_N


def normalize_fitnesses(fitnesses, eps=1e-8):
    """
    Normalize fitness scores to zero mean, unit variance.
    
    Args:
        fitnesses: (population,) - raw fitness scores
        eps: small constant for numerical stability
        
    Returns:
        normalized: (population,) - normalized fitness scores
    """
    return (fitnesses - fitnesses.mean()) / (fitnesses.std() + eps)


def compute_classification_fitness(logits, labels):
    """
    Compute fitness for classification (negative cross-entropy loss).
    
    Args:
        logits: (population, batch, num_classes) - model outputs
        labels: (batch,) - ground truth labels
        
    Returns:
        fitnesses: (population,) - fitness scores (higher is better)
    """
    population_size = logits.shape[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    labels_exp = labels.unsqueeze(0).expand(population_size, -1)
    nll = -log_probs.gather(dim=-1, index=labels_exp.unsqueeze(-1)).squeeze(-1)
    losses = nll.mean(dim=-1)  # Average over batch
    return -losses  # Negative loss = fitness (higher is better)


def mlp_forward_perturbed(
    x,
    W1, b1, W2, b2,
    A1_scaled, B1,
    A2_scaled, B2,
    hidden_dim,
    use_layernorm=True,
):
    """
    Forward pass through 2-layer MLP with low-rank perturbations.
    
    Args:
        x: (population, batch, input_dim) - input
        W1, b1: layer 1 weights and biases
        W2, b2: layer 2 weights and biases
        A1_scaled, B1: perturbation factors for W1
        A2_scaled, B2: perturbation factors for W2
        hidden_dim: hidden layer dimension (for layernorm)
        use_layernorm: whether to apply layer normalization
        
    Returns:
        logits: (population, batch, output_dim)
    """
    # Layer 1
    base1 = x @ W1.T + b1
    pert1 = apply_lowrank_perturbation(x, B1, A1_scaled)
    pre_act = base1 + pert1
    
    if use_layernorm:
        h = torch.relu(torch.nn.functional.layer_norm(pre_act, (hidden_dim,)))
    else:
        h = torch.relu(pre_act)
    
    # Layer 2
    base2 = h @ W2.T + b2
    pert2 = apply_lowrank_perturbation(h, B2, A2_scaled)
    logits = base2 + pert2
    
    return logits


def rl_forward_perturbed(
    obs, W1, b1, W2, b2, A1_scaled, B1, A2_scaled, B2, hidden_dim, use_layernorm=True
):
    """
    Forward pass through 2-layer MLP with low-rank perturbations for RL.
    
    This is for RL where each population member has ONE observation (not a batch).
    
    Args:
        obs: (population, input_dim) - one observation per population member
        W1, b1: layer 1 weights and biases
        W2, b2: layer 2 weights and biases
        A1_scaled, B1: layer 1 perturbation factors
        A2_scaled, B2: layer 2 perturbation factors
        hidden_dim: hidden layer dimension
        use_layernorm: whether to apply layer normalization
        
    Returns:
        logits: (population, output_dim)
    """
    # Layer 1: perturbed linear + layernorm + relu
    base1 = obs @ W1.T + b1  # (pop, hidden)
    # For (pop, in_dim) input, we use different einsum pattern than batched
    pert1 = torch.einsum('pi,pir,pjr->pj', obs, B1, A1_scaled)  # (pop, hidden)
    pre_act = base1 + pert1
    
    if use_layernorm:
        h = torch.relu(torch.nn.functional.layer_norm(pre_act, (hidden_dim,)))
    else:
        h = torch.relu(pre_act)
    
    # Layer 2: perturbed linear (no activation)
    base2 = h @ W2.T + b2  # (pop, output)
    pert2 = torch.einsum('pi,pir,pjr->pj', h, B2, A2_scaled)  # (pop, output)
    logits = base2 + pert2
    
    return logits


def mlp_forward_clean(x, W1, b1, W2, b2, hidden_dim, use_layernorm=True):
    """
    Forward pass through 2-layer MLP without perturbations (for eval).
    
    Args:
        x: (batch, input_dim) - input
        W1, b1: layer 1 weights and biases
        W2, b2: layer 2 weights and biases
        hidden_dim: hidden layer dimension
        use_layernorm: whether to apply layer normalization
        
    Returns:
        logits: (batch, output_dim)
    """
    pre_act = x @ W1.T + b1
    if use_layernorm:
        h = torch.relu(torch.nn.functional.layer_norm(pre_act, (hidden_dim,)))
    else:
        h = torch.relu(pre_act)
    return h @ W2.T + b2


def main(config: EggrollConfig = None):
    """CartPole-v1 with Torch EGGROLL."""
    # === Experiment-specific config (from EGGROLL paper Table 3) ===
    if config is None:
        config = EggrollConfig(
            population_size=2048,
            rank=4,
            sigma=0.2,
            lr=0.1,
            lr_decay=0.9995,
            sigma_decay=0.999,
            max_epochs=300,
        )
    
    # Network architecture
    input_dim = 4       # CartPole observation
    hidden_dim = 256    # Paper: layer_size=256
    output_dim = 2      # CartPole actions
    use_layernorm = True
    
    torch.manual_seed(config.seed)
    reset_gpu_stats()
    
    # === Initialize params ===
    W1, b1, W2, b2 = init_mlp_params(
        input_dim, hidden_dim, output_dim,
        config.dtype, scale=0.1
    )
    n_params = count_params(W1, b1, W2, b2)
    
    console.print(f"[bold]Functional EGGROLL - CartPole-v1[/bold]")
    console.print(f"Population: {config.population_size}, Rank: {config.rank}, Sigma: {config.sigma}, LR: {config.lr}")
    console.print(f"Network: {input_dim} -> {hidden_dim} -> {output_dim} ({n_params:,} params)")
    print_gpu_stats("Init ")
    console.print()
    
    # === Create vectorized envs ===
    envs = gym.make_vec("CartPole-v1", num_envs=config.population_size)
    
    # === Compile functions for speed (matching JAX jit) ===
    compiled_rl_forward = torch.compile(rl_forward_perturbed)
    compiled_es_gradient = torch.compile(compute_es_gradient)
    
    total_steps = 0
    start_time = time.perf_counter()
    current_lr = config.lr
    current_sigma = config.sigma
    
    for epoch in range(config.max_epochs):
        epoch_start = time.perf_counter()
        
        # === Generate low-rank perturbations ===
        gen = torch.Generator(device="cuda")
        gen.manual_seed(config.seed + epoch * 1000)
        
        A1_scaled, A1, B1 = generate_lowrank_perturbations(
            config.population_size, hidden_dim, input_dim, config.rank, 
            current_sigma, gen, config.dtype
        )
        A2_scaled, A2, B2 = generate_lowrank_perturbations(
            config.population_size, output_dim, hidden_dim, config.rank, 
            current_sigma, gen, config.dtype
        )
        
        # === Run episodes ===
        obs, _ = envs.reset(seed=epoch)
        episode_returns = torch.zeros(config.population_size, device="cuda", dtype=config.dtype)
        dones = torch.zeros(config.population_size, dtype=torch.bool, device="cuda")
        
        steps_this_epoch = 0
        max_steps = 500
        
        for step in range(max_steps):
            obs_t = torch.as_tensor(obs, device="cuda", dtype=config.dtype)
            
            # === Forward with perturbations ===
            logits = compiled_rl_forward(
                obs_t, W1, b1, W2, b2, A1_scaled, B1, A2_scaled, B2, 
                hidden_dim, use_layernorm=use_layernorm
            )
            
            actions = logits.argmax(dim=-1).cpu().numpy()
            obs, rewards, terminated, truncated, _ = envs.step(actions)
            
            # Count steps for non-done envs
            active = ~dones
            steps_this_epoch += active.sum().item()
            
            rewards_t = torch.as_tensor(rewards, device="cuda", dtype=config.dtype)
            done_t = torch.as_tensor(terminated | truncated, device="cuda")
            
            episode_returns += rewards_t * active.float()
            dones = dones | done_t
            
            if dones.all():
                break
        
        total_steps += steps_this_epoch
        
        # === ES Update ===
        fitnesses = normalize_fitnesses(episode_returns)
        
        # Use A_scaled (includes sigma/sqrt(rank)) for proper gradient magnitude
        grad_W1 = compiled_es_gradient(fitnesses, A1_scaled, B1, config.population_size)
        grad_W2 = compiled_es_gradient(fitnesses, A2_scaled, B2, config.population_size)
        
        W1 = W1 + current_lr * grad_W1
        W2 = W2 + current_lr * grad_W2
        
        # Decay lr and sigma
        current_lr *= config.lr_decay
        current_sigma *= config.sigma_decay
        
        # === Logging ===
        epoch_time = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - start_time
        steps_per_sec = total_steps / elapsed
        
        mean_ret = episode_returns.mean().item()
        max_ret = episode_returns.max().item()
        min_ret = episode_returns.min().item()
        
        if epoch % 10 == 0 or mean_ret >= 475:
            console.print(
                f"Epoch {epoch:3d} | "
                f"mean={mean_ret:6.1f} max={max_ret:6.1f} min={min_ret:6.1f} | "
                f"{steps_per_sec:,.0f} steps/s | "
                f"epoch={epoch_time:.2f}s"
            )
        
        if mean_ret >= 475:
            console.print(f"[bold green]Solved at epoch {epoch}![/bold green]")
            break
    
    envs.close()
    
    total_time = time.perf_counter() - start_time
    console.print()
    console.print(f"[bold]Total: {total_steps:,} steps in {total_time:.1f}s ({total_steps/total_time:,.0f} steps/s)[/bold]")
    
    # === Final evaluation ===
    console.print("\n[bold]Final evaluation (base policy, no perturbations):[/bold]")
    eval_env = gym.make("CartPole-v1")
    
    returns = []
    for ep in range(10):
        obs, _ = eval_env.reset(seed=ep + 1000)
        total_reward = 0
        done = False
        
        while not done:
            obs_t = torch.as_tensor(obs, device="cuda", dtype=config.dtype)
            logits = mlp_forward_clean(obs_t, W1, b1, W2, b2, hidden_dim, use_layernorm=use_layernorm)
            action = logits.argmax().item()
            
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        returns.append(total_reward)
    
    eval_env.close()
    
    table = Table(title="Evaluation Results")
    table.add_column("Episode", style="cyan")
    table.add_column("Return", style="green")
    
    for i, r in enumerate(returns):
        table.add_row(str(i), str(int(r)))
    
    table.add_row("[bold]Mean[/bold]", f"[bold]{sum(returns)/len(returns):.1f}[/bold]")
    console.print(table)
    print_gpu_stats("Final ")


def main_jax():
    """JAX EGGROLL benchmark using the actual reference implementation."""
    import jax
    import jax.numpy as jnp
    import optax
    import numpy as np
    
    from hyperscalees.noiser.eggroll import EggRoll
    from hyperscalees.models.common import MLP, simple_es_tree_key
    
    # === Config (same as torch) ===
    population_size = 2048
    rank = 4
    sigma = 0.2
    lr = 0.1
    lr_decay = 0.9995
    sigma_decay = 0.999
    max_epochs = 300
    seed = 42
    hidden_dim = 256
    
    console.print(f"\n[bold]JAX EGGROLL (reference implementation) - CartPole-v1[/bold]")
    console.print(f"Population: {population_size}, Rank: {rank}, Sigma: {sigma}, LR: {lr}")
    console.print()
    
    # === Create vectorized envs ===
    envs = gym.make_vec("CartPole-v1", num_envs=population_size)
    obs_dim = 4
    action_dim = 2
    
    # === Initialize model using JAX EGGROLL infrastructure ===
    key = jax.random.key(seed)
    model_key = jax.random.fold_in(key, 0)
    es_key = jax.random.fold_in(key, 1)
    
    frozen_params, params, scan_map, es_map = MLP.rand_init(
        model_key,
        in_dim=obs_dim,
        out_dim=action_dim,
        hidden_dims=[hidden_dim],  # Single hidden layer to match torch
        use_bias=True,
        activation="pqn",  # Available: relu, silu, pqn
        dtype="float32",
    )
    
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)
    
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params,
        sigma=sigma,
        lr=lr,
        solver=optax.sgd,
        rank=rank,
        noise_reuse=0,
    )
    
    n_params = sum(p.size for p in jax.tree.leaves(params))
    console.print(f"Network params: {n_params:,}")
    console.print()
    
    # === JIT compile forward functions ===
    def forward_noisy(noiser_params, params, iterinfo, obs):
        return MLP.forward(
            EggRoll, frozen_noiser_params, noiser_params, frozen_params,
            params, es_tree_key, iterinfo, obs
        )
    
    jit_forward = jax.jit(jax.vmap(
        lambda n, p, i, x: forward_noisy(n, p, i, x),
        in_axes=(None, None, 0, 0)
    ))
    
    @jax.jit
    def do_update(noiser_params, params, fitnesses, iterinfos):
        return EggRoll.do_updates(
            frozen_noiser_params, noiser_params, params,
            es_tree_key, fitnesses, iterinfos, es_map
        )
    
    # === Training loop ===
    total_steps = 0
    start_time = time.perf_counter()
    current_sigma = sigma
    
    for epoch in range(max_epochs):
        epoch_start = time.perf_counter()
        
        obs, _ = envs.reset(seed=epoch)
        episode_returns = np.zeros(population_size)
        dones = np.zeros(population_size, dtype=bool)
        
        # Create iterinfo for this epoch (epoch, thread_ids)
        iterinfo = (jnp.full(population_size, epoch, dtype=jnp.int32), jnp.arange(population_size))
        
        steps_this_epoch = 0
        max_steps = 500
        
        for step in range(max_steps):
            obs_jax = jnp.array(obs)
            action_logits = jit_forward(noiser_params, params, iterinfo, obs_jax)
            actions = np.array(jnp.argmax(action_logits, axis=-1))
            
            obs, rewards, terminated, truncated, _ = envs.step(actions)
            
            active = ~dones
            steps_this_epoch += int(active.sum())
            
            episode_returns += rewards * active
            dones = dones | terminated | truncated
            
            if dones.all():
                break
        
        total_steps += steps_this_epoch
        
        # Convert fitnesses using EGGROLL's method
        raw_scores = jnp.array(episode_returns)
        fitnesses = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
        
        # Update using EGGROLL's update
        noiser_params, params = do_update(noiser_params, params, fitnesses, iterinfo)
        
        # Decay sigma
        current_sigma *= sigma_decay
        noiser_params = {**noiser_params, "sigma": current_sigma}
        
        # === Logging ===
        epoch_time = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - start_time
        steps_per_sec = total_steps / elapsed
        
        mean_ret = float(episode_returns.mean())
        max_ret = float(episode_returns.max())
        min_ret = float(episode_returns.min())
        
        if epoch % 10 == 0 or mean_ret >= 475:
            console.print(
                f"Epoch {epoch:3d} | "
                f"mean={mean_ret:6.1f} max={max_ret:6.1f} min={min_ret:6.1f} | "
                f"{steps_per_sec:,.0f} steps/s | "
                f"epoch={epoch_time:.2f}s"
            )
        
        if mean_ret >= 475:
            console.print(f"[bold green]Solved at epoch {epoch}![/bold green]")
            break
    
    envs.close()
    
    total_time = time.perf_counter() - start_time
    console.print()
    console.print(f"[bold]Total: {total_steps:,} steps in {total_time:.1f}s ({total_steps/total_time:,.0f} steps/s)[/bold]")


# =============================================================================
# MNIST (Supervised Learning with EGGROLL)
# =============================================================================


def main_mnist_torch(config: EggrollConfig = None):
    """MNIST classification with MLP using Torch EGGROLL."""
    # === Experiment-specific config ===
    if config is None:
        config = EggrollConfig(
            population_size=4096,
            rank=4,
            sigma=0.15,
            lr=0.1,
            sigma_decay=0.999,
            max_epochs=100,
            batch_size=256,
        )
    
    # Network architecture
    input_dim = 784
    hidden_dim = 256
    output_dim = 10
    use_layernorm = True
    
    torch.manual_seed(config.seed)
    reset_gpu_stats()
    
    console.print(f"\n[bold]MNIST MLP - Torch EGGROLL[/bold]")
    console.print(f"Population: {config.population_size}, Rank: {config.rank}, Sigma: {config.sigma}, LR: {config.lr}")
    console.print(f"Network: {input_dim} -> {hidden_dim} -> {output_dim}")
    console.print()
    
    # === Load MNIST ===
    console.print("Preprocessing MNIST...")
    train_imgs, train_labels, test_imgs, test_labels = load_mnist_flat(config.dtype)
    
    # === Initialize params ===
    W1, b1, W2, b2 = init_mlp_params(
        input_dim, hidden_dim, output_dim,
        config.dtype
    )
    n_params = count_params(W1, b1, W2, b2)
    console.print(f"Model params: {n_params:,}")
    print_gpu_stats("Init ")
    console.print()
    
    # === Compile key functions for speed (matching JAX jit) ===
    compiled_forward_perturbed = torch.compile(mlp_forward_perturbed)
    compiled_forward_clean = torch.compile(mlp_forward_clean)
    compiled_es_gradient = torch.compile(compute_es_gradient)
    
    # === Training loop ===
    current_sigma = config.sigma
    start_time = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    
    for epoch in range(config.max_epochs):
        epoch_start = time.perf_counter()
        
        # Sample batch
        idx = torch.tensor(rng.integers(0, len(train_imgs), size=config.batch_size), device="cuda")
        batch_imgs = train_imgs[idx]
        batch_labels = train_labels[idx]
        
        # Generate perturbations
        gen = torch.Generator(device="cuda")
        gen.manual_seed(config.seed + epoch * 1000)
        
        A1_scaled, A1, B1 = generate_lowrank_perturbations(
            config.population_size, hidden_dim, input_dim, config.rank, 
            current_sigma, gen, config.dtype
        )
        A2_scaled, A2, B2 = generate_lowrank_perturbations(
            config.population_size, output_dim, hidden_dim, config.rank, 
            current_sigma, gen, config.dtype
        )
        
        # Forward: expand batch for population (batch, 784) -> (pop, batch, 784)
        x = batch_imgs.unsqueeze(0).expand(config.population_size, -1, -1)
        
        # MLP forward with perturbations (compiled)
        logits = compiled_forward_perturbed(
            x, W1, b1, W2, b2, A1_scaled, B1, A2_scaled, B2, 
            hidden_dim, use_layernorm=use_layernorm
        )
        
        # Compute fitness and normalize
        fitnesses = compute_classification_fitness(logits, batch_labels)
        fitnesses = normalize_fitnesses(fitnesses)
        
        # ES gradient update - use A_scaled (includes sigma/sqrt(rank))
        grad_W1 = compiled_es_gradient(fitnesses, A1_scaled, B1, config.population_size)
        grad_W2 = compiled_es_gradient(fitnesses, A2_scaled, B2, config.population_size)
        
        W1 = W1 + config.lr * grad_W1
        W2 = W2 + config.lr * grad_W2
        
        current_sigma *= config.sigma_decay
        
        # === Logging ===
        with torch.no_grad():
            test_logits = compiled_forward_clean(test_imgs, W1, b1, W2, b2, hidden_dim, use_layernorm=use_layernorm)
            test_preds = test_logits.argmax(dim=-1)
            test_acc = (test_preds == test_labels).float().mean().item() * 100
        
        epoch_time = time.perf_counter() - epoch_start
        console.print(f"Epoch {epoch:3d} | test_acc={test_acc:5.1f}% | time={epoch_time:.2f}s")
    
    total_time = time.perf_counter() - start_time
    console.print()
    console.print(f"[bold]Total training time: {total_time:.1f}s[/bold]")
    print_gpu_stats("Final ")


def main_mnist_jax():
    """MNIST classification with MLP using JAX EGGROLL."""
    import jax
    import jax.numpy as jnp
    import optax
    from torchvision import datasets, transforms
    
    from hyperscalees.noiser.eggroll import EggRoll
    from hyperscalees.models.common import MLP, simple_es_tree_key
    
    device = "cuda"
    
    # === Config (HYPERSCALE - match Torch) ===
    population_size = 4096   # 8x larger than typical ES
    rank = 4
    sigma = 0.15
    lr = 0.1
    lr_decay = 0.998
    sigma_decay = 0.999
    max_epochs = 100
    batch_size = 256
    seed = 42
    hidden_dim = 256
    
    console.print(f"\n[bold]MNIST MLP - JAX EGGROLL[/bold]")
    console.print(f"Population: {population_size}, Rank: {rank}, Sigma: {sigma}, LR: {lr}")
    console.print(f"Network: 784 -> {hidden_dim} -> 10")
    console.print()
    
    # === Load MNIST ===
    transform = transforms.ToTensor()
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    # Flatten images
    console.print("Preprocessing MNIST...")
    train_imgs = np.stack([train_data[i][0].numpy().flatten() for i in range(len(train_data))])
    train_labels = np.array([train_data[i][1] for i in range(len(train_data))])
    test_imgs = np.stack([test_data[i][0].numpy().flatten() for i in range(len(test_data))])
    test_labels = np.array([test_data[i][1] for i in range(len(test_data))])
    
    train_imgs = jnp.array(train_imgs)
    train_labels = jnp.array(train_labels)
    test_imgs = jnp.array(test_imgs)
    test_labels = jnp.array(test_labels)
    
    # === Initialize model ===
    key = jax.random.key(seed)
    model_key = jax.random.fold_in(key, 0)
    es_key = jax.random.fold_in(key, 1)
    
    frozen_params, params, scan_map, es_map = MLP.rand_init(
        model_key,
        in_dim=784,
        out_dim=10,
        hidden_dims=[hidden_dim],
        use_bias=True,
        activation="pqn",
        dtype="float32",
    )
    
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)
    
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params,
        sigma=sigma,
        lr=lr,
        solver=optax.sgd,
        rank=rank,
    )
    
    n_params = sum(p.size for p in jax.tree.leaves(params))
    console.print(f"Model params: {n_params:,}")
    console.print()
    
    # === JIT compile ===
    def forward_noisy(noiser_params, params, iterinfo, x):
        return MLP.forward(
            EggRoll, frozen_noiser_params, noiser_params, frozen_params,
            params, es_tree_key, iterinfo, x
        )
    
    jit_forward = jax.jit(jax.vmap(
        lambda n, p, i, x: forward_noisy(n, p, i, x),
        in_axes=(None, None, 0, 0)
    ))
    
    @jax.jit
    def do_update(noiser_params, params, fitnesses, iterinfos):
        return EggRoll.do_updates(
            frozen_noiser_params, noiser_params, params,
            es_tree_key, fitnesses, iterinfos, es_map
        )
    
    # Forward for eval (no perturbation) - implement manually since MLP doesn't have forward_base
    def forward_base(params, x):
        """MLP forward without perturbations."""
        # Layer 0: in -> hidden
        x = x @ params['0']['weight'].T
        if 'bias' in params['0']:
            x = x + params['0']['bias']
        # pqn activation: relu(layer_norm(x))
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        x = jax.nn.relu(x)
        
        # Layer 1: hidden -> out
        x = x @ params['1']['weight'].T
        if 'bias' in params['1']:
            x = x + params['1']['bias']
        return x
    
    jit_forward_base = jax.jit(jax.vmap(forward_base, in_axes=(None, 0)))
    
    # === Training loop ===
    current_sigma = sigma
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    
    for epoch in range(max_epochs):
        epoch_start = time.perf_counter()
        
        # Sample batch
        idx = rng.integers(0, len(train_imgs), size=batch_size)
        batch_imgs = train_imgs[idx]      # (batch, 784)
        batch_labels_np = train_labels[idx]  # (batch,)
        
        # Expand for population: (pop, batch, 784)
        batch_imgs_pop = jnp.broadcast_to(batch_imgs, (population_size, batch_size, 784))
        
        iterinfo = (jnp.full(population_size, epoch, dtype=jnp.int32), jnp.arange(population_size))
        
        # Forward
        logits = jit_forward(noiser_params, params, iterinfo, batch_imgs_pop)  # (pop, batch, 10)
        
        # Compute fitness: negative cross-entropy
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        labels_exp = jnp.broadcast_to(batch_labels_np, (population_size, batch_size))
        nll = -jnp.take_along_axis(log_probs, labels_exp[:, :, None], axis=-1).squeeze(-1)
        losses = nll.mean(axis=-1)
        raw_fitnesses = -losses
        
        fitnesses = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, raw_fitnesses)
        
        # Update
        noiser_params, params = do_update(noiser_params, params, fitnesses, iterinfo)
        
        current_sigma *= sigma_decay
        noiser_params = {**noiser_params, "sigma": current_sigma}
        
        # === Logging (every epoch) ===
        test_logits = jit_forward_base(params, test_imgs)
        test_preds = jnp.argmax(test_logits, axis=-1)
        test_acc = float((test_preds == test_labels).mean()) * 100
        
        epoch_time = time.perf_counter() - epoch_start
        console.print(f"Epoch {epoch:3d} | test_acc={test_acc:5.1f}% | time={epoch_time:.2f}s")
    
    total_time = time.perf_counter() - start_time
    console.print()
    console.print(f"[bold]Total training time: {total_time:.1f}s[/bold]")
    print_gpu_stats_jax("Final ")


# =============================================================================
# MNIST CNN - Torch EGGROLL
# =============================================================================

def cnn_forward(x, W_conv1, W_conv2, batch_size):
    """CNN feature extraction (conv layers only)."""
    x = torch.nn.functional.conv2d(x, W_conv1, padding=1)
    x = torch.relu(x)
    x = torch.nn.functional.max_pool2d(x, 2)
    x = torch.nn.functional.conv2d(x, W_conv2, padding=1)
    x = torch.relu(x)
    x = torch.nn.functional.max_pool2d(x, 2)
    return x.view(batch_size, -1)


def main_mnist_cnn_torch(config: EggrollConfig = None):
    """MNIST classification with CNN using Torch EGGROLL."""
    # === Experiment-specific config ===
    if config is None:
        config = EggrollConfig(
            population_size=4096,
            rank=4,
            sigma=0.15,
            lr=0.1,
            lr_decay=0.998,
            sigma_decay=0.999,
            max_epochs=100,
            batch_size=128,
        )
    
    torch.manual_seed(config.seed)
    reset_gpu_stats()
    
    console.print(f"\n[bold]MNIST CNN - Torch EGGROLL[/bold]")
    console.print(f"Population: {config.population_size}, Rank: {config.rank}, Sigma: {config.sigma}, LR: {config.lr}")
    console.print()
    
    # === Load MNIST ===
    console.print("Preprocessing MNIST...")
    train_imgs, train_labels, test_imgs, test_labels = load_mnist_2d(config.dtype)
    
    # === CNN Architecture ===
    # Conv1: 1 -> 16 channels, 3x3 kernel
    # Conv2: 16 -> 32 channels, 3x3 kernel  
    # After 2 maxpools: 28 -> 14 -> 7, so 32*7*7 = 1568 features
    # FC: 1568 -> 10
    
    conv1_out, conv1_in, k1 = 16, 1, 3
    conv2_out, conv2_in, k2 = 32, 16, 3
    fc_in = 32 * 7 * 7
    fc_out = 10
    
    # Initialize weights
    W_conv1 = torch.randn(conv1_out, conv1_in, k1, k1, device="cuda", dtype=config.dtype) * 0.1
    W_conv2 = torch.randn(conv2_out, conv2_in, k2, k2, device="cuda", dtype=config.dtype) * 0.1
    W_fc = torch.randn(fc_out, fc_in, device="cuda", dtype=config.dtype) / math.sqrt(fc_in)
    b_fc = torch.zeros(fc_out, device="cuda", dtype=config.dtype)
    
    n_params = count_params(W_conv1, W_conv2, W_fc, b_fc)
    console.print(f"CNN params: {n_params:,}")
    console.print(f"  Conv1: {W_conv1.shape}, Conv2: {W_conv2.shape}, FC: {W_fc.shape}")
    print_gpu_stats("Init ")
    console.print()
    
    # === Compile key functions ===
    compiled_apply_pert = torch.compile(apply_lowrank_perturbation)
    compiled_es_gradient = torch.compile(compute_es_gradient)
    
    # === Training loop ===
    current_lr = config.lr
    current_sigma = config.sigma
    start_time = time.perf_counter()
    
    for epoch in range(config.max_epochs):
        epoch_start = time.perf_counter()
        
        # Sample batch
        idx = torch.randint(0, len(train_imgs), (config.batch_size,), device="cuda")
        batch_imgs = train_imgs[idx]
        batch_labels = train_labels[idx]
        
        # Generate perturbations for FC layer only
        gen = torch.Generator(device="cuda")
        gen.manual_seed(config.seed + epoch * 1000)
        
        A_fc_scaled, A_fc, B_fc = generate_lowrank_perturbations(
            config.population_size, fc_out, fc_in, config.rank, 
            current_sigma, gen, config.dtype
        )
        
        # Forward: conv layers are shared, only FC is perturbed
        x = cnn_forward(batch_imgs, W_conv1, W_conv2, config.batch_size)
        
        # Expand for population
        x_pop = x.unsqueeze(0).expand(config.population_size, -1, -1)
        
        # FC with perturbation
        base = x_pop @ W_fc.T + b_fc
        pert = compiled_apply_pert(x_pop, B_fc, A_fc_scaled)
        logits = base + pert
        
        # Compute fitness and normalize
        fitnesses = compute_classification_fitness(logits, batch_labels)
        fitnesses = normalize_fitnesses(fitnesses)
        
        # ES gradient for FC only
        grad_fc = compiled_es_gradient(fitnesses, A_fc_scaled, B_fc, config.population_size)
        W_fc = W_fc + current_lr * grad_fc
        
        current_lr *= config.lr_decay
        current_sigma *= config.sigma_decay
        
        # === Logging ===
        with torch.no_grad():
            x_test = cnn_forward(test_imgs, W_conv1, W_conv2, len(test_imgs))
            test_logits = x_test @ W_fc.T + b_fc
            test_preds = test_logits.argmax(dim=-1)
            test_acc = (test_preds == test_labels).float().mean().item() * 100
        
        epoch_time = time.perf_counter() - epoch_start
        console.print(f"Epoch {epoch:3d} | test_acc={test_acc:5.1f}% | time={epoch_time:.2f}s")
    
    total_time = time.perf_counter() - start_time
    console.print()
    console.print(f"[bold]Total training time: {total_time:.1f}s[/bold]")
    print_gpu_stats("Final ")


# =============================================================================
# MNIST CNN - JAX EGGROLL
# =============================================================================

def main_mnist_cnn_jax():
    """MNIST classification with CNN using JAX EGGROLL (from codebase)."""
    import jax
    import jax.numpy as jnp
    import optax
    from torchvision import datasets, transforms
    
    from hyperscalees.noiser.eggroll import EggRoll
    from hyperscalees.models.common import SimpleCNN, simple_es_tree_key
    
    console.print(f"\n[bold]MNIST CNN - JAX EGGROLL (codebase)[/bold]")
    
    # === Config (match Torch) ===
    population_size = 4096   # Match Torch for fair comparison
    rank = 4
    sigma = 0.15
    lr = 0.1
    lr_decay = 0.998
    sigma_decay = 0.999
    max_epochs = 100
    batch_size = 128
    seed = 42
    
    console.print(f"Population: {population_size}, Rank: {rank}, Sigma: {sigma}, LR: {lr}")
    console.print()
    
    # === Load MNIST ===
    transform = transforms.ToTensor()
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    console.print("Preprocessing MNIST...")
    train_imgs = np.stack([train_data[i][0].numpy() for i in range(len(train_data))])  # (60000, 1, 28, 28)
    train_labels = np.array([train_data[i][1] for i in range(len(train_data))])
    test_imgs = np.stack([test_data[i][0].numpy() for i in range(len(test_data))])
    test_labels = np.array([test_data[i][1] for i in range(len(test_data))])
    
    train_imgs = jnp.array(train_imgs)
    train_labels = jnp.array(train_labels)
    test_imgs = jnp.array(test_imgs)
    test_labels = jnp.array(test_labels)
    
    # === Initialize model using JAX EGGROLL infrastructure ===
    key = jax.random.key(seed)
    model_key = jax.random.fold_in(key, 0)
    es_key = jax.random.fold_in(key, 1)
    
    frozen_params, params, scan_map, es_map = SimpleCNN.rand_init(
        model_key,
        in_channels=1,
        num_classes=10,
        dtype=jnp.float32,
    )
    
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)
    
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params,
        sigma=sigma,
        lr=lr,
        solver=optax.sgd,
        rank=rank,
        noise_reuse=0,
    )
    
    n_params = sum(p.size for p in jax.tree.leaves(params))
    console.print(f"CNN params: {n_params:,}")
    print_gpu_stats("Init ")
    console.print()
    
    # === JIT compile forward functions ===
    # Efficient pattern: conv_forward once, then vmap over fc_forward
    jit_conv_forward = jax.jit(SimpleCNN.conv_forward)
    
    def fc_forward_noisy(noiser_params, params, iterinfo, x):
        """FC forward with perturbation for a single population member."""
        return SimpleCNN.fc_forward(
            EggRoll, frozen_noiser_params, noiser_params, frozen_params,
            params, es_tree_key, iterinfo, x
        )
    
    # vmap over population (iterinfo varies, x is broadcast)
    jit_fc_forward = jax.jit(jax.vmap(
        lambda n, p, i, x: fc_forward_noisy(n, p, i, x),
        in_axes=(None, None, 0, None)  # x is NOT vmapped - shared across pop
    ))
    
    @jax.jit
    def forward_eval(params, x):
        """Forward without noise (for evaluation)."""
        features = SimpleCNN.conv_forward(params, x)
        # FC without perturbation
        return features @ params['fc']['weight'].T + params['fc']['bias']
    
    @jax.jit
    def do_update(noiser_params, params, fitnesses, iterinfos):
        return EggRoll.do_updates(
            frozen_noiser_params, noiser_params, params,
            es_tree_key, fitnesses, iterinfos, es_map
        )
    
    # === Training loop ===
    current_sigma = sigma
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    
    for epoch in range(max_epochs):
        epoch_start = time.perf_counter()
        
        # Sample batch
        idx = rng.integers(0, len(train_imgs), size=batch_size)
        batch_imgs = train_imgs[idx]      # (batch, 1, 28, 28)
        batch_labels = train_labels[idx]
        
        # Run conv layers ONCE (shared across population)
        features = jit_conv_forward(params, batch_imgs)  # (batch, 1568)
        
        # Create iterinfo for this epoch
        iterinfo = (jnp.full(population_size, epoch, dtype=jnp.int32), jnp.arange(population_size))
        
        # Run FC layer with perturbations (vmapped over population)
        logits = jit_fc_forward(noiser_params, params, iterinfo, features)  # (pop, batch, 10)
        
        # Compute fitness: negative cross-entropy
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        labels_exp = jnp.broadcast_to(batch_labels, (population_size, batch_size))
        nll = -jnp.take_along_axis(log_probs, labels_exp[:, :, None], axis=-1).squeeze(-1)
        losses = nll.mean(axis=-1)
        raw_fitnesses = -losses
        
        fitnesses = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, raw_fitnesses)
        
        # Update
        noiser_params, params = do_update(noiser_params, params, fitnesses, iterinfo)
        
        # Decay sigma
        current_sigma *= sigma_decay
        noiser_params = {**noiser_params, "sigma": current_sigma}
        
        # === Logging (every epoch) ===
        test_logits = forward_eval(params, test_imgs)
        test_preds = jnp.argmax(test_logits, axis=-1)
        test_acc = float((test_preds == test_labels).mean()) * 100
        
        epoch_time = time.perf_counter() - epoch_start
        console.print(f"Epoch {epoch:3d} | test_acc={test_acc:5.1f}% | time={epoch_time:.2f}s")
    
    total_time = time.perf_counter() - start_time
    console.print()
    console.print(f"[bold]Total training time: {total_time:.1f}s[/bold]")
    print_gpu_stats_jax("Final ")


# =============================================================================
# Inference Time Benchmark - Training vs Pure Inference
# =============================================================================

def benchmark_inference_jax():
    """
    JAX-only inference benchmark. Run in a separate process from Torch.
    
    NOTE: JAX EggRoll uses on-the-fly noise generation via seeded PRNG.
    The noise is regenerated deterministically each forward pass using
    jax.random.fold_in(key, epoch, thread_idx).
    """
    import jax
    import jax.numpy as jnp
    import optax
    from hyperscalees.noiser.eggroll import EggRoll
    from hyperscalees.models.common import MLP, simple_es_tree_key
    
    console.print(f"\n[bold blue]═══ JAX Inference Benchmark ═══[/bold blue]")
    console.print("JAX EggRoll uses on-the-fly noise (seeded PRNG regeneration)")
    console.print("Paper target for on-the-fly: ~69% of pure inference speed")
    console.print()
    
    configs = [("MLP 784->256->10", 784, 256, 10, 128)]
    population_sizes = [1024, 2048, 4096, 8192]
    rank = 4
    sigma = 0.1
    num_warmup = 5
    num_iters = 20
    
    results = []
    
    for name, in_dim, hidden_dim, out_dim, batch_size in configs:
        console.print(f"[cyan]{name}[/cyan]")
        
        for pop_size in population_sizes:
            try:
                key = jax.random.key(42)
                model_key, es_key = jax.random.split(key)
                
                frozen_params, params, scan_map, es_map = MLP.rand_init(
                    model_key, in_dim=in_dim, out_dim=out_dim,
                    hidden_dims=[hidden_dim], use_bias=False,
                    activation="relu", dtype="float32",
                )
                
                es_tree_key = simple_es_tree_key(params, es_key, scan_map)
                frozen_noiser_params, noiser_params = EggRoll.init_noiser(
                    params, sigma=sigma, lr=0.1, solver=optax.sgd, rank=rank,
                )
                
                x_jax = jnp.ones((batch_size, in_dim))
                x_batch_jax = jnp.ones((pop_size, batch_size, in_dim))
                
                @jax.jit
                def batched_inference_jax(params, x):
                    def single_forward(x):
                        h = jax.nn.relu(x @ params['0']['weight'].T)
                        return h @ params['1']['weight'].T
                    return jax.vmap(single_forward)(x)
                
                def forward_noisy(noiser_params, params, iterinfo, x):
                    return MLP.forward(
                        EggRoll, frozen_noiser_params, noiser_params, frozen_params,
                        params, es_tree_key, iterinfo, x
                    )
                
                jit_forward = jax.jit(jax.vmap(
                    lambda n, p, i, x: forward_noisy(n, p, i, x),
                    in_axes=(None, None, 0, 0)
                ))
                
                @jax.jit
                def do_update(noiser_params, params, fitnesses, iterinfos):
                    return EggRoll.do_updates(
                        frozen_noiser_params, noiser_params, params,
                        es_tree_key, fitnesses, iterinfos, es_map
                    )
                
                def run_eggroll_jax():
                    x_pop = jnp.broadcast_to(x_jax, (pop_size, batch_size, in_dim))
                    iterinfo = (jnp.zeros(pop_size, dtype=jnp.int32), jnp.arange(pop_size))
                    logits = jit_forward(noiser_params, params, iterinfo, x_pop)
                    fitnesses = logits.mean(axis=(1, 2))
                    fitnesses = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
                    return do_update(noiser_params, params, fitnesses, iterinfo)
                
                # Warmup
                _ = batched_inference_jax(params, x_batch_jax).block_until_ready()
                _ = run_eggroll_jax()
                for _ in range(num_warmup):
                    _ = batched_inference_jax(params, x_batch_jax).block_until_ready()
                    _ = run_eggroll_jax()
                
                # Time inference
                t0 = time.perf_counter()
                for _ in range(num_iters):
                    _ = batched_inference_jax(params, x_batch_jax).block_until_ready()
                inference_time = (time.perf_counter() - t0) / num_iters
                
                # Time EGGROLL
                t0 = time.perf_counter()
                for _ in range(num_iters):
                    _ = run_eggroll_jax()
                eggroll_time = (time.perf_counter() - t0) / num_iters
                
                efficiency = (inference_time / eggroll_time) * 100
                
                console.print(
                    f"  pop={pop_size:>5}: "
                    f"inference={inference_time*1000:.2f}ms, "
                    f"eggroll={eggroll_time*1000:.2f}ms, "
                    f"[bold]efficiency={efficiency:.1f}%[/bold]"
                )
                
                results.append({
                    "config": name, "pop_size": pop_size, "backend": "jax",
                    "inference_ms": inference_time * 1000,
                    "eggroll_ms": eggroll_time * 1000,
                    "efficiency": efficiency,
                })
                
                jax.clear_caches()
                
            except Exception as e:
                console.print(f"  pop={pop_size:>5}: [red]{e}[/red]")
                break
    
    return results


def benchmark_inference_torch():
    """
    Torch-only inference benchmark using torch.compile for JAX-parity performance.
    
    Measures:
    - Compiled inference vs compiled EGGROLL step
    - Shows efficiency as % of pure inference throughput
    
    Performance achieved: ~0.99x JAX speed with torch.compile
    """
    # Enable TF32 for benchmark speed (not needed for RL stability)
    torch.set_float32_matmul_precision('high')
    
    console.print(f"\n[bold green]═══ Torch Inference Benchmark (Compiled) ═══[/bold green]")
    console.print("Using torch.compile for JAX-parity performance")
    console.print("Measuring training speed as % of pure inference speed")
    console.print()
    
    device = "cuda"
    dtype = torch.float32
    
    configs = [("MLP 784->256->10", 784, 256, 10, 128)]
    population_sizes = [1024, 2048, 4096]
    rank = 4
    sigma = 0.1
    num_warmup = 20
    num_iters = 100
    
    results = []
    
    for name, in_dim, hidden_dim, out_dim, batch_size in configs:
        console.print(f"[cyan]{name}[/cyan]")
        
        # Warmup torch.compile once for this config
        console.print("  [dim]Warming up torch.compile (one-time cost)...[/dim]")
        warmup_compiled_functions(population_sizes[0], batch_size, in_dim, hidden_dim, out_dim, rank, dtype)
        
        for pop_size in population_sizes:
            try:
                reset_gpu_stats()
                torch.cuda.empty_cache()
                
                W1 = torch.randn(hidden_dim, in_dim, device=device, dtype=dtype)
                W2 = torch.randn(out_dim, hidden_dim, device=device, dtype=dtype)
                b1 = torch.zeros(hidden_dim, device=device, dtype=dtype)
                b2 = torch.zeros(out_dim, device=device, dtype=dtype)
                x_batch = torch.randn(pop_size, batch_size, in_dim, device=device, dtype=dtype)
                
                # Generate perturbations
                gen = torch.Generator(device=device).manual_seed(42)
                A1_scaled, A1, B1 = generate_lowrank_perturbations(
                    pop_size, hidden_dim, in_dim, rank, sigma, gen, dtype
                )
                A2_scaled, A2, B2 = generate_lowrank_perturbations(
                    pop_size, out_dim, hidden_dim, rank, sigma, gen, dtype
                )
                sqrt_N = math.sqrt(pop_size)
                
                # Warmup compiled functions for this pop_size
                for _ in range(num_warmup):
                    _ = compiled_batched_inference(x_batch, W1, b1, W2, b2)
                    _ = compiled_eggroll_step(x_batch, W1, b1, W2, b2, A1_scaled, B1, A2_scaled, B2, sqrt_N)
                torch.cuda.synchronize()
                
                # Time compiled inference
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(num_iters):
                    _ = compiled_batched_inference(x_batch, W1, b1, W2, b2)
                torch.cuda.synchronize()
                inference_time = (time.perf_counter() - t0) / num_iters
                
                # Time compiled EGGROLL step
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(num_iters):
                    _ = compiled_eggroll_step(x_batch, W1, b1, W2, b2, A1_scaled, B1, A2_scaled, B2, sqrt_N)
                torch.cuda.synchronize()
                eggroll_time = (time.perf_counter() - t0) / num_iters
                
                efficiency = (inference_time / eggroll_time) * 100
                
                console.print(
                    f"  pop={pop_size:>5}: "
                    f"inf={inference_time*1000:.2f}ms | "
                    f"eggroll={eggroll_time*1000:.2f}ms | "
                    f"[bold]efficiency={efficiency:.1f}%[/bold]"
                )
                
                results.append({
                    "config": name, "pop_size": pop_size, "backend": "torch_compiled",
                    "inference_ms": inference_time * 1000,
                    "eggroll_ms": eggroll_time * 1000,
                    "efficiency": efficiency,
                })
                
                del W1, W2, b1, b2, x_batch
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                console.print(f"  pop={pop_size:>5}: [red]OOM[/red]")
                break
            except Exception as e:
                console.print(f"  pop={pop_size:>5}: [red]{e}[/red]")
                import traceback
                traceback.print_exc()
                break
    
    return results


def benchmark_inference_ratio():
    """
    Benchmark EGGROLL training time vs pure inference time.
    
    Uses torch.compile for optimal performance (~0.99x JAX speed).
    
    Key metric from EGGROLL paper (Figure 8):
    - EGGROLL achieves 91% (pre-gen noise) or 69% (on-the-fly noise) of pure inference throughput
    - PPO: 34%
    - OpenES: 0.41% (0.054% with noise regen)
    
    For JAX vs Torch head-to-head comparison, see:
        benchmarks/compile_experiments.py
    """
    console.print("[bold]Torch EGGROLL Inference Benchmark (with torch.compile)[/bold]")
    console.print()
    
    torch_results = benchmark_inference_torch()
    
    # Summary
    console.print()
    console.print("[bold]Summary - Training Efficiency (% of pure inference speed)[/bold]")
    table = Table()
    table.add_column("Population", justify="right")
    table.add_column("Inference", justify="right")
    table.add_column("EGGROLL", justify="right")
    table.add_column("Efficiency", justify="right")
    
    for r in torch_results:
        table.add_row(
            f"{r['pop_size']:,}",
            f"{r['inference_ms']:.2f}ms",
            f"{r['eggroll_ms']:.2f}ms",
            f"[bold]{r['efficiency']:.1f}%[/bold]"
        )
    
    console.print(table)
    return torch_results


# =============================================================================
# Hyperscale Test - Find max population size
# =============================================================================

def main_hyperscale_torch():
    """Find maximum population size before OOM for Torch EGGROLL."""
    
    device = "cuda"
    dtype = torch.float32
    
    console.print(f"\n[bold]Hyperscale Test - Torch EGGROLL[/bold]")
    console.print("Finding maximum population size before OOM...")
    console.print()
    
    # Test configs
    configs = [
        ("CartPole MLP", 4, 256, 2),      # (name, input_dim, hidden_dim, output_dim)
        ("MNIST MLP", 784, 256, 10),
        ("MNIST CNN FC", 1568, 10, 0),    # Just the FC layer
    ]
    
    results = []
    
    for name, in_dim, hidden_dim, out_dim in configs:
        if out_dim == 0:
            # Just a single layer
            out_dim = hidden_dim
            hidden_dim = 0
        
        console.print(f"[cyan]Testing {name}...[/cyan]")
        
        pop_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        max_pop = 0
        rank = 4
        
        for pop_size in pop_sizes:
            try:
                reset_gpu_stats()
                torch.cuda.empty_cache()
                
                half_pop = pop_size // 2
                
                if hidden_dim > 0:
                    # Two-layer network
                    W1 = torch.randn(hidden_dim, in_dim, device=device, dtype=dtype)
                    W2 = torch.randn(out_dim, hidden_dim, device=device, dtype=dtype)
                    
                    # Perturbations
                    A1 = torch.randn(pop_size, hidden_dim, rank, device=device, dtype=dtype)
                    B1 = torch.randn(pop_size, in_dim, rank, device=device, dtype=dtype)
                    A2 = torch.randn(pop_size, out_dim, rank, device=device, dtype=dtype)
                    B2 = torch.randn(pop_size, hidden_dim, rank, device=device, dtype=dtype)
                    
                    # Simulate forward
                    batch_size = 256 if in_dim < 100 else 64
                    x = torch.randn(pop_size, batch_size, in_dim, device=device, dtype=dtype)
                    h = x @ W1.T
                    h = h + torch.einsum('pbi,pir,pjr->pbj', x, B1, A1)
                    h = torch.relu(h)
                    out = h @ W2.T
                    out = out + torch.einsum('pbi,pir,pjr->pbj', h, B2, A2)
                else:
                    # Single layer
                    W = torch.randn(out_dim, in_dim, device=device, dtype=dtype)
                    A = torch.randn(pop_size, out_dim, rank, device=device, dtype=dtype)
                    B = torch.randn(pop_size, in_dim, rank, device=device, dtype=dtype)
                    
                    batch_size = 64
                    x = torch.randn(pop_size, batch_size, in_dim, device=device, dtype=dtype)
                    out = x @ W.T
                    out = out + torch.einsum('pbi,pir,pjr->pbj', x, B, A)
                
                torch.cuda.synchronize()
                
                stats = get_gpu_stats()
                max_pop = pop_size
                console.print(f"  pop={pop_size:,}: ✓ ({stats['allocated_gb']:.2f}GB)")
                
                # Clean up
                del x, out
                if hidden_dim > 0:
                    del W1, W2, A1, B1, A2, B2, h
                else:
                    del W, A, B
                
            except torch.cuda.OutOfMemoryError:
                console.print(f"  pop={pop_size:,}: ✗ OOM")
                break
            except Exception as e:
                console.print(f"  pop={pop_size:,}: ✗ {e}")
                break
        
        results.append((name, max_pop))
        torch.cuda.empty_cache()
    
    console.print()
    console.print("[bold]Results:[/bold]")
    table = Table()
    table.add_column("Config")
    table.add_column("Max Population", justify="right")
    
    for name, max_pop in results:
        table.add_row(name, f"{max_pop:,}")
    
    console.print(table)
    print_gpu_stats("Final ")


def main_hyperscale_jax():
    """Find maximum population size before OOM for JAX EGGROLL."""
    import jax
    import jax.numpy as jnp
    
    console.print(f"\n[bold]Hyperscale Test - JAX EGGROLL[/bold]")
    console.print("Finding maximum population size before OOM...")
    console.print()
    
    # Test with MNIST MLP config
    in_dim = 784
    hidden_dim = 256
    out_dim = 10
    rank = 4
    
    pop_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    max_pop = 0
    
    for pop_size in pop_sizes:
        try:
            # Clear JAX cache
            jax.clear_caches()
            
            # Allocate
            W1 = jnp.ones((hidden_dim, in_dim))
            W2 = jnp.ones((out_dim, hidden_dim))
            
            A1 = jnp.ones((pop_size, hidden_dim, rank))
            B1 = jnp.ones((pop_size, in_dim, rank))
            A2 = jnp.ones((pop_size, out_dim, rank))
            B2 = jnp.ones((pop_size, hidden_dim, rank))
            
            batch_size = 64
            x = jnp.ones((pop_size, batch_size, in_dim))
            
            # Forward
            h = x @ W1.T + jnp.einsum('pbi,pir,pjr->pbj', x, B1, A1)
            h = jax.nn.relu(h)
            out = h @ W2.T + jnp.einsum('pbi,pir,pjr->pbj', h, B2, A2)
            
            # Force computation
            _ = float(out.mean())
            
            max_pop = pop_size
            console.print(f"  pop={pop_size:,}: ✓")
            
        except Exception as e:
            console.print(f"  pop={pop_size:,}: ✗ {type(e).__name__}")
            break
    
    console.print()
    console.print(f"[bold]Max JAX population: {max_pop:,}[/bold]")


# =============================================================================
# Detailed Torch Profiling - Where is time spent?
# =============================================================================

def profile_torch_breakdown():
    """
    Profile where time is spent in the Torch EGGROLL forward pass.
    
    This breaks down:
    1. Noise generation (randn)
    2. Low-rank perturbation computation (x @ B @ A.T)
    3. Base matmul (x @ W.T)
    4. Activation
    5. Fitness normalization
    6. ES gradient computation
    
    Goal: Identify bottlenecks to match JAX's ~69% efficiency target.
    """
    console.print(f"\n[bold magenta]═══ Torch EGGROLL Detailed Profiling ═══[/bold magenta]")
    console.print("Breaking down where time is spent in the forward pass")
    console.print()
    
    device = "cuda"
    dtype = torch.float32
    
    # Config
    in_dim, hidden_dim, out_dim = 784, 256, 10  # MNIST MLP
    batch_size = 128
    rank = 4
    sigma = 0.1
    num_warmup = 10
    num_iters = 50
    
    population_sizes = [1024, 2048, 4096, 8192]
    
    for pop_size in population_sizes:
        console.print(f"[cyan]Population: {pop_size:,}[/cyan]")
        
        try:
            reset_gpu_stats()
            torch.cuda.empty_cache()
            
            # Initialize
            W1 = torch.randn(hidden_dim, in_dim, device=device, dtype=dtype)
            W2 = torch.randn(out_dim, hidden_dim, device=device, dtype=dtype)
            b1 = torch.zeros(hidden_dim, device=device, dtype=dtype)
            b2 = torch.zeros(out_dim, device=device, dtype=dtype)
            x = torch.randn(batch_size, in_dim, device=device, dtype=dtype)
            x_pop = x.unsqueeze(0).expand(pop_size, -1, -1)  # (pop, batch, in)
            
            # Pre-generate noise for warmup
            gen = torch.Generator(device=device).manual_seed(42)
            A1_scaled, A1, B1 = generate_lowrank_perturbations(
                pop_size, hidden_dim, in_dim, rank, sigma, gen, dtype
            )
            A2_scaled, A2, B2 = generate_lowrank_perturbations(
                pop_size, out_dim, hidden_dim, rank, sigma, gen, dtype
            )
            
            # ========== Timing Functions ==========
            
            def time_noise_gen():
                gen = torch.Generator(device=device).manual_seed(42)
                A1_s, A1_, B1_ = generate_lowrank_perturbations(
                    pop_size, hidden_dim, in_dim, rank, sigma, gen, dtype
                )
                A2_s, A2_, B2_ = generate_lowrank_perturbations(
                    pop_size, out_dim, hidden_dim, rank, sigma, gen, dtype
                )
                return A1_s, B1_, A2_s, B2_
            
            def time_base_matmul():
                """Pure inference: x @ W1.T, relu, x @ W2.T"""
                base1 = x_pop @ W1.T + b1
                h = torch.relu(base1)
                base2 = h @ W2.T + b2
                return base2
            
            def time_perturbation():
                """Just the low-rank perturbation: x @ B @ A.T"""
                pert1 = apply_lowrank_perturbation(x_pop, B1, A1_scaled)
                h = torch.relu(x_pop @ W1.T + b1 + pert1)
                pert2 = apply_lowrank_perturbation(h, B2, A2_scaled)
                return pert2
            
            def time_full_forward():
                """Full forward with perturbations."""
                return mlp_forward_perturbed(
                    x_pop, W1, b1, W2, b2, A1_scaled, B1, A2_scaled, B2,
                    hidden_dim, use_layernorm=False
                )
            
            def time_fitness_norm():
                """Fitness normalization."""
                logits = mlp_forward_perturbed(
                    x_pop, W1, b1, W2, b2, A1_scaled, B1, A2_scaled, B2,
                    hidden_dim, use_layernorm=False
                )
                fitnesses = normalize_fitnesses(logits.mean(dim=(1, 2)))
                return fitnesses
            
            def time_es_gradient():
                """ES gradient computation."""
                logits = mlp_forward_perturbed(
                    x_pop, W1, b1, W2, b2, A1_scaled, B1, A2_scaled, B2,
                    hidden_dim, use_layernorm=False
                )
                fitnesses = normalize_fitnesses(logits.mean(dim=(1, 2)))
                # Use A_scaled (includes sigma/sqrt(rank)) for proper gradient magnitude
                grad1 = compute_es_gradient(fitnesses, A1_scaled, B1, pop_size)
                grad2 = compute_es_gradient(fitnesses, A2_scaled, B2, pop_size)
                return grad1, grad2
            
            def time_full_step():
                """Full EGGROLL step: noise gen + forward + fitness + gradient."""
                gen = torch.Generator(device=device).manual_seed(42)
                A1_s, A1_, B1_ = generate_lowrank_perturbations(
                    pop_size, hidden_dim, in_dim, rank, sigma, gen, dtype
                )
                A2_s, A2_, B2_ = generate_lowrank_perturbations(
                    pop_size, out_dim, hidden_dim, rank, sigma, gen, dtype
                )
                logits = mlp_forward_perturbed(
                    x_pop, W1, b1, W2, b2, A1_s, B1_, A2_s, B2_,
                    hidden_dim, use_layernorm=False
                )
                fitnesses = normalize_fitnesses(logits.mean(dim=(1, 2)))
                # Use A_s (scaled, includes sigma/sqrt(rank)) for proper gradient magnitude
                grad1 = compute_es_gradient(fitnesses, A1_s, B1_, pop_size)
                grad2 = compute_es_gradient(fitnesses, A2_s, B2_, pop_size)
                return grad1, grad2
            
            # Warmup
            for _ in range(num_warmup):
                _ = time_noise_gen()
                _ = time_base_matmul()
                _ = time_perturbation()
                _ = time_full_forward()
                _ = time_fitness_norm()
                _ = time_es_gradient()
                _ = time_full_step()
            torch.cuda.synchronize()
            
            # Time each component
            timings = {}
            
            def measure(name, fn, iters=num_iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    _ = fn()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) / iters * 1000  # ms
                timings[name] = elapsed
            
            measure("1_noise_gen", time_noise_gen)
            measure("2_base_matmul", time_base_matmul)
            measure("3_perturbation", time_perturbation)
            measure("4_full_forward", time_full_forward)
            measure("5_fitness_norm", time_fitness_norm)
            measure("6_es_gradient", time_es_gradient)
            measure("7_full_step", time_full_step)
            
            # Compute derived metrics
            base_inference = timings["2_base_matmul"]
            overhead_pert = timings["4_full_forward"] - base_inference
            overhead_noise = timings["1_noise_gen"]
            overhead_gradient = timings["7_full_step"] - timings["5_fitness_norm"]
            
            efficiency_pregen = (base_inference / timings["6_es_gradient"]) * 100
            efficiency_onthefly = (base_inference / timings["7_full_step"]) * 100
            
            # Print results
            console.print(f"  {'Component':<20} {'Time (ms)':>10} {'% of full step':>15}")
            console.print(f"  {'-'*20} {'-'*10} {'-'*15}")
            full = timings["7_full_step"]
            for name, t in sorted(timings.items()):
                pct = (t / full) * 100
                console.print(f"  {name[2:]:<20} {t:>10.3f} {pct:>14.1f}%")
            
            console.print()
            console.print(f"  [bold]Analysis:[/bold]")
            console.print(f"    Base inference:       {base_inference:.3f}ms")
            console.print(f"    Perturbation overhead:{overhead_pert:.3f}ms ({(overhead_pert/base_inference)*100:.1f}% of base)")
            console.print(f"    Noise gen overhead:   {overhead_noise:.3f}ms ({(overhead_noise/base_inference)*100:.1f}% of base)")
            console.print()
            console.print(f"    [green]Pre-gen efficiency:    {efficiency_pregen:.1f}%[/green] (paper target: 91%)")
            console.print(f"    [green]On-the-fly efficiency: {efficiency_onthefly:.1f}%[/green] (paper target: 69%)")
            console.print()
            
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()


def profile_torch_optimizations():
    """
    Compare different optimization strategies for the perturbation operation.
    
    Tests:
    1. einsum (current)
    2. bmm (two separate matmuls)
    3. torch.compile on einsum
    4. torch.compile on fused matmul+pert
    """
    console.print(f"\n[bold magenta]═══ Torch Optimization Comparison ═══[/bold magenta]")
    console.print("Comparing different strategies for low-rank perturbation")
    console.print()
    
    device = "cuda"
    dtype = torch.float32
    
    in_dim, hidden_dim, out_dim = 784, 256, 10
    batch_size = 128
    rank = 4
    sigma = 0.1
    num_warmup = 20
    num_iters = 100
    
    population_sizes = [1024, 2048, 4096, 8192]
    
    for pop_size in population_sizes:
        console.print(f"[cyan]Population: {pop_size:,}[/cyan]")
        
        try:
            reset_gpu_stats()
            torch.cuda.empty_cache()
            
            # Initialize
            W = torch.randn(hidden_dim, in_dim, device=device, dtype=dtype)
            x = torch.randn(pop_size, batch_size, in_dim, device=device, dtype=dtype)
            
            gen = torch.Generator(device=device).manual_seed(42)
            A_scaled, A, B = generate_lowrank_perturbations(
                pop_size, hidden_dim, in_dim, rank, sigma, gen, dtype
            )
            
            # === Different implementations ===
            
            def impl_einsum():
                return apply_lowrank_perturbation(x, B, A_scaled)
            
            def impl_bmm():
                return apply_lowrank_perturbation_bmm(x, B, A_scaled)
            
            def impl_fused():
                return fused_matmul_with_lowrank(x, W, B, A_scaled)
            
            def impl_base_only():
                return x @ W.T
            
            # Compiled versions
            compiled_einsum = torch.compile(impl_einsum, mode="reduce-overhead")
            compiled_bmm = torch.compile(impl_bmm, mode="reduce-overhead")
            compiled_fused = torch.compile(impl_fused, mode="reduce-overhead")
            compiled_base = torch.compile(impl_base_only, mode="reduce-overhead")
            
            # Warmup all
            for _ in range(num_warmup):
                _ = impl_einsum()
                _ = impl_bmm()
                _ = impl_fused()
                _ = impl_base_only()
            torch.cuda.synchronize()
            
            # Warmup compiled (triggers compilation)
            for _ in range(3):
                _ = compiled_einsum()
                _ = compiled_bmm()
                _ = compiled_fused()
                _ = compiled_base()
            torch.cuda.synchronize()
            
            # Additional warmup after compilation
            for _ in range(num_warmup):
                _ = compiled_einsum()
                _ = compiled_bmm()
                _ = compiled_fused()
                _ = compiled_base()
            torch.cuda.synchronize()
            
            timings = {}
            
            def measure(name, fn, iters=num_iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    _ = fn()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) / iters * 1000
                timings[name] = elapsed
            
            measure("base_only", impl_base_only)
            measure("einsum", impl_einsum)
            measure("bmm", impl_bmm)
            measure("fused", impl_fused)
            measure("compiled_base", compiled_base)
            measure("compiled_einsum", compiled_einsum)
            measure("compiled_bmm", compiled_bmm)
            measure("compiled_fused", compiled_fused)
            
            # Analysis
            base = timings["base_only"]
            compiled_base_t = timings["compiled_base"]
            
            console.print(f"  {'Method':<20} {'Time (ms)':>10} {'Efficiency':>12}")
            console.print(f"  {'-'*20} {'-'*10} {'-'*12}")
            
            # Print non-compiled
            console.print(f"  {'base_only':<20} {base:>10.3f} {'100.0%':>12}")
            for name in ["einsum", "bmm", "fused"]:
                t = timings[name]
                eff = (base / t) * 100
                console.print(f"  {name:<20} {t:>10.3f} {eff:>11.1f}%")
            
            # Print compiled
            console.print(f"  {'compiled_base':<20} {compiled_base_t:>10.3f} {'-':>12}")
            for name in ["compiled_einsum", "compiled_bmm", "compiled_fused"]:
                t = timings[name]
                eff = (compiled_base_t / t) * 100
                console.print(f"  {name:<20} {t:>10.3f} {eff:>11.1f}%")
            
            console.print()
            
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()


def profile_head_to_head():
    """
    Head-to-head comparison of JAX vs Torch EGGROLL.
    
    Measures identical operations on both backends:
    1. Pure inference (batched forward without perturbation)
    2. EGGROLL forward (with perturbation, pre-gen noise)
    3. Full EGGROLL step (forward + ES gradient computation)
    
    JAX implementation is the source of truth.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from hyperscalees.noiser.eggroll import EggRoll
    from hyperscalees.models.common import MLP, simple_es_tree_key
    
    console.print(f"\n[bold cyan]═══ JAX vs Torch Head-to-Head Benchmark ═══[/bold cyan]")
    console.print("Comparing Torch EGGROLL against JAX EGGROLL (source of truth)")
    console.print()
    
    # Config - identical for both
    in_dim, hidden_dim, out_dim = 784, 256, 10  # MNIST MLP
    batch_size = 128
    rank = 4
    sigma = 0.1
    num_warmup = 10
    num_iters = 50
    seed = 42
    
    population_sizes = [1024, 2048, 4096, 8192]
    
    results = []
    
    for pop_size in population_sizes:
        console.print(f"[bold]Population: {pop_size:,}[/bold]")
        
        # ===== JAX Setup =====
        try:
            jax.clear_caches()
            
            key = jax.random.key(seed)
            model_key = jax.random.fold_in(key, 0)
            es_key = jax.random.fold_in(key, 1)
            
            frozen_params, params, scan_map, es_map = MLP.rand_init(
                model_key,
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dims=[hidden_dim],
                use_bias=True,
                activation="relu",  # Simple relu for fair comparison
                dtype="float32",
            )
            
            es_tree_key = simple_es_tree_key(params, es_key, scan_map)
            
            frozen_noiser_params, noiser_params = EggRoll.init_noiser(
                params,
                sigma=sigma,
                lr=0.1,
                solver=optax.sgd,
                rank=rank,
            )
            
            x_jax = jnp.ones((batch_size, in_dim))
            x_pop_jax = jnp.broadcast_to(x_jax, (pop_size, batch_size, in_dim))
            
            # JAX pure inference (no perturbation)
            @jax.jit
            def jax_inference(params, x):
                def single_forward(x):
                    h = jax.nn.relu(x @ params['0']['weight'].T + params['0']['bias'])
                    return h @ params['1']['weight'].T + params['1']['bias']
                return jax.vmap(single_forward)(x)
            
            # JAX EGGROLL forward (with perturbation)
            def forward_noisy(noiser_params, params, iterinfo, x):
                return MLP.forward(
                    EggRoll, frozen_noiser_params, noiser_params, frozen_params,
                    params, es_tree_key, iterinfo, x
                )
            
            jit_forward_jax = jax.jit(jax.vmap(
                lambda n, p, i, x: forward_noisy(n, p, i, x),
                in_axes=(None, None, 0, 0)
            ))
            
            @jax.jit
            def jax_do_update(noiser_params, params, fitnesses, iterinfos):
                return EggRoll.do_updates(
                    frozen_noiser_params, noiser_params, params,
                    es_tree_key, fitnesses, iterinfos, es_map
                )
            
            def jax_eggroll_forward():
                iterinfo = (jnp.zeros(pop_size, dtype=jnp.int32), jnp.arange(pop_size))
                return jit_forward_jax(noiser_params, params, iterinfo, x_pop_jax)
            
            def jax_full_step():
                iterinfo = (jnp.zeros(pop_size, dtype=jnp.int32), jnp.arange(pop_size))
                logits = jit_forward_jax(noiser_params, params, iterinfo, x_pop_jax)
                fitnesses = logits.mean(axis=(1, 2))
                fitnesses = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
                return jax_do_update(noiser_params, params, fitnesses, iterinfo)
            
            # Warmup JAX
            _ = jax_inference(params, x_pop_jax).block_until_ready()
            _ = jax_eggroll_forward().block_until_ready()
            _ = jax_full_step()
            
            for _ in range(num_warmup):
                _ = jax_inference(params, x_pop_jax).block_until_ready()
                _ = jax_eggroll_forward().block_until_ready()
                _ = jax_full_step()
            
            # Time JAX
            t0 = time.perf_counter()
            for _ in range(num_iters):
                _ = jax_inference(params, x_pop_jax).block_until_ready()
            jax_inference_time = (time.perf_counter() - t0) / num_iters * 1000
            
            t0 = time.perf_counter()
            for _ in range(num_iters):
                _ = jax_eggroll_forward().block_until_ready()
            jax_eggroll_time = (time.perf_counter() - t0) / num_iters * 1000
            
            t0 = time.perf_counter()
            for _ in range(num_iters):
                _ = jax_full_step()
            jax_full_time = (time.perf_counter() - t0) / num_iters * 1000
            
            jax_efficiency = (jax_inference_time / jax_full_time) * 100
            
        except Exception as e:
            console.print(f"  [red]JAX Error: {e}[/red]")
            jax_inference_time = jax_eggroll_time = jax_full_time = jax_efficiency = float('nan')
        
        # ===== Torch Setup =====
        try:
            torch.cuda.empty_cache()
            reset_gpu_stats()
            torch.set_float32_matmul_precision('high')  # Enable TF32
            
            device = "cuda"
            dtype = torch.float32
            
            torch.manual_seed(seed)
            
            W1 = torch.randn(hidden_dim, in_dim, device=device, dtype=dtype)
            W2 = torch.randn(out_dim, hidden_dim, device=device, dtype=dtype)
            b1 = torch.zeros(hidden_dim, device=device, dtype=dtype)
            b2 = torch.zeros(out_dim, device=device, dtype=dtype)
            
            x_torch = torch.ones(batch_size, in_dim, device=device, dtype=dtype)
            x_pop_torch = x_torch.unsqueeze(0).expand(pop_size, -1, -1).contiguous()
            
            # Pre-generate perturbations
            gen = torch.Generator(device=device).manual_seed(seed)
            A1_scaled, A1, B1 = generate_lowrank_perturbations(
                pop_size, hidden_dim, in_dim, rank, sigma, gen, dtype
            )
            A2_scaled, A2, B2 = generate_lowrank_perturbations(
                pop_size, out_dim, hidden_dim, rank, sigma, gen, dtype
            )
            
            def torch_inference():
                h = torch.relu(x_pop_torch @ W1.T + b1)
                return h @ W2.T + b2
            
            def torch_eggroll_forward():
                return mlp_forward_perturbed(
                    x_pop_torch, W1, b1, W2, b2, A1_scaled, B1, A2_scaled, B2,
                    hidden_dim, use_layernorm=False
                )
            
            def torch_full_step():
                logits = mlp_forward_perturbed(
                    x_pop_torch, W1, b1, W2, b2, A1_scaled, B1, A2_scaled, B2,
                    hidden_dim, use_layernorm=False
                )
                fitnesses = normalize_fitnesses(logits.mean(dim=(1, 2)))
                # Use A_scaled (includes sigma/sqrt(rank)) for proper gradient magnitude
                grad1 = compute_es_gradient(fitnesses, A1_scaled, B1, pop_size)
                grad2 = compute_es_gradient(fitnesses, A2_scaled, B2, pop_size)
                return grad1, grad2
            
            # Compiled versions
            compiled_inference = torch.compile(torch_inference)
            compiled_forward = torch.compile(torch_eggroll_forward)
            compiled_full_step = torch.compile(torch_full_step)
            
            # Warmup eager
            for _ in range(num_warmup):
                _ = torch_inference()
                _ = torch_eggroll_forward()
                _ = torch_full_step()
            torch.cuda.synchronize()
            
            # Warmup compiled (triggers JIT)
            for _ in range(5):
                _ = compiled_inference()
                _ = compiled_forward()
                _ = compiled_full_step()
            torch.cuda.synchronize()
            
            # Additional warmup after compilation
            for _ in range(num_warmup):
                _ = compiled_inference()
                _ = compiled_forward()
                _ = compiled_full_step()
            torch.cuda.synchronize()
            
            # Time Torch eager
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_iters):
                _ = torch_inference()
            torch.cuda.synchronize()
            torch_inference_time = (time.perf_counter() - t0) / num_iters * 1000
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_iters):
                _ = torch_full_step()
            torch.cuda.synchronize()
            torch_full_time = (time.perf_counter() - t0) / num_iters * 1000
            
            torch_efficiency = (torch_inference_time / torch_full_time) * 100
            
            # Time Torch compiled
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_iters):
                _ = compiled_inference()
            torch.cuda.synchronize()
            torch_compiled_inference_time = (time.perf_counter() - t0) / num_iters * 1000
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_iters):
                _ = compiled_full_step()
            torch.cuda.synchronize()
            torch_compiled_full_time = (time.perf_counter() - t0) / num_iters * 1000
            
            torch_compiled_efficiency = (torch_compiled_inference_time / torch_compiled_full_time) * 100
            
        except Exception as e:
            console.print(f"  [red]Torch Error: {e}[/red]")
            import traceback
            traceback.print_exc()
            torch_inference_time = torch_full_time = torch_efficiency = float('nan')
            torch_compiled_inference_time = torch_compiled_full_time = torch_compiled_efficiency = float('nan')
        
        # Print results
        console.print(f"  {'Metric':<30} {'JAX':>10} {'Torch':>10} {'Compiled':>10}")
        console.print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
        
        console.print(f"  {'Pure inference (ms)':<30} {jax_inference_time:>10.2f} {torch_inference_time:>10.2f} {torch_compiled_inference_time:>10.2f}")
        console.print(f"  {'Full EGGROLL step (ms)':<30} {jax_full_time:>10.2f} {torch_full_time:>10.2f} {torch_compiled_full_time:>10.2f}")
        console.print(f"  {'Efficiency (inf/full)':<30} {jax_efficiency:>9.1f}% {torch_efficiency:>9.1f}% {torch_compiled_efficiency:>9.1f}%")
        console.print(f"  {'vs JAX speed':<30} {'-':>10} {jax_full_time/torch_full_time:>9.2f}x {jax_full_time/torch_compiled_full_time:>9.2f}x")
        console.print()
        
        results.append({
            "pop_size": pop_size,
            "jax_inference_ms": jax_inference_time,
            "jax_full_ms": jax_full_time,
            "jax_efficiency": jax_efficiency,
            "torch_inference_ms": torch_inference_time,
            "torch_full_ms": torch_full_time,
            "torch_efficiency": torch_efficiency,
            "torch_compiled_inference_ms": torch_compiled_inference_time,
            "torch_compiled_full_ms": torch_compiled_full_time,
            "torch_compiled_efficiency": torch_compiled_efficiency,
        })
        
        # Clear JAX before next iteration
        jax.clear_caches()
    
    # Summary
    console.print("[bold]Summary:[/bold]")
    console.print("  Efficiency = pure_inference_time / full_step_time")
    console.print("  Higher efficiency = EGGROLL overhead is lower relative to inference")
    console.print("  vs JAX speed > 1.0 means Torch is faster than JAX")
    console.print()
    console.print("  [green]Target: Match or exceed JAX efficiency and speed[/green]")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["cartpole", "mnist", "mnist_cnn", "hyperscale", "inference", "profile", "all"], default="cartpole")
    parser.add_argument("--backend", choices=["torch", "jax", "both"], default="both")
    args = parser.parse_args()
    
    if args.experiment == "cartpole":
        if args.backend in ("torch", "both"):
            reset_gpu_stats()
            main()
            print_gpu_stats("Final ")
        if args.backend in ("jax", "both"):
            main_jax()
    
    elif args.experiment == "mnist":
        if args.backend in ("torch", "both"):
            reset_gpu_stats()
            main_mnist_torch()
        if args.backend in ("jax", "both"):
            main_mnist_jax()
    
    elif args.experiment == "mnist_cnn":
        if args.backend in ("torch", "both"):
            main_mnist_cnn_torch()
        if args.backend in ("jax", "both"):
            main_mnist_cnn_jax()
    
    elif args.experiment == "hyperscale":
        if args.backend in ("torch", "both"):
            main_hyperscale_torch()
        if args.backend in ("jax", "both"):
            main_hyperscale_jax()
    
    elif args.experiment == "inference":
        benchmark_inference_ratio()
    
    elif args.experiment == "profile":
        profile_torch_breakdown()
    
    elif args.experiment == "all":
        console.print("[bold cyan]═══ CartPole Experiments ═══[/bold cyan]")
        if args.backend in ("torch", "both"):
            reset_gpu_stats()
            main()
            print_gpu_stats("Final ")
        if args.backend in ("jax", "both"):
            main_jax()
        
        console.print("\n[bold cyan]═══ MNIST MLP Experiments ═══[/bold cyan]")
        if args.backend in ("torch", "both"):
            reset_gpu_stats()
            main_mnist_torch()
        if args.backend in ("jax", "both"):
            main_mnist_jax()
        
        console.print("\n[bold cyan]═══ MNIST CNN Experiments ═══[/bold cyan]")
        if args.backend in ("torch", "both"):
            main_mnist_cnn_torch()
        if args.backend in ("jax", "both"):
            main_mnist_cnn_jax()
        
        console.print("\n[bold cyan]═══ Inference Benchmark ═══[/bold cyan]")
        benchmark_inference_ratio()
        
        console.print("\n[bold cyan]═══ Hyperscale Tests ═══[/bold cyan]")
        if args.backend in ("torch", "both"):
            main_hyperscale_torch()
        if args.backend in ("jax", "both"):
            main_hyperscale_jax()
