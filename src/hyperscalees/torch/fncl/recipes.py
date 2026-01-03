"""
EGGROLL Experiments & Recipes
=============================

This file contains runnable experiments demonstrating EGGROLL on RL and SL tasks.
For API documentation, see README.md and core.py.

Experiments:
    CartPole (RL)       - cartpole()
    MNIST MLP (SL)      - mnist_mlp()
    MNIST CNN (SL)      - mnist_cnn()
    
Run:
    python -m hyperscalees.torch.fncl.recipes --experiment cartpole
    python -m hyperscalees.torch.fncl.recipes --experiment mnist
    python -m hyperscalees.torch.fncl.recipes --experiment mnist_cnn
"""

import torch
import torch.nn as nn
import gymnasium as gym
import time
import math
import numpy as np
from rich.console import Console
from rich.table import Table

from .core import (
    EggrollConfig,
    get_params_dict, get_weight_shapes, generate_perturbations,
    perturbed_forward, make_perturbed_forward_fn, eggroll_step,
)

console = Console()

# =============================================================================
# GPU Utilities
# =============================================================================

def get_gpu_stats():
    """Get current GPU memory and utilization stats."""
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


def reset_gpu_stats():
    """Reset GPU peak memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


# =============================================================================
# Data Loading
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
# Experiment Helpers
# =============================================================================

def compute_classification_fitness(logits, labels):
    """Compute fitness for classification: -cross_entropy (higher = better)."""
    population_size = logits.shape[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    labels_exp = labels.unsqueeze(0).expand(population_size, -1)
    nll = -log_probs.gather(dim=-1, index=labels_exp.unsqueeze(-1)).squeeze(-1)
    return -nll.mean(dim=-1)

@torch.compile
def compute_weight_perturbation(A_scaled: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Materialize the full weight perturbation matrix from low-rank factors.
    
    WARNING: This explicitly computes A @ B.T, which is O(m*n) memory.
    Only use when materialization is unavoidable (e.g., conv2d via grouped conv).
    
    For linear layers, use perturbed_linear() instead - it computes:
        x @ B @ A.T  (two rank-r matmuls, never materializes m×n matrix)
    
    Args:
        A_scaled: (population, out_dim, rank) - scaled A factors  
        B: (population, in_dim, rank) - B factors
        
    Returns:
        delta_W: (population, out_dim, in_dim) - materialized weight perturbation
    """
    return torch.einsum('pir,pjr->pij', A_scaled, B)


def perturbed_conv2d(x, weight, perts, weight_name, padding=1):
    """
    Apply perturbed conv2d using grouped convolution.
    
    Conv layers require materializing the full perturbed weight (unlike linear
    layers which can use the efficient x @ B @ A.T formulation). We use grouped
    convolution to efficiently batch across the population.
    """
    if weight_name not in perts:
        return torch.nn.functional.conv2d(x, weight, padding=padding)
    
    A_scaled, B = perts[weight_name]
    pop_size = A_scaled.shape[0]
    C_out, C_in, k1, k2 = weight.shape
    
    if x.dim() == 4:
        x = x.unsqueeze(0).expand(pop_size, -1, -1, -1, -1)
    
    batch_size = x.shape[1]
    H, W_dim = x.shape[3], x.shape[4]
    
    W_flat = weight.reshape(C_out, -1)
    delta_W = compute_weight_perturbation(A_scaled, B)
    W_perturbed = (W_flat.unsqueeze(0) + delta_W).reshape(pop_size, C_out, C_in, k1, k2)
    
    # Grouped conv: reshape for efficient batched conv across population
    x_grouped = x.permute(1, 0, 2, 3, 4).reshape(batch_size, pop_size * C_in, H, W_dim)
    W_grouped = W_perturbed.reshape(pop_size * C_out, C_in, k1, k2)
    out_grouped = torch.nn.functional.conv2d(x_grouped, W_grouped, padding=padding, groups=pop_size)
    
    H_out, W_out = out_grouped.shape[2], out_grouped.shape[3]
    return out_grouped.reshape(batch_size, pop_size, C_out, H_out, W_out).permute(1, 0, 2, 3, 4)


def count_params(*tensors):
    """Count total number of parameters in tensors."""
    return sum(t.numel() for t in tensors)


def count_params_dict(params):
    """Count total parameters in a params dict."""
    return sum(t.numel() for t in params.values() if isinstance(t, torch.Tensor))


# =============================================================================
# EXPERIMENT: CartPole (RL)
# =============================================================================

def cartpole(config: EggrollConfig = None):
    """CartPole-v1 with EGGROLL. Architecture: 4 -> 256 (tanh) -> 2."""
    if config is None:
        config = EggrollConfig(
            population_size=2048, rank=4, sigma=0.2, lr=0.1,
            lr_decay=0.9995, sigma_decay=0.999, max_epochs=300,
        )
    
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(config.seed)
    reset_gpu_stats()
    
    # Model: 2-layer MLP using standard PyTorch
    model = nn.Sequential(
        nn.Linear(4, 256),
        nn.Tanh(),
        nn.Linear(256, 2),
    )
    # Initialize with small weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.1)
            nn.init.zeros_(m.bias)
    
    # Convert to EGGROLL format
    params = get_params_dict(model, dtype=config.dtype)
    shapes = get_weight_shapes(params)
    forward, forward_eval = make_perturbed_forward_fn(model)
    n_params = count_params_dict(params)
    
    console.print(f"[bold]Torch EGGROLL - CartPole-v1[/bold]")
    console.print(f"Population: {config.population_size}, Rank: {config.rank}, Sigma: {config.sigma}, LR: {config.lr}")
    console.print(f"Network: 4 -> 256 -> 2 ({n_params:,} params)")
    print_gpu_stats("Init ")
    console.print()
    
    # Training loop
    envs = gym.make_vec("CartPole-v1", num_envs=config.population_size)
    total_steps = 0
    start_time = time.perf_counter()
    current_lr, current_sigma = config.lr, config.sigma
    
    for epoch in range(config.max_epochs):
        epoch_start = time.perf_counter()
        
        gen = torch.Generator(device="cuda").manual_seed(config.seed + epoch * 1000)
        perts = generate_perturbations(shapes, config.population_size, config.rank, 
                                       current_sigma, gen, config.dtype)
        
        obs, _ = envs.reset(seed=epoch)
        episode_returns = torch.zeros(config.population_size, device="cuda", dtype=config.dtype)
        dones = torch.zeros(config.population_size, dtype=torch.bool, device="cuda")
        
        steps_this_epoch = 0
        for step in range(500):
            obs_t = torch.as_tensor(obs, device="cuda", dtype=config.dtype)
            logits = forward(obs_t, params, perts)
            
            actions = logits.argmax(dim=-1).cpu().numpy()
            obs, rewards, terminated, truncated, _ = envs.step(actions)
            
            active = ~dones
            steps_this_epoch += active.sum().item()
            
            rewards_t = torch.as_tensor(rewards, device="cuda", dtype=config.dtype)
            done_t = torch.as_tensor(terminated | truncated, device="cuda")
            
            episode_returns += rewards_t * active.float()
            dones = dones | done_t
            
            if dones.all():
                break
        
        total_steps += steps_this_epoch
        
        current_lr, current_sigma = eggroll_step(
            params, episode_returns, perts, current_lr, current_sigma, config
        )
        
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
    
    # Final evaluation
    console.print("\n[bold]Final evaluation (base policy, no perturbations):[/bold]")
    eval_env = gym.make("CartPole-v1")
    
    returns = []
    for ep in range(10):
        obs, _ = eval_env.reset(seed=ep + 1000)
        total_reward = 0
        done = False
        
        while not done:
            obs_t = torch.as_tensor(obs, device="cuda", dtype=config.dtype)
            logits = forward_eval(obs_t, params)
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


# =============================================================================
# EXPERIMENT: MNIST MLP (Supervised Learning)
# =============================================================================

def mnist_mlp(config: EggrollConfig = None):
    """MNIST classification with EGGROLL. Architecture: 784 -> 256 (tanh) -> 10."""
    if config is None:
        config = EggrollConfig(
            population_size=4096, rank=4, sigma=0.15, lr=0.1,
            sigma_decay=0.999, max_epochs=100, batch_size=256,
        )
    
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(config.seed)
    reset_gpu_stats()
    
    # Model: 2-layer MLP using standard PyTorch
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.Tanh(),
        nn.Linear(256, 10),
    )
    # Initialize with small weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.1)
            nn.init.zeros_(m.bias)
    
    # Convert to EGGROLL format
    params = get_params_dict(model, dtype=config.dtype)
    shapes = get_weight_shapes(params)
    forward, forward_eval = make_perturbed_forward_fn(model)
    n_params = count_params_dict(params)
    
    console.print(f"\n[bold]MNIST MLP - Torch EGGROLL[/bold]")
    console.print(f"Population: {config.population_size}, Rank: {config.rank}, Sigma: {config.sigma}, LR: {config.lr}")
    console.print(f"Network: 784 -> 256 -> 10 ({n_params:,} params)")
    print_gpu_stats("Init ")
    console.print()
    
    console.print("Preprocessing MNIST...")
    train_imgs, train_labels, test_imgs, test_labels = load_mnist_flat(config.dtype)
    
    # Training loop
    start_time = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    current_lr, current_sigma = config.lr, config.sigma
    
    for epoch in range(config.max_epochs):
        epoch_start = time.perf_counter()
        
        idx = torch.tensor(rng.integers(0, len(train_imgs), size=config.batch_size), device="cuda")
        batch_imgs, batch_labels = train_imgs[idx], train_labels[idx]
        
        gen = torch.Generator(device="cuda").manual_seed(config.seed + epoch * 1000)
        perts = generate_perturbations(shapes, config.population_size, config.rank, 
                                       current_sigma, gen, config.dtype)
        
        x = batch_imgs.unsqueeze(0).expand(config.population_size, -1, -1)
        logits = forward(x, params, perts)
        fitnesses = compute_classification_fitness(logits, batch_labels)
        current_lr, current_sigma = eggroll_step(
            params, fitnesses, perts, current_lr, current_sigma, config
        )
        
        with torch.no_grad():
            test_logits = forward_eval(test_imgs, params)
            test_preds = test_logits.argmax(dim=-1)
            test_acc = (test_preds == test_labels).float().mean().item() * 100
        
        epoch_time = time.perf_counter() - epoch_start
        console.print(f"Epoch {epoch:3d} | test_acc={test_acc:5.1f}% | time={epoch_time:.2f}s")
    
    total_time = time.perf_counter() - start_time
    console.print()
    console.print(f"[bold]Total training time: {total_time:.1f}s[/bold]")
    print_gpu_stats("Final ")


# =============================================================================
# EXPERIMENT: MNIST CNN
# =============================================================================

def mnist_cnn(config: EggrollConfig = None):
    """MNIST CNN with EGGROLL (perturbs conv + FC layers)."""
    if config is None:
        config = EggrollConfig(
            population_size=2048, rank=8, sigma=0.1, lr=0.1,
            lr_decay=0.998, sigma_decay=0.999, max_epochs=1000, batch_size=64,
        )
    
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(config.seed)
    reset_gpu_stats()
    
    console.print(f"\n[bold]MNIST CNN - Torch EGGROLL[/bold]")
    console.print(f"Population: {config.population_size}, Rank: {config.rank}, Sigma: {config.sigma}")
    console.print()
    console.print("Preprocessing MNIST...")
    train_imgs, train_labels, test_imgs, test_labels = load_mnist_2d(config.dtype)
    
    # Model: Conv1 -> ReLU -> Pool -> Conv2 -> ReLU -> Pool -> FC
    params = {
        'conv1.weight': torch.randn(16, 1, 3, 3, device="cuda", dtype=config.dtype) * 0.1,
        'conv2.weight': torch.randn(32, 16, 3, 3, device="cuda", dtype=config.dtype) * 0.1,
        'fc.weight': torch.randn(10, 32*7*7, device="cuda", dtype=config.dtype) / math.sqrt(32*7*7),
        'fc.bias': torch.zeros(10, device="cuda", dtype=config.dtype),
    }
    shapes = get_weight_shapes(params)
    n_params = count_params_dict(params)
    
    def forward(x, params, perts):
        """Forward with perturbed conv + FC layers."""
        x = perturbed_conv2d(x, params['conv1.weight'], perts, 'conv1.weight', padding=1)
        x = torch.relu(x)
        pop_size, batch_size = x.shape[0], x.shape[1]
        x = torch.nn.functional.max_pool2d(x.reshape(pop_size * batch_size, *x.shape[2:]), 2)
        x = x.reshape(pop_size, batch_size, *x.shape[1:])
        
        x = perturbed_conv2d(x, params['conv2.weight'], perts, 'conv2.weight', padding=1)
        x = torch.relu(x)
        x = torch.nn.functional.max_pool2d(x.reshape(pop_size * batch_size, *x.shape[2:]), 2)
        x = x.reshape(pop_size, batch_size, -1)
        
        return perturbed_forward(x, params['fc.weight'], params['fc.bias'], perts, 'fc.weight')
    
    def forward_eval(x, params):
        """Forward without perturbations (for eval)."""
        x = torch.nn.functional.conv2d(x, params['conv1.weight'], padding=1)
        x = torch.relu(torch.nn.functional.max_pool2d(x, 2))
        x = torch.nn.functional.conv2d(x, params['conv2.weight'], padding=1)
        x = torch.relu(torch.nn.functional.max_pool2d(x, 2))
        return x.flatten(1) @ params['fc.weight'].T + params['fc.bias']
    
    console.print(f"CNN params: {n_params:,}")
    print_gpu_stats("Init ")
    console.print()
    
    # Training loop
    current_lr, current_sigma = config.lr, config.sigma
    start_time = time.perf_counter()
    
    for epoch in range(config.max_epochs):
        epoch_start = time.perf_counter()
        
        idx = torch.randint(0, len(train_imgs), (config.batch_size,), device="cuda")
        batch_imgs, batch_labels = train_imgs[idx], train_labels[idx]
        
        gen = torch.Generator(device="cuda").manual_seed(config.seed + epoch * 1000)
        perts = generate_perturbations(shapes, config.population_size, config.rank, 
                                       current_sigma, gen, config.dtype)
        
        logits = forward(batch_imgs, params, perts)
        fitnesses = compute_classification_fitness(logits, batch_labels)
        current_lr, current_sigma = eggroll_step(
            params, fitnesses, perts, current_lr, current_sigma, config
        )
        
        with torch.no_grad():
            test_logits = forward_eval(test_imgs, params)
            test_acc = (test_logits.argmax(dim=-1) == test_labels).float().mean().item() * 100
        
        console.print(f"Epoch {epoch:3d} | test_acc={test_acc:5.1f}% | time={time.perf_counter() - epoch_start:.2f}s")
    
    total_time = time.perf_counter() - start_time
    console.print()
    console.print(f"[bold]Total training time: {total_time:.1f}s[/bold]")
    print_gpu_stats("Final ")


# =============================================================================
# Benchmarking
# =============================================================================

def hyperscale():
    """Find maximum population size before OOM."""
    dtype = torch.float32
    
    console.print(f"\n[bold]Hyperscale Test - Torch EGGROLL[/bold]")
    console.print("Finding maximum population size before OOM...")
    console.print()
    
    configs = [
        ("CartPole MLP", 4, 256, 2),
        ("MNIST MLP", 784, 256, 10),
    ]
    
    results = []
    
    for name, in_dim, hidden_dim, out_dim in configs:
        console.print(f"[cyan]Testing {name}...[/cyan]")
        
        pop_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        max_pop = 0
        rank = 4
        sigma = 0.1
        
        for pop_size in pop_sizes:
            try:
                torch._dynamo.reset()
                reset_gpu_stats()
                torch.cuda.empty_cache()
                
                params = {
                    'layer1.weight': torch.randn(hidden_dim, in_dim, device="cuda", dtype=dtype),
                    'layer1.bias': torch.zeros(hidden_dim, device="cuda", dtype=dtype),
                    'layer2.weight': torch.randn(out_dim, hidden_dim, device="cuda", dtype=dtype),
                    'layer2.bias': torch.zeros(out_dim, device="cuda", dtype=dtype),
                }
                shapes = get_weight_shapes(params)
                
                gen = torch.Generator(device="cuda").manual_seed(42)
                perts = generate_perturbations(shapes, pop_size, rank, sigma, gen, dtype)
                
                batch_size = 256 if in_dim < 100 else 64
                x = torch.randn(pop_size, batch_size, in_dim, device="cuda", dtype=dtype)
                
                h = perturbed_forward(x, params['layer1.weight'], params['layer1.bias'], perts, 'layer1.weight')
                h = torch.relu(h)
                out = perturbed_forward(h, params['layer2.weight'], params['layer2.bias'], perts, 'layer2.weight')
                
                torch.cuda.synchronize()
                
                stats = get_gpu_stats()
                max_pop = pop_size
                console.print(f"  pop={pop_size:,}: ✓ ({stats['allocated_gb']:.2f}GB)")
                
                del x, out, h, params, perts
                
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
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EGGROLL experiments")
    parser.add_argument("--experiment", choices=["cartpole", "mnist", "mnist_cnn", "hyperscale", "all"], 
                        default="cartpole", help="Experiment to run")
    args = parser.parse_args()
    
    if args.experiment == "cartpole":
        cartpole()
    elif args.experiment == "mnist":
        mnist_mlp()
    elif args.experiment == "mnist_cnn":
        mnist_cnn()
    elif args.experiment == "hyperscale":
        hyperscale()
    elif args.experiment == "all":
        console.print("[bold cyan]═══ CartPole ═══[/bold cyan]")
        cartpole()
        console.print("\n[bold cyan]═══ MNIST MLP ═══[/bold cyan]")
        mnist_mlp()
        console.print("\n[bold cyan]═══ MNIST CNN ═══[/bold cyan]")
        mnist_cnn()