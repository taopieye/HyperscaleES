#!/usr/bin/env python3
"""
EGGROLL vs Baseline ES: CartPole Benchmark

A stress-test comparing EGGROLL (low-rank ES) against standard OpenES
on the CartPole-v1 environment.

Usage:
    uv run python experiments/rl_benchmark.py
    uv run python experiments/rl_benchmark.py --population 64 --generations 50
    uv run python experiments/rl_benchmark.py --help
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import statistics

import torch
import torch.nn as nn

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

console = Console()


# ============================================================================
# Neural Network Policy
# ============================================================================

class CartPolePolicy(nn.Module):
    """MLP policy for CartPole (matches paper's architecture)."""
    
    def __init__(self, hidden_size: int = 256, n_layers: int = 3):
        super().__init__()
        
        layers = []
        in_size = 4  # CartPole observation size
        
        for _ in range(n_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.Tanh())
            in_size = hidden_size
        
        layers.append(nn.Linear(hidden_size, 2))  # 2 actions
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def get_action(self, obs: torch.Tensor) -> int:
        with torch.no_grad():
            logits = self.forward(obs)
            return logits.argmax().item()


# ============================================================================
# Fitness Evaluation
# ============================================================================

def evaluate_policy(
    model: nn.Module,
    env: "gym.Env",
    device: torch.device,
    max_steps: int = 500,
    n_episodes: int = 1,
) -> float:
    """Evaluate a policy on CartPole, return average reward."""
    total_reward = 0.0
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        
        for _ in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            action = model.get_action(obs_tensor)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
    
    return total_reward / n_episodes


def evaluate_population_batched(
    model: nn.Module,
    strategy: "EggrollStrategy",
    population_size: int,
    epoch: int,
    env_fn,  # Factory function to create envs
    device: torch.device,
    max_steps: int = 500,
) -> torch.Tensor:
    """
    Evaluate all population members in parallel using vectorized environments.
    
    This properly batches both:
    1. Environment steps (using gymnasium's vectorized envs)
    2. Policy forward passes (using batched_forward)
    """
    # Create vectorized environment with one env per population member
    envs = gym.vector.SyncVectorEnv([env_fn for _ in range(population_size)])
    
    fitnesses = torch.zeros(population_size, device=device)
    active = torch.ones(population_size, dtype=torch.bool, device=device)
    
    # Reset all environments
    obs, _ = envs.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    
    # Member IDs for batched forward - each env corresponds to one population member
    member_ids = torch.arange(population_size, device=device)
    
    with strategy.perturb(population_size=population_size, epoch=epoch) as pop:
        for step in range(max_steps):
            if not active.any():
                break
            
            # Batch forward pass for ALL active population members at once
            # This is the key efficiency gain!
            logits = pop.batched_forward(model, obs_tensor, member_ids=member_ids)
            actions = logits.argmax(dim=-1).cpu().numpy()
            
            # Step all environments in parallel
            obs, rewards, terminated, truncated, _ = envs.step(actions)
            
            # Update fitnesses and active mask
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            fitnesses += rewards_tensor * active.float()
            
            done = torch.tensor(terminated | truncated, device=device)
            active = active & ~done
            
            # Update observations for next step
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    
    envs.close()
    return fitnesses


# ============================================================================
# Benchmark Results
# ============================================================================

@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    elapsed_time: float
    

@dataclass  
class BenchmarkResult:
    """Results from a complete benchmark run."""
    strategy_name: str
    generations: List[GenerationStats] = field(default_factory=list)
    total_time: float = 0.0
    final_best: float = 0.0
    solved_generation: Optional[int] = None  # Generation where solved (>=475 reward)
    
    @property
    def best_fitness_history(self) -> List[float]:
        return [g.best_fitness for g in self.generations]
    
    @property
    def mean_fitness_history(self) -> List[float]:
        return [g.mean_fitness for g in self.generations]


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_benchmark(
    strategy_name: str,
    strategy_class: type,
    strategy_kwargs: Dict[str, Any],
    population_size: int,
    n_generations: int,
    hidden_size: int,
    n_layers: int,
    device: torch.device,
    seed: int,
    progress: Progress,
    task_id: int,
) -> BenchmarkResult:
    """Run a single benchmark with the given strategy."""
    
    from hyperscalees.torch import EggrollStrategy, OpenESStrategy
    
    # Environment factory for vectorized envs
    def make_env():
        return gym.make("CartPole-v1")
    
    # Create model (paper uses 256x3 architecture)
    torch.manual_seed(seed)
    model = CartPolePolicy(hidden_size=hidden_size, n_layers=n_layers).to(device)
    
    # Log model size
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"  [dim]{strategy_name}: {n_params:,} parameters[/dim]")
    
    # Create strategy
    strategy = strategy_class(**strategy_kwargs)
    strategy.setup(model)
    
    result = BenchmarkResult(strategy_name=strategy_name)
    start_time = time.time()
    
    best_ever = 0.0
    
    for gen in range(n_generations):
        gen_start = time.time()
        
        # Evaluate population with truly batched evaluation
        fitnesses = evaluate_population_batched(
            model=model,
            strategy=strategy,
            population_size=population_size,
            epoch=gen,
            env_fn=make_env,
            device=device,
        )
        
        # Update strategy
        strategy.step(fitnesses)
        
        # Record stats
        gen_stats = GenerationStats(
            generation=gen,
            best_fitness=fitnesses.max().item(),
            mean_fitness=fitnesses.mean().item(),
            std_fitness=fitnesses.std().item(),
            elapsed_time=time.time() - gen_start,
        )
        result.generations.append(gen_stats)
        
        # Track best ever
        if gen_stats.best_fitness > best_ever:
            best_ever = gen_stats.best_fitness
        
        # Check if solved
        if gen_stats.best_fitness >= 475 and result.solved_generation is None:
            result.solved_generation = gen
        
        # Update progress
        progress.update(task_id, advance=1, description=f"[cyan]{strategy_name}[/] Gen {gen+1} | Best: {best_ever:.0f}")
    
    result.total_time = time.time() - start_time
    result.final_best = best_ever
    
    return result


# ============================================================================
# Display Functions
# ============================================================================

def create_comparison_table(results: List[BenchmarkResult]) -> Table:
    """Create a rich table comparing benchmark results."""
    table = Table(
        title="🏁 Benchmark Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    
    table.add_column("Strategy", style="cyan", justify="left")
    table.add_column("Final Best", style="green", justify="right")
    table.add_column("Mean (last 10)", style="yellow", justify="right")
    table.add_column("Solved Gen", style="blue", justify="right")
    table.add_column("Total Time", style="white", justify="right")
    table.add_column("Time/Gen", style="dim", justify="right")
    
    for r in results:
        last_10_mean = statistics.mean([g.mean_fitness for g in r.generations[-10:]])
        solved = f"Gen {r.solved_generation}" if r.solved_generation is not None else "❌"
        time_per_gen = r.total_time / len(r.generations)
        
        table.add_row(
            r.strategy_name,
            f"{r.final_best:.0f}",
            f"{last_10_mean:.1f}",
            solved,
            f"{r.total_time:.1f}s",
            f"{time_per_gen:.2f}s",
        )
    
    return table


def create_fitness_chart(results: List[BenchmarkResult], width: int = 60) -> str:
    """Create a simple ASCII fitness chart."""
    lines = []
    lines.append("📈 Fitness Over Generations")
    lines.append("─" * width)
    
    max_fitness = max(max(r.best_fitness_history) for r in results)
    min_fitness = min(min(r.mean_fitness_history) for r in results)
    
    for result in results:
        lines.append(f"\n[cyan]{result.strategy_name}[/cyan] (best):")
        
        # Sample points for display
        history = result.best_fitness_history
        n_points = min(len(history), width - 10)
        step = max(1, len(history) // n_points)
        sampled = history[::step][:n_points]
        
        # Create bar
        bar_chars = "▁▂▃▄▅▆▇█"
        bar = ""
        for val in sampled:
            normalized = (val - min_fitness) / (max_fitness - min_fitness + 1e-8)
            idx = min(int(normalized * len(bar_chars)), len(bar_chars) - 1)
            bar += bar_chars[idx]
        
        lines.append(f"  {bar}")
        lines.append(f"  [dim]{min_fitness:.0f} → {max_fitness:.0f}[/dim]")
    
    return "\n".join(lines)


def print_header():
    """Print benchmark header."""
    header = Text()
    header.append("🥚 ", style="bold yellow")
    header.append("EGGROLL", style="bold cyan")
    header.append(" vs ", style="dim")
    header.append("OpenES", style="bold green")
    header.append(" Benchmark", style="bold white")
    
    console.print(Panel(header, box=box.DOUBLE, padding=(1, 2)))


def print_config(args: argparse.Namespace, device: torch.device):
    """Print configuration panel."""
    config_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    config_table.add_column("Key", style="dim")
    config_table.add_column("Value", style="white")
    
    config_table.add_row("Environment", "CartPole-v1")
    config_table.add_row("Device", str(device))
    config_table.add_row("Generations", str(args.generations))
    config_table.add_row("Architecture", f"{args.hidden_size}x{args.n_layers} layers")
    config_table.add_row("Seed", str(args.seed))
    config_table.add_row("", "")
    config_table.add_row("[cyan]EGGROLL[/cyan]", "")
    config_table.add_row("  Population", str(args.eggroll_pop))
    config_table.add_row("  Sigma", str(args.eggroll_sigma))
    config_table.add_row("  LR", str(args.eggroll_lr))
    config_table.add_row("  Rank", str(args.rank))
    config_table.add_row("", "")
    config_table.add_row("[green]OpenES[/green]", "")
    config_table.add_row("  Population", str(args.openes_pop))
    config_table.add_row("  Sigma", str(args.openes_sigma))
    config_table.add_row("  LR", str(args.openes_lr))
    
    console.print(Panel(config_table, title="⚙️  Configuration (Paper Defaults)", box=box.ROUNDED))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EGGROLL vs OpenES CartPole Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # General args
    parser.add_argument("--generations", "-g", type=int, default=30, help="Number of generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick test with smaller populations (128/64)")
    
    # Use paper's hyperparameters as defaults
    # EGGROLL-specific (can be overridden)
    parser.add_argument("--eggroll-pop", type=int, default=2048, help="EGGROLL population size")
    parser.add_argument("--eggroll-sigma", type=float, default=0.2, help="EGGROLL sigma")
    parser.add_argument("--eggroll-lr", type=float, default=0.1, help="EGGROLL learning rate")
    parser.add_argument("--rank", type=int, default=4, help="EGGROLL rank")
    
    # OpenES-specific (can be overridden)
    parser.add_argument("--openes-pop", type=int, default=512, help="OpenES population size")
    parser.add_argument("--openes-sigma", type=float, default=0.5, help="OpenES sigma")
    parser.add_argument("--openes-lr", type=float, default=0.1, help="OpenES learning rate")
    
    # Shared architecture (paper uses 256x3 layers)
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of hidden layers")
    
    args = parser.parse_args()
    
    # Quick mode: use smaller populations for testing
    if args.quick:
        args.eggroll_pop = 128
        args.openes_pop = 64
        args.generations = min(args.generations, 10)
        console.print("[yellow]⚡ Quick mode: pop=128/64, gen≤10[/yellow]\n")
    
    # Check dependencies
    if not GYM_AVAILABLE:
        console.print("[red]Error:[/red] gymnasium not installed. Run: [cyan]uv add gymnasium[/cyan]")
        return 1
    
    # Check CUDA
    if not torch.cuda.is_available():
        console.print("[red]Error:[/red] CUDA not available. EGGROLL requires a GPU.")
        return 1
    
    device = torch.device("cuda")
    
    # Print header and config
    print_header()
    print_config(args, device)
    
    # Import strategies
    try:
        from hyperscalees.torch import EggrollStrategy, OpenESStrategy
    except ImportError as e:
        console.print(f"[red]Error importing strategies:[/red] {e}")
        console.print("Make sure hyperscalees is installed: [cyan]uv sync[/cyan]")
        return 1
    
    # Define strategies to benchmark (using paper's hyperparameters)
    strategies = [
        (
            f"EGGROLL (pop={args.eggroll_pop}, rank={args.rank})",
            EggrollStrategy,
            {
                "sigma": args.eggroll_sigma,
                "lr": args.eggroll_lr,
                "rank": args.rank,
                "seed": args.seed,
                "antithetic": True,
            },
            args.eggroll_pop,
        ),
        (
            f"OpenES (pop={args.openes_pop})",
            OpenESStrategy,
            {
                "sigma": args.openes_sigma,
                "lr": args.openes_lr,
                "seed": args.seed,
                "antithetic": True,
            },
            args.openes_pop,
        ),
    ]
    
    results = []
    
    # Run benchmarks with progress
    console.print("\n[bold]Running benchmarks...[/bold]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        
        for name, cls, kwargs, pop_size in strategies:
            task_id = progress.add_task(f"[cyan]{name}[/]", total=args.generations)
            
            result = run_benchmark(
                strategy_name=name,
                strategy_class=cls,
                strategy_kwargs=kwargs,
                population_size=pop_size,
                n_generations=args.generations,
                hidden_size=args.hidden_size,
                n_layers=args.n_layers,
                device=device,
                seed=args.seed,
                progress=progress,
                task_id=task_id,
            )
            
            results.append(result)
    
    # Display results
    console.print()
    console.print(create_comparison_table(results))
    console.print()
    console.print(Panel(create_fitness_chart(results), title="📊 Fitness Progress", box=box.ROUNDED))
    
    # Summary
    console.print()
    eggroll_result = results[0]
    openes_result = results[1]
    
    speedup = openes_result.total_time / eggroll_result.total_time
    
    if eggroll_result.final_best >= openes_result.final_best:
        winner = "[cyan]EGGROLL[/cyan]"
        margin = eggroll_result.final_best - openes_result.final_best
    else:
        winner = "[green]OpenES[/green]"
        margin = openes_result.final_best - eggroll_result.final_best
    
    summary = f"""
🏆 Winner (by final best): {winner} (+{margin:.0f})
⏱️  Time comparison: EGGROLL {eggroll_result.total_time:.1f}s vs OpenES {openes_result.total_time:.1f}s ({speedup:.2f}x)
"""
    
    if eggroll_result.solved_generation is not None and openes_result.solved_generation is not None:
        if eggroll_result.solved_generation <= openes_result.solved_generation:
            summary += f"🎯 First to solve: [cyan]EGGROLL[/cyan] (Gen {eggroll_result.solved_generation} vs {openes_result.solved_generation})"
        else:
            summary += f"🎯 First to solve: [green]OpenES[/green] (Gen {openes_result.solved_generation} vs {eggroll_result.solved_generation})"
    elif eggroll_result.solved_generation is not None:
        summary += f"🎯 Only [cyan]EGGROLL[/cyan] solved the environment (Gen {eggroll_result.solved_generation})"
    elif openes_result.solved_generation is not None:
        summary += f"🎯 Only [green]OpenES[/green] solved the environment (Gen {openes_result.solved_generation})"
    else:
        summary += "🎯 Neither strategy solved the environment (need ≥475 reward)"
    
    console.print(Panel(summary.strip(), title="📋 Summary", box=box.ROUNDED))
    
    return 0


if __name__ == "__main__":
    exit(main())
