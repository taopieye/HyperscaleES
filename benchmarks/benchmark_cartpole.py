#!/usr/bin/env python3
"""
Benchmark: Compare EGGROLL (Torch & JAX), PPO (Rejax), and OpenES (JAX)
on CartPole-v1.

This script replicates the experiments from the EGGROLL paper (Section G.1, Figure 9),
using hyperparameters from Tables 3 and 19.

Paper experiment scale: 5e8 (500 million) steps, 10 seeds
Hyperparameters from Tables:
- EGGROLL: Table 3 - pop_size=2048, rank=4, sigma=0.2, lr=0.1, sgd, lr_decay=0.9995
- OpenES: Table 3 - pop_size=512, sigma=0.5, lr=0.1, adamw, rank_transform=True
- PPO: Table 19 - num_envs=256, gamma=0.995, clip_eps=0.2, gae_lambda=0.9

Usage:
    uv run python benchmarks/benchmark_cartpole.py
    uv run python benchmarks/benchmark_cartpole.py --methods torch_eggroll ppo
    uv run python benchmarks/benchmark_cartpole.py --num-seeds 10 --max-steps 500000000
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sys

import numpy as np

# Check available frameworks
HAS_TORCH = False
HAS_JAX = False
HAS_REJAX = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    pass

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    pass

try:
    import rejax
    HAS_REJAX = True
except ImportError:
    pass

import gymnasium as gym

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Global benchmark configuration."""
    env_name: str = "CartPole-v1"
    max_steps: int = 100_000  # Total environment steps
    eval_freq: int = 10_000   # Evaluate every N steps
    num_eval_episodes: int = 10
    seed: int = 42
    

@dataclass
class EggrollHyperparams:
    """Hyperparameters for EGGROLL (from Table 3)."""
    pop_size: int = 2048
    n_parallel_evaluations: int = 1
    rank: int = 4
    optimizer: str = "sgd"
    learning_rate: float = 0.1
    lr_decay: float = 0.9995
    sigma: float = 0.2
    sigma_decay: float = 0.999
    rank_transform: bool = False
    deterministic_policy: bool = False
    layer_size: int = 256
    n_layers: int = 3
    activation: str = "pqn"  # Parameterized quasi-Newton (tanh in practice)


@dataclass 
class OpenESHyperparams:
    """Hyperparameters for OpenES (from Table 3)."""
    pop_size: int = 512
    n_parallel_evaluations: int = 4
    optimizer: str = "adamw"
    learning_rate: float = 0.1
    lr_decay: float = 0.9995
    sigma: float = 0.5
    sigma_decay: float = 0.9995
    rank_transform: bool = True
    deterministic_policy: bool = True
    layer_size: int = 256
    n_layers: int = 3
    activation: str = "pqn"


@dataclass
class PPOHyperparams:
    """Hyperparameters for PPO (from Table 19 of the paper for CartPole)."""
    clip_eps: float = 0.2
    ent_coef: float = 0.0001
    gae_lambda: float = 0.9
    gamma: float = 0.995
    learning_rate: float = 0.0003
    max_grad_norm: float = 0.5
    layer_size: int = 256
    n_layers: int = 3
    num_envs: int = 256  # Paper uses 256
    num_epochs: int = 4
    num_minibatches: int = 32
    num_steps: int = 128
    vf_coef: float = 0.5
    normalize_obs: bool = True
    normalize_rew: bool = False


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method: str
    seed: int
    final_return: float
    best_return: float
    steps_to_solve: Optional[int]  # Steps to reach 475 avg return (solved threshold)
    total_steps: int
    wall_time_sec: float
    eval_history: List[Dict[str, float]] = field(default_factory=list)
    hyperparams: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Network Architectures
# ============================================================================

def create_torch_mlp(
    obs_dim: int,
    action_dim: int,
    hidden_size: int = 256,
    n_layers: int = 3,
    activation: str = "tanh",
) -> nn.Module:
    """Create a PyTorch MLP policy network."""
    act_fn = nn.Tanh if activation in ("tanh", "pqn") else nn.ReLU
    
    layers = []
    in_dim = obs_dim
    for i in range(n_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(act_fn())
        in_dim = hidden_size
    layers.append(nn.Linear(in_dim, action_dim))
    
    return nn.Sequential(*layers)


# ============================================================================
# Torch EGGROLL Benchmark
# ============================================================================

def run_torch_eggroll(
    config: BenchmarkConfig,
    hyperparams: EggrollHyperparams,
    seed: int,
) -> BenchmarkResult:
    """Run Torch EGGROLL on CartPole."""
    from hyperscalees.torch import EggrollStrategy
    
    print(f"\n{'='*60}")
    print(f"Running Torch EGGROLL (seed={seed})")
    print(f"{'='*60}")
    
    # Create vectorized environment
    pop_size = hyperparams.pop_size
    envs = gym.make_vec(config.env_name, num_envs=pop_size, vectorization_mode="sync")
    
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n
    
    # Create policy
    torch.manual_seed(seed)
    policy = create_torch_mlp(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=hyperparams.layer_size,
        n_layers=hyperparams.n_layers,
        activation=hyperparams.activation,
    ).cuda()
    
    # Set up strategy
    strategy = EggrollStrategy(
        sigma=hyperparams.sigma,
        lr=hyperparams.learning_rate,
        rank=hyperparams.rank,
        optimizer=hyperparams.optimizer,
        antithetic=True,
        seed=seed,
        fitness_transform="normalize" if not hyperparams.rank_transform else "rank",
    )
    strategy.setup(policy)
    
    # Training loop
    start_time = time.time()
    total_steps = 0
    eval_history = []
    best_return = -float('inf')
    steps_to_solve = None
    epoch = 0
    
    current_sigma = hyperparams.sigma
    current_lr = hyperparams.learning_rate
    
    while total_steps < config.max_steps:
        # Evaluate population
        obs, _ = envs.reset(seed=seed + epoch)
        episode_returns = np.zeros(pop_size)
        dones = np.zeros(pop_size, dtype=bool)
        
        with strategy.perturb(population_size=pop_size, epoch=epoch) as pop:
            while not dones.all():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device='cuda')
                
                with torch.no_grad():
                    action_logits = pop.batched_forward(policy, obs_tensor)
                
                if hyperparams.deterministic_policy:
                    actions = action_logits.argmax(dim=-1).cpu().numpy()
                else:
                    probs = torch.softmax(action_logits, dim=-1)
                    actions = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()
                
                obs, rewards, terminated, truncated, _ = envs.step(actions)
                episode_returns += rewards * (~dones)
                dones = dones | terminated | truncated
        
        # Count steps (approximate - one episode per member)
        total_steps += episode_returns.sum()  # Each reward of 1 = 1 step
        
        # Convert to fitness tensor
        fitnesses = torch.tensor(episode_returns, device='cuda', dtype=torch.float32)
        
        # Update
        metrics = strategy.step(fitnesses, epoch=epoch)
        
        # Decay sigma and lr
        current_sigma *= hyperparams.sigma_decay
        current_lr *= hyperparams.lr_decay
        strategy.sigma = current_sigma
        for param_group in strategy._optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Log progress
        mean_return = episode_returns.mean()
        max_return = episode_returns.max()
        
        if max_return > best_return:
            best_return = max_return
        
        if mean_return >= 475 and steps_to_solve is None:
            steps_to_solve = total_steps
        
        eval_history.append({
            "epoch": epoch,
            "total_steps": int(total_steps),
            "mean_return": float(mean_return),
            "max_return": float(max_return),
            "sigma": current_sigma,
            "lr": current_lr,
        })
        
        print(f"  Epoch {epoch}: steps={total_steps:.0f}, mean={mean_return:.1f}, max={max_return:.1f}")
        
        epoch += 1
    
    envs.close()
    wall_time = time.time() - start_time
    
    return BenchmarkResult(
        method="torch_eggroll",
        seed=seed,
        final_return=float(episode_returns.mean()),
        best_return=float(best_return),
        steps_to_solve=steps_to_solve,
        total_steps=int(total_steps),
        wall_time_sec=wall_time,
        eval_history=eval_history,
        hyperparams=asdict(hyperparams),
    )


# ============================================================================
# JAX EGGROLL Benchmark  
# ============================================================================

def run_jax_eggroll(
    config: BenchmarkConfig,
    hyperparams: EggrollHyperparams,
    seed: int,
) -> BenchmarkResult:
    """Run JAX EGGROLL on CartPole."""
    import jax
    import jax.numpy as jnp
    import optax
    from functools import partial
    
    from hyperscalees.noiser.eggroll import EggRoll
    from hyperscalees.models.common import MLP, simple_es_tree_key
    
    print(f"\n{'='*60}")
    print(f"Running JAX EGGROLL (seed={seed})")
    print(f"{'='*60}")
    
    # Create vectorized environment (using gym for fair comparison)
    pop_size = hyperparams.pop_size
    envs = gym.make_vec(config.env_name, num_envs=pop_size, vectorization_mode="sync")
    
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n
    
    # Initialize model
    key = jax.random.key(seed)
    model_key = jax.random.fold_in(key, 0)
    es_key = jax.random.fold_in(key, 1)
    
    # Map optimizer name
    optimizer_map = {
        "sgd": optax.sgd,
        "adam": optax.adam,
        "adamw": optax.adamw,
    }
    solver = optimizer_map.get(hyperparams.optimizer.lower(), optax.sgd)
    
    frozen_params, params, scan_map, es_map = MLP.rand_init(
        model_key,
        in_dim=obs_dim,
        out_dim=action_dim,
        hidden_dims=[hyperparams.layer_size] * (hyperparams.n_layers - 1),
        use_bias=True,
        activation=hyperparams.activation,  # "pqn" is a valid activation in the codebase
        dtype="float32",
    )
    
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)
    
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params,
        sigma=hyperparams.sigma,
        lr=hyperparams.learning_rate,
        solver=solver,
        rank=hyperparams.rank,
        noise_reuse=0,
    )
    
    # JIT-compile forward functions
    @jax.jit
    def forward_eval(params, obs):
        """Forward without noise (for evaluation)."""
        return MLP.forward(
            EggRoll, frozen_noiser_params, noiser_params, frozen_params,
            params, es_tree_key, None, obs
        )
    
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
    
    @jax.jit
    def do_update(noiser_params, params, fitnesses, iterinfos):
        return EggRoll.do_updates(
            frozen_noiser_params, noiser_params, params,
            es_tree_key, fitnesses, iterinfos, es_map
        )
    
    # Training loop
    start_time = time.time()
    total_steps = 0
    eval_history = []
    best_return = -float('inf')
    steps_to_solve = None
    epoch = 0
    
    current_sigma = hyperparams.sigma
    
    while total_steps < config.max_steps:
        # Reset environments
        obs, _ = envs.reset(seed=seed + epoch)
        episode_returns = np.zeros(pop_size)
        dones = np.zeros(pop_size, dtype=bool)
        
        # Create iterinfo for this epoch
        iterinfo = (jnp.full(pop_size, epoch, dtype=jnp.int32), jnp.arange(pop_size))
        
        while not dones.all():
            # Get actions from noisy policies
            obs_jax = jnp.array(obs)
            action_logits = jit_forward(noiser_params, params, iterinfo, obs_jax)
            
            if hyperparams.deterministic_policy:
                actions = np.array(jnp.argmax(action_logits, axis=-1))
            else:
                probs = jax.nn.softmax(action_logits, axis=-1)
                # Sample actions
                key, sample_key = jax.random.split(key)
                actions = np.array(jax.random.categorical(sample_key, jnp.log(probs + 1e-10)))
            
            obs, rewards, terminated, truncated, _ = envs.step(actions)
            episode_returns += rewards * (~dones)
            dones = dones | terminated | truncated
        
        total_steps += episode_returns.sum()
        
        # Convert fitnesses
        raw_scores = jnp.array(episode_returns)
        fitnesses = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
        
        # Update
        noiser_params, params = do_update(noiser_params, params, fitnesses, iterinfo)
        
        # Decay sigma
        current_sigma *= hyperparams.sigma_decay
        noiser_params = {**noiser_params, "sigma": current_sigma}
        
        # Log
        mean_return = float(episode_returns.mean())
        max_return = float(episode_returns.max())
        
        if max_return > best_return:
            best_return = max_return
        
        if mean_return >= 475 and steps_to_solve is None:
            steps_to_solve = int(total_steps)
        
        eval_history.append({
            "epoch": epoch,
            "total_steps": int(total_steps),
            "mean_return": mean_return,
            "max_return": max_return,
        })
        
        print(f"  Epoch {epoch}: steps={total_steps:.0f}, mean={mean_return:.1f}, max={max_return:.1f}")
        
        epoch += 1
    
    envs.close()
    wall_time = time.time() - start_time
    
    return BenchmarkResult(
        method="jax_eggroll",
        seed=seed,
        final_return=float(episode_returns.mean()),
        best_return=float(best_return),
        steps_to_solve=steps_to_solve,
        total_steps=int(total_steps),
        wall_time_sec=wall_time,
        eval_history=eval_history,
        hyperparams=asdict(hyperparams),
    )


# ============================================================================
# PPO Benchmark (Rejax - GPU-optimized JAX RL)
# ============================================================================

def run_ppo(
    config: BenchmarkConfig,
    hyperparams: PPOHyperparams,
    seed: int,
) -> BenchmarkResult:
    """Run PPO using Rejax (GPU-optimized JAX implementation, as used in the paper)."""
    import jax
    import jax.numpy as jnp
    from rejax import PPO
    
    print(f"\n{'='*60}")
    print(f"Running PPO / Rejax (seed={seed})")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Configure PPO with paper hyperparameters (Table 19)
    ppo = PPO.create(
        env="CartPole-v1",
        num_envs=hyperparams.num_envs,
        num_steps=hyperparams.num_steps,
        num_epochs=hyperparams.num_epochs,
        num_minibatches=hyperparams.num_minibatches,
        learning_rate=hyperparams.learning_rate,
        max_grad_norm=hyperparams.max_grad_norm,
        gamma=hyperparams.gamma,
        gae_lambda=hyperparams.gae_lambda,
        clip_eps=hyperparams.clip_eps,
        ent_coef=hyperparams.ent_coef,
        vf_coef=hyperparams.vf_coef,
        normalize_observations=hyperparams.normalize_obs,
        normalize_rewards=hyperparams.normalize_rew,
        total_timesteps=config.max_steps,
        eval_freq=config.eval_freq,
        skip_initial_evaluation=False,
    )
    
    # Train
    key = jax.random.key(seed)
    train_state, eval_state = jax.jit(ppo.train)(key)
    
    wall_time = time.time() - start_time
    
    # eval_state is a tuple of (episode_lengths, episode_returns)
    # Shape: (num_evals, num_envs)
    episode_lengths, episode_returns = eval_state
    episode_returns = np.array(episode_returns)
    
    # Build eval history
    eval_history = []
    best_return = -float('inf')
    steps_to_solve = None
    
    # Calculate steps per eval checkpoint
    steps_per_eval = config.eval_freq
    
    for i in range(len(episode_returns)):
        current_steps = (i + 1) * steps_per_eval
        mean_return = float(np.mean(episode_returns[i]))
        max_return = float(np.max(episode_returns[i]))
        
        if mean_return > best_return:
            best_return = mean_return
        
        if mean_return >= 475 and steps_to_solve is None:
            steps_to_solve = current_steps
        
        eval_history.append({
            "total_steps": current_steps,
            "mean_return": mean_return,
            "max_return": max_return,
        })
        
        print(f"  Step {current_steps}: mean={mean_return:.1f}, max={max_return:.1f}")
    
    # Final return is the last evaluation
    final_return = float(np.mean(episode_returns[-1])) if len(episode_returns) > 0 else 0.0
    
    return BenchmarkResult(
        method="ppo",
        seed=seed,
        final_return=final_return,
        best_return=best_return,
        steps_to_solve=steps_to_solve,
        total_steps=config.max_steps,
        wall_time_sec=wall_time,
        eval_history=eval_history,
        hyperparams=asdict(hyperparams),
    )


# ============================================================================
# OpenES Benchmark (in-repo JAX implementation)
# ============================================================================

def run_openes(
    config: BenchmarkConfig,
    hyperparams: OpenESHyperparams,
    seed: int,
) -> BenchmarkResult:
    """Run OpenES (in-repo JAX implementation) on CartPole."""
    import jax
    import jax.numpy as jnp
    import optax
    from functools import partial
    
    from hyperscalees.noiser.open_es import OpenES
    from hyperscalees.models.common import MLP, simple_es_tree_key
    
    print(f"\n{'='*60}")
    print(f"Running OpenES / JAX (seed={seed})")
    print(f"{'='*60}")
    
    # Create vectorized environment
    pop_size = hyperparams.pop_size
    envs = gym.make_vec(config.env_name, num_envs=pop_size, vectorization_mode="sync")
    
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n
    
    # Initialize model
    key = jax.random.key(seed)
    model_key = jax.random.fold_in(key, 0)
    es_key = jax.random.fold_in(key, 1)
    
    # Map optimizer name
    optimizer_map = {
        "sgd": optax.sgd,
        "adam": optax.adam,
        "adamw": optax.adamw,
    }
    solver = optimizer_map.get(hyperparams.optimizer.lower(), optax.adamw)
    
    frozen_params, params, scan_map, es_map = MLP.rand_init(
        model_key,
        in_dim=obs_dim,
        out_dim=action_dim,
        hidden_dims=[hyperparams.layer_size] * (hyperparams.n_layers - 1),
        use_bias=True,
        activation=hyperparams.activation,  # "pqn" is a valid activation in the codebase
        dtype="float32",
    )
    
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)
    
    frozen_noiser_params, noiser_params = OpenES.init_noiser(
        params,
        sigma=hyperparams.sigma,
        lr=hyperparams.learning_rate,
        solver=solver,
        noise_reuse=0,
    )
    
    # JIT-compile forward functions
    def forward_noisy(noiser_params, params, iterinfo, obs):
        """Forward with noise."""
        return MLP.forward(
            OpenES, frozen_noiser_params, noiser_params, frozen_params,
            params, es_tree_key, iterinfo, obs
        )
    
    jit_forward = jax.jit(jax.vmap(
        lambda n, p, i, x: forward_noisy(n, p, i, x),
        in_axes=(None, None, 0, 0)
    ))
    
    @jax.jit
    def do_update(noiser_params, params, fitnesses, iterinfos):
        return OpenES.do_updates(
            frozen_noiser_params, noiser_params, params,
            es_tree_key, fitnesses, iterinfos, es_map
        )
    
    # Training loop
    start_time = time.time()
    total_steps = 0
    eval_history = []
    best_return = -float('inf')
    steps_to_solve = None
    epoch = 0
    
    current_sigma = hyperparams.sigma
    
    while total_steps < config.max_steps:
        # Reset environments
        obs, _ = envs.reset(seed=seed + epoch)
        episode_returns = np.zeros(pop_size)
        dones = np.zeros(pop_size, dtype=bool)
        
        # Create iterinfo for this epoch
        iterinfo = (jnp.full(pop_size, epoch, dtype=jnp.int32), jnp.arange(pop_size))
        
        while not dones.all():
            # Get actions from noisy policies
            obs_jax = jnp.array(obs)
            action_logits = jit_forward(noiser_params, params, iterinfo, obs_jax)
            
            if hyperparams.deterministic_policy:
                actions = np.array(jnp.argmax(action_logits, axis=-1))
            else:
                probs = jax.nn.softmax(action_logits, axis=-1)
                # Sample actions
                key, sample_key = jax.random.split(key)
                actions = np.array(jax.random.categorical(sample_key, jnp.log(probs + 1e-10)))
            
            obs, rewards, terminated, truncated, _ = envs.step(actions)
            episode_returns += rewards * (~dones)
            dones = dones | terminated | truncated
        
        total_steps += episode_returns.sum()
        
        # Convert fitnesses (rank transform if enabled)
        raw_scores = jnp.array(episode_returns)
        if hyperparams.rank_transform:
            # Rank-based fitness shaping
            ranks = jnp.argsort(jnp.argsort(raw_scores))
            fitnesses = (ranks - ranks.mean()) / (ranks.std() + 1e-8)
        else:
            fitnesses = OpenES.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
        
        # Update
        noiser_params, params = do_update(noiser_params, params, fitnesses, iterinfo)
        
        # Decay sigma
        current_sigma *= hyperparams.sigma_decay
        noiser_params = {**noiser_params, "sigma": current_sigma}
        
        # Log
        mean_return = float(episode_returns.mean())
        max_return = float(episode_returns.max())
        
        if max_return > best_return:
            best_return = max_return
        
        if mean_return >= 475 and steps_to_solve is None:
            steps_to_solve = int(total_steps)
        
        eval_history.append({
            "epoch": epoch,
            "total_steps": int(total_steps),
            "mean_return": mean_return,
            "max_return": max_return,
        })
        
        print(f"  Epoch {epoch}: steps={total_steps:.0f}, mean={mean_return:.1f}, max={max_return:.1f}")
        
        epoch += 1
    
    envs.close()
    wall_time = time.time() - start_time
    
    return BenchmarkResult(
        method="openes",
        seed=seed,
        final_return=float(episode_returns.mean()),
        best_return=float(best_return),
        steps_to_solve=steps_to_solve,
        total_steps=int(total_steps),
        wall_time_sec=wall_time,
        eval_history=eval_history,
        hyperparams=asdict(hyperparams),
    )


# ============================================================================
# Main
# ============================================================================

def print_availability():
    """Print which methods are available."""
    print("\n" + "=" * 60)
    print("CartPole-v1 Benchmark: EGGROLL vs PPO vs OpenES")
    print("=" * 60)
    print("\nMethod availability:")
    print(f"  Torch EGGROLL: {'✓' if HAS_TORCH else '✗ (no CUDA)'}")
    print(f"  JAX EGGROLL:   {'✓' if HAS_JAX else '✗ (no JAX)'}")
    print(f"  PPO (Rejax):   {'✓' if HAS_REJAX else '✗ (install rejax)'}")
    print(f"  OpenES (JAX):  {'✓' if HAS_JAX else '✗ (no JAX)'}")


def run_all_benchmarks(
    methods: List[str],
    seeds: List[int],
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Run benchmarks for all specified methods and seeds."""
    results = []
    
    for method in methods:
        for seed in seeds:
            try:
                if method == "torch_eggroll" and HAS_TORCH:
                    result = run_torch_eggroll(config, EggrollHyperparams(), seed)
                    results.append(result)
                elif method == "jax_eggroll" and HAS_JAX:
                    result = run_jax_eggroll(config, EggrollHyperparams(), seed)
                    results.append(result)
                elif method == "ppo" and HAS_REJAX:
                    result = run_ppo(config, PPOHyperparams(), seed)
                    results.append(result)
                elif method == "openes" and HAS_JAX:
                    result = run_openes(config, OpenESHyperparams(), seed)
                    results.append(result)
                else:
                    print(f"\nSkipping {method} (not available)")
            except Exception as e:
                print(f"\nError running {method} (seed={seed}): {e}")
                import traceback
                traceback.print_exc()
    
    return results


def generate_plots(results: List[BenchmarkResult], config: BenchmarkConfig, output_dir: Path):
    """Generate matplotlib plots similar to the paper's Figure 9."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Group results by method
    methods = sorted(set(r.method for r in results))
    method_data = {m: [r for r in results if r.method == m] for m in methods}
    
    # Method display names and colors (matching paper style)
    method_config = {
        "torch_eggroll": {"name": "EGGROLL (Torch)", "color": "#E74C3C", "linestyle": "-"},
        "jax_eggroll": {"name": "EGGROLL (JAX)", "color": "#C0392B", "linestyle": "--"},
        "ppo": {"name": "PPO", "color": "#3498DB", "linestyle": "-"},
        "openes": {"name": "OpenES", "color": "#2ECC71", "linestyle": "-"},
    }
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 11
    
    # ========== Figure 1: Learning Curves ==========
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    for method in methods:
        method_results = method_data[method]
        cfg = method_config.get(method, {"name": method, "color": "gray", "linestyle": "-"})
        
        # Aggregate learning curves across seeds
        all_steps = []
        all_returns = []
        
        for r in method_results:
            if r.eval_history:
                steps = [h["total_steps"] for h in r.eval_history]
                returns = [h["mean_return"] for h in r.eval_history]
                all_steps.append(steps)
                all_returns.append(returns)
        
        if all_returns:
            # Interpolate to common x-axis
            max_steps = max(max(s) for s in all_steps)
            common_steps = np.linspace(0, max_steps, 100)
            
            interpolated_returns = []
            for steps, returns in zip(all_steps, all_returns):
                interp = np.interp(common_steps, steps, returns)
                interpolated_returns.append(interp)
            
            mean_returns = np.mean(interpolated_returns, axis=0)
            std_returns = np.std(interpolated_returns, axis=0)
            
            # Plot mean with shaded std
            ax1.plot(common_steps, mean_returns, label=cfg["name"], 
                    color=cfg["color"], linestyle=cfg["linestyle"], linewidth=2)
            ax1.fill_between(common_steps, mean_returns - std_returns, mean_returns + std_returns,
                           alpha=0.2, color=cfg["color"])
    
    ax1.axhline(y=475, color='gray', linestyle=':', alpha=0.7, label='Solved (475)')
    ax1.set_xlabel('Environment Steps')
    ax1.set_ylabel('Mean Return')
    ax1.set_title(f'{config.env_name} - Learning Curves')
    ax1.legend(loc='lower right')
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    fig1.tight_layout()
    fig1.savefig(output_dir / 'learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # ========== Figure 2: Wall-Clock Time Comparison ==========
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    
    method_names = []
    mean_times = []
    std_times = []
    colors = []
    
    for method in methods:
        method_results = method_data[method]
        cfg = method_config.get(method, {"name": method, "color": "gray"})
        
        times = [r.wall_time_sec for r in method_results]
        method_names.append(cfg["name"])
        mean_times.append(np.mean(times))
        std_times.append(np.std(times))
        colors.append(cfg["color"])
    
    bars = ax2.barh(method_names, mean_times, xerr=std_times, color=colors, alpha=0.8, capsize=5)
    ax2.set_xlabel('Wall-Clock Time (seconds)')
    ax2.set_title(f'{config.env_name} - Training Time Comparison')
    
    # Add time labels on bars
    for bar, t in zip(bars, mean_times):
        ax2.text(bar.get_width() + max(mean_times) * 0.02, bar.get_y() + bar.get_height()/2,
                f'{t:.1f}s', va='center', fontsize=10)
    
    ax2.set_xlim(right=max(mean_times) * 1.2)
    fig2.tight_layout()
    fig2.savefig(output_dir / 'training_time.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # ========== Figure 3: Speedup vs OpenES ==========
    if "openes" in method_data:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        
        openes_time = np.mean([r.wall_time_sec for r in method_data["openes"]])
        
        speedup_names = []
        speedups = []
        speedup_colors = []
        
        for method in methods:
            if method != "openes":
                method_results = method_data[method]
                cfg = method_config.get(method, {"name": method, "color": "gray"})
                
                mean_time = np.mean([r.wall_time_sec for r in method_results])
                speedup = openes_time / mean_time
                
                speedup_names.append(cfg["name"])
                speedups.append(speedup)
                speedup_colors.append(cfg["color"])
        
        bars = ax3.barh(speedup_names, speedups, color=speedup_colors, alpha=0.8)
        ax3.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='OpenES baseline')
        ax3.set_xlabel('Speedup vs OpenES')
        ax3.set_title(f'{config.env_name} - Relative Training Speed')
        
        # Add speedup labels on bars
        for bar, s in zip(bars, speedups):
            ax3.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{s:.2f}x', va='center', fontsize=10)
        
        ax3.set_xlim(right=max(speedups) * 1.2 if speedups else 2)
        fig3.tight_layout()
        fig3.savefig(output_dir / 'speedup.png', dpi=150, bbox_inches='tight')
        plt.close(fig3)
    
    # ========== Figure 4: Final Return Box Plot ==========
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    
    box_data = []
    box_labels = []
    box_colors = []
    
    for method in methods:
        method_results = method_data[method]
        cfg = method_config.get(method, {"name": method, "color": "gray"})
        
        final_returns = [r.final_return for r in method_results]
        box_data.append(final_returns)
        box_labels.append(cfg["name"])
        box_colors.append(cfg["color"])
    
    bp = ax4.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.axhline(y=475, color='gray', linestyle=':', alpha=0.7, label='Solved (475)')
    ax4.set_ylabel('Final Return')
    ax4.set_title(f'{config.env_name} - Final Return Distribution')
    ax4.legend()
    
    fig4.tight_layout()
    fig4.savefig(output_dir / 'final_returns.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    print(f"  Plots saved to: {output_dir}")
    return [
        'learning_curves.png',
        'training_time.png', 
        'speedup.png' if "openes" in method_data else None,
        'final_returns.png'
    ]


def generate_markdown_report(results: List[BenchmarkResult], config: BenchmarkConfig, plot_files: List[str] = None) -> str:
    """Generate a markdown report similar to the paper's format."""
    
    # Group results by method
    methods = sorted(set(r.method for r in results))
    method_data = {m: [r for r in results if r.method == m] for m in methods}
    
    # Method display names
    method_names = {
        "torch_eggroll": "EGGROLL (Torch)",
        "jax_eggroll": "EGGROLL (JAX)",
        "ppo": "PPO (Rejax)",
        "openes": "OpenES (JAX)",
    }
    
    report = []
    report.append("# CartPole-v1 Benchmark Results")
    report.append("")
    report.append("## Overview")
    report.append("")
    report.append("This benchmark replicates the **CartPole-v1** experiments from the EGGROLL paper")
    report.append("([Section G.1, Figure 9](https://arxiv.org/abs/...)). We compare:")
    report.append("")
    report.append("- **EGGROLL** - Low-rank Evolution Strategies with efficient gradient estimation")
    report.append("- **PPO** - Proximal Policy Optimization (gradient-based RL baseline)")
    report.append("- **OpenES** - Full-rank Evolution Strategies (ES baseline)")
    report.append("")
    report.append("### Why This Matters")
    report.append("")
    report.append("Evolution Strategies (ES) are an attractive alternative to gradient-based RL because they:")
    report.append("- Don't require backpropagation through time")
    report.append("- Are embarrassingly parallel across population members")
    report.append("- Can handle non-differentiable objectives and sparse rewards")
    report.append("")
    report.append("However, standard ES (like OpenES) scales poorly with parameter count due to full-rank")
    report.append("perturbations. **EGGROLL** addresses this by using low-rank perturbations, achieving")
    report.append("similar sample efficiency with dramatically reduced compute.")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## Experiment Configuration")
    report.append("")
    report.append(f"| Setting | Value |")
    report.append(f"|---------|-------|")
    report.append(f"| Environment | `{config.env_name}` |")
    report.append(f"| Max Steps | {config.max_steps:,} |")
    report.append(f"| Seeds | {len(results) // len(methods)} per method |")
    report.append(f"| Date | {time.strftime('%Y-%m-%d %H:%M:%S')} |")
    report.append("")
    
    # ========== Learning Curves Plot ==========
    if plot_files and 'learning_curves.png' in plot_files:
        report.append("## Learning Curves")
        report.append("")
        report.append("![Learning Curves](learning_curves.png)")
        report.append("")
        report.append("*Mean return over environment steps. Shaded regions show ±1 standard deviation across seeds.*")
        report.append("*Dashed line at 475 indicates the 'solved' threshold for CartPole-v1.*")
        report.append("")
    
    # ========== Summary Table ==========
    report.append("## Summary Table")
    report.append("")
    report.append("| Method | Final Return | Best Return | Wall Time (s) | Steps to Solve | Speedup vs OpenES |")
    report.append("|--------|--------------|-------------|---------------|----------------|-------------------|")
    
    # Calculate OpenES baseline time for speedup comparison
    openes_time = None
    if "openes" in method_data:
        openes_times = [r.wall_time_sec for r in method_data["openes"]]
        openes_time = np.mean(openes_times)
    
    for method in methods:
        method_results = method_data[method]
        
        final_returns = [r.final_return for r in method_results]
        best_returns = [r.best_return for r in method_results]
        wall_times = [r.wall_time_sec for r in method_results]
        solve_steps = [r.steps_to_solve for r in method_results if r.steps_to_solve is not None]
        
        final_str = f"{np.mean(final_returns):.1f} ± {np.std(final_returns):.1f}"
        best_str = f"{np.mean(best_returns):.1f} ± {np.std(best_returns):.1f}"
        time_str = f"{np.mean(wall_times):.1f} ± {np.std(wall_times):.1f}"
        
        if solve_steps:
            solve_str = f"{np.mean(solve_steps):,.0f} ± {np.std(solve_steps):,.0f}"
        else:
            solve_str = "—"
        
        # Calculate speedup vs OpenES
        if openes_time and method != "openes":
            speedup = openes_time / np.mean(wall_times)
            speedup_str = f"**{speedup:.2f}x**" if speedup > 1 else f"{speedup:.2f}x"
        elif method == "openes":
            speedup_str = "1.00x (baseline)"
        else:
            speedup_str = "—"
        
        name = method_names.get(method, method)
        report.append(f"| {name} | {final_str} | {best_str} | {time_str} | {solve_str} | {speedup_str} |")
    
    report.append("")
    
    # ========== Training Time Plot ==========
    if plot_files and 'training_time.png' in plot_files:
        report.append("## Training Time Comparison")
        report.append("")
        report.append("![Training Time](training_time.png)")
        report.append("")
    
    # ========== Speedup Plot ==========
    if plot_files and 'speedup.png' in plot_files:
        report.append("## Speedup vs OpenES")
        report.append("")
        report.append("![Speedup](speedup.png)")
        report.append("")
        report.append("*EGGROLL's low-rank perturbations require less compute than OpenES's full-rank perturbations,*")
        report.append("*while PPO benefits from efficient GPU parallelization of gradient computation.*")
        report.append("")
    
    # ========== Final Returns Plot ==========
    if plot_files and 'final_returns.png' in plot_files:
        report.append("## Final Return Distribution")
        report.append("")
        report.append("![Final Returns](final_returns.png)")
        report.append("")
    
    # ========== Key Findings ==========
    report.append("## Key Findings")
    report.append("")
    
    # Find best method
    best_final = max(methods, key=lambda m: np.mean([r.final_return for r in method_data[m]]))
    best_time = min(methods, key=lambda m: np.mean([r.wall_time_sec for r in method_data[m]]))
    
    report.append(f"1. **Best Final Return:** {method_names.get(best_final, best_final)}")
    report.append(f"2. **Fastest Training:** {method_names.get(best_time, best_time)}")
    
    # Check if any method solved the environment
    solved_methods = []
    for method in methods:
        solve_steps = [r.steps_to_solve for r in method_data[method] if r.steps_to_solve is not None]
        if solve_steps:
            solved_methods.append((method, np.mean(solve_steps)))
    
    if solved_methods:
        fastest_solver = min(solved_methods, key=lambda x: x[1])
        report.append(f"3. **Fastest to Solve (≥475):** {method_names.get(fastest_solver[0], fastest_solver[0])} at {fastest_solver[1]:,.0f} steps")
    else:
        report.append("3. **Solved Environment:** None reached the 475 threshold")
    
    # EGGROLL vs OpenES comparison
    if "openes" in method_data and ("torch_eggroll" in method_data or "jax_eggroll" in method_data):
        eggroll_method = "torch_eggroll" if "torch_eggroll" in method_data else "jax_eggroll"
        eggroll_time = np.mean([r.wall_time_sec for r in method_data[eggroll_method]])
        openes_time = np.mean([r.wall_time_sec for r in method_data["openes"]])
        eggroll_return = np.mean([r.final_return for r in method_data[eggroll_method]])
        openes_return = np.mean([r.final_return for r in method_data["openes"]])
        
        report.append("")
        report.append("### EGGROLL vs OpenES")
        report.append("")
        if eggroll_time < openes_time:
            report.append(f"- EGGROLL is **{openes_time/eggroll_time:.2f}x faster** than OpenES")
        else:
            report.append(f"- OpenES is **{eggroll_time/openes_time:.2f}x faster** than EGGROLL")
        
        if eggroll_return > openes_return:
            report.append(f"- EGGROLL achieves **{eggroll_return - openes_return:.1f} higher** final return")
        else:
            report.append(f"- OpenES achieves **{openes_return - eggroll_return:.1f} higher** final return")
    
    report.append("")
    
    # ========== Hyperparameters ==========
    report.append("## Hyperparameters")
    report.append("")
    report.append("All hyperparameters match those in Tables 3 and 19 of the EGGROLL paper.")
    report.append("")
    
    for method in methods:
        method_results = method_data[method]
        if method_results:
            name = method_names.get(method, method)
            hyperparams = method_results[0].hyperparams
            
            report.append(f"<details>")
            report.append(f"<summary><b>{name}</b></summary>")
            report.append("")
            report.append("| Parameter | Value |")
            report.append("|-----------|-------|")
            
            for key, value in sorted(hyperparams.items()):
                report.append(f"| `{key}` | `{value}` |")
            
            report.append("")
            report.append("</details>")
            report.append("")
    
    # ========== Individual Run Details ==========
    report.append("## Individual Run Details")
    report.append("")
    report.append("<details>")
    report.append("<summary>Click to expand per-seed results</summary>")
    report.append("")
    report.append("| Method | Seed | Final Return | Best Return | Wall Time (s) | Steps to Solve |")
    report.append("|--------|------|--------------|-------------|---------------|----------------|")
    
    for r in sorted(results, key=lambda x: (x.method, x.seed)):
        name = method_names.get(r.method, r.method)
        solve_str = f"{r.steps_to_solve:,}" if r.steps_to_solve else "—"
        report.append(f"| {name} | {r.seed} | {r.final_return:.1f} | {r.best_return:.1f} | {r.wall_time_sec:.1f} | {solve_str} |")
    
    report.append("")
    report.append("</details>")
    report.append("")
    
    # ========== Reproducibility ==========
    report.append("---")
    report.append("")
    report.append("## Reproducing These Results")
    report.append("")
    report.append("```bash")
    report.append("# Install dependencies")
    report.append("uv sync")
    report.append("")
    report.append("# Run the benchmark")
    report.append(f"uv run python benchmarks/benchmark_cartpole.py \\")
    report.append(f"    --methods {' '.join(methods)} \\")
    report.append(f"    --num-seeds {len(results) // len(methods)} \\")
    report.append(f"    --max-steps {config.max_steps}")
    report.append("```")
    report.append("")
    
    return "\n".join(report)


def print_summary(results: List[BenchmarkResult], config: BenchmarkConfig = None):
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Group by method
    methods = sorted(set(r.method for r in results))
    
    for method in methods:
        method_results = [r for r in results if r.method == method]
        
        final_returns = [r.final_return for r in method_results]
        best_returns = [r.best_return for r in method_results]
        wall_times = [r.wall_time_sec for r in method_results]
        solve_steps = [r.steps_to_solve for r in method_results if r.steps_to_solve is not None]
        
        print(f"\n{method.upper()}")
        print("-" * 40)
        print(f"  Final Return:  {np.mean(final_returns):.1f} ± {np.std(final_returns):.1f}")
        print(f"  Best Return:   {np.mean(best_returns):.1f} ± {np.std(best_returns):.1f}")
        print(f"  Wall Time:     {np.mean(wall_times):.1f}s ± {np.std(wall_times):.1f}s")
        if solve_steps:
            print(f"  Steps to Solve: {np.mean(solve_steps):.0f} ± {np.std(solve_steps):.0f}")
        else:
            print(f"  Steps to Solve: — (never reached 475)")


def save_results(results: List[BenchmarkResult], output_path: Path, config: BenchmarkConfig = None):
    """Save results to JSON, generate plots, and markdown report."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [asdict(r) for r in results],
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Generate plots and markdown report
    if config:
        output_dir = output_path.parent
        
        # Generate plots
        print("\nGenerating plots...")
        try:
            plot_files = generate_plots(results, config, output_dir)
        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")
            plot_files = []
        
        # Generate markdown report with plot references
        md_report = generate_markdown_report(results, config, plot_files)
        md_path = output_path.with_suffix(".md")
        with open(md_path, "w") as f:
            f.write(md_report)
        print(f"Markdown report saved to: {md_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Benchmark EGGROLL, PPO, and OpenES on CartPole-v1"
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["torch_eggroll", "jax_eggroll", "ppo", "openes"],
        choices=["torch_eggroll", "jax_eggroll", "ppo", "openes"],
        help="Methods to benchmark"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=3,
        help="Number of random seeds to run"
    )
    parser.add_argument(
        "--max-steps", type=int, default=100_000,
        help="Maximum environment steps per run"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=10_000,
        help="Evaluation frequency (steps)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed"
    )
    args = parser.parse_args()
    
    print_availability()
    
    config = BenchmarkConfig(
        max_steps=args.max_steps,
        eval_freq=args.eval_freq,
        seed=args.seed,
    )
    
    seeds = [args.seed + i * 100 for i in range(args.num_seeds)]
    
    print(f"\nConfiguration:")
    print(f"  Environment: {config.env_name}")
    print(f"  Max steps: {config.max_steps:,}")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds: {seeds}")
    
    # Run benchmarks
    results = run_all_benchmarks(args.methods, seeds, config)
    
    # Print summary
    if results:
        print_summary(results, config)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / f"cartpole_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    if results:
        save_results(results, output_path, config)


if __name__ == "__main__":
    main()

