"""
Test: Reinforcement Learning environment integration.

TARGET API: The strategy should integrate seamlessly with various RL environment
types, with batched evaluation as the primary pattern.

Key patterns tested:
1. Batched forward pass (batched_forward) — the main API
2. Vectorized environments (gym.vector)
3. Custom batched environments
4. Sequential evaluation (for debugging/legacy envs)
5. Multiple episodes per member
"""
import pytest
import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass

from .conftest import (
    EggrollConfig,
    make_fitnesses,
    assert_tensors_close,
    unimplemented
)


# ============================================================================
# Mock Environments for Testing
# ============================================================================

class MockGymEnv:
    """Mock standard Gym-like environment (non-batched)."""
    
    def __init__(self, obs_dim: int = 4, action_dim: int = 2, device: str = 'cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self._step_count = 0
        self._max_steps = 100
    
    def reset(self) -> Tuple[torch.Tensor, dict]:
        self._step_count = 0
        return torch.randn(self.obs_dim, device=self.device), {}
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, bool, dict]:
        self._step_count += 1
        obs = torch.randn(self.obs_dim, device=self.device)
        reward = 1.0  # Constant reward for simplicity
        terminated = self._step_count >= self._max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}


class MockVectorEnv:
    """Mock vectorized Gym-like environment."""
    
    def __init__(self, num_envs: int, obs_dim: int = 4, action_dim: int = 2, device: str = 'cpu'):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self._step_counts = None
        self._max_steps = 100
    
    def reset(self) -> Tuple[torch.Tensor, dict]:
        self._step_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        return torch.randn(self.num_envs, self.obs_dim, device=self.device), {}
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        assert actions.shape[0] == self.num_envs, \
            f"Actions batch size ({actions.shape[0]}) must match num_envs ({self.num_envs})"
        self._step_counts += 1
        obs = torch.randn(self.num_envs, self.obs_dim, device=self.device)
        rewards = torch.ones(self.num_envs, device=self.device)
        terminated = self._step_counts >= self._max_steps
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return obs, rewards, terminated, truncated, {}


class MockBatchedEnv:
    """
    Mock environment that REQUIRES batched actions.
    
    This simulates custom GPU-accelerated environments where you must
    submit actions for all batch elements simultaneously.
    """
    
    def __init__(self, batch_size: int, obs_dim: int = 16, action_dim: int = 4, device: str = 'cpu'):
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self._step_count = 0
        self._max_steps = 200
    
    def reset(self) -> torch.Tensor:
        """Returns obs: (batch_size, obs_dim)"""
        self._step_count = 0
        return torch.randn(self.batch_size, self.obs_dim, device=self.device)
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            actions: (batch_size, action_dim) - MUST be batched!
        Returns:
            obs, rewards, dones
        """
        if actions.shape != (self.batch_size, self.action_dim):
            raise ValueError(
                f"Actions must be shape ({self.batch_size}, {self.action_dim}), "
                f"got {actions.shape}"
            )
        
        self._step_count += 1
        obs = torch.randn(self.batch_size, self.obs_dim, device=self.device)
        rewards = torch.randn(self.batch_size, device=self.device)  # Random rewards
        dones = torch.full((self.batch_size,), self._step_count >= self._max_steps, device=self.device)
        return obs, rewards, dones


# ============================================================================
# Batched Forward Tests (Primary API)
# ============================================================================

class TestBatchedForward:
    """Test batched_forward — the recommended way to evaluate populations."""

    def test_batched_forward_basic(self, simple_mlp, eggroll_config, device):
        """
        batched_forward applies different perturbation to each batch element.
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        population_size = 8
        x = torch.randn(population_size, 8, device=device)
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            outputs = pop.batched_forward(simple_mlp, x)
        
        assert outputs.shape[0] == population_size, \
            f"Output batch size should equal population_size ({population_size}), got {outputs.shape[0]}"
        assert outputs.shape[1] == 2, \
            f"Output dim should match model output (2), got {outputs.shape[1]}"

    def test_batched_forward_different_outputs(self, simple_mlp, eggroll_config, device):
        """
        Different population members should produce different outputs.
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        population_size = 8
        # Same input for all members to isolate perturbation effect
        x_single = torch.randn(1, 8, device=device)
        x = x_single.expand(population_size, -1)
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            outputs = pop.batched_forward(simple_mlp, x)
        
        # Check that different members produce different outputs
        for i in range(population_size - 1):
            assert not torch.allclose(outputs[i], outputs[i + 1], atol=1e-6), \
                f"Members {i} and {i+1} should produce different outputs with same input"

    def test_batched_forward_on_gpu(self, simple_mlp, eggroll_config, device):
        """
        batched_forward should keep everything on GPU.
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        population_size = 8
        x = torch.randn(population_size, 8, device=device)
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            outputs = pop.batched_forward(simple_mlp, x)
        
        assert outputs.device.type == "cuda", \
            f"Outputs should stay on GPU, got device {outputs.device}"


# ============================================================================
# Custom Batched Environment Tests
# ============================================================================

class TestCustomBatchedEnv:
    """Test integration with environments that require batched actions."""

    def test_batched_env_rejects_unbatched(self, eggroll_config):
        """
        Mock env should reject non-batched actions (validates test setup).
        """
        env = MockBatchedEnv(batch_size=8, action_dim=4)
        env.reset()
        
        # Single action should fail
        with pytest.raises(ValueError, match="must be shape"):
            env.step(torch.randn(4))  # Missing batch dim

    def test_batched_env_accepts_correct_shape(self, eggroll_config, device):
        """
        Mock env should accept correctly shaped batched actions.
        """
        env = MockBatchedEnv(batch_size=8, action_dim=4, device=device)
        obs = env.reset()
        
        actions = torch.randn(8, 4, device=device)
        next_obs, rewards, dones = env.step(actions)
        
        assert next_obs.shape == (8, 16), \
            f"Obs shape should be (8, 16), got {next_obs.shape}"
        assert rewards.shape == (8,), \
            f"Rewards shape should be (8,), got {rewards.shape}"

    def test_batched_env_with_strategy(self, device, eggroll_config):
        """
        Strategy should work with batched environments.
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Simple policy: obs -> action
        policy = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # 4 action dims
        ).to(device)
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(policy)
        
        population_size = 8
        env = MockBatchedEnv(batch_size=population_size, obs_dim=16, action_dim=4, device=device)
        
        obs = env.reset()
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            actions = pop.batched_forward(policy, obs)
        
        assert actions.shape == (population_size, 4), \
            f"Actions shape should be ({population_size}, 4), got {actions.shape}"
        
        # Should be able to step the environment
        next_obs, rewards, dones = env.step(actions)
        
        assert rewards.shape == (population_size,), \
            f"Should get reward for each population member"


# ============================================================================
# Vectorized Gym Tests
# ============================================================================

class TestVectorizedGym:
    """Test integration with gym.vector environments."""

    def test_vectorized_gym_shapes(self, device, eggroll_config):
        """
        Vectorized gym environment should have correct shapes.
        """
        population_size = 8
        env = MockVectorEnv(num_envs=population_size, obs_dim=4, action_dim=2, device=device)
        
        obs, _ = env.reset()
        assert obs.shape == (population_size, 4), \
            f"Obs shape should be ({population_size}, 4), got {obs.shape}"
        
        actions = torch.zeros(population_size, device=device, dtype=torch.long)
        obs, rewards, terminated, truncated, _ = env.step(actions)
        
        assert rewards.shape == (population_size,), \
            f"Rewards shape should be ({population_size},), got {rewards.shape}"

    def test_vectorized_gym_with_strategy(self, device, eggroll_config):
        """
        Strategy should work with vectorized gym-like environments.
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Simple policy
        policy = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # 2 action dims
        ).to(device)
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(policy)
        
        population_size = 8
        env = MockVectorEnv(num_envs=population_size, obs_dim=4, action_dim=2, device=device)
        
        obs, _ = env.reset()
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            actions = pop.batched_forward(policy, obs)
        
        assert actions.shape == (population_size, 2), \
            f"Actions shape should be ({population_size}, 2), got {actions.shape}"


# ============================================================================
# Sequential Evaluation Tests (for debugging/legacy)
# ============================================================================

class TestSequentialEvaluation:
    """
    Test sequential evaluation pattern.
    
    This is slower than batched_forward but useful for:
    - Debugging individual population members  
    - Legacy environments that can't be batched
    - Prototyping before implementing batched evaluation
    """

    def test_batched_population_evaluation(self, simple_mlp, eggroll_config, device):
        """
        Should be able to evaluate entire population in one batched call.
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        population_size = 8
        # Each population member gets the same input
        x = torch.randn(1, 8, device=device).expand(population_size, -1)
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            outputs = pop.batched_forward(simple_mlp, x)
        
        assert outputs.shape[0] == population_size, \
            f"Should get {population_size} outputs, got {outputs.shape[0]}"
        
        # Different members should give different outputs
        for i in range(population_size - 1):
            assert not torch.allclose(outputs[i], outputs[i + 1], atol=1e-6), \
                f"Batched outputs {i} and {i+1} should differ (different perturbations)"

    def test_perturbation_deterministic_per_member(self, simple_mlp, eggroll_config, device):
        """
        Same member_id should get same perturbation across batched_forward calls.
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        population_size = 8
        x = torch.randn(1, 8, device=device).expand(population_size, -1)
        
        # Multiple batched_forward calls in same context should give same results
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            output1 = pop.batched_forward(simple_mlp, x)
            output2 = pop.batched_forward(simple_mlp, x)
            output3 = pop.batched_forward(simple_mlp, x)
        
        # All calls should be identical (same perturbations for same member_ids)
        assert torch.allclose(output1, output2, atol=1e-6), \
            "Multiple batched_forward calls should be identical"
        assert torch.allclose(output2, output3, atol=1e-6), \
            "Multiple batched_forward calls should be identical"


# ============================================================================
# Full Training Loop Tests
# ============================================================================

class TestFullTrainingLoop:
    """Test complete ES training loops with environments."""

    def test_single_epoch_loop(self, device, eggroll_config):
        """
        Complete single epoch: perturb -> evaluate -> step.
        """
        from hyperscalees.torch import EggrollStrategy
        
        policy = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        ).to(device)
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(policy)
        
        population_size = 8
        env = MockBatchedEnv(batch_size=population_size, obs_dim=16, action_dim=4, device=device)
        
        # Single training step
        obs = env.reset()
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            actions = pop.batched_forward(policy, obs)
            _, rewards, _ = env.step(actions)
        
        # Use rewards as fitness
        fitnesses = rewards
        metrics = strategy.step(fitnesses)
        
        assert isinstance(metrics, dict), \
            f"step() should return metrics dict, got {type(metrics)}"
        assert strategy.total_steps == 1, \
            f"After one step, total_steps should be 1, got {strategy.total_steps}"

    def test_multi_epoch_loop(self, device, eggroll_config):
        """
        Multiple epochs should accumulate updates.
        """
        from hyperscalees.torch import EggrollStrategy
        
        policy = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        ).to(device)
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(policy)
        
        initial_weight = policy[0].weight.clone()
        
        population_size = 8
        env = MockBatchedEnv(batch_size=population_size, obs_dim=16, action_dim=4, device=device)
        
        num_epochs = 5
        for epoch in range(num_epochs):
            obs = env.reset()
            
            with strategy.perturb(population_size=population_size, epoch=epoch) as pop:
                actions = pop.batched_forward(policy, obs)
                _, rewards, _ = env.step(actions)
            
            strategy.step(rewards)
        
        final_weight = policy[0].weight
        
        assert strategy.total_steps == num_epochs, \
            f"After {num_epochs} epochs, total_steps should be {num_epochs}, got {strategy.total_steps}"
        
        # Weights should have changed
        weight_delta = (final_weight - initial_weight).norm().item()
        assert weight_delta > 1e-6, \
            f"Weights should change after {num_epochs} epochs, delta={weight_delta:.2e}"


# ============================================================================
# Fitness Aggregation Tests
# ============================================================================

class TestFitnessAggregation:
    """Test fitness aggregation patterns common in RL."""

    def test_fitness_from_episode_return(self, device, eggroll_config):
        """
        Fitness can be computed from episode returns.
        """
        from hyperscalees.torch import EggrollStrategy
        
        policy = nn.Linear(4, 2).to(device)
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(policy)
        
        population_size = 8
        
        # Simulate episode returns as fitness
        episode_returns = torch.randn(population_size, device=device)
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 4, device=device)
            pop.batched_forward(policy, x)
        
        # Episode returns become fitness
        metrics = strategy.step(episode_returns)
        
        assert "fitness_mean" in metrics, \
            "Metrics should include fitness statistics"

    def test_normalized_fitness(self, device, eggroll_config):
        """
        Fitness normalization should work with RL returns.
        """
        from hyperscalees.torch import EggrollStrategy
        
        policy = nn.Linear(4, 2).to(device)
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(policy)
        
        population_size = 8
        
        # Episode returns with high variance (common in RL)
        episode_returns = torch.tensor([100.0, -50.0, 200.0, 10.0, -100.0, 150.0, 0.0, 75.0], device=device)
        
        normalized = strategy.normalize_fitnesses(episode_returns)
        
        # Should be centered around 0 with bounded values
        mean = normalized.mean().item()
        assert abs(mean) < 1e-5, \
            f"Normalized fitness should have zero mean, got {mean}"


# ============================================================================
# Determinism Tests for RL
# ============================================================================

class TestRLDeterminism:
    """Test reproducibility in RL setting."""

    def test_same_seed_same_actions(self, device, eggroll_config):
        """
        Same seed should produce same actions for same observations.
        """
        from hyperscalees.torch import EggrollStrategy
        
        def run_with_seed(seed):
            policy = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
            ).to(device)
            
            # Reset weights to same values
            torch.manual_seed(123)
            for p in policy.parameters():
                nn.init.normal_(p)
            
            strategy = EggrollStrategy(
                sigma=eggroll_config.sigma,
                lr=eggroll_config.lr,
                rank=eggroll_config.rank,
                seed=seed
            )
            strategy.setup(policy)
            
            torch.manual_seed(456)  # Same obs for both runs
            obs = torch.randn(8, 4, device=device)
            
            with strategy.perturb(population_size=8, epoch=0) as pop:
                actions = pop.batched_forward(policy, obs)
            
            return actions
        
        actions1 = run_with_seed(42)
        actions2 = run_with_seed(42)
        
        assert torch.allclose(actions1, actions2), \
            "Same seed should produce identical actions"
