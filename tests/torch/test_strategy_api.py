"""
Test: Strategy API contract for PyTorch implementation.

TARGET API: All evolution strategies (Eggroll, OpenES, etc.) should follow
the same interface contract, making them interchangeable via the Strategy pattern.

DESIGN PHILOSOPHY: Inspired by PyTorch Lightning's Trainer and EvoTorch's Problem/Algorithm
separation, but simpler. The Strategy encapsulates the ES algorithm while the model
remains a standard nn.Module.

Key principles:
1. GPU required — EGGROLL's speed comes from batched GPU ops
2. Setup once, perturb many times
3. Context managers for perturbation scope
4. Automatic parameter discovery
5. Serializable state
"""
import pytest
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional, ContextManager
from dataclasses import dataclass

from conftest import (
    EggrollConfig, OpenESConfig, 
    compute_matrix_rank, make_fitnesses, assert_tensors_close,
    unimplemented
)


# ============================================================================
# GPU Requirement Tests
# ============================================================================

class TestGPURequirement:
    """
    Verify EGGROLL checks for GPU availability.
    
    EGGROLL needs a GPU to deliver on its performance promises. These tests
    ensure the implementation gives clear feedback when GPU isn't available.
    """

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_setup_rejects_cpu_model(self, device):
        """
        setup() should raise RuntimeError if model is on CPU.
        
        TARGET API:
            model = nn.Linear(10, 10).cpu()
            strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=4)
            
            with pytest.raises(RuntimeError, match="CUDA|GPU"):
                strategy.setup(model)  # Clear error about needing GPU
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_setup_accepts_gpu_model(self, device):
        """
        setup() should work with a model on CUDA.
        
        TARGET API:
            model = nn.Linear(10, 10).cuda()
            strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=4)
            strategy.setup(model)  # Works fine
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_error_message_is_helpful(self, device):
        """
        The error message should explain why GPU is needed and suggest alternatives.
        
        A good error message helps users understand:
        - Why EGGROLL needs a GPU (batched perturbations)
        - What they can do instead (Colab, different library, etc.)
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_rejects_model_moved_to_cpu_after_setup(self, device):
        """
        Should detect if model is moved to CPU after setup.
        
        TARGET API:
            model = nn.Linear(10, 10).cuda()
            strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=4)
            strategy.setup(model)
            
            model = model.cpu()  # Oops
            
            with pytest.raises(RuntimeError, match="CUDA|GPU"):
                with strategy.perturb(population_size=64, epoch=0) as pop:
                    pop.batched_forward(model, x)
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_rejects_cpu_input_tensors(self, device):
        """
        Should reject CPU tensors passed to batched_forward.
        
        TARGET API:
            model = nn.Linear(10, 10).cuda()
            strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=4)
            strategy.setup(model)
            
            x_cpu = torch.randn(64, 10)  # Forgot .cuda()
            
            with strategy.perturb(population_size=64, epoch=0) as pop:
                with pytest.raises(RuntimeError, match="CUDA|GPU"):
                    pop.batched_forward(model, x_cpu)
        """
        pass


# ============================================================================
# Target Interface Definitions (what we're testing against)
# ============================================================================

class BaseStrategy(ABC):
    """
    Abstract base class defining the Evolution Strategy interface.
    
    All ES implementations must conform to this interface.
    """
    
    @abstractmethod
    def setup(self, model: nn.Module) -> None:
        """
        Attach strategy to a model and discover parameters.
        
        This should:
        - Store reference to model
        - Identify which parameters to evolve
        - Initialize optimizer state
        - Prepare perturbation infrastructure
        """
        pass
    
    @abstractmethod
    def perturb(self, population_size: int, epoch: int = 0) -> ContextManager:
        """
        Context manager for applying perturbations.
        
        Primary usage (batched):
            with strategy.perturb(population_size=64, epoch=0) as pop:
                outputs = pop.batched_forward(model, x_batch)
        
        Alternative (sequential, for debugging):
            with strategy.perturb(population_size=64, epoch=0) as pop:
                for member_id in pop.iterate():
                    output = model(x)
        """
        pass
    
    @abstractmethod
    def step(self, fitnesses: torch.Tensor) -> Dict[str, Any]:
        """
        Update model parameters based on fitness scores.
        
        Returns:
            Dict with update metrics (e.g., gradient norm, param delta)
        """
        pass
    
    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Get serializable state for checkpointing."""
        pass
    
    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        pass
    
    @property
    @abstractmethod
    def sigma(self) -> float:
        """Current noise scale."""
        pass
    
    @property
    @abstractmethod
    def lr(self) -> float:
        """Current learning rate."""
        pass


# ============================================================================
# Strategy API Contract Tests
# ============================================================================

@pytest.mark.usefixtures("device")
class TestStrategyInterface:
    """Verify all strategy implementations follow the interface contract."""

    # List of strategy classes to test (parametrized)
    # These will be imported from the actual implementation once it exists
    STRATEGY_CONFIGS = [
        pytest.param("eggroll", EggrollConfig(), id="Eggroll"),
        pytest.param("open_es", OpenESConfig(), id="OpenES"),
    ]

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    @pytest.mark.parametrize("strategy_type,config", STRATEGY_CONFIGS)
    def test_strategy_instantiation(self, strategy_type, config):
        """
        Strategies should be instantiable with a config object.
        
        TARGET API:
            strategy = EggrollStrategy(**config.__dict__)
            # or
            strategy = EggrollStrategy.from_config(config)
        """
        # Import will be from: hyperscalees.torch.strategies
        # from hyperscalees.torch import EggrollStrategy, OpenESStrategy
        
        # For now, just define what we expect
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    @pytest.mark.parametrize("strategy_type,config", STRATEGY_CONFIGS)
    def test_setup_attaches_to_model(self, strategy_type, config, simple_mlp):
        """
        setup() should attach strategy to model and discover parameters.
        
        TARGET API:
            strategy = EggrollStrategy(sigma=0.1, lr=0.01)
            strategy.setup(model)
            assert strategy.model is model
            assert len(list(strategy.parameters())) > 0
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    @pytest.mark.parametrize("strategy_type,config", STRATEGY_CONFIGS)
    def test_perturb_returns_context_manager(self, strategy_type, config, simple_mlp):
        """
        perturb() should return a context manager with batched_forward.
        
        TARGET API:
            with strategy.perturb(population_size=64, epoch=0) as pop:
                outputs = pop.batched_forward(model, x_batch)
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    @pytest.mark.parametrize("strategy_type,config", STRATEGY_CONFIGS)
    def test_step_updates_parameters(self, strategy_type, config, simple_mlp, device):
        """
        step() should update parameters based on fitnesses.
        
        TARGET API:
            # After collecting fitnesses for population
            metrics = strategy.step(fitnesses)
            assert "param_delta" in metrics
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    @pytest.mark.parametrize("strategy_type,config", STRATEGY_CONFIGS)
    def test_state_dict_serialization(self, strategy_type, config, simple_mlp):
        """
        Strategy state should be serializable for checkpointing.
        
        TARGET API:
            state = strategy.state_dict()
            torch.save(state, "checkpoint.pt")
            
            new_strategy = EggrollStrategy(**config.__dict__)
            new_strategy.setup(model)
            new_strategy.load_state_dict(torch.load("checkpoint.pt"))
        """
        pass


# ============================================================================
# Perturbation Context Tests
# ============================================================================

class TestPerturbationContext:
    """Verify perturbation context manager behavior."""

    @pytest.mark.skip(reason="PerturbationContext not yet implemented")
    def test_context_provides_batched_forward(self, simple_mlp, device):
        """
        Context should provide batched_forward for efficient evaluation.
        
        TARGET API:
            with strategy.perturb(population_size=8) as pop:
                assert pop.population_size == 8
                outputs = pop.batched_forward(model, x_batch)  # Shape: (8, output_dim)
        """
        pass

    @pytest.mark.skip(reason="PerturbationContext not yet implemented")
    def test_context_provides_iterate_for_debugging(self, simple_mlp, device):
        """
        Context should provide iterate() for sequential debugging.
        
        TARGET API:
            with strategy.perturb(population_size=8) as pop:
                for member_id in pop.iterate():
                    output = model(x)  # Uses this member's perturbation
        """
        pass

    @pytest.mark.skip(reason="PerturbationContext not yet implemented")
    def test_context_restores_parameters_on_exit(self, simple_mlp, device):
        """
        Exiting context should restore original parameters.
        
        TARGET API:
            original_params = {n: p.clone() for n, p in model.named_parameters()}
            
            with strategy.perturb(population_size=8):
                pass  # Parameters may be modified inside
            
            # Parameters restored outside
            for n, p in model.named_parameters():
                assert torch.equal(p, original_params[n])
        """
        pass

    @pytest.mark.skip(reason="PerturbationContext not yet implemented")
    def test_nested_contexts_raise_error(self, simple_mlp, device):
        """
        Nested perturbation contexts should raise an error.
        
        TARGET API:
            with strategy.perturb(population_size=8):
                with pytest.raises(RuntimeError):
                    with strategy.perturb(population_size=8):
                        pass
        """
        pass


# ============================================================================
# Eval Mode Tests  
# ============================================================================

class TestEvalMode:
    """Verify evaluation (no perturbation) behavior."""

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_no_perturbation_outside_context(self, simple_mlp, batch_input_small):
        """
        Outside perturb() context, model should use unperturbed parameters.
        
        TARGET API:
            strategy.setup(model)
            
            # No perturbation - standard forward pass
            output1 = model(x)
            output2 = model(x)
            
            assert torch.equal(output1, output2)  # Deterministic
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_eval_method_for_explicit_no_noise(self, simple_mlp, batch_input_small):
        """
        Strategy should have explicit eval mode for clarity.
        
        TARGET API:
            with strategy.eval():
                # Guaranteed no perturbation
                output = model(x)
        """
        pass


# ============================================================================
# Configuration Tests
# ============================================================================

class TestStrategyConfiguration:
    """Verify strategy configuration handling."""

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_sigma_is_readable(self):
        """
        sigma should be accessible as a property.
        
        TARGET API:
            strategy = EggrollStrategy(sigma=0.1, lr=0.01)
            assert strategy.sigma == 0.1
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_lr_is_readable(self):
        """
        lr should be accessible as a property.
        
        TARGET API:
            strategy = EggrollStrategy(sigma=0.1, lr=0.01)
            assert strategy.lr == 0.01
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_sigma_can_be_updated(self):
        """
        sigma should be updatable for annealing schedules.
        
        TARGET API:
            strategy.sigma = 0.05  # Reduce noise over time
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_lr_can_be_updated(self):
        """
        lr should be updatable for learning rate schedules.
        
        TARGET API:
            strategy.lr = 0.005  # Decay learning rate
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_from_config_classmethod(self):
        """
        Strategy should be constructible from config dataclass.
        
        TARGET API:
            config = EggrollConfig(sigma=0.1, lr=0.01, rank=4)
            strategy = EggrollStrategy.from_config(config)
        """
        pass


# ============================================================================
# Parameter Discovery Tests
# ============================================================================

class TestParameterDiscovery:
    """Verify automatic parameter discovery and categorization."""

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_discovers_all_weight_matrices(self, simple_mlp):
        """
        setup() should find all 2D weight parameters (candidates for low-rank).
        
        TARGET API:
            strategy.setup(model)
            weights = list(strategy.weight_parameters())
            assert len(weights) == 3  # 3 Linear layers
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_discovers_all_biases(self, mlp_with_bias):
        """
        setup() should find all bias parameters.
        
        TARGET API:
            strategy.setup(model)
            biases = list(strategy.bias_parameters())
            assert len(biases) == 2  # 2 layers with bias
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_exclude_parameter_by_name(self, simple_mlp):
        """
        Should be able to exclude specific parameters from evolution.
        
        TARGET API:
            strategy.setup(model)
            strategy.freeze("0.weight")  # Freeze first layer
            
            # Or at setup time:
            strategy.setup(model, exclude=["0.weight"])
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_include_only_specific_parameters(self, simple_mlp):
        """
        Should be able to evolve only specific parameters.
        
        TARGET API:
            strategy.setup(model, include=["2.weight", "4.weight"])
            # Only last two linear layers
        """
        pass


# ============================================================================
# Callback / Hook Tests
# ============================================================================

class TestCallbacks:
    """Verify callback/hook system for extensibility."""

    @pytest.mark.skip(reason="Callback system not yet implemented")
    def test_on_step_callback(self, simple_mlp):
        """
        Should support callbacks after each step.
        
        TARGET API:
            metrics_history = []
            
            def on_step(metrics):
                metrics_history.append(metrics)
            
            strategy.register_callback("on_step", on_step)
            strategy.step(fitnesses)
            
            assert len(metrics_history) == 1
        """
        pass

    @pytest.mark.skip(reason="Callback system not yet implemented")
    def test_on_perturb_callback(self, simple_mlp):
        """
        Should support callbacks when perturbations are generated.
        
        TARGET API:
            perturbation_counts = [0]
            
            def on_perturb(population_size, epoch):
                perturbation_counts[0] += population_size
            
            strategy.register_callback("on_perturb", on_perturb)
        """
        pass


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Verify proper error handling and messages."""

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_perturb_before_setup_raises(self):
        """
        Calling perturb() before setup() should raise informative error.
        
        TARGET API:
            strategy = EggrollStrategy(sigma=0.1, lr=0.01)
            
            with pytest.raises(RuntimeError, match="setup"):
                with strategy.perturb(population_size=8):
                    pass
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_step_before_setup_raises(self):
        """
        Calling step() before setup() should raise informative error.
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_step_with_wrong_fitness_size_raises(self, simple_mlp):
        """
        Fitness tensor size must match population size.
        
        TARGET API:
            with strategy.perturb(population_size=8):
                # ... collect fitnesses
                pass
            
            wrong_fitnesses = torch.randn(16)  # Wrong size
            with pytest.raises(ValueError, match="population"):
                strategy.step(wrong_fitnesses)
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")
    def test_negative_population_size_raises(self, simple_mlp):
        """
        Population size must be positive.
        """
        pass

    @pytest.mark.skip(reason="Strategy classes not yet implemented")  
    def test_odd_population_with_antithetic_warns(self, simple_mlp):
        """
        Antithetic sampling with odd population should warn.
        
        TARGET API:
            strategy = EggrollStrategy(sigma=0.1, lr=0.01, antithetic=True)
            strategy.setup(model)
            
            with pytest.warns(UserWarning, match="antithetic"):
                with strategy.perturb(population_size=7):  # Odd!
                    pass
        """
        pass
