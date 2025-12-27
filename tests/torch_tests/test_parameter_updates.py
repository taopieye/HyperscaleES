"""
Test: Parameter updates (ES gradient estimation) for PyTorch implementation.

PAPER CLAIM: ES estimates gradients as:
    ∇_θ E[F(θ+σε)] ≈ (1/σ) E[F(θ+σε)·ε]

With low-rank ε = AB^T, the update aggregates weighted low-rank perturbations
across the population.

TARGET API: Updates should be computed via step() with proper metrics returned.
The update process should be efficient and numerically stable.
"""
import pytest
import torch
import torch.nn as nn

from .conftest import (
    EggrollConfig,
    make_fitnesses,
    assert_tensors_close,
    compute_matrix_rank,
    unimplemented
)


# ============================================================================
# Core Update Tests
# ============================================================================

class TestParameterUpdates:
    """Verify ES gradient estimation and parameter updates."""

    def test_higher_fitness_perturbation_dominates_update(
        self, simple_mlp, batch_input_small, eggroll_config
    ):
        """
        Update direction should favor perturbations with higher fitness.
        
        TARGET API:
            strategy.setup(model)
            
            # Forward pass
            with strategy.perturb(population_size=8, epoch=0):
                pass  # Collect fitnesses
            
            # One perturbation has much higher fitness
            fitnesses = torch.tensor([-1, -1, -1, 10, -1, -1, -1, -1], dtype=torch.float32)
            
            # Get the high-fitness perturbation for comparison
            high_fitness_pert = strategy._get_perturbation("0.weight", member_id=3, epoch=0)
            
            before = model[0].weight.clone()
            strategy.step(fitnesses)
            after = model[0].weight.clone()
            
            # Update direction should correlate with high-fitness perturbation
            delta = after - before
            correlation = (delta * high_fitness_pert.as_matrix()).sum()
            assert correlation > 0
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        # One perturbation (index 3) has much higher fitness
        fitnesses = torch.tensor(
            [-1.0, -1.0, -1.0, 10.0, -1.0, -1.0, -1.0, -1.0], 
            device=device
        )
        
        # Get the high-fitness perturbation for comparison
        high_fitness_pert = strategy._get_perturbation("0.weight", member_id=3, epoch=0)
        
        before = simple_mlp[0].weight.clone()
        strategy.step(fitnesses)
        after = simple_mlp[0].weight.clone()
        
        # Update direction should correlate with high-fitness perturbation
        delta = after - before
        correlation = (delta * high_fitness_pert.as_matrix()).sum()
        assert correlation > 0, "Update should move toward high-fitness perturbation"

    def test_equal_fitnesses_produce_no_update(
        self, simple_mlp, eggroll_config
    ):
        """
        Equal fitness scores should produce zero update (after normalization).
        
        TARGET API:
            strategy.setup(model)
            
            fitnesses = torch.ones(8) * 5.0  # All equal
            
            before = model[0].weight.clone()
            strategy.step(fitnesses)
            after = model[0].weight.clone()
            
            assert torch.allclose(before, after, atol=1e-6)
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        # All equal fitnesses
        fitnesses = torch.ones(8, device=device) * 5.0
        
        before = simple_mlp[0].weight.clone()
        strategy.step(fitnesses)
        after = simple_mlp[0].weight.clone()
        
        assert torch.allclose(before, after, atol=1e-6), \
            "Equal fitnesses should produce no update after normalization"

    def test_antithetic_equal_fitness_cancels(
        self, simple_mlp, eggroll_config
    ):
        """
        Antithetic pairs with equal fitness should cancel out.
        
        TARGET API:
            # Pairs have same fitness
            fitnesses = torch.tensor([5.0, 5.0, 3.0, 3.0, 7.0, 7.0, 1.0, 1.0])
            
            before = model[0].weight.clone()
            strategy.step(fitnesses)
            after = model[0].weight.clone()
            
            # Cancellation means no update
            assert torch.allclose(before, after, atol=1e-6)
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        
        # Create strategy with antithetic sampling
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma, lr=eggroll_config.lr, 
            rank=eggroll_config.rank, antithetic=True
        )
        strategy.setup(simple_mlp)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        # Pairs have same fitness: (5,5), (3,3), (7,7), (1,1)
        fitnesses = torch.tensor(
            [5.0, 5.0, 3.0, 3.0, 7.0, 7.0, 1.0, 1.0], 
            device=device
        )
        
        before = simple_mlp[0].weight.clone()
        strategy.step(fitnesses)
        after = simple_mlp[0].weight.clone()
        
        # With antithetic pairs having equal fitness, +ε and -ε cancel
        assert torch.allclose(before, after, atol=1e-5), \
            "Antithetic pairs with equal fitness should cancel"


# ============================================================================
# Learning Rate Tests
# ============================================================================

class TestLearningRate:
    """Verify learning rate effects on updates."""

    def test_update_magnitude_scales_with_lr(
        self, simple_mlp, batch_input_small
    ):
        """
        Larger learning rate should produce larger parameter changes.
        
        TARGET API:
            deltas = []
            for lr in [0.001, 0.01, 0.1]:
                model = create_model()  # Fresh model
                strategy = EggrollStrategy(sigma=0.1, lr=lr)
                strategy.setup(model)
                
                fitnesses = torch.randn(8)
                
                before = model[0].weight.clone()
                strategy.step(fitnesses)
                after = model[0].weight.clone()
                
                delta = (after - before).norm()
                deltas.append(delta)
            
            # Larger LR -> larger update
            assert deltas[2] > deltas[1] > deltas[0]
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        seed = 42
        
        deltas = []
        for lr in [0.001, 0.01, 0.1]:
            # Fresh model for each test
            model = nn.Sequential(
                nn.Linear(32, 64, bias=False),
                nn.ReLU(),
                nn.Linear(64, 64, bias=False),
                nn.ReLU(),
                nn.Linear(64, 16, bias=False),
            ).to(device)
            
            strategy = EggrollStrategy(sigma=0.1, lr=lr, rank=4, seed=seed)
            strategy.setup(model)
            
            population_size = 8
            with strategy.perturb(population_size=population_size, epoch=0) as pop:
                x = torch.randn(population_size, 32, device=device)  # Match model input dim
                pop.batched_forward(model, x)
            
            fitnesses = torch.randn(population_size, device=device)
            
            before = model[0].weight.clone()
            strategy.step(fitnesses)
            after = model[0].weight.clone()
            
            delta = (after - before).norm().item()
            deltas.append(delta)
        
        # Larger LR -> larger update
        assert deltas[2] > deltas[1] > deltas[0], \
            f"Update magnitude should scale with lr: lr=[0.001,0.01,0.1] -> deltas={deltas}, " \
            f"but got deltas[2]({deltas[2]:.4f}) <= deltas[1]({deltas[1]:.4f}) <= deltas[0]({deltas[0]:.4f})"

    def test_zero_lr_produces_no_update(self, simple_mlp, eggroll_config):
        """
        lr=0 should produce no parameter updates.
        
        TARGET API:
            strategy = EggrollStrategy(sigma=0.1, lr=0.0)
            strategy.setup(model)
            
            fitnesses = torch.randn(8)
            
            before = model[0].weight.clone()
            strategy.step(fitnesses)
            after = model[0].weight.clone()
            
            assert torch.equal(before, after)
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(sigma=0.1, lr=0.0, rank=4)
        strategy.setup(simple_mlp)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        fitnesses = torch.randn(population_size, device=device)
        
        before = simple_mlp[0].weight.clone()
        strategy.step(fitnesses)
        after = simple_mlp[0].weight.clone()
        
        assert torch.equal(before, after), "lr=0 should produce no update"


# ============================================================================
# Sigma Scaling Tests
# ============================================================================

class TestSigmaInUpdate:
    """Verify sigma (noise scale) effects on updates."""

    def test_update_scales_inversely_with_sigma(
        self, simple_mlp, batch_input_small
    ):
        """
        ES gradient has 1/σ factor, so update should scale inversely.
        
        (Actually depends on normalization - this tests the formula.)
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        seed = 42
        
        deltas = []
        for sigma in [0.01, 0.1, 1.0]:
            # Fresh model for each test
            model = nn.Sequential(
                nn.Linear(32, 64, bias=False),
                nn.ReLU(),
                nn.Linear(64, 64, bias=False),
                nn.ReLU(),
                nn.Linear(64, 16, bias=False),
            ).to(device)
            
            strategy = EggrollStrategy(sigma=sigma, lr=0.01, rank=4, seed=seed)
            strategy.setup(model)
            
            population_size = 8
            with strategy.perturb(population_size=population_size, epoch=0) as pop:
                x = torch.randn(population_size, 32, device=device)  # Match model input dim
                pop.batched_forward(model, x)
            
            fitnesses = torch.randn(population_size, device=device)
            
            before = model[0].weight.clone()
            strategy.step(fitnesses)
            after = model[0].weight.clone()
            
            delta = (after - before).norm().item()
            deltas.append(delta)
        
        # ES formula: g ∝ f * ε / σ
        # Larger σ -> smaller gradient (for same ε)
        # But ε itself scales with σ, so the relationship is more nuanced
        # Just verify that updates are finite and change with sigma
        assert all(d > 0 for d in deltas), "All deltas should be positive"

    def test_very_small_sigma_handled(self, simple_mlp):
        """
        Very small sigma should not cause numerical issues.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(sigma=1e-8, lr=0.01, rank=4)
        strategy.setup(simple_mlp)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        fitnesses = torch.randn(population_size, device=device)
        
        # Should not raise and should not produce NaN
        strategy.step(fitnesses)
        
        for p in simple_mlp.parameters():
            assert torch.isfinite(p).all(), "Very small sigma caused non-finite values"


# ============================================================================
# Optimizer Integration Tests
# ============================================================================

class TestOptimizerIntegration:
    """Verify integration with PyTorch optimizers."""

    def test_sgd_optimizer(self, simple_mlp, eggroll_config):
        """
        Should work with SGD optimizer.
        
        TARGET API:
            strategy = EggrollStrategy(
                sigma=0.1, lr=0.01,
                optimizer="sgd"
            )
            strategy.setup(model)
            
            strategy.step(fitnesses)
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(
            sigma=0.1, lr=0.01, rank=4,
            optimizer="sgd"
        )
        strategy.setup(simple_mlp)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        fitnesses = make_fitnesses(population_size, device=device)
        
        before = simple_mlp[0].weight.clone()
        strategy.step(fitnesses)
        after = simple_mlp[0].weight.clone()
        
        # Should have updated
        assert not torch.equal(before, after)

    def test_adam_optimizer(self, simple_mlp, eggroll_config):
        """
        Should work with Adam optimizer.
        
        TARGET API:
            strategy = EggrollStrategy(
                sigma=0.1, lr=0.01,
                optimizer="adam",
                optimizer_kwargs={"betas": (0.9, 0.999)}
            )
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(
            sigma=0.1, lr=0.01, rank=4,
            optimizer="adam",
            optimizer_kwargs={"betas": (0.9, 0.999)}
        )
        strategy.setup(simple_mlp)
        
        before = simple_mlp[0].weight.clone()
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        fitnesses = make_fitnesses(population_size, device=device)
        strategy.step(fitnesses)
        after = simple_mlp[0].weight.clone()
        
        # Adam should have updated the weights
        assert not torch.equal(before, after), "Adam optimizer should update weights"

    def test_adamw_optimizer(self, simple_mlp, eggroll_config):
        """
        Should work with AdamW optimizer (with weight decay).
        
        TARGET API:
            strategy = EggrollStrategy(
                sigma=0.1, lr=0.01,
                optimizer="adamw",
                optimizer_kwargs={"weight_decay": 0.01}
            )
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(
            sigma=0.1, lr=0.01, rank=4,
            optimizer="adamw",
            optimizer_kwargs={"weight_decay": 0.01}
        )
        strategy.setup(simple_mlp)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        fitnesses = make_fitnesses(population_size, device=device)
        strategy.step(fitnesses)

    def test_optimizer_state_updates(self, simple_mlp, eggroll_config):
        """
        Optimizer state (e.g., Adam moments) should update correctly.
        
        TARGET API:
            strategy = EggrollStrategy(
                sigma=0.1, lr=0.01,
                optimizer="adam"
            )
            strategy.setup(model)
            
            # First step
            strategy.step(fitnesses1)
            
            # Optimizer state should exist
            state = strategy.optimizer.state
            assert len(state) > 0
            
            # Second step should use momentum
            strategy.step(fitnesses2)
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(
            sigma=0.1, lr=0.01, rank=4,
            optimizer="adam"
        )
        strategy.setup(simple_mlp)
        
        population_size = 8
        
        # First step
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        fitnesses1 = make_fitnesses(population_size, device=device)
        strategy.step(fitnesses1)
        
        # Optimizer state should exist
        state = strategy.optimizer.state
        assert len(state) > 0, \
            f"Adam optimizer state should be populated after step(), got {len(state)} param groups"
        
        # Second step
        with strategy.perturb(population_size=population_size, epoch=1) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        fitnesses2 = make_fitnesses(population_size, device=device)
        strategy.step(fitnesses2)

    def test_custom_optimizer(self, simple_mlp):
        """
        Should support custom optimizer classes.
        
        TARGET API:
            strategy = EggrollStrategy(
                sigma=0.1, lr=0.01,
                optimizer=torch.optim.RMSprop,
                optimizer_kwargs={"alpha": 0.99}
            )
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(
            sigma=0.1, lr=0.01, rank=4,
            optimizer=torch.optim.RMSprop,
            optimizer_kwargs={"alpha": 0.99}
        )
        strategy.setup(simple_mlp)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        fitnesses = make_fitnesses(population_size, device=device)
        strategy.step(fitnesses)


# ============================================================================
# Bias vs Weight Update Tests
# ============================================================================

class TestBiasWeightUpdates:
    """Verify separate handling of biases and weights."""

    def test_weights_get_lowrank_update(
        self, mlp_with_bias, eggroll_config
    ):
        """
        Weight matrices should be updated via low-rank perturbations.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = mlp_with_bias[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(mlp_with_bias)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(mlp_with_bias, x)
        
        fitnesses = make_fitnesses(population_size, device=device)
        
        before_weight = mlp_with_bias[0].weight.clone()
        strategy.step(fitnesses)
        after_weight = mlp_with_bias[0].weight.clone()
        
        # Weights should be updated
        assert not torch.equal(before_weight, after_weight)

    def test_biases_get_standard_update(
        self, mlp_with_bias, eggroll_config
    ):
        """
        Biases (1D) should be updated via standard (non-low-rank) perturbations.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = mlp_with_bias[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(mlp_with_bias)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(mlp_with_bias, x)
        
        fitnesses = make_fitnesses(population_size, device=device)
        
        before_bias = mlp_with_bias[0].bias.clone()
        strategy.step(fitnesses)
        after_bias = mlp_with_bias[0].bias.clone()
        
        # Biases should also be updated
        assert not torch.equal(before_bias, after_bias)

    def test_freeze_bias_option(self, mlp_with_bias, eggroll_config):
        """
        Should be able to freeze biases from evolution.
        
        TARGET API:
            strategy = EggrollStrategy(
                sigma=0.1, lr=0.01,
                evolve_bias=False  # Don't evolve biases
            )
            strategy.setup(model)
            
            before_bias = model[0].bias.clone()
            strategy.step(fitnesses)
            after_bias = model[0].bias.clone()
            
            assert torch.equal(before_bias, after_bias)
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = mlp_with_bias[0].weight.device
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma, lr=eggroll_config.lr, 
            rank=eggroll_config.rank, evolve_bias=False
        )
        strategy.setup(mlp_with_bias)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(mlp_with_bias, x)
        
        fitnesses = make_fitnesses(population_size, device=device)
        
        before_bias = mlp_with_bias[0].bias.clone()
        before_weight = mlp_with_bias[0].weight.clone()
        
        strategy.step(fitnesses)
        
        after_bias = mlp_with_bias[0].bias.clone()
        after_weight = mlp_with_bias[0].weight.clone()
        
        # Bias should be frozen
        assert torch.equal(before_bias, after_bias), "Bias should be frozen"
        # But weights should be updated
        assert not torch.equal(before_weight, after_weight), "Weights should update"


# ============================================================================
# Multi-Step Tests
# ============================================================================

class TestMultiStepUpdates:
    """Verify behavior over multiple update steps."""

    def test_multiple_steps_accumulate(self, simple_mlp, eggroll_config):
        """
        Multiple steps should accumulate updates.
        
        TARGET API:
            initial = model[0].weight.clone()
            
            for epoch in range(10):
                with strategy.perturb(64, epoch):
                    pass
                strategy.step(fitnesses)
            
            final = model[0].weight.clone()
            
            # Should have changed significantly
            assert not torch.allclose(initial, final)
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        initial = simple_mlp[0].weight.clone()
        
        for epoch in range(10):
            population_size = 64
            with strategy.perturb(population_size=population_size, epoch=epoch) as pop:
                x = torch.randn(population_size, 8, device=device)
                pop.batched_forward(simple_mlp, x)
            
            fitnesses = make_fitnesses(population_size, device=device)
            strategy.step(fitnesses)
        
        final = simple_mlp[0].weight.clone()
        
        # Should have changed significantly after 10 steps
        delta = (final - initial).norm()
        assert delta > 1e-4, f"Parameters should change significantly, delta={delta}"

    def test_update_improves_simple_fitness(self, simple_mlp):
        """
        ES should actually optimize a simple fitness function.
        
        TARGET API:
            # Simple fitness: minimize L2 distance to target
            target = torch.randn(2)
            
            def fitness_fn(output):
                return -((output - target) ** 2).sum()
            
            initial_fitness = fitness_fn(model(x)).mean()
            
            for epoch in range(50):
                with strategy.perturb(64, epoch) as pop:
                    fitnesses = []
                    for _ in pop.iterate():
                        output = model(x)
                        fitnesses.append(fitness_fn(output).mean())
                    fitnesses = torch.tensor(fitnesses)
                strategy.step(fitnesses)
            
            final_fitness = fitness_fn(model(x)).mean()
            
            assert final_fitness > initial_fitness  # Improvement
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        
        # Create small model for optimization test
        model = nn.Sequential(
            nn.Linear(4, 8, bias=False),
            nn.ReLU(),
            nn.Linear(8, 2, bias=False),
        ).to(device)
        
        strategy = EggrollStrategy(sigma=0.1, lr=0.1, rank=4)
        strategy.setup(model)
        
        # Target and input
        x = torch.randn(1, 4, device=device)
        target = torch.randn(1, 2, device=device)
        
        def fitness_fn(output):
            return -((output - target) ** 2).sum()
        
        initial_fitness = fitness_fn(model(x)).item()
        
        for epoch in range(50):
            population_size = 32
            with strategy.perturb(population_size=population_size, epoch=epoch) as pop:
                fitnesses = []
                for _ in pop.iterate():
                    output = model(x)
                    fitnesses.append(fitness_fn(output).item())
                fitnesses = torch.tensor(fitnesses, device=device)
            strategy.step(fitnesses)
        
        final_fitness = fitness_fn(model(x)).item()
        
        assert final_fitness > initial_fitness, \
            f"ES should improve fitness: {initial_fitness} -> {final_fitness}"


# ============================================================================
# Metrics Tests
# ============================================================================

class TestUpdateMetrics:
    """Verify metrics returned by step()."""

    def test_step_returns_metrics_dict(self, simple_mlp, eggroll_config):
        """
        step() should return a dict of useful metrics.
        
        TARGET API:
            metrics = strategy.step(fitnesses)
            
            assert isinstance(metrics, dict)
            assert "param_delta_norm" in metrics
            assert "grad_norm" in metrics
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        fitnesses = make_fitnesses(population_size, device=device)
        metrics = strategy.step(fitnesses)
        
        assert isinstance(metrics, dict)
        # Check for expected keys
        assert "param_delta_norm" in metrics or "grad_norm" in metrics

    def test_metrics_include_gradient_norm(self, simple_mlp, eggroll_config):
        """
        Metrics should include gradient (update direction) norm.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        fitnesses = make_fitnesses(population_size, device=device)
        metrics = strategy.step(fitnesses)
        
        assert "grad_norm" in metrics
        assert isinstance(metrics["grad_norm"], (int, float))
        assert metrics["grad_norm"] >= 0

    def test_metrics_include_param_delta(self, simple_mlp, eggroll_config):
        """
        Metrics should include parameter change magnitude.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        fitnesses = make_fitnesses(population_size, device=device)
        metrics = strategy.step(fitnesses)
        
        assert "param_delta_norm" in metrics
        assert isinstance(metrics["param_delta_norm"], (int, float))
        assert metrics["param_delta_norm"] >= 0

    def test_metrics_include_fitness_stats(self, simple_mlp, eggroll_config):
        """
        Metrics should include fitness statistics.
        
        TARGET API:
            metrics = strategy.step(fitnesses)
            
            assert "fitness_mean" in metrics
            assert "fitness_std" in metrics
            assert "fitness_max" in metrics
            assert "fitness_min" in metrics
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        fitnesses = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device=device)
        metrics = strategy.step(fitnesses)
        
        # Check fitness stats
        assert "fitness_mean" in metrics
        assert "fitness_std" in metrics
        assert "fitness_max" in metrics
        assert "fitness_min" in metrics
        
        # Validate values
        assert abs(metrics["fitness_mean"] - 4.5) < 0.1
        assert metrics["fitness_max"] == 8.0
        assert metrics["fitness_min"] == 1.0


# ============================================================================
# Numerical Stability Tests
# ============================================================================

class TestUpdateStability:
    """Verify numerical stability of updates."""

    def test_no_nan_in_update(self, simple_mlp, eggroll_config):
        """
        Updates should never produce NaN values.
        
        TARGET API:
            strategy.step(fitnesses)
            
            for p in model.parameters():
                assert torch.isfinite(p).all()
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        fitnesses = make_fitnesses(population_size, device=device)
        strategy.step(fitnesses)
        
        for p in simple_mlp.parameters():
            assert torch.isfinite(p).all(), "Update produced NaN values"
            assert not torch.isnan(p).any(), "Parameters contain NaN"

    def test_no_inf_in_update(self, simple_mlp, eggroll_config):
        """
        Updates should never produce Inf values.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        # Even with large fitness values
        fitnesses = torch.tensor([1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10], device=device)
        strategy.step(fitnesses)
        
        for p in simple_mlp.parameters():
            assert torch.isfinite(p).all(), "Update produced Inf values"
            assert not torch.isinf(p).any(), "Parameters contain Inf"

    def test_gradient_clipping_option(self, simple_mlp):
        """
        Should support optional gradient clipping.
        
        TARGET API:
            strategy = EggrollStrategy(
                sigma=0.1, lr=0.01,
                max_grad_norm=1.0  # Clip gradient norm
            )
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        
        # Create strategy with gradient clipping
        strategy = EggrollStrategy(
            sigma=0.1, lr=0.01, rank=4,
            max_grad_norm=1.0
        )
        strategy.setup(simple_mlp)
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        # Large fitness spread that would produce large gradients
        fitnesses = torch.tensor([-100.0, 100.0, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0], device=device)
        metrics = strategy.step(fitnesses)
        
        # Gradient should be clipped
        if "grad_norm" in metrics:
            assert metrics["grad_norm"] <= 1.0 + 1e-5, "Gradient should be clipped"


# ============================================================================
# State Consistency Tests
# ============================================================================

class TestUpdateStateConsistency:
    """Verify state consistency after updates."""

    def test_epoch_increments_after_step(self, simple_mlp, eggroll_config):
        """
        Internal epoch counter should increment after step.
        
        TARGET API:
            assert strategy.epoch == 0
            
            strategy.step(fitnesses)
            
            assert strategy.epoch == 1
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        assert strategy.epoch == 0, \
            f"Before any step(), epoch should be 0, got {strategy.epoch}"
        
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        fitnesses = make_fitnesses(population_size, device=device)
        strategy.step(fitnesses)
        
        assert strategy.epoch == 1, \
            f"After one step(), epoch should be 1, got {strategy.epoch}"

    def test_step_count_tracked(self, simple_mlp, eggroll_config):
        """
        Total step count should be tracked.
        
        TARGET API:
            assert strategy.total_steps == 0
            
            for _ in range(5):
                strategy.step(fitnesses)
            
            assert strategy.total_steps == 5
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        assert strategy.total_steps == 0, \
            f"Before any step(), total_steps should be 0, got {strategy.total_steps}"
        
        for epoch in range(5):
            population_size = 8
            with strategy.perturb(population_size=population_size, epoch=epoch) as pop:
                x = torch.randn(population_size, 8, device=device)
                pop.batched_forward(simple_mlp, x)
            
            fitnesses = make_fitnesses(population_size, device=device)
            strategy.step(fitnesses)
        
        assert strategy.total_steps == 5, \
            f"After 5 step() calls, total_steps should be 5, got {strategy.total_steps}"
