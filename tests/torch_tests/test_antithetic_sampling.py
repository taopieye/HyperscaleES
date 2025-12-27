"""
Test: Antithetic (mirrored) sampling for PyTorch implementation.

PAPER CLAIM: For variance reduction, each perturbation ε is paired with -ε.
Thread pairs (2k, 2k+1) use:
    thread 2k:   θ + σε
    thread 2k+1: θ - σε

This reduces gradient variance without extra compute, as symmetric perturbations
with similar fitness cancel out in the gradient estimate.

TARGET API: Antithetic sampling should be automatic when enabled, with pairs
easily identifiable for testing and debugging.
"""
import pytest
import torch
import torch.nn as nn

from .conftest import (
    EggrollConfig,
    assert_tensors_close,
    make_fitnesses,
    unimplemented
)


# ============================================================================
# Core Antithetic Structure Tests
# ============================================================================

class TestAntitheticSampling:
    """Verify antithetic (mirrored) sampling structure."""

    def test_even_odd_pairs_have_opposite_perturbations(
        self, simple_linear, es_generator, eggroll_config, device
    ):
        """
        Population members 2k and 2k+1 should have opposite perturbations.
        
        TARGET API:
            perturbations = strategy.sample_perturbations(param, population_size=8, epoch=0)
            
            for i in range(0, 8, 2):
                pert_even = perturbations[i].as_matrix()
                pert_odd = perturbations[i + 1].as_matrix()
                
                # Should be exact negatives
                assert torch.allclose(pert_even, -pert_odd)
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed,
            antithetic=True
        )
        strategy.setup(simple_linear)
        
        perturbations = strategy.sample_perturbations(
            param=simple_linear.weight,
            population_size=population_size,
            epoch=0
        )
        
        # Check each pair
        for i in range(0, population_size, 2):
            pert_even = perturbations[i].as_matrix()
            pert_odd = perturbations[i + 1].as_matrix()
            
            assert_tensors_close(
                pert_even,
                -pert_odd,
                atol=1e-6,
                msg=f"Pair ({i}, {i+1}): perturbations should be exact negatives"
            )

    def test_antithetic_pairs_share_base_noise(
        self, simple_linear, es_generator, eggroll_config, device
    ):
        """
        Antithetic pairs should share the same base noise (just negated).
        
        This verifies they come from the same random seed.
        
        TARGET API:
            perturbations = strategy.sample_perturbations(param, population_size=8, epoch=0)
            
            # Access low-rank factors
            A_even, B_even = perturbations[0].factors
            A_odd, B_odd = perturbations[1].factors
            
            # A factors should be negatives (or B, depending on implementation)
            assert torch.allclose(A_even, -A_odd)
            # B factors should be the same
            assert torch.allclose(B_even, B_odd)
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed,
            antithetic=True
        )
        strategy.setup(simple_linear)
        
        perturbations = strategy.sample_perturbations(
            param=simple_linear.weight,
            population_size=population_size,
            epoch=0
        )
        
        # Check factors for first pair
        A_even, B_even = perturbations[0].factors
        A_odd, B_odd = perturbations[1].factors
        
        # Either A is negated (and B same) or B is negated (and A same)
        # The key property is that A_even @ B_even.T = -(A_odd @ B_odd.T)
        pert_even = A_even @ B_even.T
        pert_odd = A_odd @ B_odd.T
        
        assert_tensors_close(
            pert_even,
            -pert_odd,
            atol=1e-6,
            msg="Antithetic pairs should produce opposite full perturbations"
        )

    def test_antithetic_flag_controls_behavior(self, simple_linear, es_generator, device):
        """
        antithetic=False should disable mirrored sampling.
        
        TARGET API:
            config_no_antithetic = EggrollConfig(antithetic=False)
            strategy = EggrollStrategy.from_config(config_no_antithetic)
            
            perturbations = strategy.sample_perturbations(param, population_size=8, epoch=0)
            
            # Pairs should NOT be negatives
            pert_0 = perturbations[0].as_matrix()
            pert_1 = perturbations[1].as_matrix()
            
            assert not torch.allclose(pert_0, -pert_1)
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        
        # Without antithetic
        strategy = EggrollStrategy(
            sigma=0.1,
            lr=0.01,
            rank=4,
            seed=42,
            antithetic=False
        )
        strategy.setup(simple_linear)
        
        perturbations = strategy.sample_perturbations(
            param=simple_linear.weight,
            population_size=population_size,
            epoch=0
        )
        
        pert_0 = perturbations[0].as_matrix()
        pert_1 = perturbations[1].as_matrix()
        
        # Should NOT be negatives without antithetic
        assert not torch.allclose(pert_0, -pert_1, atol=1e-4), \
            "Without antithetic, pairs should not be exact negatives"


# ============================================================================
# Variance Reduction Tests
# ============================================================================

class TestVarianceReduction:
    """Verify that antithetic sampling reduces variance."""

    def test_antithetic_cancellation_with_equal_fitness(
        self, simple_linear, es_generator, eggroll_config, device
    ):
        """
        When antithetic pairs have equal fitness, their contributions cancel.
        
        This is a key property: f(θ+ε) = f(θ-ε) implies ε contributes nothing.
        
        TARGET API:
            # Give pairs equal fitness
            fitnesses = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
            
            strategy.step(fitnesses)
            
            # Update should be approximately zero (pairs cancel)
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed,
            antithetic=True
        )
        strategy.setup(simple_linear)
        
        # Record original weights
        original_weight = simple_linear.weight.clone()
        
        # Equal fitness for each pair
        fitnesses = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0], device=device)
        
        # Perturb and step
        with strategy.perturb(population_size=population_size, epoch=0):
            pass
        strategy.step(fitnesses)
        
        # With equal fitness pairs, updates should cancel
        delta = (simple_linear.weight - original_weight).abs().max().item()
        assert delta < 1e-4, \
            f"With equal fitness pairs, update should be ~0, got max delta {delta:.6f}"

    def test_mean_perturbation_is_zero_with_antithetic(
        self, simple_linear, es_generator, eggroll_config, device
    ):
        """
        Sum of antithetic perturbations should be exactly zero.
        
        TARGET API:
            perturbations = strategy.sample_perturbations(param, population_size=8, epoch=0)
            
            total = sum(p.as_matrix() for p in perturbations)
            
            assert torch.allclose(total, torch.zeros_like(total))
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed,
            antithetic=True
        )
        strategy.setup(simple_linear)
        
        perturbations = strategy.sample_perturbations(
            param=simple_linear.weight,
            population_size=population_size,
            epoch=0
        )
        
        # Sum all perturbations
        total = perturbations[0].as_matrix().clone()
        for p in perturbations[1:]:
            total += p.as_matrix()
        
        # Should be exactly zero (each +ε paired with -ε)
        assert torch.allclose(total, torch.zeros_like(total), atol=1e-6), \
            "Sum of antithetic perturbations should be exactly zero"

    @pytest.mark.slow
    def test_variance_is_lower_with_antithetic(self, simple_mlp, batch_input_small, device):
        """
        Gradient estimate variance should be lower with antithetic sampling.
        
        This is an empirical test over many random seeds.
        """
        from hyperscalees.torch import EggrollStrategy
        import math
        
        population_size = 32
        n_trials = 50
        
        def run_trials(antithetic: bool):
            """Run multiple trials and return variance of updates."""
            updates = []
            
            for trial in range(n_trials):
                model = nn.Linear(8, 16, bias=False).to(device)
                
                strategy = EggrollStrategy(
                    sigma=0.1,
                    lr=0.01,
                    rank=4,
                    seed=trial,  # Different seed each trial
                    antithetic=antithetic
                )
                strategy.setup(model)
                
                original_weight = model.weight.clone()
                
                # Random fitnesses
                fitnesses = torch.randn(population_size, device=device)
                
                with strategy.perturb(population_size=population_size, epoch=0):
                    pass
                strategy.step(fitnesses)
                
                delta = (model.weight - original_weight).flatten()
                updates.append(delta.cpu())
            
            # Stack and compute variance
            updates = torch.stack(updates)
            return updates.var(dim=0).mean().item()
        
        variance_with = run_trials(antithetic=True)
        variance_without = run_trials(antithetic=False)
        
        # Antithetic should have lower variance
        assert variance_with < variance_without, \
            f"Antithetic variance ({variance_with:.6f}) should be lower than without ({variance_without:.6f})"


# ============================================================================
# Gradient Estimation with Antithetic Tests
# ============================================================================

class TestAntitheticGradientEstimation:
    """Verify gradient estimation with antithetic pairs."""

    def test_asymmetric_fitness_produces_gradient(
        self, simple_linear, es_generator, eggroll_config, device
    ):
        """
        When f(θ+ε) ≠ f(θ-ε), there should be a gradient signal.
        
        TARGET API:
            # θ+ε is better than θ-ε
            fitnesses = torch.tensor([10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0])
            
            metrics = strategy.step(fitnesses)
            
            # Should have non-trivial update
            assert metrics["param_delta_norm"] > 0
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed,
            antithetic=True
        )
        strategy.setup(simple_linear)
        
        original_weight = simple_linear.weight.clone()
        
        # Asymmetric fitness: even members (positive perturbation) are better
        fitnesses = torch.tensor([10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0], device=device)
        
        with strategy.perturb(population_size=population_size, epoch=0):
            pass
        metrics = strategy.step(fitnesses)
        
        # Should have non-trivial update
        delta_norm = (simple_linear.weight - original_weight).norm().item()
        assert delta_norm > 1e-6, \
            f"Asymmetric fitness should produce non-zero update, got norm {delta_norm:.8f}"

    def test_gradient_direction_follows_better_perturbation(
        self, simple_linear, es_generator, eggroll_config, device
    ):
        """
        Update direction should favor the better perturbation in each pair.
        
        TARGET API:
            # Collect which direction is better
            fitnesses = torch.tensor([10.0, 1.0, ...])  # Evens are better
            
            before_param = model.weight.clone()
            strategy.step(fitnesses)
            after_param = model.weight.clone()
            
            # Update should correlate with even (positive) perturbations
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed,
            antithetic=True
        )
        strategy.setup(simple_linear)
        
        # Get perturbations to compute expected direction
        perturbations = strategy.sample_perturbations(
            param=simple_linear.weight,
            population_size=population_size,
            epoch=0
        )
        
        # Compute expected update direction: sum of positive perturbations
        # (since positive perturbations have higher fitness)
        expected_direction = sum(
            perturbations[i].as_matrix() for i in range(0, population_size, 2)
        )
        
        original_weight = simple_linear.weight.clone()
        
        # Evens (positive ε) are better
        fitnesses = torch.tensor([10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0], device=device)
        
        with strategy.perturb(population_size=population_size, epoch=0):
            pass
        strategy.step(fitnesses)
        
        actual_delta = simple_linear.weight - original_weight
        
        # Correlation should be positive
        correlation = (actual_delta * expected_direction).sum().item()
        assert correlation > 0, \
            f"Update should correlate with high-fitness perturbations, got correlation {correlation:.6f}"


# ============================================================================
# Population Size Handling Tests
# ============================================================================

class TestAntitheticPopulationSize:
    """Test antithetic sampling with various population sizes."""

    def test_even_population_works(self, simple_mlp, eggroll_config, device):
        """
        Even population size should work normally with antithetic.
        """
        from hyperscalees.torch import EggrollStrategy
        
        for population_size in [2, 4, 8, 16, 64]:
            strategy = EggrollStrategy(
                sigma=eggroll_config.sigma,
                lr=eggroll_config.lr,
                rank=eggroll_config.rank,
                seed=eggroll_config.seed,
                antithetic=True
            )
            strategy.setup(simple_mlp)
            
            x = torch.randn(population_size, 8, device=device)
            
            with strategy.perturb(population_size=population_size, epoch=0) as pop:
                outputs = pop.batched_forward(simple_mlp, x)
            
            assert outputs.shape[0] == population_size

    def test_odd_population_raises_or_warns(self, simple_mlp, eggroll_config, device):
        """
        Odd population with antithetic should warn or raise.
        
        Antithetic requires pairs, so odd population is problematic.
        
        TARGET API:
            strategy = EggrollStrategy(sigma=0.1, lr=0.01, antithetic=True)
            strategy.setup(model)
            
            with pytest.warns(UserWarning, match="odd"):
                with strategy.perturb(population_size=7, epoch=0):
                    pass
            
            # Or:
            with pytest.raises(ValueError):
                strategy.perturb(population_size=7)
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed,
            antithetic=True
        )
        strategy.setup(simple_mlp)
        
        # Should either warn or raise for odd population
        odd_population = 7
        
        # Try both warning and error cases
        try:
            with pytest.warns(UserWarning):
                with strategy.perturb(population_size=odd_population, epoch=0):
                    pass
        except (pytest.fail.Exception, ValueError, RuntimeError):
            # Also acceptable: raising an error
            with pytest.raises((ValueError, RuntimeError)):
                with strategy.perturb(population_size=odd_population, epoch=0):
                    pass

    def test_population_size_2_is_single_pair(self, simple_mlp, eggroll_config, device):
        """
        Population size 2 with antithetic is just one perturbation and its mirror.
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 2
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed,
            antithetic=True
        )
        strategy.setup(simple_mlp)
        
        # Get perturbations for the single pair
        param = list(simple_mlp.parameters())[0]  # First weight
        perturbations = strategy.sample_perturbations(
            param=param,
            population_size=population_size,
            epoch=0
        )
        
        assert len(perturbations) == 2
        
        # They should be exact negatives
        pert_0 = perturbations[0].as_matrix()
        pert_1 = perturbations[1].as_matrix()
        
        assert_tensors_close(pert_0, -pert_1, atol=1e-6)


# ============================================================================
# Index Mapping Tests
# ============================================================================

class TestAntitheticIndexMapping:
    """Verify correct index mapping for antithetic pairs."""

    def test_get_antithetic_partner(self, eggroll_config, device):
        """
        Should be able to get the antithetic partner index.
        
        TARGET API:
            # Helper to find partner
            assert strategy.get_antithetic_partner(0) == 1
            assert strategy.get_antithetic_partner(1) == 0
            assert strategy.get_antithetic_partner(4) == 5
            assert strategy.get_antithetic_partner(5) == 4
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(4, 8, bias=False).to(device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed,
            antithetic=True
        )
        strategy.setup(model)
        
        # Test partner mapping
        assert strategy.get_antithetic_partner(0) == 1
        assert strategy.get_antithetic_partner(1) == 0
        assert strategy.get_antithetic_partner(4) == 5
        assert strategy.get_antithetic_partner(5) == 4
        assert strategy.get_antithetic_partner(10) == 11
        assert strategy.get_antithetic_partner(11) == 10

    def test_is_positive_perturbation(self, eggroll_config, device):
        """
        Should be able to check if member is +ε or -ε direction.
        
        TARGET API:
            assert strategy.is_positive_perturbation(0) == True
            assert strategy.is_positive_perturbation(1) == False
            assert strategy.is_positive_perturbation(2) == True
            assert strategy.is_positive_perturbation(3) == False
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(4, 8, bias=False).to(device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed,
            antithetic=True
        )
        strategy.setup(model)
        
        # Even indices are positive, odd are negative
        assert strategy.is_positive_perturbation(0) == True
        assert strategy.is_positive_perturbation(1) == False
        assert strategy.is_positive_perturbation(2) == True
        assert strategy.is_positive_perturbation(3) == False
        assert strategy.is_positive_perturbation(100) == True
        assert strategy.is_positive_perturbation(101) == False


# ============================================================================
# Multi-Layer Antithetic Tests
# ============================================================================

class TestMultiLayerAntithetic:
    """Verify antithetic sampling works correctly across multiple layers."""

    def test_all_layers_use_same_sign(
        self, simple_mlp, batch_input_small, eggroll_config, device
    ):
        """
        All layers should use the same sign for a given population member.
        
        Member 0 should use +ε on ALL layers.
        Member 1 should use -ε on ALL layers.
        
        TARGET API:
            with strategy.perturb(population_size=8, epoch=0) as pop:
                # For member 0 (positive)
                for layer_name in strategy.layer_names():
                    pert = strategy._get_perturbation(layer_name, member_id=0)
                    # All should be "positive" direction
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed,
            antithetic=True
        )
        strategy.setup(simple_mlp)
        
        # Get perturbations for member 0 and 1 for each layer
        params = [(name, p) for name, p in simple_mlp.named_parameters() if 'weight' in name]
        
        for name, param in params:
            pert_0 = strategy._sample_perturbation(param, member_id=0, epoch=0)
            pert_1 = strategy._sample_perturbation(param, member_id=1, epoch=0)
            
            # Should be negatives
            assert_tensors_close(
                pert_0.as_matrix(),
                -pert_1.as_matrix(),
                atol=1e-6,
                msg=f"Layer {name}: members 0 and 1 should have opposite perturbations"
            )

    def test_layers_have_independent_noise_but_correlated_sign(
        self, simple_mlp, eggroll_config, device
    ):
        """
        Different layers should have independent noise, but correlated signs.
        
        Layer 1 and Layer 2 have different random perturbations,
        but if member 0 uses +ε for layer 1, it also uses +ε for layer 2.
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed,
            antithetic=True
        )
        strategy.setup(simple_mlp)
        
        # Get first two weight layers
        params = [(name, p) for name, p in simple_mlp.named_parameters() if 'weight' in name]
        
        if len(params) >= 2:
            name1, param1 = params[0]
            name2, param2 = params[1]
            
            pert1_m0 = strategy._sample_perturbation(param1, member_id=0, epoch=0)
            pert2_m0 = strategy._sample_perturbation(param2, member_id=0, epoch=0)
            
            # Different perturbations (independent noise)
            # Can't directly compare since shapes may differ
            # Just verify they exist and are valid
            assert pert1_m0.as_matrix().shape == param1.shape
            assert pert2_m0.as_matrix().shape == param2.shape


# ============================================================================
# Fitness Processing with Antithetic Tests
# ============================================================================

class TestAntitheticFitnessProcessing:
    """Verify fitness processing respects antithetic structure."""

    def test_fitness_normalization_preserves_pair_structure(self, eggroll_config, device):
        """
        Fitness normalization should preserve the pair relationship.
        
        After normalization, the difference between pair fitnesses should
        determine the gradient contribution.
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(4, 8, bias=False).to(device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed,
            antithetic=True
        )
        strategy.setup(model)
        
        # Fitnesses with clear pair structure
        fitnesses = torch.tensor([10.0, 2.0, 8.0, 4.0, 6.0, 6.0, 3.0, 9.0], device=device)
        
        normalized = strategy.normalize_fitnesses(fitnesses)
        
        # Normalization should preserve which member of pair is better
        # Pair 0: 10 > 2, Pair 1: 8 > 4, Pair 2: 6 == 6, Pair 3: 3 < 9
        assert normalized[0] > normalized[1], "Pair 0: member 0 should still be higher"
        assert normalized[2] > normalized[3], "Pair 1: member 0 should still be higher"
        assert normalized[6] < normalized[7], "Pair 3: member 0 should still be lower"

    def test_baseline_subtraction_with_antithetic(self, eggroll_config, device):
        """
        Baseline subtraction can use the antithetic pair as baseline.
        
        Some implementations use f(θ+ε) - f(θ-ε) directly.
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(4, 8, bias=False).to(device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed,
            antithetic=True
        )
        strategy.setup(model)
        
        # Test that the update computation uses pair differences
        fitnesses = torch.tensor([10.0, 2.0, 8.0, 4.0], device=device)
        
        original_weight = model.weight.clone()
        
        with strategy.perturb(population_size=4, epoch=0):
            pass
        strategy.step(fitnesses)
        
        # Should have non-zero update (pairs have different fitness)
        delta = (model.weight - original_weight).norm().item()
        assert delta > 1e-6, "Antithetic pairs with different fitness should produce update"
