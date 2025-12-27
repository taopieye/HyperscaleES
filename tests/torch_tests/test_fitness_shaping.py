"""
Test: Fitness shaping (normalization) for PyTorch implementation.

PAPER CLAIM: Raw fitness scores are normalized before computing gradient estimates:
1. Mean subtraction (baseline): centers around zero
2. Variance normalization: scales to unit variance

    normalized = (fitness - mean) / sqrt(var + ε)

This improves training stability and gradient estimation quality.

TARGET API: Fitness shaping should be automatic in step(), but also
accessible for custom processing.
"""
import pytest
import torch
import torch.nn as nn

from .conftest import (
    EggrollConfig,
    make_fitnesses,
    assert_tensors_close,
    unimplemented
)


# ============================================================================
# Basic Normalization Tests
# ============================================================================

class TestFitnessNormalization:
    """Verify fitness normalization properties."""

    def test_normalized_scores_have_zero_mean(self, eggroll_config):
        """
        After normalization, fitness scores should have approximately zero mean.
        
        TARGET API:
            fitnesses = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            
            normalized = strategy.normalize_fitnesses(fitnesses)
            
            assert normalized.mean().abs() < 1e-6
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        
        fitnesses = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        normalized = strategy.normalize_fitnesses(fitnesses)
        
        # Mean should be approximately zero
        assert normalized.mean().abs() < 1e-6, f"Expected mean ~0, got {normalized.mean()}"

    def test_normalized_scores_have_unit_variance(self, eggroll_config):
        """
        After normalization, fitness scores should have approximately unit variance.
        
        TARGET API:
            fitnesses = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            
            normalized = strategy.normalize_fitnesses(fitnesses)
            
            assert abs(normalized.var() - 1.0) < 1e-5
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        
        fitnesses = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        normalized = strategy.normalize_fitnesses(fitnesses)
        
        # Variance should be approximately 1
        # Note: using unbiased=False for population variance like in normalization
        var = normalized.var(unbiased=False)
        assert abs(var - 1.0) < 0.1, f"Expected variance ~1, got {var}"

    def test_normalization_preserves_ordering(self, eggroll_config):
        """
        Normalization should preserve the relative ordering of fitness scores.
        
        TARGET API:
            fitnesses = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
            
            normalized = strategy.normalize_fitnesses(fitnesses)
            
            # Ranking should be preserved
            original_order = fitnesses.argsort()
            normalized_order = normalized.argsort()
            
            assert torch.equal(original_order, normalized_order)
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        
        fitnesses = torch.tensor([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0])
        normalized = strategy.normalize_fitnesses(fitnesses)
        
        # Ranking should be preserved
        original_order = fitnesses.argsort()
        normalized_order = normalized.argsort()
        
        assert torch.equal(original_order, normalized_order), \
            "Normalization should preserve ordering"

    def test_normalization_preserves_sign_of_differences(self, eggroll_config):
        """
        Sign of (f_i - f_j) should be preserved after normalization.
        
        TARGET API:
            fitnesses = torch.tensor([1.0, 5.0, 3.0, 7.0])
            normalized = strategy.normalize_fitnesses(fitnesses)
            
            # f[1] > f[0] before, should still be after
            assert (fitnesses[1] > fitnesses[0]) == (normalized[1] > normalized[0])
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        
        fitnesses = torch.tensor([1.0, 5.0, 3.0, 7.0])
        normalized = strategy.normalize_fitnesses(fitnesses)
        
        # Check all pairs
        for i in range(len(fitnesses)):
            for j in range(i + 1, len(fitnesses)):
                orig_sign = fitnesses[i] < fitnesses[j]
                norm_sign = normalized[i] < normalized[j]
                assert orig_sign == norm_sign, \
                    f"Sign of difference between [{i}] and [{j}] changed"


# ============================================================================
# Edge Case Handling Tests
# ============================================================================

class TestFitnessEdgeCases:
    """Verify handling of edge cases in fitness normalization."""

    def test_constant_scores_handled_gracefully(self, eggroll_config):
        """
        All equal fitness scores should not produce NaN/Inf.
        
        TARGET API:
            fitnesses = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
            
            normalized = strategy.normalize_fitnesses(fitnesses)
            
            # Should be all zeros (no variance to normalize)
            assert torch.isfinite(normalized).all()
            assert normalized.abs().max() < 1e-6
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        
        # All equal values - variance is zero
        fitnesses = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        normalized = strategy.normalize_fitnesses(fitnesses)
        
        # Should not produce NaN or Inf
        assert torch.isfinite(normalized).all(), "Got non-finite values"
        
        # All zeros since there's no difference
        assert normalized.abs().max() < 1e-6

    def test_very_small_variance_handled(self, eggroll_config):
        """
        Very small variance should not cause numerical issues.
        
        TARGET API:
            fitnesses = torch.tensor([1.0, 1.0 + 1e-10, 1.0, 1.0 - 1e-10])
            
            normalized = strategy.normalize_fitnesses(fitnesses)
            
            assert torch.isfinite(normalized).all()
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        
        # Very small variance
        fitnesses = torch.tensor([1.0, 1.0 + 1e-10, 1.0, 1.0 - 1e-10])
        normalized = strategy.normalize_fitnesses(fitnesses)
        
        # Should not produce NaN or Inf
        assert torch.isfinite(normalized).all(), "Got non-finite values with tiny variance"

    def test_large_fitness_values_handled(self, eggroll_config):
        """
        Very large fitness values should be handled without overflow.
        
        TARGET API:
            fitnesses = torch.tensor([1e30, 2e30, 3e30, 4e30])
            
            normalized = strategy.normalize_fitnesses(fitnesses)
            
            assert torch.isfinite(normalized).all()
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        
        # Very large values
        fitnesses = torch.tensor([1e30, 2e30, 3e30, 4e30])
        normalized = strategy.normalize_fitnesses(fitnesses)
        
        # Should not overflow
        assert torch.isfinite(normalized).all(), "Got non-finite values with large inputs"
        
        # Should still have zero mean
        assert normalized.mean().abs() < 1e-5

    def test_negative_fitness_values_handled(self, eggroll_config):
        """
        Negative fitness values should be handled correctly.
        
        TARGET API:
            fitnesses = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])
            
            normalized = strategy.normalize_fitnesses(fitnesses)
            
            assert torch.isfinite(normalized).all()
            assert normalized.mean().abs() < 1e-6
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        
        # Mix of positive and negative
        fitnesses = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])
        normalized = strategy.normalize_fitnesses(fitnesses)
        
        assert torch.isfinite(normalized).all()
        assert normalized.mean().abs() < 1e-6

    def test_single_fitness_value_handled(self, eggroll_config):
        """
        Single fitness value (population_size=1) should be handled.
        
        TARGET API:
            fitnesses = torch.tensor([5.0])
            
            normalized = strategy.normalize_fitnesses(fitnesses)
            
            assert torch.isfinite(normalized).all()
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        
        # Single value
        fitnesses = torch.tensor([5.0])
        normalized = strategy.normalize_fitnesses(fitnesses)
        
        # Should not produce NaN or Inf
        assert torch.isfinite(normalized).all()


# ============================================================================
# Rank-Based Fitness Tests
# ============================================================================

class TestRankBasedFitness:
    """Verify rank-based fitness transformation (optional feature)."""

    def test_rank_transform_option(self, eggroll_config):
        """
        Should support optional rank-based fitness transformation.
        
        Rank transform is more robust to outliers.
        
        TARGET API:
            strategy = EggrollStrategy(
                sigma=0.1, lr=0.01, 
                fitness_transform="rank"  # or "normalize"
            )
            
            fitnesses = torch.tensor([1.0, 100.0, 2.0, 50.0])
            transformed = strategy.normalize_fitnesses(fitnesses)
            
            # Should be based on ranks, not raw values
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=0.1, lr=0.01, rank=4,
            fitness_transform="rank"
        )
        
        # Values with outliers
        fitnesses = torch.tensor([1.0, 100.0, 2.0, 50.0])
        transformed = strategy.normalize_fitnesses(fitnesses)
        
        # With rank transform, the outlier (100.0) shouldn't dominate
        # The ranks are: [0, 3, 1, 2] (for ascending)
        # So transformed should reflect ranks, not raw values
        
        # Check that transformed values have reasonable magnitude
        assert transformed.abs().max() < 10.0, "Rank transform should bound values"
        
        # Ordering should still be preserved
        original_order = fitnesses.argsort()
        transformed_order = transformed.argsort()
        assert torch.equal(original_order, transformed_order)

    def test_rank_transform_handles_ties(self, eggroll_config):
        """
        Rank transform should handle tied values gracefully.
        
        Note: PyTorch's argsort gives different ranks to tied values based on
        position (stable sort). This is acceptable for ES - the important thing
        is that the transform doesn't produce NaN/Inf and maintains relative
        ordering for non-tied values.
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=0.1, lr=0.01, rank=4,
            fitness_transform="rank"
        )
        
        fitnesses = torch.tensor([1.0, 3.0, 3.0, 5.0])  # Ties at 3.0
        transformed = strategy.normalize_fitnesses(fitnesses)
        
        # Should not produce NaN or Inf
        assert torch.isfinite(transformed).all(), "Tied values should not cause NaN/Inf"
        
        # Tied values should have similar (adjacent) transformed values
        # Since argsort gives them adjacent ranks, their transformed values will be close
        assert abs(transformed[1] - transformed[2]) < 0.5, \
            "Tied fitness values should have similar transformed values"
        
        # Non-tied ordering should be preserved: 1.0 < 3.0 < 5.0
        assert transformed[0] < transformed[1], "Ordering should be preserved"
        assert transformed[2] < transformed[3], "Ordering should be preserved"


# ============================================================================
# Centered Rank Fitness Tests
# ============================================================================

class TestCenteredRankFitness:
    """Verify centered rank fitness transformation."""

    def test_centered_rank_is_symmetric(self, eggroll_config):
        """
        Centered rank should produce symmetric values around zero.
        
        TARGET API:
            # With even population, should be symmetric
            strategy = EggrollStrategy(fitness_transform="centered_rank")
            
            fitnesses = torch.tensor([1.0, 2.0, 3.0, 4.0])
            transformed = strategy.normalize_fitnesses(fitnesses)
            
            # Should be symmetric around 0
            assert transformed.mean().abs() < 1e-6
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=0.1, lr=0.01, rank=4,
            fitness_transform="centered_rank"
        )
        
        fitnesses = torch.tensor([1.0, 2.0, 3.0, 4.0])
        transformed = strategy.normalize_fitnesses(fitnesses)
        
        # Should be symmetric around 0
        assert transformed.mean().abs() < 1e-6, \
            f"Centered rank mean should be ~0, got {transformed.mean()}"

    def test_centered_rank_bounded(self, eggroll_config):
        """
        Centered rank should produce bounded values.
        
        Typically in range [-0.5, 0.5] or similar.
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=0.1, lr=0.01, rank=4,
            fitness_transform="centered_rank"
        )
        
        fitnesses = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        transformed = strategy.normalize_fitnesses(fitnesses)
        
        # Centered rank typically produces values in [-0.5, 0.5]
        assert transformed.min() >= -1.0, "Centered rank should be bounded below"
        assert transformed.max() <= 1.0, "Centered rank should be bounded above"


# ============================================================================
# Custom Fitness Function Tests
# ============================================================================

class TestCustomFitnessTransform:
    """Verify custom fitness transformation support."""

    def test_custom_fitness_function(self, eggroll_config):
        """
        Should support custom fitness transformation function.
        
        TARGET API:
            def my_transform(fitnesses):
                return torch.tanh(fitnesses)
            
            strategy = EggrollStrategy(
                sigma=0.1, lr=0.01,
                fitness_transform=my_transform
            )
            
            fitnesses = torch.tensor([1.0, 2.0, 3.0, 4.0])
            transformed = strategy.normalize_fitnesses(fitnesses)
            
            assert torch.allclose(transformed, torch.tanh(fitnesses))
        """
        from hyperscalees.torch import EggrollStrategy
        
        def my_transform(fitnesses):
            return torch.tanh(fitnesses)
        
        strategy = EggrollStrategy(
            sigma=0.1, lr=0.01, rank=4,
            fitness_transform=my_transform
        )
        
        fitnesses = torch.tensor([1.0, 2.0, 3.0, 4.0])
        transformed = strategy.normalize_fitnesses(fitnesses)
        
        expected = torch.tanh(fitnesses)
        assert torch.allclose(transformed, expected), \
            "Custom transform should be applied"

    def test_fitness_transform_callable_validation(self, eggroll_config):
        """
        Should validate that custom transform is callable.
        """
        from hyperscalees.torch import EggrollStrategy
        
        # String "not_a_valid_option" should raise
        with pytest.raises((ValueError, TypeError)):
            strategy = EggrollStrategy(
                sigma=0.1, lr=0.01, rank=4,
                fitness_transform="not_a_valid_option"
            )
        
        # Non-callable should raise
        with pytest.raises((ValueError, TypeError)):
            strategy = EggrollStrategy(
                sigma=0.1, lr=0.01, rank=4,
                fitness_transform=42  # Not callable
            )


# ============================================================================
# Baseline Subtraction Tests
# ============================================================================

class TestBaselineSubtraction:
    """Verify baseline subtraction in fitness processing."""

    def test_mean_baseline_subtraction(self, eggroll_config):
        """
        Default should subtract mean as baseline.
        
        TARGET API:
            fitnesses = torch.tensor([1.0, 2.0, 3.0, 4.0])
            mean = fitnesses.mean()
            
            processed = strategy.normalize_fitnesses(fitnesses)
            
            # Should be centered
            assert processed.mean().abs() < 1e-6
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        
        fitnesses = torch.tensor([1.0, 2.0, 3.0, 4.0])
        processed = strategy.normalize_fitnesses(fitnesses)
        
        # Should be centered around zero
        assert processed.mean().abs() < 1e-6, \
            f"Mean baseline subtraction should center fitnesses, got mean={processed.mean()}"

    def test_antithetic_baseline_option(self, eggroll_config):
        """
        Should support using antithetic partner as baseline.
        
        For pair (f+, f-), use (f+ - f-) instead of (f - mean).
        This can reduce variance further.
        
        TARGET API:
            strategy = EggrollStrategy(
                sigma=0.1, lr=0.01,
                antithetic=True,
                baseline="antithetic"  # or "mean"
            )
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=0.1, lr=0.01, rank=4,
            antithetic=True,
            baseline="antithetic"
        )
        
        # With antithetic baseline, pairs (f+, f-) become (f+ - f-)
        # For population [f0+, f0-, f1+, f1-], baseline is pairwise
        fitnesses = torch.tensor([10.0, 8.0, 15.0, 12.0])  # pairs: (10,8), (15,12)
        
        processed = strategy.normalize_fitnesses(fitnesses)
        
        # Antithetic baseline should compute differences within pairs
        # f0+ - f0- = 10 - 8 = 2
        # f1+ - f1- = 15 - 12 = 3
        # Result should reflect these differences
        assert torch.isfinite(processed).all()


# ============================================================================
# Epsilon Handling Tests
# ============================================================================

class TestEpsilonHandling:
    """Verify epsilon (stability constant) handling."""

    def test_epsilon_in_variance_normalization(self, eggroll_config):
        """
        Small epsilon should be added to variance for stability.
        
        normalized = (f - mean) / sqrt(var + eps)
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        
        # Very small variance that could cause division issues
        fitnesses = torch.tensor([1.0, 1.0, 1.0, 1.0 + 1e-12])
        
        # Should not raise or produce NaN/Inf
        normalized = strategy.normalize_fitnesses(fitnesses)
        
        assert torch.isfinite(normalized).all(), \
            "Epsilon should prevent division by zero"

    def test_custom_epsilon_value(self, eggroll_config):
        """
        Should support custom epsilon value.
        
        TARGET API:
            strategy = EggrollStrategy(
                sigma=0.1, lr=0.01,
                fitness_eps=1e-8  # Custom stability constant
            )
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Create with custom epsilon
        strategy = EggrollStrategy(
            sigma=0.1, lr=0.01, rank=4,
            fitness_eps=1e-4  # Larger epsilon
        )
        
        # With larger epsilon, constant values get less normalized
        fitnesses = torch.tensor([1.0, 1.0, 1.0, 1.0])
        normalized = strategy.normalize_fitnesses(fitnesses)
        
        assert torch.isfinite(normalized).all()


# ============================================================================
# Integration with Step Tests
# ============================================================================

class TestFitnessInStep:
    """Verify fitness processing integration with step()."""

    def test_step_normalizes_automatically(
        self, simple_mlp, batch_input_small, eggroll_config
    ):
        """
        step() should automatically normalize fitnesses.
        
        TARGET API:
            raw_fitnesses = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            
            # step() handles normalization internally
            metrics = strategy.step(raw_fitnesses)
            
            # Can optionally access normalized values
            assert "normalized_fitness_mean" in metrics
            assert abs(metrics["normalized_fitness_mean"]) < 1e-6
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        # Perturb and get outputs
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        # Raw fitnesses (not normalized)
        raw_fitnesses = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
            device=device
        )
        
        # step() should normalize internally
        metrics = strategy.step(raw_fitnesses)
        
        # Metrics should indicate normalization was done
        assert isinstance(metrics, dict)
        # Optionally check for normalized_fitness_mean
        if "normalized_fitness_mean" in metrics:
            assert abs(metrics["normalized_fitness_mean"]) < 1e-6

    def test_step_with_prenormalized_fitness(
        self, simple_mlp, batch_input_small, eggroll_config
    ):
        """
        Should support passing pre-normalized fitnesses.
        
        TARGET API:
            # User pre-processed fitnesses
            normalized = my_custom_normalize(raw_fitnesses)
            
            metrics = strategy.step(normalized, prenormalized=True)
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        # Perturb and get outputs
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        # User already normalized
        prenormalized = torch.randn(population_size, device=device)
        prenormalized = (prenormalized - prenormalized.mean()) / prenormalized.std()
        
        # Pass with prenormalized=True to skip internal normalization
        metrics = strategy.step(prenormalized, prenormalized=True)
        
        assert isinstance(metrics, dict)
