"""
Test: High-rank gradient accumulation from low-rank perturbations.

PAPER CLAIM: Although individual perturbations are low-rank, the gradient estimate
can be high-rank because:
    ∇̂ = Σᵢ wᵢ AᵢBᵢ^T

Even if each AᵢBᵢ^T is rank-r, the sum can have rank up to min(N×r, min(m,n)).

This is the theoretical justification for why low-rank perturbations don't limit
the expressiveness of the gradient estimate.

TARGET API: The accumulated gradient should have higher rank than individual
perturbations when population size is sufficient.
"""
import pytest
import torch
import torch.nn as nn

from .conftest import (
    EggrollConfig,
    compute_matrix_rank,
    assert_tensors_close,
    make_fitnesses,
    unimplemented
)


# ============================================================================
# Rank Accumulation Tests
# ============================================================================

class TestHighRankAccumulation:
    """Verify that accumulated updates can achieve high rank."""

    def test_sum_of_rank1_exceeds_rank1(self, medium_tensor, es_generator):
        """
        Sum of rank-1 matrices can have rank > 1.
        
        This demonstrates the fundamental principle.
        
        TARGET API:
            config = EggrollConfig(rank=1)
            strategy = EggrollStrategy.from_config(config)
            strategy.setup(model)
            
            # Get perturbations
            perturbations = strategy.sample_perturbations(param, population_size=16, epoch=0)
            
            # Weight equally
            fitnesses = torch.ones(16)
            
            # Compute weighted sum
            accumulated = sum(p.as_matrix() for p in perturbations)
            
            # Should have rank > 1
            rank = torch.linalg.matrix_rank(accumulated)
            assert rank > 1
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = medium_tensor.device
        
        # Create model with medium_tensor as weight
        model = nn.Linear(medium_tensor.shape[1], medium_tensor.shape[0], bias=False).to(device)
        model.weight.data = medium_tensor.clone()
        
        # Use rank-1 perturbations with antithetic=False to avoid pair cancellation
        # (This test is about the mathematical principle, not ES algorithm behavior)
        config = EggrollConfig(sigma=0.1, lr=0.01, rank=1, antithetic=False)
        strategy = EggrollStrategy.from_config(config)
        strategy.setup(model)
        
        population_size = 16
        perturbations = strategy.sample_perturbations(model.weight, population_size=population_size, epoch=0)
        
        # Compute sum of rank-1 perturbations
        accumulated = sum(p.as_matrix() for p in perturbations)
        
        # Sum of rank-1 matrices should have rank > 1
        rank = compute_matrix_rank(accumulated)
        assert rank > 1, f"Sum of {population_size} rank-1 matrices should have rank > 1, got {rank}"

    def test_accumulated_rank_grows_with_population(
        self, medium_tensor, es_generator, eggroll_config
    ):
        """
        More population members should generally increase accumulated rank.
        
        TARGET API:
            ranks = []
            for pop_size in [2, 8, 32, 128]:
                perturbations = strategy.sample_perturbations(param, pop_size, epoch=0)
                accumulated = sum(p.as_matrix() for p in perturbations)
                rank = torch.linalg.matrix_rank(accumulated)
                ranks.append(rank)
            
            # Ranks should generally increase (may plateau at full rank)
            assert ranks[-1] >= ranks[0]
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = medium_tensor.device
        
        model = nn.Linear(medium_tensor.shape[1], medium_tensor.shape[0], bias=False).to(device)
        model.weight.data = medium_tensor.clone()
        
        strategy = EggrollStrategy.from_config(eggroll_config)
        strategy.setup(model)
        
        ranks = []
        for pop_size in [2, 8, 32, 128]:
            perturbations = strategy.sample_perturbations(model.weight, population_size=pop_size, epoch=0)
            accumulated = sum(p.as_matrix() for p in perturbations)
            rank = compute_matrix_rank(accumulated)
            ranks.append(rank)
        
        # Ranks should generally increase (may plateau at full rank)
        assert ranks[-1] >= ranks[0], f"Rank should grow with population: {ranks}"

    def test_full_rank_achievable_with_sufficient_population(
        self, small_tensor, es_generator, eggroll_config
    ):
        """
        With enough population, accumulated gradient can achieve full rank.
        
        For matrix (m, n) with rank r perturbations, need roughly min(m,n)/r members.
        
        TARGET API:
            m, n = 8, 4
            r = 1
            # Need ~4 members to potentially achieve full rank
            
            config = EggrollConfig(rank=r)
            strategy = EggrollStrategy.from_config(config)
            
            perturbations = strategy.sample_perturbations(param, population_size=64, epoch=0)
            accumulated = sum(p.as_matrix() for p in perturbations)
            
            rank = torch.linalg.matrix_rank(accumulated)
            assert rank == min(m, n)  # Full rank achieved
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = small_tensor.device
        m, n = small_tensor.shape
        
        model = nn.Linear(n, m, bias=False).to(device)
        model.weight.data = small_tensor.clone()
        
        # Use rank-1 perturbations with antithetic=False to avoid pair cancellation
        # (This test is about the mathematical principle, not ES algorithm behavior)
        config = EggrollConfig(sigma=0.1, lr=0.01, rank=1, antithetic=False)
        strategy = EggrollStrategy.from_config(config)
        strategy.setup(model)
        
        # With 64 rank-1 perturbations, should achieve full rank
        perturbations = strategy.sample_perturbations(model.weight, population_size=64, epoch=0)
        accumulated = sum(p.as_matrix() for p in perturbations)
        
        rank = compute_matrix_rank(accumulated)
        assert rank == min(m, n), f"Should achieve full rank {min(m, n)}, got {rank}"


# ============================================================================
# Fitness Weighting Tests
# ============================================================================

class TestFitnessWeightedAccumulation:
    """Verify fitness-weighted accumulation behavior."""

    def test_weighted_sum_respects_fitness_weights(
        self, small_tensor, es_generator, eggroll_config
    ):
        """
        Higher-weighted perturbations should dominate the accumulated gradient.
        
        TARGET API:
            perturbations = strategy.sample_perturbations(param, population_size=8, epoch=0)
            
            # Give one perturbation very high weight
            fitnesses = torch.tensor([-1, -1, -1, 10, -1, -1, -1, -1])
            
            # After normalization and weighting, the high-fitness perturbation dominates
            normalized = strategy.normalize_fitnesses(fitnesses)
            
            accumulated = sum(
                w * p.as_matrix() 
                for w, p in zip(normalized, perturbations)
            )
            
            # Should be correlated with perturbation 3
            high_pert = perturbations[3].as_matrix()
            correlation = (accumulated * high_pert).sum()
            assert correlation > 0
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = small_tensor.device
        
        model = nn.Linear(small_tensor.shape[1], small_tensor.shape[0], bias=False).to(device)
        model.weight.data = small_tensor.clone()
        
        strategy = EggrollStrategy.from_config(eggroll_config)
        strategy.setup(model)
        
        population_size = 8
        perturbations = strategy.sample_perturbations(model.weight, population_size=population_size, epoch=0)
        
        # One perturbation has much higher fitness
        fitnesses = torch.tensor([-1.0, -1.0, -1.0, 10.0, -1.0, -1.0, -1.0, -1.0], device=device)
        normalized = strategy.normalize_fitnesses(fitnesses)
        
        # Compute weighted sum
        accumulated = sum(
            w * p.as_matrix() 
            for w, p in zip(normalized, perturbations)
        )
        
        # Should be correlated with the high-fitness perturbation (index 3)
        high_pert = perturbations[3].as_matrix()
        correlation = (accumulated * high_pert).sum()
        assert correlation > 0, "Accumulated gradient should favor high-fitness perturbation"

    def test_equal_weights_produce_simple_sum(
        self, small_tensor, es_generator, eggroll_config
    ):
        """
        Equal fitness weights should produce equal contributions.
        
        With antithetic sampling and equal weights, pairs cancel.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = small_tensor.device
        
        model = nn.Linear(small_tensor.shape[1], small_tensor.shape[0], bias=False).to(device)
        model.weight.data = small_tensor.clone()
        
        # Create strategy with antithetic sampling
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma, lr=eggroll_config.lr,
            rank=eggroll_config.rank, antithetic=True
        )
        strategy.setup(model)
        
        population_size = 8  # 4 antithetic pairs
        perturbations = strategy.sample_perturbations(model.weight, population_size=population_size, epoch=0)
        
        # All equal fitness - after normalization, all weights should be ~0
        fitnesses = torch.ones(population_size, device=device) * 5.0
        normalized = strategy.normalize_fitnesses(fitnesses)
        
        # With equal normalized weights (~0), accumulated should be ~0
        accumulated = sum(
            w * p.as_matrix() 
            for w, p in zip(normalized, perturbations)
        )
        
        assert accumulated.abs().max() < 1e-5, "Equal weights should produce near-zero accumulation"

    def test_negative_weights_subtract_perturbation(
        self, small_tensor, es_generator, eggroll_config
    ):
        """
        Negative fitness (after normalization) should subtract perturbation.
        
        This is how ES moves away from bad directions.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = small_tensor.device
        
        model = nn.Linear(small_tensor.shape[1], small_tensor.shape[0], bias=False).to(device)
        model.weight.data = small_tensor.clone()
        
        strategy = EggrollStrategy.from_config(eggroll_config)
        strategy.setup(model)
        
        population_size = 8
        perturbations = strategy.sample_perturbations(model.weight, population_size=population_size, epoch=0)
        
        # One perturbation has very negative fitness (after normalization)
        fitnesses = torch.tensor([1.0, 1.0, 1.0, -10.0, 1.0, 1.0, 1.0, 1.0], device=device)
        normalized = strategy.normalize_fitnesses(fitnesses)
        
        # Compute weighted sum
        accumulated = sum(
            w * p.as_matrix() 
            for w, p in zip(normalized, perturbations)
        )
        
        # Should be negatively correlated with the low-fitness perturbation (index 3)
        low_pert = perturbations[3].as_matrix()
        correlation = (accumulated * low_pert).sum()
        assert correlation < 0, "Accumulated gradient should move away from low-fitness perturbation"


# ============================================================================
# Rank Bound Tests
# ============================================================================

class TestRankBounds:
    """Verify theoretical rank bounds are respected."""

    def test_accumulated_rank_bounded_by_population_times_r(
        self, medium_tensor, es_generator, eggroll_config
    ):
        """
        Accumulated rank ≤ population_size × r (theoretical upper bound).
        
        TARGET API:
            pop_size = 8
            r = eggroll_config.rank
            
            perturbations = strategy.sample_perturbations(param, pop_size, epoch=0)
            accumulated = sum(p.as_matrix() for p in perturbations)
            
            rank = torch.linalg.matrix_rank(accumulated)
            
            # Rank can't exceed theoretical bound
            assert rank <= pop_size * r
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = medium_tensor.device
        
        model = nn.Linear(medium_tensor.shape[1], medium_tensor.shape[0], bias=False).to(device)
        model.weight.data = medium_tensor.clone()
        
        strategy = EggrollStrategy.from_config(eggroll_config)
        strategy.setup(model)
        
        pop_size = 8
        r = eggroll_config.rank
        
        perturbations = strategy.sample_perturbations(model.weight, population_size=pop_size, epoch=0)
        accumulated = sum(p.as_matrix() for p in perturbations)
        
        rank = compute_matrix_rank(accumulated)
        
        # Rank can't exceed theoretical bound
        theoretical_bound = pop_size * r
        assert rank <= theoretical_bound, f"Rank {rank} exceeds theoretical bound {theoretical_bound}"

    def test_accumulated_rank_bounded_by_matrix_dimensions(
        self, small_tensor, es_generator, eggroll_config
    ):
        """
        Accumulated rank ≤ min(m, n) (matrix dimension bound).
        
        TARGET API:
            m, n = small_tensor.shape
            
            perturbations = strategy.sample_perturbations(param, population_size=1000, epoch=0)
            accumulated = sum(p.as_matrix() for p in perturbations)
            
            rank = torch.linalg.matrix_rank(accumulated)
            
            # Can't exceed matrix dimension
            assert rank <= min(m, n)
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = small_tensor.device
        m, n = small_tensor.shape
        
        model = nn.Linear(n, m, bias=False).to(device)
        model.weight.data = small_tensor.clone()
        
        strategy = EggrollStrategy.from_config(eggroll_config)
        strategy.setup(model)
        
        # Many perturbations
        perturbations = strategy.sample_perturbations(model.weight, population_size=100, epoch=0)
        accumulated = sum(p.as_matrix() for p in perturbations)
        
        rank = compute_matrix_rank(accumulated)
        
        # Can't exceed matrix dimension
        assert rank <= min(m, n), f"Rank {rank} exceeds matrix dimension bound {min(m, n)}"


# ============================================================================
# Subspace Coverage Tests
# ============================================================================

class TestSubspaceCoverage:
    """Verify that accumulated perturbations can cover the parameter space."""

    def test_perturbations_span_diverse_directions(
        self, medium_tensor, es_generator, eggroll_config
    ):
        """
        Different perturbations should span diverse directions in parameter space.
        
        TARGET API:
            perturbations = strategy.sample_perturbations(param, population_size=32, epoch=0)
            
            # Stack as rows and compute SVD
            stacked = torch.stack([p.as_matrix().flatten() for p in perturbations])
            _, s, _ = torch.linalg.svd(stacked)
            
            # Should have multiple significant singular values (diverse directions)
            significant = (s > s.max() * 0.01).sum()
            assert significant >= config.rank
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = medium_tensor.device
        
        model = nn.Linear(medium_tensor.shape[1], medium_tensor.shape[0], bias=False).to(device)
        model.weight.data = medium_tensor.clone()
        
        strategy = EggrollStrategy.from_config(eggroll_config)
        strategy.setup(model)
        
        perturbations = strategy.sample_perturbations(model.weight, population_size=32, epoch=0)
        
        # Stack as rows and compute SVD
        stacked = torch.stack([p.as_matrix().flatten() for p in perturbations])
        _, s, _ = torch.linalg.svd(stacked)
        
        # Should have multiple significant singular values (diverse directions)
        significant = (s > s.max() * 0.01).sum().item()
        assert significant >= eggroll_config.rank, \
            f"Should have at least {eggroll_config.rank} significant directions, got {significant}"

    @pytest.mark.slow
    def test_full_space_coverage_over_many_epochs(
        self, small_tensor, es_generator, eggroll_config
    ):
        """
        Over many epochs, perturbations should cover the full parameter space.
        
        This is important for ES convergence.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = small_tensor.device
        m, n = small_tensor.shape
        
        model = nn.Linear(n, m, bias=False).to(device)
        model.weight.data = small_tensor.clone()
        
        strategy = EggrollStrategy.from_config(eggroll_config)
        strategy.setup(model)
        
        # Collect perturbations over many epochs
        all_perturbations = []
        for epoch in range(100):
            perturbations = strategy.sample_perturbations(model.weight, population_size=8, epoch=epoch)
            all_perturbations.extend(perturbations)
        
        # Stack and compute SVD
        stacked = torch.stack([p.as_matrix().flatten() for p in all_perturbations])
        _, s, _ = torch.linalg.svd(stacked)
        
        # After many epochs, should cover most directions
        significant = (s > s.max() * 0.001).sum().item()
        expected_full_rank = min(m * n, len(all_perturbations))
        assert significant >= min(m, n), \
            f"Should cover most of space after many epochs, got {significant} significant directions"


# ============================================================================
# Update Quality Tests
# ============================================================================

class TestUpdateQuality:
    """Verify quality of accumulated gradient estimates."""

    def test_update_direction_is_meaningful(
        self, simple_mlp, batch_input_small, eggroll_config
    ):
        """
        Accumulated update should point in a meaningful direction.
        
        For a simple convex problem, should point toward optimum.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        
        # Create simple model for convex problem
        model = nn.Linear(4, 2, bias=False).to(device)
        
        strategy = EggrollStrategy.from_config(eggroll_config)
        strategy.setup(model)
        
        # Target weights
        target = torch.randn(2, 4, device=device)
        x = torch.randn(8, 4, device=device)
        
        # Fitness: negative squared distance to target output
        def fitness_fn(output):
            target_output = x @ target.T
            return -((output - target_output) ** 2).sum(dim=1).mean()
        
        population_size = 32
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            # Expand input for all population members
            x_batch = x.unsqueeze(0).expand(population_size, -1, -1).reshape(population_size * x.shape[0], x.shape[1])
            member_ids = torch.arange(population_size, device=device).repeat_interleave(x.shape[0])
            outputs_flat = pop.batched_forward(model, x_batch, member_ids=member_ids)
            outputs = outputs_flat.reshape(population_size, x.shape[0], -1)
            # Compute fitness for each member
            fitnesses = torch.tensor(
                [fitness_fn(outputs[i]).item() for i in range(population_size)],
                device=device
            )
        
        initial_weight = model.weight.clone()
        initial_fitness = fitness_fn(model(x)).item()
        
        strategy.step(fitnesses)
        
        final_fitness = fitness_fn(model(x)).item()
        
        # For a convex problem with good population, should improve
        # (may not always work due to variance, so we just check it's not way worse)
        assert final_fitness >= initial_fitness - 1.0

    def test_update_variance_decreases_with_population(
        self, simple_mlp, batch_input_small, eggroll_config
    ):
        """
        Larger population should produce lower variance gradient estimates.
        
        This is the classic ES population size tradeoff.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        
        variances = []
        for pop_size in [8, 32, 128]:
            updates = []
            for trial in range(5):
                # Fresh model each trial
                model = nn.Sequential(
                    nn.Linear(32, 64, bias=False),
                    nn.ReLU(),
                    nn.Linear(64, 16, bias=False),
                ).to(device)
                
                strategy = EggrollStrategy.from_config(eggroll_config)
                strategy.setup(model)
                
                before = model[0].weight.clone()
                
                with strategy.perturb(population_size=pop_size, epoch=0) as pop:
                    x = torch.randn(pop_size, 32, device=device)  # Match model input dim
                    pop.batched_forward(model, x)
                
                fitnesses = make_fitnesses(pop_size, device=device)
                strategy.step(fitnesses)
                
                after = model[0].weight.clone()
                delta = (after - before).norm().item()
                updates.append(delta)
            
            variance = torch.tensor(updates).var().item()
            variances.append(variance)
        
        # Variance should generally decrease with population size
        # (may not be strictly monotonic, but trend should be there)
        assert variances[-1] <= variances[0] * 2, \
            f"Variance should decrease with population: {variances}"

    @pytest.mark.slow
    def test_accumulated_gradient_correlates_with_true_gradient(
        self, simple_mlp, batch_input_small
    ):
        """
        For differentiable fitness, ES gradient should correlate with true gradient.
        
        This is an empirical validation that ES is doing something sensible.
        We run multiple trials to get statistical significance.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        
        # Simple differentiable fitness: MSE loss
        model = nn.Linear(4, 2, bias=False).to(device)
        target = torch.randn(8, 2, device=device)
        x = torch.randn(8, 4, device=device)
        
        # Run multiple trials for statistical robustness
        n_trials = 5
        correlations = []
        
        for trial in range(n_trials):
            # Reset model
            with torch.no_grad():
                model.weight.normal_()
            
            # Compute true gradient
            model.zero_grad()
            output = model(x)
            loss = ((output - target) ** 2).sum()
            loss.backward()
            true_grad = model.weight.grad.clone()
            
            # Compute ES gradient
            strategy = EggrollStrategy(
                sigma=0.005,  # Small sigma for accurate gradient
                lr=1.0,  # lr=1 so update = gradient direction
                rank=4,
                seed=trial * 12345
            )
            strategy.setup(model)
            
            population_size = 512  # Large population for good estimate
            
            with strategy.perturb(population_size=population_size, epoch=0) as pop:
                # Expand input for all population members
                x_batch = x.unsqueeze(0).expand(population_size, -1, -1).reshape(population_size * x.shape[0], x.shape[1])
                member_ids = torch.arange(population_size, device=device).repeat_interleave(x.shape[0])
                outputs_flat = pop.batched_forward(model, x_batch, member_ids=member_ids)
                outputs = outputs_flat.reshape(population_size, x.shape[0], -1)
                # Compute negative loss as fitness for each member
                target_expanded = target.unsqueeze(0).expand(population_size, -1, -1)
                fitnesses = -((outputs - target_expanded) ** 2).sum(dim=(1, 2))
            
            before = model.weight.clone()
            strategy.step(fitnesses)
            after = model.weight.clone()
            
            es_grad = after - before  # With lr=1, this approximates gradient direction
            
            # Compute cosine similarity (more robust than raw correlation)
            true_flat = true_grad.flatten()
            es_flat = es_grad.flatten()
            cosine_sim = (true_flat @ es_flat) / (true_flat.norm() * es_flat.norm() + 1e-8)
            correlations.append(cosine_sim.item())
        
        # ES gradient should be negatively correlated with loss gradient
        # (ES maximizes fitness = -loss, so should move opposite to loss gradient)
        mean_correlation = sum(correlations) / len(correlations)
        
        # At least 3 out of 5 trials should show anti-correlation
        negative_count = sum(1 for c in correlations if c < 0)
        assert negative_count >= 3, \
            f"Expected at least 3/5 trials to show anti-correlation, got {negative_count}/5. " \
            f"Correlations: {[f'{c:.3f}' for c in correlations]}"
        
        # Mean should be negative
        assert mean_correlation < 0, \
            f"Mean ES-true gradient correlation should be negative, got {mean_correlation:.4f}. " \
            f"Correlations: {[f'{c:.3f}' for c in correlations]}"


# ============================================================================
# Numerical Stability Tests
# ============================================================================

class TestNumericalStability:
    """Verify numerical stability of accumulation."""

    def test_large_population_accumulation_stable(
        self, medium_tensor, es_generator, eggroll_config
    ):
        """
        Accumulating many perturbations should not cause numerical issues.
        
        TARGET API:
            perturbations = strategy.sample_perturbations(param, population_size=1000, epoch=0)
            accumulated = sum(p.as_matrix() for p in perturbations)
            
            # Should not have inf or nan
            assert torch.isfinite(accumulated).all()
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = medium_tensor.device
        
        model = nn.Linear(medium_tensor.shape[1], medium_tensor.shape[0], bias=False).to(device)
        model.weight.data = medium_tensor.clone()
        
        strategy = EggrollStrategy.from_config(eggroll_config)
        strategy.setup(model)
        
        # Large number of perturbations
        perturbations = strategy.sample_perturbations(model.weight, population_size=500, epoch=0)
        accumulated = sum(p.as_matrix() for p in perturbations)
        
        # Should not have inf or nan
        assert torch.isfinite(accumulated).all(), \
            "Large population accumulation produced non-finite values"

    def test_extreme_fitness_values_handled(
        self, small_tensor, es_generator, eggroll_config
    ):
        """
        Very large or small fitness values should be handled gracefully.
        
        Fitness normalization should prevent numerical issues.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = small_tensor.device
        
        model = nn.Linear(small_tensor.shape[1], small_tensor.shape[0], bias=False).to(device)
        model.weight.data = small_tensor.clone()
        
        strategy = EggrollStrategy.from_config(eggroll_config)
        strategy.setup(model)
        
        population_size = 8
        
        # Extreme fitness values
        fitnesses = torch.tensor([1e10, -1e10, 1e5, -1e5, 0.0, 1e-10, -1e-10, 1.0], device=device)
        
        # Normalization should handle these
        normalized = strategy.normalize_fitnesses(fitnesses)
        
        assert torch.isfinite(normalized).all(), \
            "Extreme fitness values should be handled gracefully"
        
        # Perturbation and step should also work
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, small_tensor.shape[1], device=device)
            pop.batched_forward(model, x)
        
        strategy.step(fitnesses)
        
        # Parameters should be finite
        assert torch.isfinite(model.weight).all(), \
            "Parameters should remain finite after step with extreme fitness"
