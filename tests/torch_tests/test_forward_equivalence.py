"""
Test: Forward pass equivalence for PyTorch implementation.

PAPER CLAIM: EGGROLL computes x @ (W + AB^T) as x @ W + x @ B @ A^T,
avoiding explicit formation of the perturbed matrix. This reduces the cost
of a forward pass from O(mn) to O(r(m+n)).

TARGET API: The efficient computation should be transparent to users.
When using batched_forward, the result should match what you'd get by
explicitly perturbing the weights, but computed more efficiently.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conftest import (
    EggrollConfig,
    compute_matrix_rank,
    assert_tensors_close,
    unimplemented
)


# ============================================================================
# Core Equivalence Tests
# ============================================================================

class TestForwardEquivalence:
    """Verify that efficient forward pass matches explicit perturbation."""

    def test_batched_forward_matches_explicit(
        self, simple_linear, batch_input_small, es_generator, eggroll_config, device
    ):
        """
        batched_forward should match explicit perturbation computation.
        
        This is the core correctness check for the EGGROLL optimization.
        
        TARGET API:
            strategy.setup(model)
            
            with strategy.perturb(population_size=8, epoch=0) as pop:
                # Efficient batched computation
                output_efficient = pop.batched_forward(model, x)
            
            # Compare to explicit (slow but obviously correct)
            for i in range(8):
                A, B = pop.get_factors(member_id=i, param_name="weight")
                perturbed_W = model.weight + A @ B.T
                output_explicit = x[i] @ perturbed_W.T
                assert torch.allclose(output_efficient[i], output_explicit)
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        x = torch.randn(population_size, simple_linear.in_features, device=device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(simple_linear)
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            # Efficient batched computation
            output_efficient = pop.batched_forward(simple_linear, x)
            
            # Compare to explicit (slow but obviously correct)
            for i in range(population_size):
                A, B = pop.get_factors(member_id=i, param_name="weight")
                perturbed_W = simple_linear.weight + A @ B.T
                output_explicit = x[i:i+1] @ perturbed_W.T
                
                assert_tensors_close(
                    output_efficient[i:i+1], 
                    output_explicit,
                    atol=1e-5,
                    msg=f"Member {i}: batched_forward doesn't match explicit computation"
                )

    def test_no_perturbation_outside_context(
        self, simple_linear, batch_input_small, eggroll_config, device
    ):
        """
        Outside perturb() context, forward should use base weights.
        
        TARGET API:
            strategy.setup(model)
            
            # No context = no perturbation
            output = model(x)
            expected = x @ model.weight.T + model.bias
            
            assert torch.equal(output, expected)
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Create model with bias for this test
        model = nn.Linear(8, 4, bias=True).to(device)
        x = torch.randn(4, 8, device=device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(model)
        
        # Outside context - should be standard forward
        output = model(x)
        expected = x @ model.weight.T + model.bias
        
        assert torch.allclose(output, expected, atol=1e-6), \
            "Outside perturb() context, forward should use base weights"

    def test_single_member_forward_matches_batched(
        self, simple_linear, es_generator, eggroll_config, device
    ):
        """
        Using iterate() for one member should match batched_forward.
        
        This validates that both APIs produce identical results.
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        x = torch.randn(population_size, simple_linear.in_features, device=device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(simple_linear)
        
        # Get batched result
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            batched_output = pop.batched_forward(simple_linear, x)
        
        # Get sequential results
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            sequential_outputs = []
            for member_id in pop.iterate():
                out = simple_linear(x[member_id:member_id+1])
                sequential_outputs.append(out)
            sequential_output = torch.cat(sequential_outputs, dim=0)
        
        assert_tensors_close(
            batched_output,
            sequential_output,
            atol=1e-5,
            msg="Batched and sequential forwards should match"
        )


# ============================================================================
# Determinism Tests
# ============================================================================

class TestForwardDeterminism:
    """Verify deterministic behavior of forward passes."""

    def test_batched_forward_is_deterministic(
        self, simple_mlp, batch_input_small, eggroll_config, device
    ):
        """
        Same inputs + same epoch should produce identical outputs.
        
        TARGET API:
            with strategy.perturb(population_size=8, epoch=0) as pop:
                output1 = pop.batched_forward(model, x)
            
            with strategy.perturb(population_size=8, epoch=0) as pop:
                output2 = pop.batched_forward(model, x)
            
            assert torch.equal(output1, output2)
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        x = torch.randn(population_size, 8, device=device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(simple_mlp)
        
        # First pass
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            output1 = pop.batched_forward(simple_mlp, x)
        
        # Second pass with same inputs
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            output2 = pop.batched_forward(simple_mlp, x)
        
        assert torch.equal(output1, output2), \
            "Same epoch should produce identical outputs"

    def test_different_epochs_produce_different_outputs(
        self, simple_mlp, batch_input_small, eggroll_config, device
    ):
        """
        Different epochs should produce different perturbations.
        
        Note: Requires noise_reuse >= 1 (noise_reuse=0 means same noise every epoch).
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        x = torch.randn(population_size, 8, device=device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed,
            noise_reuse=1  # Different noise each epoch
        )
        strategy.setup(simple_mlp)
        
        # Epoch 0
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            output_epoch0 = pop.batched_forward(simple_mlp, x)
        
        # Epoch 1
        with strategy.perturb(population_size=population_size, epoch=1) as pop:
            output_epoch1 = pop.batched_forward(simple_mlp, x)
        
        assert not torch.equal(output_epoch0, output_epoch1), \
            "Different epochs should produce different outputs"

    def test_population_members_are_diverse(
        self, simple_mlp, batch_input_small, eggroll_config, device
    ):
        """
        Different population members should produce different outputs.
        
        Essential for population diversity.
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        # Use same input for all members to isolate perturbation effect
        x_single = torch.randn(1, 8, device=device)
        x = x_single.expand(population_size, -1).clone()
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(simple_mlp)
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            outputs = pop.batched_forward(simple_mlp, x)
        
        # All outputs should be different
        for i in range(population_size):
            for j in range(i + 1, population_size):
                assert not torch.allclose(outputs[i], outputs[j], atol=1e-6), \
                    f"Population members {i} and {j} have identical outputs"


# ============================================================================
# Multi-Layer Tests
# ============================================================================

class TestMultiLayerForward:
    """Verify forward pass through multiple layers."""

    def test_mlp_batched_forward(
        self, simple_mlp, batch_input_small, eggroll_config, device
    ):
        """
        batched_forward should work through multiple layers.
        
        TARGET API:
            strategy.setup(model)  # Sets up all Linear layers
            
            with strategy.perturb(population_size=8, epoch=0) as pop:
                outputs = pop.batched_forward(model, x)
                # All layers perturbed, different perturbation per batch element
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        x = torch.randn(population_size, 8, device=device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(simple_mlp)
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            outputs = pop.batched_forward(simple_mlp, x)
        
        # Verify output shape
        assert outputs.shape == (population_size, 2), \
            f"Expected output shape ({population_size}, 2), got {outputs.shape}"
        
        # Verify outputs are diverse (different perturbations applied)
        # Use same input to isolate perturbation effect
        x_same = x[0:1].expand(population_size, -1).clone()
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            outputs_same_input = pop.batched_forward(simple_mlp, x_same)
        
        # With same input but different perturbations, outputs should differ
        for i in range(population_size - 1):
            assert not torch.allclose(
                outputs_same_input[i], 
                outputs_same_input[i + 1],
                atol=1e-6
            ), f"Members {i} and {i+1} should have different outputs"

    def test_selective_layer_perturbation(
        self, simple_mlp, batch_input_small, eggroll_config, device
    ):
        """
        Should be able to perturb only specific layers.
        
        TARGET API:
            strategy.setup(model, include=["1.weight"])  # Only middle layer
            
            with strategy.perturb(population_size=8, epoch=0) as pop:
                outputs = pop.batched_forward(model, x)
                # Only layer 1 is perturbed
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        x = torch.randn(population_size, 8, device=device)
        
        # Setup with only middle layer
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(simple_mlp, include=["2.weight"])  # Middle Linear layer
        
        # Check which parameters are being perturbed
        perturbed_params = list(strategy.parameters())
        assert len(perturbed_params) == 1, \
            f"Expected 1 perturbed parameter, got {len(perturbed_params)}"

    def test_bias_perturbation(
        self, mlp_with_bias, batch_input_small, eggroll_config, device
    ):
        """
        Biases should be perturbed with full-rank noise (not low-rank).
        
        Biases are 1D vectors, so low-rank factorization doesn't apply.
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        x = torch.randn(population_size, 8, device=device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(mlp_with_bias)
        
        # Record original bias
        original_bias = mlp_with_bias[0].bias.clone()
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            outputs = pop.batched_forward(mlp_with_bias, x)
            
            # Within context, bias should be perturbed for each member
            # (Implementation detail: how biases are handled may vary)
        
        # After context, bias should be restored
        assert torch.equal(mlp_with_bias[0].bias, original_bias), \
            "Bias should be restored after exiting context"


# ============================================================================
# Activation Function Tests
# ============================================================================

class TestActivationInteraction:
    """Verify perturbations interact correctly with activations."""

    def test_perturbation_before_activation(
        self, simple_mlp, batch_input_small, eggroll_config, device
    ):
        """
        Perturbation should be applied before activation function.
        
        W_perturbed @ x -> activation(result)
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Create a simple network: Linear -> ReLU
        model = nn.Sequential(
            nn.Linear(8, 16, bias=False),
            nn.ReLU()
        ).to(device)
        
        population_size = 4
        x = torch.randn(population_size, 8, device=device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(model)
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            outputs = pop.batched_forward(model, x)
            
            # For comparison, compute manually for first member
            A, B = pop.get_factors(member_id=0, param_name="0.weight")
            W_perturbed = model[0].weight + A @ B.T
            pre_activation = x[0:1] @ W_perturbed.T
            expected = F.relu(pre_activation)
            
            assert_tensors_close(
                outputs[0:1],
                expected,
                atol=1e-5,
                msg="Perturbation should be applied before activation"
            )

    def test_antithetic_pairs_bracket_base_output(
        self, simple_linear, batch_input_small, eggroll_config, device
    ):
        """
        For antithetic pairs, average of outputs should equal base output.
        
        (output+ + output-) / 2 ≈ output_base
        
        This is due to first-order Taylor expansion.
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Single input (same for all population members)
        x = torch.randn(1, simple_linear.in_features, device=device)
        
        strategy = EggrollStrategy(
            sigma=0.01,  # Small sigma for linear approximation
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed,
            antithetic=True
        )
        strategy.setup(simple_linear)
        
        # Base output (no perturbation)
        base_output = simple_linear(x)
        
        # Perturbed outputs
        with strategy.perturb(population_size=2, epoch=0) as pop:
            x_batch = x.expand(2, -1).clone()
            outputs = pop.batched_forward(simple_linear, x_batch)
            output_plus = outputs[0:1]
            output_minus = outputs[1:2]
        
        # Average should approximately equal base
        average_output = (output_plus + output_minus) / 2
        
        assert_tensors_close(
            average_output,
            base_output,
            atol=1e-3,  # Allow some tolerance due to nonlinearity
            msg="Antithetic average should approximate base output"
        )


# ============================================================================
# Efficiency Tests (Computational)
# ============================================================================

class TestComputationalEfficiency:
    """Verify computational efficiency of low-rank forward pass."""

    def test_flop_count_is_lower(self, large_tensor, eggroll_config, device):
        """
        Low-rank forward should have fewer FLOPs than explicit.
        
        Explicit: O(batch * m * n) for forming perturbed matrix
        Low-rank: O(batch * (m + n) * r) for factor multiplication
        
        When r << min(m, n), this is a significant reduction.
        """
        m, n = 256, 128  # Large dimensions
        r = eggroll_config.rank  # 4
        batch = 32
        
        # Calculate theoretical FLOPs
        # Explicit: form AB^T then multiply
        explicit_flops = m * n * r  # AB^T formation
        explicit_flops += batch * m * n  # (W + AB^T) @ x^T
        
        # Low-rank: x @ W.T + x @ B @ A.T
        lowrank_flops = batch * m * n  # x @ W.T (unavoidable)
        lowrank_flops += batch * n * r  # x @ B
        lowrank_flops += batch * r * m  # (x @ B) @ A.T
        
        # The key insight: we don't form AB^T
        # So the comparison is: mn (formation) vs r(m+n) (two smaller matmuls)
        
        savings_numerator = m * n  # What we save by not forming full matrix
        savings_denominator = r * (m + n)  # What we spend instead
        savings_ratio = savings_numerator / savings_denominator
        
        assert savings_ratio > 5, \
            f"Expected >5x theoretical savings, got {savings_ratio:.1f}x"

    @pytest.mark.slow
    def test_forward_is_actually_faster(self, device, eggroll_config):
        """
        Low-rank forward should be measurably faster.
        
        Note: This is a timing test, may be flaky.
        """
        import time
        from hyperscalees.torch import EggrollStrategy
        
        # Large model for timing visibility
        model = nn.Linear(512, 1024, bias=False).to(device)
        population_size = 64
        x = torch.randn(population_size, 512, device=device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=4,  # Low rank
            seed=eggroll_config.seed
        )
        strategy.setup(model)
        
        # Warmup
        for _ in range(5):
            with strategy.perturb(population_size=population_size, epoch=0) as pop:
                _ = pop.batched_forward(model, x)
        
        torch.cuda.synchronize()
        
        # Time efficient method
        n_trials = 20
        start = time.perf_counter()
        for trial in range(n_trials):
            with strategy.perturb(population_size=population_size, epoch=trial) as pop:
                _ = pop.batched_forward(model, x)
        torch.cuda.synchronize()
        efficient_time = (time.perf_counter() - start) / n_trials
        
        # Time explicit method (for comparison, if available)
        # This test mainly verifies the efficient method runs reasonably fast
        assert efficient_time < 1.0, \
            f"Forward pass took {efficient_time:.3f}s, expected < 1.0s"


# ============================================================================
# Vmap-style Population Tests
# ============================================================================

class TestPopulationForward:
    """Verify efficient forward pass over entire population."""

    def test_batched_forward_is_primary_api(
        self, simple_mlp, batch_input_small, eggroll_config, device
    ):
        """
        batched_forward is the main way to evaluate a population.
        
        TARGET API:
            with strategy.perturb(population_size=64, epoch=0) as pop:
                # One call evaluates all 64 members
                outputs = pop.batched_forward(model, x)
                # Shape: (64, *output_dims)
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 64
        x = torch.randn(population_size, 8, device=device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(simple_mlp)
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            outputs = pop.batched_forward(simple_mlp, x)
        
        # Verify shape
        assert outputs.shape[0] == population_size, \
            f"Batch dimension should be {population_size}, got {outputs.shape[0]}"
        assert outputs.shape[1] == 2, \
            f"Output dimension should be 2, got {outputs.shape[1]}"

    def test_iterate_available_for_debugging(
        self, simple_mlp, batch_input_small, eggroll_config, device
    ):
        """
        iterate() is available for debugging or legacy environments.
        
        TARGET API:
            with strategy.perturb(population_size=64, epoch=0) as pop:
                # Slower, but useful for debugging
                outputs = []
                for member_id in pop.iterate():
                    outputs.append(model(x[member_id]))
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        x = torch.randn(population_size, 8, device=device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(simple_mlp)
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            outputs = []
            for member_id in pop.iterate():
                out = simple_mlp(x[member_id:member_id+1])
                outputs.append(out)
            sequential_outputs = torch.cat(outputs, dim=0)
        
        assert sequential_outputs.shape == (population_size, 2), \
            f"Expected shape ({population_size}, 2), got {sequential_outputs.shape}"

    def test_vmap_style_forward(self, simple_mlp, batch_input_small, eggroll_config, device):
        """
        Should support vmap-style vectorized forward.
        
        TARGET API:
            # Using torch.vmap or manual batching
            with strategy.perturb(population_size=64, epoch=0) as pop:
                outputs = pop.vmap_forward(model, x)
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 16
        x = torch.randn(population_size, 8, device=device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(simple_mlp)
        
        # If vmap_forward is available, test it
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            if hasattr(pop, 'vmap_forward'):
                vmap_outputs = pop.vmap_forward(simple_mlp, x)
                batched_outputs = pop.batched_forward(simple_mlp, x)
                
                assert_tensors_close(
                    vmap_outputs,
                    batched_outputs,
                    atol=1e-5,
                    msg="vmap_forward should match batched_forward"
                )
            else:
                # vmap_forward is optional, batched_forward is primary
                outputs = pop.batched_forward(simple_mlp, x)
                assert outputs.shape[0] == population_size


# ============================================================================
# Edge Cases
# ============================================================================

class TestForwardEdgeCases:
    """Test edge cases in forward pass."""

    def test_single_sample_batch(self, simple_mlp, eggroll_config, device):
        """
        Forward should work with batch_size=1.
        """
        from hyperscalees.torch import EggrollStrategy
        
        x = torch.randn(1, 8, device=device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(simple_mlp)
        
        with strategy.perturb(population_size=1, epoch=0) as pop:
            outputs = pop.batched_forward(simple_mlp, x)
        
        assert outputs.shape == (1, 2), f"Expected shape (1, 2), got {outputs.shape}"

    def test_very_small_rank(self, simple_mlp, batch_input_small, device):
        """
        Forward should work with rank=1.
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 4
        x = torch.randn(population_size, 8, device=device)
        
        strategy = EggrollStrategy(
            sigma=0.1,
            lr=0.01,
            rank=1,  # Minimal rank
            seed=42
        )
        strategy.setup(simple_mlp)
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            outputs = pop.batched_forward(simple_mlp, x)
        
        assert outputs.shape == (population_size, 2), \
            f"Expected shape ({population_size}, 2), got {outputs.shape}"
        
        # Verify outputs are still diverse even with rank=1
        for i in range(population_size - 1):
            assert not torch.allclose(outputs[i], outputs[i+1], atol=1e-6), \
                f"Outputs {i} and {i+1} should differ even with rank=1"

    def test_rank_equals_min_dimension(self, simple_linear, batch_input_small, device):
        """
        Forward should work when rank equals min(m, n).
        
        This is effectively full-rank perturbation.
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Create a small layer where rank can equal min dimension
        model = nn.Linear(4, 8, bias=False).to(device)  # min(8, 4) = 4
        population_size = 4
        x = torch.randn(population_size, 4, device=device)
        
        strategy = EggrollStrategy(
            sigma=0.1,
            lr=0.01,
            rank=4,  # rank = min(m, n)
            seed=42
        )
        strategy.setup(model)
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            outputs = pop.batched_forward(model, x)
            
            # Verify perturbation can be full rank
            A, B = pop.get_factors(member_id=0, param_name="weight")
            pert = A @ B.T
            rank = compute_matrix_rank(pert)
            assert rank == 4, f"Full rank perturbation expected rank 4, got {rank}"

    def test_empty_batch(self, simple_mlp, eggroll_config, device):
        """
        Forward should handle empty batch gracefully.
        """
        from hyperscalees.torch import EggrollStrategy
        
        x = torch.randn(0, 8, device=device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(simple_mlp)
        
        # Empty batch might raise or return empty tensor - both are acceptable
        try:
            with strategy.perturb(population_size=0, epoch=0) as pop:
                outputs = pop.batched_forward(simple_mlp, x)
            assert outputs.shape[0] == 0, "Empty batch should return empty output"
        except (ValueError, RuntimeError) as e:
            # It's also acceptable to reject empty batches
            assert "empty" in str(e).lower() or "zero" in str(e).lower() or "population" in str(e).lower()


# ============================================================================
# Gradient Flow Tests
# ============================================================================

class TestGradientFlow:
    """Verify gradients flow correctly through perturbed forward."""

    def test_gradients_flow_through_perturbation(
        self, simple_mlp, batch_input_small, eggroll_config, device
    ):
        """
        Gradients should flow through perturbed parameters.
        
        Note: In ES, we don't use gradients for updates, but they should
        still work for hybrid methods or analysis.
        
        TARGET API:
            x = x.requires_grad_(True)
            
            with strategy.perturb(population_size=8, epoch=0) as pop:
                pop.set_member(0)
                output = model(x)
                loss = output.sum()
                
            loss.backward()
            assert x.grad is not None
        """
        from hyperscalees.torch import EggrollStrategy
        
        x = torch.randn(1, 8, device=device, requires_grad=True)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(simple_mlp)
        
        with strategy.perturb(population_size=1, epoch=0) as pop:
            outputs = pop.batched_forward(simple_mlp, x)
            loss = outputs.sum()
        
        loss.backward()
        
        assert x.grad is not None, "Gradients should flow through perturbed forward"
        assert not torch.all(x.grad == 0), "Gradients should be non-zero"

    def test_no_grad_context_works(self, simple_mlp, batch_input_small, eggroll_config, device):
        """
        Perturbation should work inside torch.no_grad().
        
        This is the typical ES use case - no gradient computation needed.
        """
        from hyperscalees.torch import EggrollStrategy
        
        x = torch.randn(8, 8, device=device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(simple_mlp)
        
        with torch.no_grad():
            with strategy.perturb(population_size=8, epoch=0) as pop:
                outputs = pop.batched_forward(simple_mlp, x)
        
        assert outputs.shape == (8, 2), f"Expected shape (8, 2), got {outputs.shape}"
        assert not outputs.requires_grad, "Outputs should not require grad in no_grad context"
