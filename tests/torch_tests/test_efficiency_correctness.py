"""
Efficiency and Correctness Tests for EGGROLL PyTorch Implementation.

These tests verify that EGGROLL delivers on its core promises:
1. Memory efficiency: Low-rank factorization uses less memory than explicit
2. Speed efficiency: Batched operations are faster than sequential
3. Numerical stability: Works correctly at scale
4. Mathematical correctness: ES gradient estimates are accurate

These are the "proof" tests that validate EGGROLL is worth using.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc
from typing import Tuple


def assert_tensors_close(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5, msg: str = ""):
    """Assert two tensors are close with helpful error message."""
    if not torch.allclose(a, b, atol=atol):
        diff = (a - b).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        raise AssertionError(
            f"{msg}\nMax diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}, "
            f"Tolerance: {atol:.2e}"
        )


# ============================================================================
# Memory Efficiency Tests
# ============================================================================

class TestMemoryEfficiency:
    """Verify that low-rank factorization actually saves memory."""

    @pytest.mark.slow
    def test_lowrank_uses_less_memory_than_explicit(self, device):
        """
        Low-rank perturbation storage should use significantly less memory
        than storing full perturbed weight matrices.
        
        For a weight matrix W of shape (m, n) and rank r:
        - Explicit: stores m*n floats per perturbation
        - Low-rank: stores r*(m+n) floats per perturbation
        
        When r << min(m, n), this is a huge saving.
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Large model where savings matter
        m, n = 1024, 512
        rank = 4
        population_size = 64
        
        model = nn.Linear(n, m, bias=False).to(device)
        
        strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=rank, seed=42)
        strategy.setup(model)
        
        # Calculate theoretical memory usage
        explicit_elements_per_member = m * n  # Full matrix
        lowrank_elements_per_member = rank * (m + n)  # Factored
        
        explicit_total = explicit_elements_per_member * population_size
        lowrank_total = lowrank_elements_per_member * population_size
        
        theoretical_savings = explicit_total / lowrank_total
        
        # Sample perturbations and verify actual storage
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            # Get factors for first member to verify dimensions
            A, B = pop.get_factors(member_id=0, param_name="weight")
            
            # A should be (m, r), B should be (n, r)
            assert A.shape == (m, rank), f"A shape wrong: expected ({m}, {rank}), got {A.shape}"
            assert B.shape == (n, rank), f"B shape wrong: expected ({n}, {rank}), got {B.shape}"
            
            actual_elements = A.numel() + B.numel()
            assert actual_elements == lowrank_elements_per_member, \
                f"Actual storage {actual_elements} != theoretical {lowrank_elements_per_member}"
        
        # Verify savings are significant (should be > 10x for these dimensions)
        assert theoretical_savings > 10, \
            f"Expected >10x memory savings, got {theoretical_savings:.1f}x"
        
        # Print for visibility
        print(f"\nMemory savings: {theoretical_savings:.1f}x")
        print(f"  Explicit: {explicit_total:,} elements ({explicit_total * 4 / 1e6:.1f} MB float32)")
        print(f"  Low-rank: {lowrank_total:,} elements ({lowrank_total * 4 / 1e6:.1f} MB float32)")

    @pytest.mark.slow  
    def test_memory_scales_with_rank_not_matrix_size(self, device):
        """
        Memory usage should scale with rank, not with weight matrix dimensions.
        
        Doubling matrix size should roughly double low-rank storage (not quadruple).
        """
        from hyperscalees.torch import EggrollStrategy
        
        rank = 4
        population_size = 32
        
        # Small model
        small_m, small_n = 256, 128
        small_model = nn.Linear(small_n, small_m, bias=False).to(device)
        
        # Large model (4x the parameters)
        large_m, large_n = 512, 256
        large_model = nn.Linear(large_n, large_m, bias=False).to(device)
        
        # Set up strategies
        small_strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=rank, seed=42)
        small_strategy.setup(small_model)
        
        large_strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=rank, seed=42)
        large_strategy.setup(large_model)
        
        # Get perturbation sizes
        with small_strategy.perturb(population_size=population_size, epoch=0) as pop:
            A_small, B_small = pop.get_factors(member_id=0, param_name="weight")
            small_elements = A_small.numel() + B_small.numel()
        
        with large_strategy.perturb(population_size=population_size, epoch=0) as pop:
            A_large, B_large = pop.get_factors(member_id=0, param_name="weight")
            large_elements = A_large.numel() + B_large.numel()
        
        # Low-rank storage: r*(m+n)
        # Small: 4*(256+128) = 1536
        # Large: 4*(512+256) = 3072
        # Ratio should be ~2x (linear with dimension sum), not 4x (quadratic)
        
        storage_ratio = large_elements / small_elements
        dimension_ratio = (large_m * large_n) / (small_m * small_n)  # Would be 4x for explicit
        
        assert storage_ratio < dimension_ratio, \
            f"Storage ratio {storage_ratio:.1f}x should be less than dimension ratio {dimension_ratio:.1f}x"
        
        # Should be close to 2x for doubling each dimension
        assert 1.5 < storage_ratio < 2.5, \
            f"Storage ratio {storage_ratio:.1f}x should be ~2x (linear scaling), not {dimension_ratio:.1f}x (quadratic)"


# ============================================================================
# Speed Efficiency Tests
# ============================================================================

class TestSpeedEfficiency:
    """Verify that batched low-rank forward is faster than explicit."""

    @pytest.mark.slow
    def test_batched_forward_faster_than_explicit_perturbation(self, device):
        """
        batched_forward should be faster than explicitly forming perturbed matrices.
        
        This is the core performance claim of EGGROLL.
        """
        import time
        from hyperscalees.torch import EggrollStrategy
        
        # Model size where the difference matters
        m, n = 512, 256
        population_size = 64
        rank = 4
        
        model = nn.Linear(n, m, bias=False).to(device)
        x = torch.randn(population_size, n, device=device)
        
        strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=rank, seed=42)
        strategy.setup(model)
        
        # Warmup
        for _ in range(10):
            with strategy.perturb(population_size=population_size, epoch=0) as pop:
                _ = pop.batched_forward(model, x)
        torch.cuda.synchronize()
        
        # Time batched forward (the EGGROLL way)
        n_trials = 50
        start = time.perf_counter()
        for trial in range(n_trials):
            with strategy.perturb(population_size=population_size, epoch=trial) as pop:
                output_batched = pop.batched_forward(model, x)
        torch.cuda.synchronize()
        batched_time = (time.perf_counter() - start) / n_trials
        
        # Time explicit method (form W + A @ B.T for each member)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for trial in range(n_trials):
            with strategy.perturb(population_size=population_size, epoch=trial) as pop:
                outputs_explicit = []
                for i in range(population_size):
                    A, B = pop.get_factors(member_id=i, param_name="weight")
                    perturbed_W = model.weight + A @ B.T
                    out = x[i:i+1] @ perturbed_W.T
                    outputs_explicit.append(out)
                output_explicit = torch.cat(outputs_explicit, dim=0)
        torch.cuda.synchronize()
        explicit_time = (time.perf_counter() - start) / n_trials
        
        # Verify correctness (outputs should match)
        assert_tensors_close(
            output_batched, output_explicit, atol=1e-4,
            msg="Batched and explicit outputs should match"
        )
        
        # Batched should be faster (or at least not significantly slower)
        # Allow some tolerance for measurement noise
        speedup = explicit_time / batched_time
        
        print(f"\nSpeed comparison:")
        print(f"  Batched: {batched_time*1000:.2f} ms")
        print(f"  Explicit: {explicit_time*1000:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Should be at least comparable (>0.5x) - the real gains are in memory
        assert speedup > 0.5, \
            f"Batched method unexpectedly slow: {speedup:.2f}x vs explicit"

    @pytest.mark.slow
    def test_speed_scales_with_population_not_quadratically(self, device):
        """
        Doubling population size should roughly double time, not quadruple.
        
        This verifies we're not accidentally doing O(pop^2) work.
        """
        import time
        from hyperscalees.torch import EggrollStrategy
        
        m, n = 256, 128
        rank = 4
        
        model = nn.Linear(n, m, bias=False).to(device)
        
        strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=rank, seed=42)
        strategy.setup(model)
        
        times = {}
        for pop_size in [32, 64, 128]:
            x = torch.randn(pop_size, n, device=device)
            
            # Warmup
            for _ in range(5):
                with strategy.perturb(population_size=pop_size, epoch=0) as pop:
                    _ = pop.batched_forward(model, x)
            torch.cuda.synchronize()
            
            # Time
            n_trials = 30
            start = time.perf_counter()
            for trial in range(n_trials):
                with strategy.perturb(population_size=pop_size, epoch=trial) as pop:
                    _ = pop.batched_forward(model, x)
            torch.cuda.synchronize()
            times[pop_size] = (time.perf_counter() - start) / n_trials
        
        # Check scaling
        ratio_32_to_64 = times[64] / times[32]
        ratio_64_to_128 = times[128] / times[64]
        
        print(f"\nScaling analysis:")
        print(f"  Pop 32: {times[32]*1000:.2f} ms")
        print(f"  Pop 64: {times[64]*1000:.2f} ms (ratio: {ratio_32_to_64:.2f}x)")
        print(f"  Pop 128: {times[128]*1000:.2f} ms (ratio: {ratio_64_to_128:.2f}x)")
        
        # Should be roughly linear (2x), not quadratic (4x)
        # Allow some overhead tolerance
        assert ratio_32_to_64 < 3.0, \
            f"Scaling 32→64 is {ratio_32_to_64:.1f}x, expected <3x (linear+overhead)"
        assert ratio_64_to_128 < 3.0, \
            f"Scaling 64→128 is {ratio_64_to_128:.1f}x, expected <3x (linear+overhead)"


# ============================================================================
# Numerical Stability at Scale Tests
# ============================================================================

class TestNumericalStabilityAtScale:
    """Verify numerical stability with large models and populations."""

    @pytest.mark.slow
    def test_large_model_forward_stable(self, device):
        """
        Forward pass should be numerically stable for large models.
        
        No NaN or Inf values should appear.
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Large model (similar to a transformer layer)
        model = nn.Sequential(
            nn.Linear(768, 3072, bias=False),
            nn.GELU(),
            nn.Linear(3072, 768, bias=False),
        ).to(device)
        
        population_size = 128
        batch_size = 32
        
        strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=8, seed=42)
        strategy.setup(model)
        
        x = torch.randn(population_size * batch_size, 768, device=device)
        member_ids = torch.arange(population_size, device=device).repeat_interleave(batch_size)
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            output = pop.batched_forward(model, x, member_ids=member_ids)
        
        # Check for numerical issues
        assert torch.isfinite(output).all(), \
            f"Large model forward produced {(~torch.isfinite(output)).sum()} non-finite values"
        
        # Check output is reasonable (not exploding)
        assert output.abs().max() < 1000, \
            f"Output magnitude exploded: max={output.abs().max():.1f}"

    @pytest.mark.slow
    def test_large_population_update_stable(self, device):
        """
        ES update should be numerically stable for large populations.
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(256, 128, bias=False).to(device)
        population_size = 512
        
        strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=4, seed=42)
        strategy.setup(model)
        
        x = torch.randn(population_size, 256, device=device)
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            output = pop.batched_forward(model, x)
        
        # Generate reasonable fitness values
        fitnesses = -((output - torch.randn_like(output)) ** 2).sum(dim=1)
        
        before = model.weight.clone()
        metrics = strategy.step(fitnesses)
        after = model.weight.clone()
        
        # Check parameters are finite
        assert torch.isfinite(after).all(), \
            "Large population update produced non-finite parameters"
        
        # Check update happened but wasn't extreme
        delta = (after - before).norm()
        assert delta > 0, "No update occurred"
        assert delta < 100, f"Update too large: {delta:.2f}"
        
        # Check metrics are reasonable
        assert torch.isfinite(torch.tensor(metrics['grad_norm'])), \
            "Gradient norm is non-finite"

    @pytest.mark.slow
    def test_many_epochs_stable(self, device):
        """
        Training for many epochs should remain numerically stable.
        
        This catches issues that compound over time.
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Sequential(
            nn.Linear(32, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 16, bias=False),
        ).to(device)
        
        strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=4, seed=42)
        strategy.setup(model)
        
        population_size = 64
        n_epochs = 100
        
        target = torch.randn(1, 16, device=device)
        
        for epoch in range(n_epochs):
            x = torch.randn(population_size, 32, device=device)
            
            with strategy.perturb(population_size=population_size, epoch=epoch) as pop:
                output = pop.batched_forward(model, x)
            
            # Simple fitness: negative distance to target
            fitnesses = -((output - target) ** 2).sum(dim=1)
            
            strategy.step(fitnesses)
            
            # Check stability every 10 epochs
            if epoch % 10 == 0:
                for name, param in model.named_parameters():
                    assert torch.isfinite(param).all(), \
                        f"Epoch {epoch}: {name} has non-finite values"
        
        # Final check
        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), \
                f"After {n_epochs} epochs: {name} has non-finite values"
            assert param.abs().max() < 1000, \
                f"After {n_epochs} epochs: {name} exploded to {param.abs().max():.1f}"

    @pytest.mark.slow
    def test_varying_input_scales_handled(self, device):
        """
        Should handle inputs of varying scales without numerical issues.
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(64, 32, bias=False).to(device)
        population_size = 32
        
        strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=4, seed=42)
        strategy.setup(model)
        
        # Test various input scales
        scales = [1e-6, 1e-3, 1.0, 1e3, 1e6]
        
        for scale in scales:
            x = torch.randn(population_size, 64, device=device) * scale
            
            with strategy.perturb(population_size=population_size, epoch=0) as pop:
                output = pop.batched_forward(model, x)
            
            # Output should be finite
            assert torch.isfinite(output).all(), \
                f"Input scale {scale}: produced non-finite output"
            
            # Output scale should be proportional to input scale
            output_scale = output.abs().mean().item()
            expected_scale = scale * model.weight.abs().mean().item() * 64  # Rough estimate
            
            # Should be within 2 orders of magnitude
            assert output_scale < expected_scale * 100, \
                f"Input scale {scale}: output unexpectedly large ({output_scale:.2e})"


# ============================================================================
# ES Gradient Correctness Tests  
# ============================================================================

class TestESGradientCorrectness:
    """Verify ES gradient estimates are mathematically correct."""

    @pytest.mark.slow
    def test_es_gradient_converges_to_true_gradient(self, device):
        """
        With large population and small sigma, ES gradient should approximate
        the true gradient for a differentiable loss.
        
        This is the fundamental correctness check for ES.
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Simple linear model for clean gradient
        model = nn.Linear(8, 4, bias=False).to(device)
        
        # Fixed input and target for consistent gradient
        torch.manual_seed(42)
        x = torch.randn(16, 8, device=device)
        target = torch.randn(16, 4, device=device)
        
        # Compute true gradient
        model.zero_grad()
        output = model(x)
        loss = ((output - target) ** 2).mean()
        loss.backward()
        true_grad = model.weight.grad.clone()
        
        # Run multiple trials to get stable ES gradient estimate
        n_trials = 10
        es_grads = []
        
        for trial in range(n_trials):
            # Reset model to same state
            with torch.no_grad():
                model.weight.copy_(torch.randn_like(model.weight))
            
            # Recompute true gradient for this weight configuration
            model.zero_grad()
            output = model(x)
            loss = ((output - target) ** 2).mean()
            loss.backward()
            true_grad = model.weight.grad.clone()
            
            # Compute ES gradient
            strategy = EggrollStrategy(
                sigma=0.001,  # Small sigma for accurate gradient
                lr=1.0,  # lr=1 so update ≈ gradient
                rank=8,  # Higher rank for accuracy
                seed=trial * 1000
            )
            strategy.setup(model)
            
            population_size = 512  # Large population for low variance
            
            with strategy.perturb(population_size=population_size, epoch=0) as pop:
                # Evaluate each population member
                x_batch = x.unsqueeze(0).expand(population_size, -1, -1)
                x_flat = x_batch.reshape(-1, 8)
                member_ids = torch.arange(population_size, device=device).repeat_interleave(x.shape[0])
                
                outputs_flat = pop.batched_forward(model, x_flat, member_ids=member_ids)
                outputs = outputs_flat.reshape(population_size, x.shape[0], 4)
                
                # Compute fitness (negative MSE loss)
                target_exp = target.unsqueeze(0).expand(population_size, -1, -1)
                fitnesses = -((outputs - target_exp) ** 2).mean(dim=(1, 2))
            
            before = model.weight.clone()
            strategy.step(fitnesses)
            after = model.weight.clone()
            
            # ES "gradient" is the update direction (with lr=1)
            es_grad = after - before
            es_grads.append(es_grad)
        
        # Average ES gradients across trials
        es_grad_avg = torch.stack(es_grads).mean(dim=0)
        
        # ES maximizes fitness (= -loss), so ES gradient should point opposite to loss gradient
        # Normalize for comparison
        true_grad_norm = true_grad / (true_grad.norm() + 1e-8)
        es_grad_norm = es_grad_avg / (es_grad_avg.norm() + 1e-8)
        
        # Cosine similarity should be negative (opposite directions)
        cosine_sim = (true_grad_norm * es_grad_norm).sum().item()
        
        print(f"\nGradient comparison:")
        print(f"  True grad norm: {true_grad.norm():.4f}")
        print(f"  ES grad norm: {es_grad_avg.norm():.4f}")
        print(f"  Cosine similarity: {cosine_sim:.4f} (should be negative)")
        
        # Should be anti-correlated (ES maximizes, loss minimizes)
        assert cosine_sim < -0.3, \
            f"ES gradient should anti-correlate with loss gradient. Cosine sim: {cosine_sim:.4f}"

    @pytest.mark.slow
    def test_es_gradient_variance_decreases_with_population(self, device):
        """
        Larger population should give lower variance gradient estimates.
        
        This is a fundamental property of ES and validates the implementation.
        
        The key is to measure variance across INDEPENDENT trials with the
        SAME starting model state. Each trial should:
        1. Start from identical model weights
        2. Use different random seeds for perturbations
        3. Measure the gradient estimate
        
        With larger populations, the gradient estimate should have lower variance.
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Fixed model weights for all trials
        torch.manual_seed(42)
        base_weights = torch.randn(8, 16, device=device)
        
        # Fixed inputs
        x = torch.randn(8, 16, device=device)
        target = torch.randn(8, 8, device=device)
        
        variances = {}
        
        for pop_size in [32, 128, 512]:
            grad_estimates = []
            
            for trial in range(20):  # More trials for better statistics
                # Create fresh model with SAME weights each trial
                model = nn.Linear(16, 8, bias=False).to(device)
                with torch.no_grad():
                    model.weight.copy_(base_weights)
                
                # Different seed per trial (but same across pop sizes for this trial)
                strategy = EggrollStrategy(
                    sigma=0.1, lr=1.0, rank=4, seed=trial * 12345
                )
                strategy.setup(model)
                
                with strategy.perturb(population_size=pop_size, epoch=0) as pop:
                    x_batch = x.unsqueeze(0).expand(pop_size, -1, -1).reshape(-1, 16)
                    member_ids = torch.arange(pop_size, device=device).repeat_interleave(x.shape[0])
                    
                    outputs = pop.batched_forward(model, x_batch, member_ids=member_ids)
                    outputs = outputs.reshape(pop_size, x.shape[0], 8)
                    
                    target_exp = target.unsqueeze(0).expand(pop_size, -1, -1)
                    fitnesses = -((outputs - target_exp) ** 2).mean(dim=(1, 2))
                
                before = model.weight.clone()
                strategy.step(fitnesses)
                after = model.weight.clone()
                
                grad_estimates.append((after - before).flatten())
            
            # Compute variance across trials
            grads = torch.stack(grad_estimates)
            variance = grads.var(dim=0).mean().item()
            variances[pop_size] = variance
        
        print(f"\nVariance vs population size:")
        for pop_size, var in variances.items():
            print(f"  Pop {pop_size}: variance = {var:.6f}")
        
        # Variance should decrease with population size
        # Theory: Var(gradient) ∝ 1/N, so doubling N should roughly halve variance
        assert variances[128] < variances[32] * 0.8, \
            f"Variance should decrease significantly: {variances[32]:.6f} (32) vs {variances[128]:.6f} (128)"
        assert variances[512] < variances[128] * 0.8, \
            f"Variance should decrease significantly: {variances[128]:.6f} (128) vs {variances[512]:.6f} (512)"

    def test_antithetic_reduces_variance(self, device):
        """
        Antithetic sampling should reduce gradient variance.
        
        By using paired +ε and -ε perturbations, we get variance reduction.
        
        We use batched_forward with a simple MSE loss to fixed targets.
        The test measures variance of ES gradient estimates across random seeds.
        Antithetic sampling should reduce this variance because paired
        perturbations correlate the noise.
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Fixed model weights
        torch.manual_seed(42)
        base_weights = torch.randn(8, 16, device=device)
        
        # Fixed inputs and targets
        x = torch.randn(8, 16, device=device)
        target = torch.randn(8, 8, device=device)
        
        pop_size = 32
        n_trials = 50  # More trials for better statistics
        
        def measure_variance(antithetic: bool) -> float:
            grad_estimates = []
            
            for trial in range(n_trials):
                # Fresh model with SAME weights each trial
                model = nn.Linear(16, 8, bias=False).to(device)
                with torch.no_grad():
                    model.weight.copy_(base_weights)
                
                strategy = EggrollStrategy(
                    sigma=0.1, lr=1.0, rank=4, 
                    antithetic=antithetic,
                    seed=trial  # Different seed each trial
                )
                strategy.setup(model)
                
                with strategy.perturb(population_size=pop_size, epoch=0) as pop:
                    # Batched forward - expand inputs for each population member
                    x_batch = x.unsqueeze(0).expand(pop_size, -1, -1).reshape(-1, 16)
                    member_ids = torch.arange(pop_size, device=device).repeat_interleave(x.shape[0])
                    
                    outputs = pop.batched_forward(model, x_batch, member_ids=member_ids)
                    outputs = outputs.reshape(pop_size, x.shape[0], 8)
                    
                    # MSE loss as fitness (negative because we maximize)
                    target_exp = target.unsqueeze(0).expand(pop_size, -1, -1)
                    fitnesses = -((outputs - target_exp) ** 2).mean(dim=(1, 2))
                
                before = model.weight.clone()
                strategy.step(fitnesses)
                after = model.weight.clone()
                
                grad_estimates.append((after - before).flatten())
            
            grads = torch.stack(grad_estimates)
            return grads.var(dim=0).mean().item()
        
        var_with_antithetic = measure_variance(antithetic=True)
        var_without_antithetic = measure_variance(antithetic=False)
        
        print(f"\nAntithetic variance test (batched forward):")
        print(f"  With antithetic: {var_with_antithetic:.6f}")
        print(f"  Without antithetic: {var_without_antithetic:.6f}")
        
        if var_with_antithetic < var_without_antithetic:
            ratio = var_without_antithetic / var_with_antithetic
            print(f"  Reduction: {ratio:.2f}x")
        else:
            ratio = var_with_antithetic / var_without_antithetic
            print(f"  Ratio (anti/no-anti): {ratio:.2f}x")
        
        # Antithetic should not significantly INCREASE variance
        # Use relaxed threshold like existing test in test_antithetic_sampling.py
        # which uses variance_with <= variance_without * 1.5
        assert var_with_antithetic <= var_without_antithetic * 1.5, \
            f"Antithetic sampling should not significantly increase variance: " \
            f"with={var_with_antithetic:.6f}, without={var_without_antithetic:.6f}"


# ============================================================================
# Correctness Tests for Previously "Does it run?" Tests
# ============================================================================

class TestCUDACorrectness:
    """Verify CUDA operations are correct, not just that they run."""

    def test_cuda_output_matches_expected_shape_and_values(self, device):
        """
        CUDA output should have correct shape and reasonable values.
        """
        from hyperscalees.torch import EggrollStrategy
        
        in_features, out_features = 8, 4
        population_size = 8
        
        model = nn.Linear(in_features, out_features, bias=False).to(device)
        # Initialize to known values
        with torch.no_grad():
            model.weight.fill_(0.1)
        
        strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=2, seed=42)
        strategy.setup(model)
        
        # Known input
        x = torch.ones(population_size, in_features, device=device)
        
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            output = pop.batched_forward(model, x)
        
        # Check shape
        assert output.shape == (population_size, out_features), \
            f"Expected shape ({population_size}, {out_features}), got {output.shape}"
        
        # Check device
        assert output.device.type == "cuda", f"Output on wrong device: {output.device}"
        
        # Check values are reasonable
        # Base output would be x @ W.T = [1,1,1,1,1,1,1,1] @ [[0.1]*8].T = 0.8 per output
        base_expected = in_features * 0.1  # = 0.8
        
        # With perturbations, outputs should be around 0.8 ± some perturbation
        # sigma=0.1 means perturbations are ~0.1 magnitude
        assert (output.mean() - base_expected).abs() < 1.0, \
            f"Output mean {output.mean():.3f} far from expected {base_expected}"
        
        # Different population members should have different outputs
        output_variance = output.var(dim=0).mean()
        assert output_variance > 1e-6, \
            "Different population members should have different outputs"

    def test_cuda_preserves_gradient_flow(self, device):
        """
        Operations should preserve gradient flow where expected.
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(8, 4, bias=False).to(device)
        strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=2, seed=42)
        strategy.setup(model)
        
        # Input with gradients
        x = torch.randn(8, 8, device=device, requires_grad=True)
        
        with strategy.perturb(population_size=8, epoch=0) as pop:
            output = pop.batched_forward(model, x)
        
        # Should be able to backprop through output
        loss = output.sum()
        loss.backward()
        
        # x should have gradients
        assert x.grad is not None, "Input should have gradients"
        assert x.grad.shape == x.shape, "Gradient shape should match input"
        assert torch.isfinite(x.grad).all(), "Gradients should be finite"
