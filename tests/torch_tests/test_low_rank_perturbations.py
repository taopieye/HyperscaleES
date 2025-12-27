"""
Test: Low-rank perturbation structure for PyTorch implementation.

PAPER CLAIM: EGGROLL generates perturbations as AB^T where A ∈ R^{m×r}, B ∈ R^{n×r}
with r << min(m,n). This reduces auxiliary storage from mn to r(m+n).

TARGET API: Perturbations should be generated via a clean interface that
abstracts away the low-rank details while exposing them for testing.

    perturbation = strategy.sample_perturbation(param, member_id, epoch)
    A, B = perturbation.factors  # Low-rank factors
    full = perturbation.as_matrix()  # Reconstructed (for testing)
"""
import pytest
import torch
import torch.nn as nn
from typing import Tuple

from .conftest import (
    EggrollConfig,
    compute_matrix_rank,
    assert_tensors_close,
    unimplemented
)

# Import from the implementation (will exist after implementation)
# from hyperscalees.torch import EggrollStrategy


# ============================================================================
# Perturbation Structure Tests
# ============================================================================

class TestLowRankPerturbationStructure:
    """Verify that EGGROLL perturbations have the claimed low-rank structure."""

    def test_perturbation_returns_low_rank_factors(
        self, simple_linear, es_generator, eggroll_config, device
    ):
        """
        Perturbation should provide access to low-rank factors (A, B).
        
        TARGET API:
            perturbation = strategy._sample_perturbation(
                param=weight,
                member_id=0,
                epoch=0
            )
            A, B = perturbation.factors
            
            # A should be m x r, B should be n x r
            assert A.shape == (m, r)
            assert B.shape == (n, r)
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Setup
        m, n = simple_linear.weight.shape  # out_features, in_features
        r = eggroll_config.rank
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=r,
            seed=eggroll_config.seed
        )
        strategy.setup(simple_linear)
        
        # Sample perturbation
        perturbation = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=0,
            epoch=0
        )
        
        # Verify factors exist and have correct shapes
        A, B = perturbation.factors
        
        assert A.shape == (m, r), f"Expected A shape ({m}, {r}), got {A.shape}"
        assert B.shape == (n, r), f"Expected B shape ({n}, {r}), got {B.shape}"

    def test_reconstructed_perturbation_has_correct_rank(
        self, simple_linear, es_generator, eggroll_config, device
    ):
        """
        The reconstructed perturbation A @ B.T should have rank at most r.
        
        TARGET API:
            perturbation = strategy._sample_perturbation(param, member_id=0, epoch=0)
            full_matrix = perturbation.as_matrix()  # A @ B.T
            
            rank = torch.linalg.matrix_rank(full_matrix)
            assert rank <= config.rank
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(simple_linear)
        
        # Sample perturbation and reconstruct full matrix
        perturbation = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=0,
            epoch=0
        )
        full_matrix = perturbation.as_matrix()  # Should be A @ B.T
        
        # Verify rank
        rank = compute_matrix_rank(full_matrix)
        assert rank <= eggroll_config.rank, \
            f"Perturbation matrix rank ({rank}) exceeds configured rank limit ({eggroll_config.rank}). " \
            f"Matrix shape: {full_matrix.shape}"

    @pytest.mark.parametrize("rank", [1, 2, 4, 8, 16])
    def test_rank_parameter_controls_perturbation_rank(
        self, device, rank
    ):
        """
        The rank parameter should directly control the maximum rank of perturbations.
        
        This is crucial for the memory/compute tradeoff discussed in the paper.
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Create a model with enough dimensions for all rank values
        model = nn.Linear(32, 64, bias=False).to(device)
        
        config = EggrollConfig(sigma=0.1, lr=0.01, rank=rank, seed=42)
        strategy = EggrollStrategy.from_config(config)
        strategy.setup(model)
        
        perturbation = strategy._sample_perturbation(
            param=model.weight,
            member_id=0,
            epoch=0
        )
        
        # Verify factor dimensions match requested rank
        A, B = perturbation.factors
        assert A.shape[1] == rank, f"A rank dimension should be {rank}, got {A.shape[1]}"
        assert B.shape[1] == rank, f"B rank dimension should be {rank}, got {B.shape[1]}"
        
        # Verify reconstructed matrix has correct rank
        full_matrix = perturbation.as_matrix()
        actual_rank = compute_matrix_rank(full_matrix)
        assert actual_rank <= rank, \
            f"Full matrix rank {actual_rank} exceeds configured rank {rank}"

    def test_rank_one_is_outer_product(self, device):
        """
        At rank=1, the perturbation should be a simple outer product.
        
        TARGET API:
            config = EggrollConfig(rank=1)
            strategy = EggrollStrategy.from_config(config)
            
            perturbation = strategy._sample_perturbation(param, 0, 0)
            A, B = perturbation.factors
            
            assert A.shape[1] == 1  # Column vector
            assert B.shape[1] == 1  # Column vector
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(8, 16, bias=False).to(device)
        
        config = EggrollConfig(sigma=0.1, lr=0.01, rank=1, seed=42)
        strategy = EggrollStrategy.from_config(config)
        strategy.setup(model)
        
        perturbation = strategy._sample_perturbation(
            param=model.weight,
            member_id=0,
            epoch=0
        )
        A, B = perturbation.factors
        
        # Should be column vectors (m x 1) and (n x 1)
        assert A.shape[1] == 1, f"A should be column vector, got shape {A.shape}"
        assert B.shape[1] == 1, f"B should be column vector, got shape {B.shape}"
        
        # Verify it's truly an outer product: A @ B.T = outer(a, b)
        full_matrix = perturbation.as_matrix()
        expected = A.squeeze(-1).unsqueeze(1) @ B.squeeze(-1).unsqueeze(0)
        assert_tensors_close(full_matrix, expected, msg="Rank-1 should be outer product")

    def test_perturbation_has_correct_dtype(self, device, eggroll_config):
        """
        Perturbation factors should match parameter dtype.
        
        TARGET API:
            param = torch.randn(8, 4, dtype=torch.float16)
            perturbation = strategy._sample_perturbation(param, 0, 0)
            A, B = perturbation.factors
            
            assert A.dtype == param.dtype
            assert B.dtype == param.dtype
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Test with float16
        model_fp16 = nn.Linear(8, 16, bias=False).to(device).half()
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(model_fp16)
        
        perturbation = strategy._sample_perturbation(
            param=model_fp16.weight,
            member_id=0,
            epoch=0
        )
        A, B = perturbation.factors
        
        assert A.dtype == model_fp16.weight.dtype, \
            f"A dtype {A.dtype} doesn't match param dtype {model_fp16.weight.dtype}"
        assert B.dtype == model_fp16.weight.dtype, \
            f"B dtype {B.dtype} doesn't match param dtype {model_fp16.weight.dtype}"

    def test_perturbation_on_correct_device(self, device, eggroll_config):
        """
        Perturbation factors should be on the same device as parameter.
        
        TARGET API:
            param = torch.randn(8, 4, device=device)
            perturbation = strategy._sample_perturbation(param, 0, 0)
            A, B = perturbation.factors
            
            assert A.device == param.device
            assert B.device == param.device
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(8, 16, bias=False).to(device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(model)
        
        perturbation = strategy._sample_perturbation(
            param=model.weight,
            member_id=0,
            epoch=0
        )
        A, B = perturbation.factors
        
        assert A.device == model.weight.device, \
            f"A device {A.device} doesn't match param device {model.weight.device}"
        assert B.device == model.weight.device, \
            f"B device {B.device} doesn't match param device {model.weight.device}"


# ============================================================================
# Storage Efficiency Tests
# ============================================================================

class TestStorageEfficiency:
    """Verify storage savings from low-rank structure."""

    def test_storage_savings_calculation(self, device, eggroll_config):
        """
        Verify storage savings: r(m+n) << mn for reasonable r.
        
        PAPER: "reducing the auxiliary storage from mn to r(m+n) per layer"
        
        TARGET API:
            perturbation = strategy._sample_perturbation(param, 0, 0)
            
            # Can query storage stats
            stats = perturbation.storage_stats()
            assert stats["full_rank_elements"] == m * n
            assert stats["low_rank_elements"] == r * (m + n)
            assert stats["savings_ratio"] > 10  # Significant savings
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Large layer to demonstrate savings
        m, n = 256, 128
        r = eggroll_config.rank  # 4
        
        model = nn.Linear(n, m, bias=False).to(device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=r,
            seed=eggroll_config.seed
        )
        strategy.setup(model)
        
        perturbation = strategy._sample_perturbation(
            param=model.weight,
            member_id=0,
            epoch=0
        )
        
        # Calculate storage metrics
        stats = perturbation.storage_stats()
        
        expected_full = m * n  # 32768
        expected_low = r * (m + n)  # 4 * 384 = 1536
        expected_ratio = expected_full / expected_low  # ~21x
        
        assert stats["full_rank_elements"] == expected_full, \
            f"Full rank storage should be {expected_full} elements (m*n), got {stats['full_rank_elements']}"
        assert stats["low_rank_elements"] == expected_low, \
            f"Low rank storage should be {expected_low} elements (r*(m+n)), got {stats['low_rank_elements']}"
        assert stats["savings_ratio"] > 10, \
            f"Expected >10x storage savings for rank {r} on {m}x{n} matrix, got only {stats['savings_ratio']:.1f}x. " \
            f"Full: {stats['full_rank_elements']}, Low-rank: {stats['low_rank_elements']}"

    def test_memory_usage_scales_with_rank(self, device):
        """
        Memory usage should scale linearly with rank, not quadratically.
        
        TARGET API:
            # Compare memory for different ranks
            for rank in [1, 2, 4, 8]:
                config = EggrollConfig(rank=rank)
                strategy = EggrollStrategy.from_config(config)
                perturbation = strategy._sample_perturbation(param, 0, 0)
                
                # Memory should scale as O(rank), not O(rank^2)
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(64, 128, bias=False).to(device)
        m, n = model.weight.shape
        
        element_counts = []
        for rank in [1, 2, 4, 8]:
            config = EggrollConfig(sigma=0.1, lr=0.01, rank=rank, seed=42)
            strategy = EggrollStrategy.from_config(config)
            strategy.setup(model)
            
            perturbation = strategy._sample_perturbation(
                param=model.weight,
                member_id=0,
                epoch=0
            )
            
            A, B = perturbation.factors
            elements = A.numel() + B.numel()
            element_counts.append(elements)
        
        # Verify linear scaling: doubling rank should approximately double elements
        # (within tolerance for different m, n contributions)
        for i in range(len(element_counts) - 1):
            ratio = element_counts[i + 1] / element_counts[i]
            # Should be approximately 2 (linear scaling)
            assert 1.5 < ratio < 2.5, \
                f"Expected ~2x scaling, got {ratio:.2f}x"


# ============================================================================
# Sigma Scaling Tests
# ============================================================================

class TestSigmaScaling:
    """Verify that sigma (noise scale) is applied correctly."""

    def test_perturbation_magnitude_scales_with_sigma(self, device):
        """
        Larger sigma should produce larger perturbations.
        
        TARGET API:
            for sigma in [0.01, 0.1, 1.0]:
                strategy = EggrollStrategy(sigma=sigma, lr=0.01, rank=4)
                strategy.setup(model)
                perturbation = strategy._sample_perturbation(param, 0, 0)
                
                magnitude = perturbation.as_matrix().norm()
                # Magnitude should scale with sigma
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(16, 32, bias=False).to(device)
        
        magnitudes = []
        for sigma in [0.01, 0.1, 1.0]:
            strategy = EggrollStrategy(sigma=sigma, lr=0.01, rank=4, seed=42)
            strategy.setup(model)
            
            perturbation = strategy._sample_perturbation(
                param=model.weight,
                member_id=0,
                epoch=0
            )
            magnitude = perturbation.as_matrix().norm().item()
            magnitudes.append(magnitude)
        
        # Magnitude should increase with sigma
        assert magnitudes[1] > magnitudes[0], \
            f"sigma=0.1 magnitude ({magnitudes[1]:.4f}) should exceed sigma=0.01 ({magnitudes[0]:.4f})"
        assert magnitudes[2] > magnitudes[1], \
            f"sigma=1.0 magnitude ({magnitudes[2]:.4f}) should exceed sigma=0.1 ({magnitudes[1]:.4f})"
        
        # Should scale roughly linearly with sigma
        ratio_1 = magnitudes[1] / magnitudes[0]  # Should be ~10
        ratio_2 = magnitudes[2] / magnitudes[1]  # Should be ~10
        assert 5 < ratio_1 < 20, f"Expected ~10x scaling, got {ratio_1:.1f}x"
        assert 5 < ratio_2 < 20, f"Expected ~10x scaling, got {ratio_2:.1f}x"

    def test_sigma_zero_produces_zero_perturbation(self, device):
        """
        sigma=0 should produce zero perturbation (no noise).
        
        TARGET API:
            strategy = EggrollStrategy(sigma=0.0, lr=0.01, rank=4)
            perturbation = strategy._sample_perturbation(param, 0, 0)
            
            assert perturbation.as_matrix().abs().max() == 0
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(8, 16, bias=False).to(device)
        
        strategy = EggrollStrategy(sigma=0.0, lr=0.01, rank=4, seed=42)
        strategy.setup(model)
        
        perturbation = strategy._sample_perturbation(
            param=model.weight,
            member_id=0,
            epoch=0
        )
        
        full_matrix = perturbation.as_matrix()
        max_val = full_matrix.abs().max().item()
        assert max_val == 0, \
            f"sigma=0 should produce all-zero perturbation, got max absolute value {max_val:.2e}"

    def test_perturbation_normalized_by_sqrt_rank(self, device):
        """
        Perturbations should be normalized by sqrt(rank) for consistent magnitude.
        
        This ensures that changing rank doesn't drastically change perturbation scale.
        
        PAPER: The sigma is divided by sqrt(rank) to normalize.
        """
        from hyperscalees.torch import EggrollStrategy
        import math
        
        model = nn.Linear(64, 128, bias=False).to(device)
        
        magnitudes = []
        ranks = [1, 4, 16]
        
        for rank in ranks:
            strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=rank, seed=42)
            strategy.setup(model)
            
            perturbation = strategy._sample_perturbation(
                param=model.weight,
                member_id=0,
                epoch=0
            )
            magnitude = perturbation.as_matrix().norm().item()
            magnitudes.append(magnitude)
        
        # With proper sqrt(rank) normalization, magnitudes should be similar
        # across different ranks (within a factor of ~2-3)
        max_mag = max(magnitudes)
        min_mag = min(magnitudes)
        ratio = max_mag / min_mag
        
        assert ratio < 3.0, \
            f"Magnitudes should be similar across ranks, but ratio is {ratio:.2f}x"


# ============================================================================
# Batch Perturbation Tests
# ============================================================================

class TestBatchPerturbation:
    """Verify efficient batch perturbation generation."""

    def test_batch_perturbation_generation(self, device, eggroll_config):
        """
        Should be able to generate perturbations for entire population at once.
        
        TARGET API:
            # Generate all perturbations for population
            perturbations = strategy.sample_perturbations(
                param=weight,
                population_size=64,
                epoch=0
            )
            
            assert len(perturbations) == 64  # Should return one perturbation per population member
            # Or as batched tensors:
            # A_batch: (64, m, r), B_batch: (64, n, r)
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(8, 16, bias=False).to(device)
        population_size = 64
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(model)
        
        # Generate batch of perturbations
        perturbations = strategy.sample_perturbations(
            param=model.weight,
            population_size=population_size,
            epoch=0
        )
        
        assert len(perturbations) == population_size, \
            f"Expected {population_size} perturbations, got {len(perturbations)}"
        
        # Verify each perturbation has correct structure
        m, n = model.weight.shape
        r = eggroll_config.rank
        
        for i, pert in enumerate(perturbations):
            A, B = pert.factors
            assert A.shape == (m, r), f"Perturbation {i}: A shape mismatch"
            assert B.shape == (n, r), f"Perturbation {i}: B shape mismatch"

    def test_batch_perturbations_are_independent(self, device, eggroll_config):
        """
        Different population members should have different perturbations.
        
        TARGET API:
            perturbations = strategy.sample_perturbations(param, population_size=8, epoch=0)
            
            # Each perturbation should be unique
            for i in range(8):
                for j in range(i + 1, 8):
                    assert not torch.allclose(
                        perturbations[i].as_matrix(),
                        perturbations[j].as_matrix()
                    )
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(8, 16, bias=False).to(device)
        population_size = 8
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(model)
        
        perturbations = strategy.sample_perturbations(
            param=model.weight,
            population_size=population_size,
            epoch=0
        )
        
        # Check all pairs are different
        for i in range(population_size):
            for j in range(i + 1, population_size):
                mat_i = perturbations[i].as_matrix()
                mat_j = perturbations[j].as_matrix()
                
                assert not torch.allclose(mat_i, mat_j), \
                    f"Perturbations {i} and {j} are identical (should be independent)"


# ============================================================================
# Perturbation Properties Tests
# ============================================================================

class TestPerturbationProperties:
    """Verify mathematical properties of perturbations."""

    def test_perturbation_mean_is_approximately_zero(self, device, eggroll_config):
        """
        Perturbations should have approximately zero mean (unbiased noise).
        
        TARGET API:
            perturbations = strategy.sample_perturbations(param, population_size=1000, epoch=0)
            
            sum_perturbation = sum(p.as_matrix() for p in perturbations)
            mean_perturbation = sum_perturbation / len(perturbations)
            
            assert mean_perturbation.abs().max() < 0.1  # Should be close to zero
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(16, 32, bias=False).to(device)
        population_size = 1000
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(model)
        
        perturbations = strategy.sample_perturbations(
            param=model.weight,
            population_size=population_size,
            epoch=0
        )
        
        # Compute mean perturbation
        sum_pert = perturbations[0].as_matrix().clone()
        for p in perturbations[1:]:
            sum_pert += p.as_matrix()
        mean_pert = sum_pert / population_size
        
        # Mean should be close to zero (law of large numbers)
        max_mean = mean_pert.abs().max().item()
        assert max_mean < 0.1, \
            f"Mean perturbation should be ~0, but max element is {max_mean:.4f}"

    def test_perturbation_entries_are_normally_distributed(self, device, eggroll_config):
        """
        Perturbation entries should be approximately normally distributed.
        
        TARGET API:
            perturbation = strategy._sample_perturbation(param, 0, 0)
            entries = perturbation.as_matrix().flatten()
            
            # Check distribution is approximately normal
            # (Shapiro-Wilk or similar test)
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Large matrix for statistical significance
        model = nn.Linear(64, 128, bias=False).to(device)
        
        strategy = EggrollStrategy(
            sigma=1.0,  # Use sigma=1 for easier verification
            lr=0.01,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(model)
        
        perturbation = strategy._sample_perturbation(
            param=model.weight,
            member_id=0,
            epoch=0
        )
        
        entries = perturbation.as_matrix().flatten()
        
        # Basic statistics check (since low-rank, won't be exactly normal)
        mean = entries.mean().item()
        std = entries.std().item()
        
        # Mean should be close to 0
        assert abs(mean) < 0.5, f"Mean {mean:.4f} should be close to 0"
        
        # Std should be reasonable (not checking exact value due to low-rank structure)
        assert 0.01 < std < 10.0, f"Std {std:.4f} seems unreasonable"

    def test_factors_are_normalized(self, device, eggroll_config):
        """
        Individual factors A and B should have controlled magnitude.
        
        This prevents numerical issues with very large or small factors.
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(32, 64, bias=False).to(device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(model)
        
        perturbation = strategy._sample_perturbation(
            param=model.weight,
            member_id=0,
            epoch=0
        )
        
        A, B = perturbation.factors
        
        # Check factors don't have extreme values
        a_max = A.abs().max().item()
        b_max = B.abs().max().item()
        
        # Factors should be reasonable (not exploding or vanishing)
        assert 1e-6 < a_max < 1e6, f"A has extreme values: max={a_max}"
        assert 1e-6 < b_max < 1e6, f"B has extreme values: max={b_max}"


# ============================================================================
# Perturbation Dataclass Tests
# ============================================================================

class TestPerturbationDataclass:
    """Test the Perturbation dataclass/namedtuple interface."""

    def test_perturbation_is_immutable(self, device, eggroll_config):
        """
        Perturbation should be immutable once created.
        
        TARGET API:
            perturbation = strategy._sample_perturbation(param, 0, 0)
            
            with pytest.raises(AttributeError):
                perturbation.factors = (new_A, new_B)
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(8, 16, bias=False).to(device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(model)
        
        perturbation = strategy._sample_perturbation(
            param=model.weight,
            member_id=0,
            epoch=0
        )
        
        # Attempting to modify should raise
        new_A = torch.randn_like(perturbation.factors[0])
        new_B = torch.randn_like(perturbation.factors[1])
        
        with pytest.raises((AttributeError, TypeError)):
            perturbation.factors = (new_A, new_B)

    def test_perturbation_has_metadata(self, device, eggroll_config):
        """
        Perturbation should include metadata for debugging/logging.
        
        TARGET API:
            perturbation = strategy._sample_perturbation(param, member_id=5, epoch=2)
            
            assert perturbation.member_id == 5
            assert perturbation.epoch == 2
            assert perturbation.rank == config.rank
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(8, 16, bias=False).to(device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(model)
        
        perturbation = strategy._sample_perturbation(
            param=model.weight,
            member_id=5,
            epoch=2
        )
        
        # Check metadata is accessible
        assert perturbation.member_id == 5, f"member_id should be 5, got {perturbation.member_id}"
        assert perturbation.epoch == 2, f"epoch should be 2, got {perturbation.epoch}"
        assert perturbation.rank == eggroll_config.rank, \
            f"rank should be {eggroll_config.rank}, got {perturbation.rank}"

    def test_perturbation_repr_is_informative(self, device, eggroll_config):
        """
        Perturbation __repr__ should be informative for debugging.
        
        TARGET API:
            perturbation = strategy._sample_perturbation(param, 0, 0)
            repr_str = repr(perturbation)
            
            assert "rank" in repr_str
            assert "shape" in repr_str
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(8, 16, bias=False).to(device)
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=eggroll_config.seed
        )
        strategy.setup(model)
        
        perturbation = strategy._sample_perturbation(
            param=model.weight,
            member_id=0,
            epoch=0
        )
        
        repr_str = repr(perturbation)
        
        # Should contain useful info
        assert "rank" in repr_str.lower() or str(eggroll_config.rank) in repr_str, \
            f"repr should mention rank: {repr_str}"
        assert "16" in repr_str or "8" in repr_str, \
            f"repr should mention shape dimensions: {repr_str}"
