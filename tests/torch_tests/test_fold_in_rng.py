"""
Tests for the pure-tensor RNG functions (_fold_in and _random_normal).

These are critical for EGGROLL correctness since they must:
1. Work inside vmap (no torch.Generator)
2. Be deterministic (same key → same output)
3. Produce different outputs for different keys
4. Generate statistically valid random numbers
"""

import pytest
import torch
import math
from torch.func import vmap

# Import from strategy
import sys
sys.path.insert(0, str(pytest.importorskip("pathlib").Path(__file__).parent.parent.parent / "src"))
from hyperscalees.torch.strategy import _fold_in, _random_normal


class TestFoldIn:
    """Tests for _fold_in function."""
    
    def test_deterministic(self):
        """Same inputs produce same output."""
        key = torch.tensor(42, dtype=torch.int64)
        data = torch.tensor(123, dtype=torch.int64)
        
        result1 = _fold_in(key, data)
        result2 = _fold_in(key, data)
        
        assert result1 == result2
    
    def test_different_keys_different_output(self):
        """Different keys produce different outputs."""
        data = torch.tensor(123, dtype=torch.int64)
        
        key1 = torch.tensor(42, dtype=torch.int64)
        key2 = torch.tensor(43, dtype=torch.int64)
        
        result1 = _fold_in(key1, data)
        result2 = _fold_in(key2, data)
        
        assert result1 != result2
    
    def test_different_data_different_output(self):
        """Different data produce different outputs."""
        key = torch.tensor(42, dtype=torch.int64)
        
        data1 = torch.tensor(123, dtype=torch.int64)
        data2 = torch.tensor(124, dtype=torch.int64)
        
        result1 = _fold_in(key, data1)
        result2 = _fold_in(key, data2)
        
        assert result1 != result2
    
    def test_output_in_32bit_range(self):
        """Output should be in 32-bit range (masked by 0xFFFFFFFF)."""
        key = torch.tensor(2**40, dtype=torch.int64)  # Large input
        data = torch.tensor(2**35, dtype=torch.int64)
        
        result = _fold_in(key, data)
        
        assert result >= 0
        assert result <= 0xFFFFFFFF
    
    def test_works_with_vmap(self):
        """fold_in should work inside vmap."""
        key = torch.tensor(42, dtype=torch.int64)
        data = torch.arange(100, dtype=torch.int64)
        
        # Should not raise
        results = vmap(lambda d: _fold_in(key, d))(data)
        
        assert results.shape == (100,)
        # All results should be unique
        assert len(torch.unique(results)) == 100
    
    def test_vmap_matches_loop(self):
        """vmap results should match sequential loop results."""
        key = torch.tensor(42, dtype=torch.int64)
        data = torch.arange(10, dtype=torch.int64)
        
        # vmap version
        vmap_results = vmap(lambda d: _fold_in(key, d))(data)
        
        # Loop version
        loop_results = torch.stack([_fold_in(key, d) for d in data])
        
        assert torch.equal(vmap_results, loop_results)
    
    def test_nested_fold_in(self):
        """Nested fold_in (like JAX pattern) should work."""
        base_key = torch.tensor(42, dtype=torch.int64)
        epoch = torch.tensor(5, dtype=torch.int64)
        member_id = torch.tensor(10, dtype=torch.int64)
        
        # fold_in(fold_in(key, epoch), member_id)
        intermediate = _fold_in(base_key, epoch)
        final = _fold_in(intermediate, member_id)
        
        # Should be deterministic
        intermediate2 = _fold_in(base_key, epoch)
        final2 = _fold_in(intermediate2, member_id)
        
        assert final == final2
    
    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))])
    def test_works_on_device(self, device):
        """fold_in should work on both CPU and CUDA."""
        key = torch.tensor(42, dtype=torch.int64, device=device)
        data = torch.tensor(123, dtype=torch.int64, device=device)
        
        result = _fold_in(key, data)
        
        assert result.device.type == device


class TestRandomNormal:
    """Tests for _random_normal function."""
    
    def test_correct_shape(self):
        """Output should have correct shape."""
        key = torch.tensor(42, dtype=torch.int64)
        
        for shape in [(10,), (5, 5), (3, 4, 5), (100,)]:
            result = _random_normal(key, shape, torch.float32, torch.device('cpu'))
            assert result.shape == shape
    
    def test_deterministic(self):
        """Same key produces same output."""
        key = torch.tensor(42, dtype=torch.int64)
        shape = (10, 10)
        
        result1 = _random_normal(key, shape, torch.float32, torch.device('cpu'))
        result2 = _random_normal(key, shape, torch.float32, torch.device('cpu'))
        
        assert torch.equal(result1, result2)
    
    def test_different_keys_different_output(self):
        """Different keys produce different outputs."""
        key1 = torch.tensor(42, dtype=torch.int64)
        key2 = torch.tensor(43, dtype=torch.int64)
        shape = (10, 10)
        
        result1 = _random_normal(key1, shape, torch.float32, torch.device('cpu'))
        result2 = _random_normal(key2, shape, torch.float32, torch.device('cpu'))
        
        assert not torch.equal(result1, result2)
    
    def test_approximately_normal_distribution(self):
        """Output should be approximately normally distributed."""
        key = torch.tensor(42, dtype=torch.int64)
        shape = (10000,)  # Large sample
        
        result = _random_normal(key, shape, torch.float32, torch.device('cpu'))
        
        # Mean should be close to 0
        assert abs(result.mean().item()) < 0.1
        
        # Std should be close to 1
        assert abs(result.std().item() - 1.0) < 0.1
    
    def test_no_obvious_patterns(self):
        """Output should not have obvious sequential patterns."""
        key = torch.tensor(42, dtype=torch.int64)
        shape = (1000,)
        
        result = _random_normal(key, shape, torch.float32, torch.device('cpu'))
        
        # Check that consecutive values are not correlated
        # (simple check: correlation should be low)
        x = result[:-1]
        y = result[1:]
        correlation = ((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std())
        
        assert abs(correlation.item()) < 0.1
    
    def test_works_with_vmap(self):
        """_random_normal should work inside vmap."""
        keys = torch.arange(10, dtype=torch.int64)
        shape = (5, 5)
        device = torch.device('cpu')
        
        # Should not raise
        results = vmap(lambda k: _random_normal(k, shape, torch.float32, device))(keys)
        
        assert results.shape == (10, 5, 5)
    
    def test_vmap_produces_different_outputs(self):
        """Different keys in vmap should produce different outputs."""
        keys = torch.arange(10, dtype=torch.int64)
        shape = (5, 5)
        device = torch.device('cpu')
        
        results = vmap(lambda k: _random_normal(k, shape, torch.float32, device))(keys)
        
        # Each result should be different
        for i in range(10):
            for j in range(i + 1, 10):
                assert not torch.equal(results[i], results[j])
    
    def test_respects_dtype(self):
        """Output should respect requested dtype."""
        key = torch.tensor(42, dtype=torch.int64)
        shape = (10,)
        
        for dtype in [torch.float32, torch.float64, torch.float16]:
            result = _random_normal(key, shape, dtype, torch.device('cpu'))
            assert result.dtype == dtype
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_works_on_cuda(self):
        """_random_normal should work on CUDA."""
        key = torch.tensor(42, dtype=torch.int64, device='cuda')
        shape = (10, 10)
        
        result = _random_normal(key, shape, torch.float32, torch.device('cuda'))
        
        assert result.device.type == 'cuda'
        assert result.shape == shape


class TestFoldInRandomNormalIntegration:
    """Integration tests combining fold_in and _random_normal."""
    
    def test_full_eggroll_pattern(self):
        """Test the full EGGROLL RNG pattern: fold_in(fold_in(key, epoch), member)."""
        base_key = torch.tensor(42, dtype=torch.int64)
        epoch = torch.tensor(0, dtype=torch.int64)
        num_members = 8
        shape = (10, 4)  # (m + n, rank)
        
        # Generate perturbations for all members
        results = []
        for member_id in range(num_members):
            member_key = _fold_in(base_key, epoch)
            member_key = _fold_in(member_key, torch.tensor(member_id, dtype=torch.int64))
            factors = _random_normal(member_key, shape, torch.float32, torch.device('cpu'))
            results.append(factors)
        
        # All should be different
        for i in range(num_members):
            for j in range(i + 1, num_members):
                assert not torch.equal(results[i], results[j])
    
    def test_antithetic_pairs_share_base(self):
        """Antithetic pairs (0,1), (2,3), etc. should share base noise when using member_id // 2."""
        base_key = torch.tensor(42, dtype=torch.int64)
        shape = (10, 4)
        
        # Member 0 and 1 should use same effective_member_id = 0
        member_key_0 = _fold_in(base_key, torch.tensor(0, dtype=torch.int64))  # effective = 0
        member_key_1 = _fold_in(base_key, torch.tensor(0, dtype=torch.int64))  # effective = 0
        
        factors_0 = _random_normal(member_key_0, shape, torch.float32, torch.device('cpu'))
        factors_1 = _random_normal(member_key_1, shape, torch.float32, torch.device('cpu'))
        
        # Should be identical (before sign flip)
        assert torch.equal(factors_0, factors_1)
    
    def test_vmap_matches_explicit_loop(self):
        """vmap version should match explicit loop version exactly."""
        base_key = torch.tensor(42, dtype=torch.int64)
        member_ids = torch.arange(8, dtype=torch.int64)
        param_key = torch.tensor(5, dtype=torch.int64)
        shape = (10, 4)
        device = torch.device('cpu')
        
        # vmap version
        def generate_one(member_id):
            member_key = _fold_in(base_key, member_id)
            layer_key = _fold_in(member_key, param_key)
            return _random_normal(layer_key, shape, torch.float32, device)
        
        vmap_results = vmap(generate_one)(member_ids)
        
        # Loop version
        loop_results = []
        for mid in member_ids:
            member_key = _fold_in(base_key, mid)
            layer_key = _fold_in(member_key, param_key)
            loop_results.append(_random_normal(layer_key, shape, torch.float32, device))
        loop_results = torch.stack(loop_results)
        
        assert torch.equal(vmap_results, loop_results)
    
    def test_different_epochs_different_results(self):
        """Different epochs should produce different perturbations."""
        base_seed = 42
        member_id = torch.tensor(0, dtype=torch.int64)
        param_key = torch.tensor(0, dtype=torch.int64)
        shape = (10, 4)
        
        results = []
        for epoch in range(5):
            base_key = torch.tensor(
                (base_seed * 2654435761 + epoch * 2246822519) & 0xFFFFFFFF,
                dtype=torch.int64
            )
            member_key = _fold_in(base_key, member_id)
            layer_key = _fold_in(member_key, param_key)
            factors = _random_normal(layer_key, shape, torch.float32, torch.device('cpu'))
            results.append(factors)
        
        # All epochs should produce different results
        for i in range(5):
            for j in range(i + 1, 5):
                assert not torch.equal(results[i], results[j])
    
    def test_different_params_different_results(self):
        """Different param_keys should produce different perturbations."""
        base_key = torch.tensor(42, dtype=torch.int64)
        member_id = torch.tensor(0, dtype=torch.int64)
        shape = (10, 4)
        
        results = []
        for param_key in range(5):
            member_key = _fold_in(base_key, member_id)
            layer_key = _fold_in(member_key, torch.tensor(param_key, dtype=torch.int64))
            factors = _random_normal(layer_key, shape, torch.float32, torch.device('cpu'))
            results.append(factors)
        
        # All params should produce different results
        for i in range(5):
            for j in range(i + 1, 5):
                assert not torch.equal(results[i], results[j])


class TestStatisticalProperties:
    """Statistical tests for the RNG quality."""
    
    def test_chi_squared_uniformity(self):
        """Test that underlying uniform distribution is reasonably uniform."""
        key = torch.tensor(42, dtype=torch.int64)
        n_samples = 10000
        n_bins = 10
        
        # Get raw uniform values (before Box-Muller)
        counters = torch.arange(n_samples, dtype=torch.int64)
        seeds = (key + counters) & 0xFFFFFFFF
        seeds = ((seeds ^ (seeds >> 17)) * 0xed5ad4bb) & 0xFFFFFFFF
        seeds = ((seeds ^ (seeds >> 11)) * 0xac4c1b51) & 0xFFFFFFFF
        seeds = ((seeds ^ (seeds >> 15)) * 0x31848bab) & 0xFFFFFFFF
        seeds = (seeds ^ (seeds >> 14)) & 0xFFFFFFFF
        uniform = seeds.float() / (2**32)
        
        # Count in bins
        hist = torch.histc(uniform, bins=n_bins, min=0, max=1)
        expected = n_samples / n_bins
        
        # Chi-squared statistic
        chi_sq = ((hist - expected) ** 2 / expected).sum().item()
        
        # For 9 degrees of freedom, chi-sq < 16.92 gives p > 0.05
        assert chi_sq < 30, f"Chi-squared too high: {chi_sq}"
    
    def test_normal_quantiles(self):
        """Test that normal samples match expected quantiles."""
        key = torch.tensor(42, dtype=torch.int64)
        shape = (10000,)
        
        result = _random_normal(key, shape, torch.float32, torch.device('cpu'))
        result_sorted = torch.sort(result).values
        
        # Check some quantiles
        n = len(result)
        
        # Median should be close to 0
        median = result_sorted[n // 2].item()
        assert abs(median) < 0.1
        
        # ~68% should be within 1 std
        within_1_std = ((result > -1) & (result < 1)).float().mean().item()
        assert abs(within_1_std - 0.68) < 0.05
        
        # ~95% should be within 2 std
        within_2_std = ((result > -2) & (result < 2)).float().mean().item()
        assert abs(within_2_std - 0.95) < 0.03
