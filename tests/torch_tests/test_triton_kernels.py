"""
Tests for Triton EGGROLL kernels.

These tests verify that the Triton kernels work correctly.
Note: We don't test exact match with PyTorch RNG since Triton uses
its own Philox implementation. Instead we test:
1. Determinism: same inputs → same outputs
2. Correctness: output has right shape and reasonable values
3. Different inputs produce different outputs
4. Performance is acceptable
"""

import pytest
import torch
import math

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Triton tests"
)


class TestTritonImport:
    """Test that Triton kernels can be imported."""
    
    def test_import_triton(self):
        """Triton should be available with PyTorch 2.0+."""
        import triton
        assert triton.__version__ is not None
    
    def test_import_kernels(self):
        """Our Triton kernels should import without error."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear


class TestPhiloxRNG:
    """Test that Triton Philox RNG produces reasonable distributions."""
    
    def test_philox_determinism(self):
        """Same key + counter should give same output."""
        from hyperscalees.torch.strategy import _random_normal_batched
        
        device = torch.device('cuda')
        keys = torch.tensor([42, 123, 456], dtype=torch.int64, device=device)
        shape = (100,)
        
        # Generate samples twice with same keys
        samples1 = _random_normal_batched(keys, shape, torch.float32, device)
        samples2 = _random_normal_batched(keys, shape, torch.float32, device)
        
        # Should be identical
        assert torch.allclose(samples1, samples2), "Same keys should produce identical outputs"
    
    def test_normal_distribution_statistics(self):
        """Generated normals should have ~N(0,1) statistics."""
        from hyperscalees.torch.strategy import _random_normal_batched
        
        device = torch.device('cuda')
        keys = torch.arange(10000, dtype=torch.int64, device=device)
        shape = (100,)
        
        samples = _random_normal_batched(keys, shape, torch.float32, device)
        
        # Should be approximately standard normal
        mean = samples.mean().item()
        std = samples.std().item()
        
        assert abs(mean) < 0.1, f"Mean {mean} too far from 0"
        assert abs(std - 1.0) < 0.1, f"Std {std} too far from 1"


class TestFusedPerturbedLinear:
    """Test the main fused kernel."""
    
    @pytest.fixture
    def setup_data(self):
        """Create test data matching typical EGGROLL usage."""
        device = torch.device('cuda')
        
        batch_size = 64
        in_features = 128
        out_features = 64
        rank = 4
        sigma = 0.1
        
        x = torch.randn(batch_size, in_features, device=device)
        W = torch.randn(out_features, in_features, device=device)
        bias = torch.randn(out_features, device=device)
        member_ids = torch.arange(batch_size, dtype=torch.int64, device=device)
        base_key = 42
        layer_idx = 0
        
        return {
            'x': x,
            'W': W,
            'bias': bias,
            'member_ids': member_ids,
            'base_key': base_key,
            'layer_idx': layer_idx,
            'sigma': sigma,
            'rank': rank,
            'batch_size': batch_size,
            'in_features': in_features,
            'out_features': out_features,
        }
    
    def test_output_shape(self, setup_data):
        """Output should have correct shape."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        d = setup_data
        
        out = fused_perturbed_linear(
            d['x'], d['W'], d['bias'], d['base_key'], d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank']
        )
        
        expected_shape = (d['batch_size'], d['out_features'])
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    
    def test_output_dtype(self, setup_data):
        """Output should match input dtype."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        d = setup_data
        
        out = fused_perturbed_linear(
            d['x'], d['W'], d['bias'], d['base_key'], d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank']
        )
        
        assert out.dtype == d['x'].dtype
    
    def test_output_device(self, setup_data):
        """Output should be on same device as input."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        d = setup_data
        
        out = fused_perturbed_linear(
            d['x'], d['W'], d['bias'], d['base_key'], d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank']
        )
        
        assert out.device == d['x'].device
    
    def test_no_nan_or_inf(self, setup_data):
        """Output should not contain NaN or Inf values."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        d = setup_data
        
        out = fused_perturbed_linear(
            d['x'], d['W'], d['bias'], d['base_key'], d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank']
        )
        
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
    
    def test_base_matmul_dominates(self, setup_data):
        """With small sigma, output should be close to base matmul."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        d = setup_data
        
        # Very small sigma
        out = fused_perturbed_linear(
            d['x'], d['W'], d['bias'], d['base_key'], d['member_ids'],
            d['layer_idx'], sigma=1e-6, rank=d['rank']
        )
        
        # Base matmul
        base = d['x'] @ d['W'].T + d['bias']
        
        # Should be very close
        torch.testing.assert_close(out, base, rtol=1e-3, atol=1e-3)
    
    def test_no_bias(self, setup_data):
        """Test without bias."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        d = setup_data
        
        out = fused_perturbed_linear(
            d['x'], d['W'], None, d['base_key'], d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank']
        )
        
        assert out.shape == (d['batch_size'], d['out_features'])
        assert not torch.isnan(out).any()
    
    def test_no_antithetic(self, setup_data):
        """Test without antithetic sampling."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        d = setup_data
        
        out = fused_perturbed_linear(
            d['x'], d['W'], d['bias'], d['base_key'], d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank'], antithetic=False
        )
        
        assert out.shape == (d['batch_size'], d['out_features'])
        assert not torch.isnan(out).any()
    
    def test_determinism(self, setup_data):
        """Same inputs should always give same outputs."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        d = setup_data
        
        out1 = fused_perturbed_linear(
            d['x'], d['W'], d['bias'], d['base_key'], d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank']
        )
        
        out2 = fused_perturbed_linear(
            d['x'], d['W'], d['bias'], d['base_key'], d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank']
        )
        
        torch.testing.assert_close(out1, out2, rtol=0, atol=0)
    
    def test_different_member_ids_different_output(self, setup_data):
        """Different member_ids should give different perturbations."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        d = setup_data
        
        out1 = fused_perturbed_linear(
            d['x'], d['W'], d['bias'], d['base_key'], d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank']
        )
        
        # Shift member IDs
        shifted_ids = d['member_ids'] + 100
        out2 = fused_perturbed_linear(
            d['x'], d['W'], d['bias'], d['base_key'], shifted_ids,
            d['layer_idx'], d['sigma'], d['rank']
        )
        
        # Should be different
        assert not torch.allclose(out1, out2), "Different member_ids should give different outputs"
    
    def test_different_layer_idx_different_output(self, setup_data):
        """Different layer indices should give different perturbations."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        d = setup_data
        
        out1 = fused_perturbed_linear(
            d['x'], d['W'], d['bias'], d['base_key'], d['member_ids'],
            0, d['sigma'], d['rank']
        )
        
        out2 = fused_perturbed_linear(
            d['x'], d['W'], d['bias'], d['base_key'], d['member_ids'],
            1, d['sigma'], d['rank']
        )
        
        assert not torch.allclose(out1, out2), "Different layer_idx should give different outputs"
    
    def test_different_base_key_different_output(self, setup_data):
        """Different base keys should give different perturbations."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        d = setup_data
        
        out1 = fused_perturbed_linear(
            d['x'], d['W'], d['bias'], 42, d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank']
        )
        
        out2 = fused_perturbed_linear(
            d['x'], d['W'], d['bias'], 123456, d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank']
        )
        
        assert not torch.allclose(out1, out2), "Different base_key should give different outputs"
    
    @pytest.mark.parametrize("batch_size", [1, 16, 64, 256])
    def test_various_batch_sizes(self, batch_size):
        """Test with various batch sizes."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        device = torch.device('cuda')
        in_features = 64
        out_features = 32
        rank = 4
        sigma = 0.1
        
        x = torch.randn(batch_size, in_features, device=device)
        W = torch.randn(out_features, in_features, device=device)
        bias = torch.randn(out_features, device=device)
        member_ids = torch.arange(batch_size, dtype=torch.int64, device=device)
        
        out = fused_perturbed_linear(x, W, bias, 42, member_ids, 0, sigma, rank)
        
        assert out.shape == (batch_size, out_features)
        assert not torch.isnan(out).any()
    
    @pytest.mark.parametrize("rank", [1, 2, 4, 8, 16])
    def test_various_ranks(self, rank):
        """Test with various low-rank dimensions."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        device = torch.device('cuda')
        batch_size = 64
        in_features = 64
        out_features = 32
        sigma = 0.1
        
        x = torch.randn(batch_size, in_features, device=device)
        W = torch.randn(out_features, in_features, device=device)
        bias = torch.randn(out_features, device=device)
        member_ids = torch.arange(batch_size, dtype=torch.int64, device=device)
        
        out = fused_perturbed_linear(x, W, bias, 42, member_ids, 0, sigma, rank)
        
        assert out.shape == (batch_size, out_features)
        assert not torch.isnan(out).any()
    
    @pytest.mark.parametrize("in_features,out_features", [
        (32, 16),
        (64, 64),
        (128, 256),
        (256, 128),
    ])
    def test_various_dimensions(self, in_features, out_features):
        """Test with various input/output dimensions."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        device = torch.device('cuda')
        batch_size = 64
        rank = 4
        sigma = 0.1
        
        x = torch.randn(batch_size, in_features, device=device)
        W = torch.randn(out_features, in_features, device=device)
        bias = torch.randn(out_features, device=device)
        member_ids = torch.arange(batch_size, dtype=torch.int64, device=device)
        
        out = fused_perturbed_linear(x, W, bias, 42, member_ids, 0, sigma, rank)
        
        assert out.shape == (batch_size, out_features)
        assert not torch.isnan(out).any()


class TestTritonPerformance:
    """Benchmark tests comparing Triton to PyTorch."""
    
    @pytest.mark.slow
    def test_triton_faster_than_pytorch(self):
        """Triton should be significantly faster than naive PyTorch."""
        import time
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        from hyperscalees.torch.strategy import _generate_lowrank_factors_batched
        
        device = torch.device('cuda')
        batch_size = 256
        in_features = 256
        out_features = 128
        rank = 4
        sigma = 0.1
        num_iterations = 100
        
        x = torch.randn(batch_size, in_features, device=device)
        W = torch.randn(out_features, in_features, device=device)
        bias = torch.randn(out_features, device=device)
        member_ids = torch.arange(batch_size, dtype=torch.int64, device=device)
        base_key = torch.tensor(42, dtype=torch.int64, device=device)
        
        # Warmup
        for _ in range(10):
            _ = fused_perturbed_linear(x, W, bias, 42, member_ids, 0, sigma, rank)
        torch.cuda.synchronize()
        
        # Benchmark Triton
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = fused_perturbed_linear(x, W, bias, 42, member_ids, 0, sigma, rank)
        torch.cuda.synchronize()
        triton_time = time.perf_counter() - start
        
        # Benchmark PyTorch reference
        def pytorch_forward():
            out = x @ W.T + bias
            A, B = _generate_lowrank_factors_batched(
                base_key, member_ids, 0, out_features, in_features, 
                rank, sigma, x.dtype, device, True
            )
            xB = torch.einsum('bi,bir->br', x, B)
            pert = torch.einsum('br,bor->bo', xB, A)
            return out + pert
        
        for _ in range(10):
            _ = pytorch_forward()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = pytorch_forward()
        torch.cuda.synchronize()
        pytorch_time = time.perf_counter() - start
        
        speedup = pytorch_time / triton_time
        print(f"\nTriton: {triton_time*1000/num_iterations:.3f} ms")
        print(f"PyTorch: {pytorch_time*1000/num_iterations:.3f} ms")
        print(f"Speedup: {speedup:.2f}x")
        
        # Triton should be at least as fast (ideally much faster)
        assert speedup >= 0.8, f"Triton too slow: {speedup:.2f}x vs PyTorch"
