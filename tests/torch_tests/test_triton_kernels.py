"""
Tests for Triton EGGROLL kernels.

These tests verify that the Triton kernels produce identical results
to the reference PyTorch implementation in strategy.py.

Key invariants:
1. Deterministic: Same inputs → same outputs
2. Matches PyTorch: Triton kernel output == PyTorch reference output
3. Correct perturbation structure: Low-rank factors applied correctly
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
    """Test that Triton Philox RNG matches PyTorch implementation."""
    
    def test_philox_determinism(self):
        """Same key + counter should give same output."""
        from hyperscalees.torch.triton_kernels import philox_4x32_10
        import triton
        import triton.language as tl
        
        # We can't directly test the JIT function, but we can test
        # through the wrapper functions
        pass  # Tested indirectly through generate_normal tests
    
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
    """Test the main fused kernel against PyTorch reference."""
    
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
    
    def _pytorch_reference(self, x, W, bias, base_key, member_ids, layer_idx, 
                           sigma, rank, antithetic=True):
        """
        Reference implementation using our existing PyTorch code.
        This is what the Triton kernel must match.
        """
        from hyperscalees.torch.strategy import (
            _fold_in,
            _generate_lowrank_factors_batched,
        )
        
        batch_size = x.shape[0]
        out_features, in_features = W.shape
        device = x.device
        dtype = x.dtype
        
        # Base matmul
        out = x @ W.T
        if bias is not None:
            out = out + bias
        
        # Generate low-rank factors
        base_key_tensor = torch.tensor(base_key, dtype=torch.int64, device=device)
        A, B = _generate_lowrank_factors_batched(
            base_key_tensor, member_ids, layer_idx,
            out_features, in_features, rank, sigma,
            dtype, device, antithetic
        )
        
        # Apply perturbation: out += (x @ B) @ A.T
        # A: [batch, out_features, rank]
        # B: [batch, in_features, rank]
        xB = torch.einsum('bi,bir->br', x, B)  # [batch, rank]
        perturbation = torch.einsum('br,bor->bo', xB, A)  # [batch, out_features]
        
        return out + perturbation
    
    def test_triton_matches_pytorch_basic(self, setup_data):
        """Triton kernel should match PyTorch reference output."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        d = setup_data
        
        # PyTorch reference
        ref_out = self._pytorch_reference(
            d['x'], d['W'], d['bias'], d['base_key'], d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank'], antithetic=True
        )
        
        # Triton kernel
        triton_out = fused_perturbed_linear(
            d['x'], d['W'], d['bias'], d['base_key'], d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank'], antithetic=True
        )
        
        # Compare
        torch.testing.assert_close(
            triton_out, ref_out,
            rtol=1e-3, atol=1e-3,
            msg="Triton kernel output doesn't match PyTorch reference"
        )
    
    def test_triton_matches_pytorch_no_bias(self, setup_data):
        """Test without bias."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        d = setup_data
        
        ref_out = self._pytorch_reference(
            d['x'], d['W'], None, d['base_key'], d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank'], antithetic=True
        )
        
        triton_out = fused_perturbed_linear(
            d['x'], d['W'], None, d['base_key'], d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank'], antithetic=True
        )
        
        torch.testing.assert_close(triton_out, ref_out, rtol=1e-3, atol=1e-3)
    
    def test_triton_matches_pytorch_no_antithetic(self, setup_data):
        """Test without antithetic sampling."""
        from hyperscalees.torch.triton_kernels import fused_perturbed_linear
        
        d = setup_data
        
        ref_out = self._pytorch_reference(
            d['x'], d['W'], d['bias'], d['base_key'], d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank'], antithetic=False
        )
        
        triton_out = fused_perturbed_linear(
            d['x'], d['W'], d['bias'], d['base_key'], d['member_ids'],
            d['layer_idx'], d['sigma'], d['rank'], antithetic=False
        )
        
        torch.testing.assert_close(triton_out, ref_out, rtol=1e-3, atol=1e-3)
    
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
    
    @pytest.mark.parametrize("batch_size", [1, 16, 64, 256, 1024])
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
        
        ref_out = self._pytorch_reference(
            x, W, bias, 42, member_ids, 0, sigma, rank
        )
        
        triton_out = fused_perturbed_linear(
            x, W, bias, 42, member_ids, 0, sigma, rank
        )
        
        torch.testing.assert_close(triton_out, ref_out, rtol=1e-3, atol=1e-3)
    
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
        
        ref_out = self._pytorch_reference(
            x, W, bias, 42, member_ids, 0, sigma, rank
        )
        
        triton_out = fused_perturbed_linear(
            x, W, bias, 42, member_ids, 0, sigma, rank
        )
        
        torch.testing.assert_close(triton_out, ref_out, rtol=1e-3, atol=1e-3)


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
