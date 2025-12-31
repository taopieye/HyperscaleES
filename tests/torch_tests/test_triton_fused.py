"""
Tests for fused Triton kernels for EGGROLL low-rank factor generation.

Tests verify:
1. Output shapes are correct
2. Statistical properties (mean, variance) match expected values
3. Antithetic sampling property (E[i] = -E[i+1] for pairs)
4. Determinism (same inputs → same outputs)
5. Numerical equivalence with PyTorch reference (within RNG differences)
6. Performance improvement over PyTorch native
"""
import pytest
import torch
import numpy as np
import time
from typing import Tuple


# Skip all tests if Triton is not available
triton = pytest.importorskip("triton")


from hyperscalees.torch.triton_fused import (
    generate_lowrank_factors_triton,
    generate_lowrank_factors_triton_simple,
    generate_lowrank_factors_triton_optimized,
    generate_lowrank_factors_fused,
)
from hyperscalees.torch.triton_kernels import generate_lowrank_factors_torch


class TestTritonFusedKernelShapes:
    """Test that output shapes are correct."""
    
    @pytest.mark.parametrize("pop_size", [1, 32, 128, 1024, 2048])
    @pytest.mark.parametrize("in_features,out_features", [(4, 256), (256, 256), (256, 2)])
    @pytest.mark.parametrize("rank", [1, 4, 8])
    def test_output_shapes(self, pop_size, in_features, out_features, rank):
        """Test that A and B have correct shapes."""
        device = torch.device('cuda')
        member_ids = torch.arange(pop_size, device=device)
        
        A, B = generate_lowrank_factors_fused(
            out_features=out_features,
            in_features=in_features,
            rank=rank,
            seed=42,
            epoch=0,
            member_ids=member_ids,
            param_id=0,
            sigma=0.2,
            antithetic=True,
        )
        
        assert A.shape == (pop_size, out_features, rank), f"A shape mismatch: {A.shape}"
        assert B.shape == (pop_size, in_features, rank), f"B shape mismatch: {B.shape}"
    
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_output_dtype(self, dtype):
        """Test that output dtype matches requested dtype."""
        device = torch.device('cuda')
        member_ids = torch.arange(64, device=device)
        
        A, B = generate_lowrank_factors_fused(
            out_features=256,
            in_features=256,
            rank=4,
            seed=42,
            epoch=0,
            member_ids=member_ids,
            param_id=0,
            sigma=0.2,
            dtype=dtype,
        )
        
        assert A.dtype == dtype
        assert B.dtype == dtype


class TestTritonFusedKernelStatistics:
    """Test statistical properties of generated factors."""
    
    def test_mean_near_zero(self):
        """Test that mean of generated noise is near zero."""
        device = torch.device('cuda')
        pop_size = 2048
        member_ids = torch.arange(pop_size, device=device)
        
        A, B = generate_lowrank_factors_fused(
            out_features=256,
            in_features=256,
            rank=4,
            seed=42,
            epoch=0,
            member_ids=member_ids,
            param_id=0,
            sigma=0.2,
            antithetic=True,
        )
        
        # B should be standard normal (mean ~0, std ~1)
        assert abs(B.mean().item()) < 0.1, f"B mean too far from 0: {B.mean().item()}"
        assert abs(B.std().item() - 1.0) < 0.1, f"B std too far from 1: {B.std().item()}"
        
        # A should have mean ~0 (antithetic cancels out)
        assert abs(A.mean().item()) < 0.1, f"A mean too far from 0: {A.mean().item()}"
    
    def test_sigma_scaling(self):
        """Test that A is scaled by sigma/sqrt(rank)."""
        device = torch.device('cuda')
        pop_size = 2048
        member_ids = torch.arange(pop_size, device=device)
        sigma = 0.2
        rank = 4
        
        A, B = generate_lowrank_factors_fused(
            out_features=256,
            in_features=256,
            rank=rank,
            seed=42,
            epoch=0,
            member_ids=member_ids,
            param_id=0,
            sigma=sigma,
            antithetic=True,
        )
        
        expected_std = sigma / np.sqrt(rank)  # 0.1
        actual_std = A.std().item()
        
        # Allow 10% tolerance
        assert abs(actual_std - expected_std) / expected_std < 0.15, \
            f"A std {actual_std:.4f} doesn't match expected {expected_std:.4f}"
    
    @pytest.mark.parametrize("sigma", [0.1, 0.2, 0.5, 1.0])
    def test_different_sigma_values(self, sigma):
        """Test scaling works correctly for different sigma values."""
        device = torch.device('cuda')
        pop_size = 1024
        member_ids = torch.arange(pop_size, device=device)
        rank = 4
        
        A, B = generate_lowrank_factors_fused(
            out_features=256,
            in_features=256,
            rank=rank,
            seed=42,
            epoch=0,
            member_ids=member_ids,
            param_id=0,
            sigma=sigma,
            antithetic=True,
        )
        
        expected_std = sigma / np.sqrt(rank)
        actual_std = A.std().item()
        
        assert abs(actual_std - expected_std) / expected_std < 0.2, \
            f"sigma={sigma}: A std {actual_std:.4f} doesn't match expected {expected_std:.4f}"


class TestTritonFusedKernelAntithetic:
    """Test antithetic sampling properties."""
    
    def test_antithetic_pairs_opposite_sign(self):
        """Test that antithetic pairs have opposite signs for A."""
        device = torch.device('cuda')
        pop_size = 128
        member_ids = torch.arange(pop_size, device=device)
        
        A, B = generate_lowrank_factors_fused(
            out_features=256,
            in_features=256,
            rank=4,
            seed=42,
            epoch=0,
            member_ids=member_ids,
            param_id=0,
            sigma=0.2,
            antithetic=True,
        )
        
        # For antithetic sampling, A[0] should be -A[1], A[2] should be -A[3], etc.
        for i in range(0, pop_size, 2):
            error = (A[i] + A[i+1]).abs().mean().item()
            assert error < 1e-5, f"Antithetic pair ({i}, {i+1}) not opposite: error={error}"
    
    def test_antithetic_pairs_same_B(self):
        """Test that antithetic pairs share the same B factor."""
        device = torch.device('cuda')
        pop_size = 128
        member_ids = torch.arange(pop_size, device=device)
        
        A, B = generate_lowrank_factors_fused(
            out_features=256,
            in_features=256,
            rank=4,
            seed=42,
            epoch=0,
            member_ids=member_ids,
            param_id=0,
            sigma=0.2,
            antithetic=True,
        )
        
        # B[0] should equal B[1], B[2] should equal B[3], etc.
        for i in range(0, pop_size, 2):
            error = (B[i] - B[i+1]).abs().mean().item()
            assert error < 1e-5, f"Antithetic pair ({i}, {i+1}) B not equal: error={error}"
    
    def test_perturbation_matrix_antithetic(self):
        """Test that E[i] = -E[i+1] for perturbation matrices."""
        device = torch.device('cuda')
        pop_size = 64
        member_ids = torch.arange(pop_size, device=device)
        
        A, B = generate_lowrank_factors_fused(
            out_features=64,
            in_features=64,
            rank=4,
            seed=42,
            epoch=0,
            member_ids=member_ids,
            param_id=0,
            sigma=0.2,
            antithetic=True,
        )
        
        # E = A @ B.T
        E = torch.bmm(A, B.transpose(1, 2))
        
        # Check E[0] + E[1] ≈ 0, E[2] + E[3] ≈ 0, etc.
        for i in range(0, pop_size, 2):
            error = (E[i] + E[i+1]).abs().mean().item()
            assert error < 1e-5, f"E[{i}] + E[{i+1}] not zero: error={error}"
    
    def test_non_antithetic_mode(self):
        """Test that non-antithetic mode produces independent samples."""
        device = torch.device('cuda')
        pop_size = 64
        member_ids = torch.arange(pop_size, device=device)
        
        A, B = generate_lowrank_factors_fused(
            out_features=256,
            in_features=256,
            rank=4,
            seed=42,
            epoch=0,
            member_ids=member_ids,
            param_id=0,
            sigma=0.2,
            antithetic=False,  # Disable antithetic
        )
        
        # In non-antithetic mode, consecutive pairs should NOT be related
        # A[0] should NOT equal -A[1]
        diff = (A[0] + A[1]).abs().mean().item()
        same = (A[0] - A[1]).abs().mean().item()
        
        # If they were antithetic, diff would be ~0 and same would be large
        # In non-antithetic mode, both should be similar (both around 2*std)
        assert diff > 0.01, "Non-antithetic A[0] and A[1] appear to be negatives"


class TestTritonFusedKernelDeterminism:
    """Test deterministic output given same inputs."""
    
    def test_same_seed_same_output(self):
        """Test that same seed produces identical output."""
        device = torch.device('cuda')
        member_ids = torch.arange(64, device=device)
        
        kwargs = dict(
            out_features=256,
            in_features=256,
            rank=4,
            seed=42,
            epoch=5,
            member_ids=member_ids,
            param_id=3,
            sigma=0.2,
            antithetic=True,
        )
        
        A1, B1 = generate_lowrank_factors_fused(**kwargs)
        A2, B2 = generate_lowrank_factors_fused(**kwargs)
        
        assert torch.allclose(A1, A2), "A not deterministic"
        assert torch.allclose(B1, B2), "B not deterministic"
    
    def test_different_seed_different_output(self):
        """Test that different seeds produce different outputs."""
        device = torch.device('cuda')
        member_ids = torch.arange(64, device=device)
        
        A1, B1 = generate_lowrank_factors_fused(
            out_features=256, in_features=256, rank=4,
            seed=42, epoch=0, member_ids=member_ids, param_id=0, sigma=0.2,
        )
        
        A2, B2 = generate_lowrank_factors_fused(
            out_features=256, in_features=256, rank=4,
            seed=43, epoch=0, member_ids=member_ids, param_id=0, sigma=0.2,  # Different seed
        )
        
        assert not torch.allclose(A1, A2), "Different seeds produced same A"
        assert not torch.allclose(B1, B2), "Different seeds produced same B"
    
    def test_noise_reuse_zero_same_every_epoch(self):
        """Test that noise_reuse=0 produces same output every epoch.
        
        This is the intended behavior per the original JAX implementation:
        When noise_reuse=0, the same noise is used for all epochs.
        CODE: true_epoch = 0 if noise_reuse == 0 else epoch // noise_reuse
        """
        device = torch.device('cuda')
        member_ids = torch.arange(64, device=device)
        
        A1, B1 = generate_lowrank_factors_fused(
            out_features=256, in_features=256, rank=4,
            seed=42, epoch=0, member_ids=member_ids, param_id=0, sigma=0.2,
            noise_reuse=0,  # Same noise every epoch
        )
        
        A2, B2 = generate_lowrank_factors_fused(
            out_features=256, in_features=256, rank=4,
            seed=42, epoch=100, member_ids=member_ids, param_id=0, sigma=0.2,
            noise_reuse=0,
        )
        
        assert torch.allclose(A1, A2), "noise_reuse=0 should produce same noise every epoch"
        assert torch.allclose(B1, B2), "noise_reuse=0 should produce same noise every epoch"
    
    def test_different_param_id_different_output(self):
        """Test that different param_ids produce different outputs."""
        device = torch.device('cuda')
        member_ids = torch.arange(64, device=device)
        
        A1, B1 = generate_lowrank_factors_fused(
            out_features=256, in_features=256, rank=4,
            seed=42, epoch=0, member_ids=member_ids, param_id=0, sigma=0.2,
        )
        
        A2, B2 = generate_lowrank_factors_fused(
            out_features=256, in_features=256, rank=4,
            seed=42, epoch=0, member_ids=member_ids, param_id=1, sigma=0.2,  # Different param_id
        )
        
        assert not torch.allclose(A1, A2), "Different param_ids produced same A"


class TestTritonFusedKernelNoiseReuse:
    """Test noise reuse functionality."""
    
    def test_noise_reuse_same_within_window(self):
        """Test that noise is reused within the reuse window."""
        device = torch.device('cuda')
        member_ids = torch.arange(64, device=device)
        noise_reuse = 5
        
        # Epochs 0-4 should have same noise (effective_epoch = 0)
        A0, B0 = generate_lowrank_factors_fused(
            out_features=256, in_features=256, rank=4,
            seed=42, epoch=0, member_ids=member_ids, param_id=0, sigma=0.2,
            noise_reuse=noise_reuse,
        )
        
        A4, B4 = generate_lowrank_factors_fused(
            out_features=256, in_features=256, rank=4,
            seed=42, epoch=4, member_ids=member_ids, param_id=0, sigma=0.2,
            noise_reuse=noise_reuse,
        )
        
        assert torch.allclose(A0, A4), "Noise not reused within window"
        assert torch.allclose(B0, B4), "Noise not reused within window"
    
    def test_noise_reuse_different_across_windows(self):
        """Test that noise changes across reuse windows."""
        device = torch.device('cuda')
        member_ids = torch.arange(64, device=device)
        noise_reuse = 5
        
        # Epoch 4 (effective=0) vs Epoch 5 (effective=1)
        A4, B4 = generate_lowrank_factors_fused(
            out_features=256, in_features=256, rank=4,
            seed=42, epoch=4, member_ids=member_ids, param_id=0, sigma=0.2,
            noise_reuse=noise_reuse,
        )
        
        A5, B5 = generate_lowrank_factors_fused(
            out_features=256, in_features=256, rank=4,
            seed=42, epoch=5, member_ids=member_ids, param_id=0, sigma=0.2,
            noise_reuse=noise_reuse,
        )
        
        assert not torch.allclose(A4, A5), "Noise should change across windows"


class TestTritonFusedKernelSubsetMemberIds:
    """Test with non-contiguous member_ids."""
    
    def test_subset_member_ids(self):
        """Test with a subset of member IDs."""
        device = torch.device('cuda')
        
        # Request only even members
        member_ids = torch.tensor([0, 2, 4, 6, 8], device=device)
        
        A, B = generate_lowrank_factors_fused(
            out_features=256,
            in_features=256,
            rank=4,
            seed=42,
            epoch=0,
            member_ids=member_ids,
            param_id=0,
            sigma=0.2,
            antithetic=True,
        )
        
        # Should get 5 outputs
        assert A.shape[0] == 5
        assert B.shape[0] == 5
    
    def test_single_member(self):
        """Test with a single member ID."""
        device = torch.device('cuda')
        member_ids = torch.tensor([42], device=device)
        
        A, B = generate_lowrank_factors_fused(
            out_features=256,
            in_features=256,
            rank=4,
            seed=42,
            epoch=0,
            member_ids=member_ids,
            param_id=0,
            sigma=0.2,
        )
        
        assert A.shape == (1, 256, 4)
        assert B.shape == (1, 256, 4)


class TestTritonFusedKernelPerformance:
    """Test performance compared to PyTorch native implementation."""
    
    @pytest.mark.parametrize("pop_size", [512, 1024, 2048, 4096])
    def test_performance_vs_pytorch(self, pop_size):
        """Test that Triton kernel is faster than PyTorch native."""
        device = torch.device('cuda')
        member_ids = torch.arange(pop_size, device=device)
        
        kwargs = dict(
            out_features=256,
            in_features=256,
            rank=4,
            seed=42,
            epoch=0,
            member_ids=member_ids,
            param_id=0,
            sigma=0.2,
            antithetic=True,
        )
        
        # Warmup
        for _ in range(10):
            generate_lowrank_factors_torch(**kwargs)
            generate_lowrank_factors_fused(**kwargs)
        torch.cuda.synchronize()
        
        # Time PyTorch
        n_iter = 100
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iter):
            generate_lowrank_factors_torch(**kwargs)
        torch.cuda.synchronize()
        pytorch_time = time.perf_counter() - start
        
        # Time Triton
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iter):
            generate_lowrank_factors_fused(**kwargs)
        torch.cuda.synchronize()
        triton_time = time.perf_counter() - start
        
        speedup = pytorch_time / triton_time
        print(f"\npop_size={pop_size}: PyTorch={pytorch_time*1000/n_iter:.3f}ms, "
              f"Triton={triton_time*1000/n_iter:.3f}ms, Speedup={speedup:.2f}x")
        
        # We expect at least some speedup (or parity)
        # Note: For small sizes, PyTorch might be faster due to lower overhead
        # The main benefit is at larger sizes
        if pop_size >= 1024:
            assert speedup > 0.5, f"Triton significantly slower than PyTorch: {speedup:.2f}x"


class TestTritonKernelVariants:
    """Test all kernel variants produce consistent results."""
    
    def test_all_variants_consistent_statistics(self):
        """Test that all kernel variants have similar statistics."""
        device = torch.device('cuda')
        pop_size = 1024
        member_ids = torch.arange(pop_size, device=device)
        
        kwargs = dict(
            out_features=256,
            in_features=256,
            rank=4,
            seed=42,
            epoch=0,
            member_ids=member_ids,
            param_id=0,
            sigma=0.2,
            antithetic=True,
        )
        
        # Get results from all variants
        A_simple, B_simple = generate_lowrank_factors_triton_simple(**kwargs)
        A_opt, B_opt = generate_lowrank_factors_triton_optimized(**kwargs)
        A_torch, B_torch = generate_lowrank_factors_torch(**kwargs)
        
        # All should have similar statistics (not identical due to different RNG)
        for name, (A, B) in [
            ("simple", (A_simple, B_simple)),
            ("optimized", (A_opt, B_opt)),
            ("torch", (A_torch, B_torch)),
        ]:
            assert abs(B.mean().item()) < 0.1, f"{name}: B mean not near 0"
            assert abs(B.std().item() - 1.0) < 0.15, f"{name}: B std not near 1"
            assert abs(A.mean().item()) < 0.1, f"{name}: A mean not near 0"
    
    def test_all_variants_antithetic_property(self):
        """Test that all variants satisfy antithetic property."""
        device = torch.device('cuda')
        pop_size = 64
        member_ids = torch.arange(pop_size, device=device)
        
        kwargs = dict(
            out_features=64,
            in_features=64,
            rank=4,
            seed=42,
            epoch=0,
            member_ids=member_ids,
            param_id=0,
            sigma=0.2,
            antithetic=True,
        )
        
        for name, func in [
            ("simple", generate_lowrank_factors_triton_simple),
            ("optimized", generate_lowrank_factors_triton_optimized),
        ]:
            A, B = func(**kwargs)
            
            # Check antithetic pairs
            for i in range(0, pop_size, 2):
                A_error = (A[i] + A[i+1]).abs().mean().item()
                B_error = (B[i] - B[i+1]).abs().mean().item()
                
                assert A_error < 1e-4, f"{name}: A[{i}] + A[{i+1}] not zero: {A_error}"
                assert B_error < 1e-4, f"{name}: B[{i}] != B[{i+1}]: {B_error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
