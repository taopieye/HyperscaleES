"""
EGGROLL Functional Core Tests
=============================

These tests verify the fundamental mathematical properties claimed in
the EGGROLL paper and demonstrate key design decisions.

Paper: "Evolution Strategies at the Hyperscale" (Sarkar et al.)
https://www.alphaxiv.org/abs/2511.16652

Run tests:
    pytest src/hyperscalees/torch/fncl/tests/ -v

Guide:
    1. Low-rank Structure (TestLowRankStructure)
       - Verifies perturbations are rank-r as claimed
       - Demonstrates memory savings: r(m+n) << mn
    
    2. Forward Equivalence (TestForwardEquivalence)
       - The core optimization: x @ B @ A.T instead of x @ (A @ B.T)
       - Proves efficient path = explicit path
    
    3. Antithetic Sampling (TestAntitheticSampling)
       - Thread pairs (2k, 2k+1) have opposite perturbations
       - Standard variance reduction for ES
    
    4. High-Rank Accumulation (TestHighRankAccumulation)
       - Individual perturbations are low-rank
       - But the ES update (weighted sum) is high-rank!
    
    5. ES Gradient Estimate (TestESGradient)
       - Validates the gradient computation formula
       - Shows fitness-weighted sum produces correct update direction
"""
import pytest
import torch
import torch.nn as nn
import math

from hyperscalees.torch.core import (
    EggrollConfig,
    # Dict-Based API
    get_params_dict,
    get_weight_shapes,
    generate_perturbations,
    eggroll_step,
    perturbed_forward,
    make_perturbed_forward_fn,
    # Raw Primitives
    generate_lowrank_perturbations,
    perturbed_linear,
    apply_lowrank_perturbation,
    compute_es_gradient,
    normalize_fitnesses,
)


# ============================================================================
# Test Constants
# ============================================================================

# Default test dimensions (chosen to make rank tests meaningful)
DEFAULT_RANK = 4
DEFAULT_SIGMA = 0.1
DEFAULT_LR = 0.1
DEFAULT_POPULATION_SIZE = 8

# Numerical tolerances
ATOL_TIGHT = 1e-6      # For exact mathematical equalities (e.g., antithetic pairs)
ATOL_LOOSE = 1e-5      # For computed equivalences (e.g., forward pass comparison)

# Random seeds for reproducibility
SEED_BASE = 42
SEED_ALT = 123

# Alignment thresholds
MIN_COSINE_SIMILARITY = 0.5  # Minimum alignment for gradient direction tests


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def device():
    """EGGROLL requires CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("EGGROLL requires CUDA")
    return torch.device("cuda")


@pytest.fixture
def generator(device):
    """Deterministic CUDA generator."""
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED_BASE)
    return gen


@pytest.fixture
def small_weight(device):
    """Small weight matrix for detailed inspection: (8, 4) - rank up to 4."""
    return torch.randn(8, 4, device=device, dtype=torch.float32) * DEFAULT_SIGMA


@pytest.fixture
def medium_weight(device):
    """Medium weight matrix for rank accumulation tests: (64, 32)."""
    return torch.randn(64, 32, device=device, dtype=torch.float32) * DEFAULT_SIGMA


@pytest.fixture
def simple_model(device):
    """Two-layer MLP for integration tests: 4 -> 16 -> 2."""
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.Tanh(),
        nn.Linear(16, 2),
    )
    return model.to(device)


def compute_matrix_rank(matrix: torch.Tensor, tol: float = 1e-5) -> int:
    """Compute numerical rank via SVD."""
    s = torch.linalg.svdvals(matrix.float())
    return int((s > tol).sum().item())


# ============================================================================
# Test 1: Low-Rank Perturbation Structure
# ============================================================================

class TestLowRankStructure:
    """
    PAPER CLAIM: EGGROLL generates perturbations ε = AB^T where A ∈ R^{m×r}, B ∈ R^{n×r}
    with r << min(m,n). This reduces auxiliary storage from mn to r(m+n).
    
    WHY IT MATTERS:
        - For a 256×784 weight (typical MLP), full ES stores 200K floats per population member
        - With rank=4, EGGROLL stores only 4×(256+784) = 4,160 floats - a 48x reduction!
        - This enables population sizes of 2048-65536 that would otherwise OOM
    """

    def test_perturbation_factors_have_correct_shape(self, device, generator):
        """
        generate_lowrank_perturbations returns (A, B) with shapes (pop, m, r) and (pop, n, r).
        
        This is the fundamental storage optimization: we never materialize the full m×n matrix.
        """
        pop_size = 16
        out_dim, in_dim = 64, 32
        
        A_scaled, A, B = generate_lowrank_perturbations(
            pop_size, out_dim, in_dim, DEFAULT_RANK, DEFAULT_SIGMA, generator, torch.float32
        )
        
        # Verify shapes match the paper's specification
        assert A_scaled.shape == (pop_size, out_dim, DEFAULT_RANK), \
            f"A_scaled shape {A_scaled.shape} != expected ({pop_size}, {out_dim}, {DEFAULT_RANK})"
        assert A.shape == (pop_size, out_dim, DEFAULT_RANK)
        assert B.shape == (pop_size, in_dim, DEFAULT_RANK)

    def test_reconstructed_perturbation_has_rank_at_most_r(self, device, generator):
        """
        The materialized perturbation A @ B.T should have rank ≤ r.
        
        MATH: rank(AB^T) ≤ min(rank(A), rank(B)) ≤ r
        
        This is provable: the product of an (m×r) and (r×n) matrix has rank at most r.
        """
        out_dim, in_dim = 32, 16
        
        A_scaled, _, B = generate_lowrank_perturbations(
            DEFAULT_POPULATION_SIZE, out_dim, in_dim, DEFAULT_RANK, DEFAULT_SIGMA, generator, torch.float32
        )
        
        # Check each population member
        for i in range(DEFAULT_POPULATION_SIZE):
            perturbation = A_scaled[i] @ B[i].T  # (out_dim, in_dim)
            computed_rank = compute_matrix_rank(perturbation)
            
            assert computed_rank <= DEFAULT_RANK, \
                f"Member {i}: perturbation rank {computed_rank} > specified rank {DEFAULT_RANK}"

    @pytest.mark.parametrize("rank", [1, 2, 4, 8])
    def test_rank_parameter_controls_perturbation_rank(self, device, rank):
        """
        The rank parameter directly controls the perturbation rank.
        
        Generically (with probability 1), random matrices achieve full rank.
        So rank(AB^T) = r exactly, not just ≤ r.
        """
        gen = torch.Generator(device=device).manual_seed(SEED_ALT)
        pop_size = 4
        out_dim, in_dim = 32, 16  # min(m,n) = 16, so rank up to 16 is achievable
        
        A_scaled, _, B = generate_lowrank_perturbations(
            pop_size, out_dim, in_dim, rank, DEFAULT_SIGMA, gen, torch.float32
        )
        
        for i in range(pop_size):
            perturbation = A_scaled[i] @ B[i].T
            computed_rank = compute_matrix_rank(perturbation)
            
            # With random Gaussian entries, rank should be exactly r
            assert computed_rank == rank, \
                f"Member {i}: expected rank {rank}, got {computed_rank}"

    def test_storage_complexity_is_r_times_m_plus_n(self, device, generator):
        """
        PAPER CLAIM: Auxiliary storage reduced from O(mn) to O(r(m+n)) per layer.
        
        MATH:
            Full-rank ES stores the full perturbation matrix: m × n floats
            EGGROLL stores factors A ∈ R^{m×r} and B ∈ R^{n×r}: r×m + r×n = r(m+n) floats
        
        This test allocates the tensors and measures their memory.
        """
        pop_size = 2  # Minimum for antithetic sampling
        m, n, r = 256, 784, DEFAULT_RANK  # MNIST hidden layer
        
        # Generate the actual low-rank factors
        A_scaled, A, B = generate_lowrank_perturbations(
            pop_size, m, n, r, DEFAULT_SIGMA, generator, torch.float32
        )
        
        # Measure actual memory of low-rank factors (what EGGROLL stores)
        # A_scaled: (pop, m, r), B: (pop, n, r)
        eggroll_bytes = A_scaled.numel() * A_scaled.element_size() + B.numel() * B.element_size()
        
        # What full-rank ES would store: (pop, m, n)
        full_perturbation = A_scaled[0] @ B[0].T  # Materialize to get actual size
        full_rank_bytes = full_perturbation.numel() * full_perturbation.element_size() * pop_size
        
        # Verify actual tensor sizes match the theoretical formula
        expected_eggroll_elements = pop_size * r * (m + n)
        
        assert A_scaled.numel() + B.numel() == expected_eggroll_elements, \
            f"EGGROLL factors have {A_scaled.numel() + B.numel()} elements, expected {expected_eggroll_elements}"
        
        assert full_perturbation.numel() == m * n, \
            f"Full perturbation has {full_perturbation.numel()} elements, expected {m * n}"
        
        # The core claim: low-rank storage < full-rank storage
        assert eggroll_bytes < full_rank_bytes, \
            f"EGGROLL {eggroll_bytes} bytes should be < full-rank {full_rank_bytes} bytes"


# ============================================================================
# Test 2: Forward Pass Equivalence
# ============================================================================

class TestForwardEquivalence:
    """
    PAPER CLAIM: Computing x @ (W + AB^T) naively requires forming the mn perturbation.
    Instead: x @ (W + AB^T)^T = x @ W^T + x @ B @ A^T
    
    This reduces cost from O(mn) to O(r(m+n)) per forward pass.
    
    WHY IT MATTERS:
        - The efficient formula never materializes the full perturbed weight
        - Two rank-r matmuls are cheaper than one full matmul when r << min(m,n)
        - This is THE key insight that makes low-rank ES practical
    """

    def test_perturbed_linear_equals_explicit_perturbation(self, device, generator):
        """
        CORE CORRECTNESS: perturbed_linear(x, W, b, A, B) == x @ (W + AB^T)^T + b
        
        This proves the efficient implementation matches the mathematical definition.
        """
        batch_size, in_dim, out_dim = 8, 16, 32
        pop_size = 4
        
        # Setup
        x = torch.randn(pop_size, batch_size, in_dim, device=device)
        W = torch.randn(out_dim, in_dim, device=device)
        b = torch.randn(out_dim, device=device)
        
        A_scaled, _, B = generate_lowrank_perturbations(
            pop_size, out_dim, in_dim, DEFAULT_RANK, DEFAULT_SIGMA, generator, torch.float32
        )
        
        # Method 1: Efficient implementation (never forms full perturbation)
        output_efficient = perturbed_linear(x, W, b, A_scaled, B)
        
        # Method 2: Explicit perturbation (slow but obviously correct)
        output_explicit = torch.zeros_like(output_efficient)
        for i in range(pop_size):
            perturbation = A_scaled[i] @ B[i].T  # (out_dim, in_dim)
            perturbed_W = W + perturbation
            output_explicit[i] = x[i] @ perturbed_W.T + b
        
        assert torch.allclose(output_efficient, output_explicit, atol=ATOL_LOOSE), \
            "Efficient perturbed_linear must equal explicit perturbation"

    def test_apply_lowrank_perturbation_is_two_matmuls(self, device, generator):
        """
        MATH: x @ B @ A^T is computed as two matmuls, never forming B @ A^T.
        
        This is the computational trick: (m,n) @ (n,r) @ (r,m) uses O(r(m+n)) flops
        vs O(mn) to form the perturbation first.
        """
        batch_size, in_dim, out_dim = 16, 32, 64
        pop_size = DEFAULT_POPULATION_SIZE
        
        x = torch.randn(pop_size, batch_size, in_dim, device=device)
        A_scaled, _, B = generate_lowrank_perturbations(
            pop_size, out_dim, in_dim, DEFAULT_RANK, DEFAULT_SIGMA, generator, torch.float32
        )
        
        # Efficient: x @ B @ A.T (computed left-to-right)
        efficient = apply_lowrank_perturbation(x, B, A_scaled)
        
        # Explicit: form A @ B.T first, then multiply
        explicit = torch.zeros(pop_size, batch_size, out_dim, device=device)
        for i in range(pop_size):
            full_pert = A_scaled[i] @ B[i].T  # This is what we avoid!
            explicit[i] = x[i] @ full_pert.T
        
        assert torch.allclose(efficient, explicit, atol=ATOL_LOOSE), \
            "apply_lowrank_perturbation must match explicit A @ B.T computation"

    def test_make_perturbed_forward_fn_traces_sequential_model(self, simple_model, device):
        """
        make_perturbed_forward_fn uses torch.fx to auto-generate forward functions.
        
        DESIGN: This lets users define models with standard PyTorch, then convert
        to EGGROLL format automatically. The traced forward replaces nn.Linear
        with perturbed_forward calls.
        """
        params = get_params_dict(simple_model)
        shapes = get_weight_shapes(params)
        forward, forward_eval = make_perturbed_forward_fn(simple_model)
        
        # Generate perturbations
        gen = torch.Generator(device=device).manual_seed(SEED_BASE)
        perts = generate_perturbations(shapes, population_size=4, rank=DEFAULT_RANK, sigma=DEFAULT_SIGMA, 
                                       generator=gen, dtype=torch.float32)
        
        # Test forward with perturbations
        batch_size, model_in_dim, model_out_dim = 8, 4, 2
        x = torch.randn(4, batch_size, model_in_dim, device=device)  # (pop, batch, in_dim)
        output = forward(x, params, perts)
        
        assert output.shape == (4, batch_size, model_out_dim), \
            f"Expected output shape (4, {batch_size}, {model_out_dim}), got {output.shape}"
        
        # Test eval forward (no perturbations)
        x_eval = torch.randn(batch_size, model_in_dim, device=device)  # (batch, in_dim)
        output_eval = forward_eval(x_eval, params)
        
        assert output_eval.shape == (batch_size, model_out_dim), \
            f"Expected eval output shape ({batch_size}, {model_out_dim}), got {output_eval.shape}"


# ============================================================================
# Test 3: Antithetic (Mirrored) Sampling
# ============================================================================

class TestAntitheticSampling:
    """
    PAPER CLAIM: Thread pairs (2k, 2k+1) use opposite-sign perturbations: ±σε.
    
    MATH: For ε ~ N(0, I), we evaluate both θ + σε and θ - σε.
    This halves variance of the gradient estimate without extra forward passes
    (since we'd compute pop/2 unique perturbations anyway).
    
    WHY IT MATTERS:
        - Variance reduction is crucial for ES with limited compute
        - If f(θ+ε) ≈ f(θ-ε), contributions cancel → noise reduction
        - If f(θ+ε) >> f(θ-ε), the update direction is clear
    """

    def test_even_odd_pairs_have_negated_A_factors(self, device, generator):
        """
        For antithetic pairs, A_even = -A_odd (scaled factor contains the sign).
        B stays the same since the sign is absorbed into A.
        
        CODE: A_pos / -A_pos concatenation in generate_lowrank_perturbations
        """
        pop_size = DEFAULT_POPULATION_SIZE
        out_dim, in_dim = 16, 8
        
        A_scaled, _, B = generate_lowrank_perturbations(
            pop_size, out_dim, in_dim, DEFAULT_RANK, DEFAULT_SIGMA, generator, torch.float32
        )
        
        half = pop_size // 2
        for i in range(half):
            # Even member (i) and odd member (i + half) should be negatives
            A_even = A_scaled[i]
            A_odd = A_scaled[i + half]
            
            assert torch.allclose(A_even, -A_odd, atol=ATOL_TIGHT), \
                f"A factors for members {i} and {i + half} should be negatives"
            
            # B factors should be identical
            B_even = B[i]
            B_odd = B[i + half]
            
            assert torch.allclose(B_even, B_odd, atol=ATOL_TIGHT), \
                f"B factors for members {i} and {i + half} should be identical"

    def test_perturbation_matrices_are_exact_negatives(self, device, generator):
        """
        The full perturbation A @ B.T is negated between antithetic pairs.
        
        MATH: A_pos @ B.T and (-A_pos) @ B.T = -(A_pos @ B.T)
        """
        pop_size = DEFAULT_POPULATION_SIZE
        out_dim, in_dim = 16, 8
        
        A_scaled, _, B = generate_lowrank_perturbations(
            pop_size, out_dim, in_dim, DEFAULT_RANK, DEFAULT_SIGMA, generator, torch.float32
        )
        
        half = pop_size // 2
        for i in range(half):
            pert_even = A_scaled[i] @ B[i].T
            pert_odd = A_scaled[i + half] @ B[i + half].T
            
            assert torch.allclose(pert_even, -pert_odd, atol=ATOL_TIGHT), \
                f"Perturbations for members {i} and {i + half} should be negatives"

    def test_antithetic_pairs_bracket_base_output(self, device, generator):
        """
        PROPERTY: (output_+ + output_-) / 2 = unperturbed output
        
        This is a consequence of linearity: if y = Wx + b, then
        (W + ε)x + (W - ε)x = 2Wx, so the average is the unperturbed output.
        """
        batch_size, in_dim, out_dim = 4, 8, 16
        pop_size = 2  # One antithetic pair
        
        x = torch.randn(batch_size, in_dim, device=device)
        W = torch.randn(out_dim, in_dim, device=device)
        b = torch.randn(out_dim, device=device)
        
        A_scaled, _, B = generate_lowrank_perturbations(
            pop_size, out_dim, in_dim, DEFAULT_RANK, DEFAULT_SIGMA, generator, torch.float32
        )
        
        # Expand x for population
        x_pop = x.unsqueeze(0).expand(pop_size, -1, -1)  # (2, batch, in_dim)
        
        # Perturbed outputs
        output_perturbed = perturbed_linear(x_pop, W, b, A_scaled, B)
        
        # Unperturbed output
        output_base = x @ W.T + b
        
        # Average of antithetic pair should equal base
        output_avg = (output_perturbed[0] + output_perturbed[1]) / 2
        
        assert torch.allclose(output_avg, output_base, atol=ATOL_LOOSE), \
            "Average of antithetic outputs should equal unperturbed output"


# ============================================================================
# Test 4: High-Rank Accumulation
# ============================================================================

class TestHighRankAccumulation:
    """
    PAPER CLAIM: "Although individual perturbations are low-rank, the expression on the
    right side is actually high-rank, due to the properties of sums of low-rank matrices."
    
    MATH: Each ε_i = A_i B_i^T has rank r. But Σ w_i ε_i can have rank up to min(Nr, m, n).
    
    WHY IT MATTERS:
        - Low-rank perturbations don't mean low-rank updates!
        - The final gradient estimate spans a high-dimensional subspace
        - This is why EGGROLL is as expressive as full-rank ES
    """

    def test_sum_of_rank1_exceeds_rank1(self, device):
        """
        Sum of N independent rank-1 matrices generically has rank > 1.
        
        MATH: Each outer product uv^T is rank-1, but Σ u_i v_i^T is generically full-rank
        (unless there are linear dependencies, which has measure zero for random matrices).
        """
        out_dim, in_dim = 16, 8
        rank_one = 1  # Rank-1 perturbations for this test
        num_perturbations = DEFAULT_POPULATION_SIZE
        
        accumulated = torch.zeros(out_dim, in_dim, device=device)
        
        for i in range(num_perturbations):
            gen = torch.Generator(device=device).manual_seed(i * 100)
            A_scaled, _, B = generate_lowrank_perturbations(
                2, out_dim, in_dim, rank_one, DEFAULT_SIGMA, gen, torch.float32
            )
            # Use only the first (even) member of each pair
            accumulated += A_scaled[0] @ B[0].T
        
        # Individual perturbations are rank-1, but sum should be higher
        accumulated_rank = compute_matrix_rank(accumulated)
        
        assert accumulated_rank > 1, \
            f"Sum of {num_perturbations} rank-1 matrices should have rank > 1, got {accumulated_rank}"
        
        print(f"\nSum of {num_perturbations} rank-1 perturbations has rank {accumulated_rank}")

    def test_accumulated_rank_approaches_full_rank(self, medium_weight, device):
        """
        With enough population members, the accumulated update approaches full rank.
        
        For (64, 32) matrix: min(m,n) = 32. With 32+ rank-1 perturbations,
        we should achieve (nearly) full rank.
        """
        m, n = medium_weight.shape  # 64, 32
        max_rank = min(m, n)  # 32
        rank_one = 1
        num_perturbations = 64  # More than max_rank to ensure full coverage
        
        accumulated = torch.zeros(m, n, device=device)
        
        for i in range(num_perturbations):
            gen = torch.Generator(device=device).manual_seed(i * 100)
            A_scaled, _, B = generate_lowrank_perturbations(
                2, m, n, rank_one, DEFAULT_SIGMA, gen, torch.float32
            )
            accumulated += A_scaled[0] @ B[0].T
        
        accumulated_rank = compute_matrix_rank(accumulated)
        
        # Should be close to max_rank
        rank_tolerance = 2
        assert accumulated_rank >= max_rank - rank_tolerance, \
            f"With {num_perturbations} rank-1 perturbations, expected rank ~{max_rank}, got {accumulated_rank}"

    def test_higher_rank_accumulates_faster(self, device):
        """
        Higher-rank perturbations reach full rank with fewer population members.
        
        With rank=4 perturbations, we need ~8 members to span rank-32 space.
        With rank=1, we need ~32 members.
        """
        out_dim, in_dim = 32, 32  # Square for simplicity
        max_rank = 32
        num_perturbations = DEFAULT_POPULATION_SIZE  # Fixed number of perturbations
        
        results = []
        for rank in [1, 2, 4, 8]:
            accumulated = torch.zeros(out_dim, in_dim, device=device)
            
            for i in range(num_perturbations):
                gen = torch.Generator(device=device).manual_seed(i * 100 + rank)
                A_scaled, _, B = generate_lowrank_perturbations(
                    2, out_dim, in_dim, rank, DEFAULT_SIGMA, gen, torch.float32
                )
                accumulated += A_scaled[0] @ B[0].T
            
            accumulated_rank = compute_matrix_rank(accumulated)
            results.append((rank, accumulated_rank))
        
        # Higher rank should lead to higher accumulated rank (monotonic)
        print(f"\nRank accumulation with {num_perturbations} perturbations:")
        for r, acc_r in results:
            print(f"  rank={r} perturbations → accumulated rank {acc_r}")
        
        # Verify monotonicity
        ranks = [r[1] for r in results]
        assert ranks == sorted(ranks), \
            f"Accumulated rank should increase with perturbation rank: {results}"


# ============================================================================
# Test 5: ES Gradient Estimate
# ============================================================================

class TestESGradient:
    """
    PAPER FORMULA: ∇_W ≈ (1/√N) Σ f_i · A_i · B_i^T
    
    where f_i = normalized fitness for population member i.
    
    DESIGN DECISIONS:
        1. We use einsum for efficient batched gradient computation
        2. Normalization (zero-mean, unit-variance) is critical for stable updates
        3. The √N scaling follows the paper's variance normalization
    """

    def test_normalize_fitnesses_produces_zero_mean_unit_var(self, device):
        """
        normalize_fitnesses should produce zero mean and approximately unit variance.
        
        This is standard practice in ES to make the learning rate scale-invariant.
        """
        fitnesses = torch.tensor([100., 200., 150., 175., 225., 125., 180., 160.], device=device)
        
        normalized = normalize_fitnesses(fitnesses)
        
        assert abs(normalized.mean().item()) < ATOL_TIGHT, \
            f"Normalized fitnesses should have zero mean, got {normalized.mean().item()}"
        
        # Variance should be close to 1 (not exactly 1 due to Bessel's correction)
        std_tolerance = 0.2
        assert abs(normalized.std().item() - 1.0) < std_tolerance, \
            f"Normalized fitnesses should have ~unit std, got {normalized.std().item()}"

    def test_compute_es_gradient_formula(self, device, generator):
        """
        VERIFY: compute_es_gradient implements (1/√N) Σ f_i · A_i · B_i^T
        
        We compute this explicitly and compare to the einsum implementation.
        """
        pop_size = DEFAULT_POPULATION_SIZE
        out_dim, in_dim = 16, 8
        
        A_scaled, _, B = generate_lowrank_perturbations(
            pop_size, out_dim, in_dim, DEFAULT_RANK, DEFAULT_SIGMA, generator, torch.float32
        )
        
        # Random normalized fitnesses
        fitnesses = normalize_fitnesses(torch.randn(pop_size, device=device))
        
        # Method 1: Use compute_es_gradient (einsum implementation)
        grad_efficient = compute_es_gradient(fitnesses, A_scaled, B, pop_size)
        
        # Method 2: Explicit loop (obviously correct)
        grad_explicit = torch.zeros(out_dim, in_dim, device=device)
        sqrt_N = math.sqrt(pop_size)
        for i in range(pop_size):
            grad_explicit += fitnesses[i] * (A_scaled[i] @ B[i].T)
        grad_explicit /= sqrt_N
        
        assert torch.allclose(grad_efficient, grad_explicit, atol=ATOL_LOOSE), \
            "compute_es_gradient must match explicit formula"

    def test_gradient_direction_favors_high_fitness_perturbations(self, device, generator):
        """
        The gradient should point toward perturbations with high fitness.
        
        INTUITION: If perturbation A_0 @ B_0^T leads to high fitness,
        the gradient should be aligned with that direction.
        """
        pop_size = 4
        out_dim, in_dim = 8, 4
        rank = 2
        
        A_scaled, _, B = generate_lowrank_perturbations(
            pop_size, out_dim, in_dim, rank, DEFAULT_SIGMA, generator, torch.float32
        )
        
        # Make member 0 have much higher fitness
        fitnesses = torch.tensor([10., -10., 0.1, -0.1], device=device)  # Antithetic pairs
        fitnesses = normalize_fitnesses(fitnesses)
        
        grad = compute_es_gradient(fitnesses, A_scaled, B, pop_size)
        
        # The gradient should be most aligned with the high-fitness perturbation
        pert_0 = A_scaled[0] @ B[0].T
        
        # Compute cosine similarity
        grad_flat = grad.flatten()
        pert_flat = pert_0.flatten()
        
        cosine_sim = (grad_flat @ pert_flat) / (grad_flat.norm() * pert_flat.norm())
        
        assert cosine_sim > MIN_COSINE_SIMILARITY, \
            f"Gradient should be aligned with high-fitness perturbation, cosine_sim = {cosine_sim:.3f}"


# ============================================================================
# Test 6: Integration - eggroll_step
# ============================================================================

class TestEggrollStep:
    """
    eggroll_step is the high-level API that combines:
    1. normalize_fitnesses
    2. compute_gradients (for all params)
    3. update_params (in-place)
    4. Apply lr/sigma decay from config
    
    These tests verify the update is mathematically correct: W_new = W_old + lr * grad
    """

    def test_eggroll_step_applies_correct_update(self, device, generator):
        """
        VERIFY: eggroll_step computes W_new = W_old + lr * grad exactly.
        
        We manually compute the expected gradient and verify the update matches.
        This proves the full pipeline (normalize → gradient → update) is correct.
        """
        # Simple single-layer case for clarity
        out_dim, in_dim = 16, 8
        pop_size = DEFAULT_POPULATION_SIZE
        
        # Create a simple model and extract params
        model = nn.Linear(in_dim, out_dim, bias=True).to(device)
        params = get_params_dict(model)
        shapes = get_weight_shapes(params)
        
        config = EggrollConfig(population_size=pop_size, rank=DEFAULT_RANK, sigma=DEFAULT_SIGMA, lr=DEFAULT_LR,
                               lr_decay=1.0, sigma_decay=1.0)  # No decay for this test
        
        gen = torch.Generator(device=device).manual_seed(SEED_BASE)
        perts = generate_perturbations(shapes, pop_size, DEFAULT_RANK, DEFAULT_SIGMA, gen, torch.float32)
        
        # Save original weight
        W_old = params['weight'].clone()
        
        # Create specific fitnesses (not random - we want to verify exact computation)
        fitnesses = torch.tensor([2., -2., 1., -1., 0.5, -0.5, 0.1, -0.1], device=device)
        
        # Manually compute expected gradient using the formula from TestESGradient
        A_scaled, B = perts['weight']
        f_normalized = normalize_fitnesses(fitnesses)
        sqrt_N = math.sqrt(pop_size)
        expected_grad = torch.zeros(out_dim, in_dim, device=device)
        for i in range(pop_size):
            expected_grad += f_normalized[i] * (A_scaled[i] @ B[i].T)
        expected_grad /= sqrt_N
        
        W_expected = W_old + DEFAULT_LR * expected_grad
        
        # Perform the actual step
        eggroll_step(params, fitnesses, perts, DEFAULT_LR, DEFAULT_SIGMA, config)
        
        # Verify the update is exactly correct
        assert torch.allclose(params['weight'], W_expected, atol=ATOL_LOOSE), \
            "eggroll_step should apply W_new = W_old + lr * grad exactly"

    def test_decay_parameters_updated_correctly(self, device):
        """
        Verify lr_decay and sigma_decay are applied correctly.
        
        This is a simple sanity check - the decay math is trivial but 
        mistakes here would break training schedules.
        """
        model = nn.Linear(8, 4).to(device)
        params = get_params_dict(model)
        shapes = get_weight_shapes(params)
        
        lr, sigma = DEFAULT_LR, 0.2
        lr_decay, sigma_decay = 0.99, 0.995
        
        config = EggrollConfig(population_size=4, rank=2, sigma=sigma, lr=lr,
                               lr_decay=lr_decay, sigma_decay=sigma_decay)
        
        gen = torch.Generator(device=device).manual_seed(SEED_BASE)
        perts = generate_perturbations(shapes, 4, 2, sigma, gen, torch.float32)
        fitnesses = torch.randn(4, device=device)
        
        new_lr, new_sigma = eggroll_step(params, fitnesses, perts, lr, sigma, config)
        
        decay_atol = 1e-8
        assert abs(new_lr - lr * lr_decay) < decay_atol, \
            f"Expected lr={lr * lr_decay}, got {new_lr}"
        assert abs(new_sigma - sigma * sigma_decay) < decay_atol, \
            f"Expected sigma={sigma * sigma_decay}, got {new_sigma}"

    def test_high_fitness_perturbation_increases_alignment(self, device):
        """
        VERIFY: After update, weights are more aligned with the high-fitness perturbation.
        
        This is the core behavior we want: ES should move params toward
        perturbations that produced high fitness.
        
        MATH: If member 0 has the highest fitness, then after the update,
        cos(W_new - W_old, perturbation_0) should be positive.
        """
        out_dim, in_dim = 16, 8
        pop_size = 4
        lr = 0.5  # Higher lr to make update more pronounced
        
        model = nn.Linear(in_dim, out_dim, bias=False).to(device)
        params = get_params_dict(model)
        shapes = get_weight_shapes(params)
        
        gen = torch.Generator(device=device).manual_seed(SEED_BASE)
        perts = generate_perturbations(shapes, pop_size, DEFAULT_RANK, DEFAULT_SIGMA, gen, torch.float32)
        
        W_old = params['weight'].clone()
        
        # Member 0 has high fitness, member 1 (its antithetic pair) has low fitness
        # Members 2,3 are neutral
        fitnesses = torch.tensor([10., -10., 0., 0.], device=device)
        
        config = EggrollConfig(population_size=pop_size, rank=DEFAULT_RANK, sigma=DEFAULT_SIGMA, lr=lr,
                               lr_decay=1.0, sigma_decay=1.0)
        
        eggroll_step(params, fitnesses, perts, lr, DEFAULT_SIGMA, config)
        
        # Compute the actual update direction
        W_delta = params['weight'] - W_old
        
        # Get the perturbation that had high fitness
        A_scaled, B = perts['weight']
        pert_0 = A_scaled[0] @ B[0].T  # High-fitness perturbation
        
        # Compute alignment (cosine similarity)
        delta_flat = W_delta.flatten()
        pert_flat = pert_0.flatten()
        cosine_sim = (delta_flat @ pert_flat) / (delta_flat.norm() * pert_flat.norm())
        
        # The alignment won't be perfect (1.0) because:
        # 1. Antithetic pairs both contribute (with opposite signs)
        # 2. Other population members add noise
        # But with extreme fitness difference (10 vs 0), alignment should be strong
        assert cosine_sim > MIN_COSINE_SIMILARITY, \
            f"Update should be positively aligned with high-fitness perturbation, got cosine_sim={cosine_sim:.3f}"
        
        print(f"\nUpdate alignment with high-fitness perturbation: {cosine_sim:.4f}")


# ============================================================================
# Test 7: Dict-Based API Convenience
# ============================================================================

class TestDictBasedAPI:
    """
    The Dict-Based API (get_params_dict, get_weight_shapes, generate_perturbations, etc.)
    is the recommended interface. It handles multiple layers automatically.
    """

    def test_get_weight_shapes_detects_weights(self, simple_model, device):
        """
        get_weight_shapes should return shapes for all weight tensors (not biases).
        
        DESIGN: We only perturb weight matrices, not biases.
        Biases are 1D and low-rank perturbation doesn't apply cleanly.
        """
        params = get_params_dict(simple_model)
        shapes = get_weight_shapes(params)
        
        # Model structure: Linear(4→16), ReLU, Linear(16→2)
        num_weight_layers = 2
        assert len(shapes) == num_weight_layers
        assert '0.weight' in shapes
        assert '2.weight' in shapes
        
        # Should NOT include biases
        assert '0.bias' not in shapes
        assert '2.bias' not in shapes
        
        # Verify shapes match layer dimensions (out_dim, in_dim)
        layer0_out, layer0_in = 16, 4
        layer2_out, layer2_in = 2, 16
        assert shapes['0.weight'] == (layer0_out, layer0_in)
        assert shapes['2.weight'] == (layer2_out, layer2_in)

    def test_generate_perturbations_handles_multiple_layers(self, simple_model, device):
        """
        generate_perturbations should create perturbations for all weight layers.
        """
        params = get_params_dict(simple_model)
        shapes = get_weight_shapes(params)
        
        gen = torch.Generator(device=device).manual_seed(SEED_BASE)
        perts = generate_perturbations(shapes, population_size=DEFAULT_POPULATION_SIZE, rank=DEFAULT_RANK, sigma=DEFAULT_SIGMA,
                                       generator=gen, dtype=torch.float32)
        
        # Should have perturbations for each weight
        assert len(perts) == len(shapes)
        
        for name, (out_dim, in_dim) in shapes.items():
            assert name in perts
            A_scaled, B = perts[name]
            assert A_scaled.shape == (DEFAULT_POPULATION_SIZE, out_dim, DEFAULT_RANK)
            assert B.shape == (DEFAULT_POPULATION_SIZE, in_dim, DEFAULT_RANK)

    def test_perturbed_forward_uses_correct_perturbation(self, simple_model, device):
        """
        perturbed_forward should look up the correct perturbation by weight_name.
        """
        params = get_params_dict(simple_model)
        shapes = get_weight_shapes(params)
        
        pop_size = 4
        gen = torch.Generator(device=device).manual_seed(SEED_BASE)
        perts = generate_perturbations(shapes, population_size=pop_size, rank=2, sigma=DEFAULT_SIGMA,
                                       generator=gen, dtype=torch.float32)
        
        # Apply perturbed forward manually for one layer
        batch_size = 8
        model_in_dim, hidden_dim = 4, 16
        x = torch.randn(pop_size, batch_size, model_in_dim, device=device)
        
        output = perturbed_forward(x, params['0.weight'], params['0.bias'], 
                                   perts, '0.weight')
        
        # Verify output shape
        assert output.shape == (pop_size, batch_size, hidden_dim)
        
        # Verify perturbation was applied (compare to unperturbed)
        unperturbed = x @ params['0.weight'].T + params['0.bias']
        assert not torch.allclose(output, unperturbed), \
            "perturbed_forward should differ from unperturbed forward"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
