"""
Shared fixtures for PyTorch EGGROLL test suite.

These fixtures provide common test infrastructure for the PyTorch port,
following PyTorch conventions while maintaining test coverage parity
with the JAX implementation.

Note: EGGROLL-Torch needs a CUDA GPU. Tests will skip if no GPU is available.
"""
import pytest
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# GPU Requirement - This is non-negotiable
# ============================================================================

def _require_gpu():
    """
    Check that a CUDA GPU is available.
    
    EGGROLL-Torch needs a GPU to deliver on its promise of fast batched
    perturbations. On CPU, you'd lose the speed advantage that makes 
    EGGROLL worth using over simpler ES implementations.
    
    Raises:
        RuntimeError: If no CUDA device is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n" + "="*70 + "\n"
            "EGGROLL needs a CUDA GPU\n"
            "="*70 + "\n\n"
            "EGGROLL-Torch is designed for GPU-accelerated batched perturbations.\n"
            "On CPU, you'd lose the speed advantage that makes it worth using.\n\n"
            "A few options:\n"
            "  • Use a machine with an NVIDIA GPU\n"
            "  • Try Google Colab (free GPU tier)\n"
            "  • For CPU-only work, check out OpenAI's ES or other CPU-friendly libraries\n\n"
            "If you're seeing this in CI, make sure your runner has GPU support.\n"
            + "="*70
        )


# ============================================================================
# Device fixtures
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def require_gpu():
    """
    Session-scoped fixture that enforces GPU availability.
    
    This runs once at the start of the test session and fails loudly
    if no CUDA device is detected. All EGGROLL tests require GPU.
    """
    _require_gpu()


@pytest.fixture
def device(require_gpu):
    """
    Get CUDA device. Fails if no GPU available.
    
    Note: There is no CPU fallback. EGGROLL is GPU-only.
    """
    return torch.device("cuda")


# ============================================================================
# Random generator fixtures
# ============================================================================

@pytest.fixture
def base_generator(device):
    """
    Base random generator for reproducible tests.
    
    Note: PyTorch generators are device-specific. This creates a CUDA generator
    since all EGGROLL operations happen on GPU.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    return gen


@pytest.fixture
def cpu_generator():
    """
    CPU generator for operations that require CPU RNG.
    
    Some PyTorch operations (e.g., DataLoader shuffling) require CPU generators.
    Use sparingly - most EGGROLL ops should use GPU generators.
    """
    return torch.Generator(device='cpu').manual_seed(42)


@pytest.fixture
def model_generator(device):
    """Generator for model initialization (GPU)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(42 + 0)
    return gen


@pytest.fixture
def es_generator(device):
    """Generator for ES perturbation generation (GPU)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(42 + 1)
    return gen


# ============================================================================
# Tensor fixtures
# ============================================================================

@pytest.fixture
def small_tensor(device):
    """Small 2D tensor for detailed inspection (8x4 matrix)."""
    return torch.ones(8, 4, dtype=torch.float32, device=device) * 0.1


@pytest.fixture
def medium_tensor(device):
    """Medium 2D tensor for rank tests (64x32 matrix)."""
    gen = torch.Generator().manual_seed(123)
    return torch.randn(64, 32, dtype=torch.float32, generator=gen).to(device) * 0.1


@pytest.fixture
def large_tensor(device):
    """Larger 2D tensor for scalability tests (256x128 matrix)."""
    gen = torch.Generator().manual_seed(456)
    return torch.randn(256, 128, dtype=torch.float32, generator=gen).to(device) * 0.1


# ============================================================================
# Configuration dataclasses (clean, typed configs)
# ============================================================================

@dataclass
class EggrollConfig:
    """Configuration for Eggroll strategy."""
    sigma: float = 0.1
    lr: float = 0.01
    rank: int = 4
    antithetic: bool = True
    noise_reuse: int = 0
    optimizer: str = "sgd"
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    seed: int = 42


@dataclass
class OpenESConfig:
    """Configuration for OpenES strategy."""
    sigma: float = 0.1
    lr: float = 0.01
    antithetic: bool = True
    noise_reuse: int = 0
    optimizer: str = "sgd"
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    seed: int = 42


# ============================================================================
# Strategy configuration fixtures
# ============================================================================

@pytest.fixture
def eggroll_config():
    """Standard EggRoll configuration."""
    return EggrollConfig()


@pytest.fixture
def eggroll_rank1_config():
    """EggRoll with rank=1 (minimal low-rank)."""
    return EggrollConfig(rank=1)


@pytest.fixture
def eggroll_high_rank_config():
    """EggRoll with higher rank for expressiveness tests."""
    return EggrollConfig(rank=16)


@pytest.fixture
def open_es_config():
    """Standard OpenES configuration."""
    return OpenESConfig()


# ============================================================================
# Model fixtures
# ============================================================================

@pytest.fixture
def simple_linear(device):
    """Simple linear layer for basic tests."""
    layer = nn.Linear(4, 8, bias=False)
    layer.to(device)
    return layer


@pytest.fixture
def simple_mlp(device):
    """Simple MLP for integration tests."""
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )
    model.to(device)
    return model


@pytest.fixture
def deep_mlp(device):
    """Deeper MLP for scalability tests."""
    layers = []
    dims = [32, 64, 64, 64, 32, 16]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    model.to(device)
    return model


@pytest.fixture
def mlp_with_bias(device):
    """MLP with biases for testing non-lora parameter handling."""
    model = nn.Sequential(
        nn.Linear(8, 16, bias=True),
        nn.ReLU(),
        nn.Linear(16, 2, bias=True)
    )
    model.to(device)
    return model


# ============================================================================
# Population fixtures
# ============================================================================

@pytest.fixture
def small_population_size():
    """Small population for detailed tests."""
    return 8


@pytest.fixture
def medium_population_size():
    """Medium population for standard tests."""
    return 64


@pytest.fixture
def large_population_size():
    """Larger population for scalability tests."""
    return 256


# ============================================================================
# Input fixtures
# ============================================================================

@pytest.fixture
def batch_input_small(device):
    """Small batch input for basic tests."""
    gen = torch.Generator().manual_seed(789)
    return torch.randn(4, 8, dtype=torch.float32, generator=gen).to(device)


@pytest.fixture
def batch_input_medium(device):
    """Medium batch input for standard tests."""
    gen = torch.Generator().manual_seed(101112)
    return torch.randn(32, 8, dtype=torch.float32, generator=gen).to(device)


# ============================================================================
# Helper functions (importable by tests)
# ============================================================================

def compute_matrix_rank(tensor: torch.Tensor, tol: float = 1e-5) -> int:
    """
    Compute numerical rank of a matrix using SVD.
    
    A singular value is considered zero if it's less than tol * max(singular_values).
    """
    s = torch.linalg.svdvals(tensor)
    threshold = tol * s.max()
    return int((s > threshold).sum().item())


def make_fitnesses(population_size: int, pattern: str = "random", device: torch.device = None) -> torch.Tensor:
    """
    Create fitness tensors with various patterns for testing.
    
    Patterns:
    - "random": Random fitnesses
    - "uniform": All equal (should produce zero update)
    - "one_hot": One high, rest low (should produce directed update)
    - "gradient": Linearly increasing
    - "antithetic_cancel": Pairs have equal fitness
    """
    device = device or torch.device("cpu")
    
    if pattern == "random":
        gen = torch.Generator().manual_seed(999)
        return torch.randn(population_size, generator=gen).to(device)
    
    elif pattern == "uniform":
        return torch.ones(population_size, device=device) * 5.0
    
    elif pattern == "one_hot":
        fitnesses = torch.ones(population_size, device=device) * -1.0
        fitnesses[population_size // 2] = 10.0  # One high fitness
        return fitnesses
    
    elif pattern == "gradient":
        return torch.linspace(0, 1, population_size, device=device)
    
    elif pattern == "antithetic_cancel":
        # Pairs (0,1), (2,3), etc. have same fitness
        base = torch.randn(population_size // 2)
        return torch.repeat_interleave(base, 2).to(device)
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def assert_tensors_close(
    actual: torch.Tensor, 
    expected: torch.Tensor, 
    atol: float = 1e-5, 
    rtol: float = 1e-5,
    msg: str = ""
):
    """Assert two tensors are close with helpful error message."""
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        diff = (actual - expected).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        raise AssertionError(
            f"{msg}\n"
            f"Max difference: {max_diff:.2e}, Mean difference: {mean_diff:.2e}\n"
            f"Tolerance: atol={atol}, rtol={rtol}"
        )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters by type in a model."""
    counts = {"total": 0, "trainable": 0, "frozen": 0}
    for p in model.parameters():
        counts["total"] += p.numel()
        if p.requires_grad:
            counts["trainable"] += p.numel()
        else:
            counts["frozen"] += p.numel()
    return counts


# ============================================================================
# Marks for test categorization
# ============================================================================

# Mark tests that require CUDA
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

# Mark tests for unimplemented features (expected to fail)
unimplemented = pytest.mark.skip(reason="Feature not yet implemented")

# Mark slow tests
slow = pytest.mark.slow


# ============================================================================
# Type aliases for cleaner test signatures
# ============================================================================

Tensor = torch.Tensor
Module = nn.Module
Generator = torch.Generator
