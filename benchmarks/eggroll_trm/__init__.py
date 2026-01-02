"""
EGGROLL-Optimized TRM (Tiny Recursive Model)

This implements TRM with EGGROLL-friendly architecture:
- Functional parameter dict (no nn.Module) for vmap compatibility
- Rank-1 perturbations injected at linear layers
- Memory-efficient batched forward pass
- Minimal intermediate activations

Key design principles from the Perceiver implementation:
1. Share base computation (W @ x) across population
2. Only add perturbation term (σ * (x @ B) * A) per population member
3. Process data in minibatches to bound memory
4. Use deterministic perturbation generation for gradient consistency

Architecture (TRM-style recursive):
1. Input projection
2. Recursive z refinement (L_cycles of tiny network)
3. y update after each H_cycle
4. Output head

References:
- TRM: "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)
- EGGROLL: "Evolution Strategies at the Hyperscale" (arXiv:2511.16652)
"""

from .core import (
    init_trm_params,
    trm_forward,
    get_param_shapes,
    count_parameters,
)

from .batched_forward import (
    forward_population_batched,
    perturbed_linear_population,
)

from .perturbation import (
    generate_perturbations,
    hash_combine,
)

__all__ = [
    "init_trm_params",
    "trm_forward",
    "get_param_shapes",
    "count_parameters",
    "forward_population_batched",
    "perturbed_linear_population",
    "generate_perturbations",
    "hash_combine",
]
