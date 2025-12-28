"""
PyTorch EGGROLL Implementation.

This module provides a PyTorch port of the EGGROLL evolution strategy with
GPU-accelerated batched perturbations using Triton kernels.

The key insight: generate perturbations on-the-fly using parallel PRNG (Philox)
rather than pre-materializing noise tensors. This enables O(rank * (m+n)) memory
instead of O(m * n) per population member.

Usage:
    from hyperscalees.torch import EggrollStrategy
    
    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2)).cuda()
    strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=4)
    strategy.setup(model)
    
    for epoch in range(100):
        x = torch.randn(64, 8, device='cuda')
        with strategy.perturb(population_size=64, epoch=epoch) as pop:
            outputs = pop.batched_forward(model, x)
            fitnesses = -((outputs - targets) ** 2).mean(dim=-1)
        strategy.step(fitnesses)
"""

from .strategy import EggrollStrategy
from .perturbation import PerturbationContext, Perturbation
from .module import ESModule

__all__ = [
    "EggrollStrategy",
    "PerturbationContext", 
    "Perturbation",
    "ESModule",
]
