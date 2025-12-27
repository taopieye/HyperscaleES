"""
PyTorch EGGROLL - Low-rank Evolution Strategy for GPU-accelerated optimization.

This module provides a PyTorch implementation of EGGROLL (Efficient Gradient-free 
Gaussian Rank-One Learning for Low-rank) evolution strategy, designed for 
GPU-accelerated batched perturbations.

Main classes:
- EggrollStrategy: Low-rank evolution strategy with efficient batched perturbations
- OpenESStrategy: Standard OpenAI Evolution Strategy (full-rank perturbations)
- ESModule: Wrapper for ES-compatible models
- LowRankLinear: Linear layer optimized for low-rank perturbations

Example usage:
    from hyperscalees.torch import EggrollStrategy
    
    model = nn.Sequential(
        nn.Linear(8, 32), nn.ReLU(),
        nn.Linear(32, 2)
    ).cuda()
    
    strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=4)
    strategy.setup(model)
    
    for epoch in range(100):
        with strategy.perturb(population_size=64, epoch=epoch) as pop:
            outputs = pop.batched_forward(model, x_batch)
            fitnesses = compute_fitness(outputs)
        
        strategy.step(fitnesses)
"""

from .strategy import EggrollStrategy, OpenESStrategy, BaseStrategy
from .perturbation import Perturbation, PerturbationContext
from .module import ESModule, LowRankLinear

__all__ = [
    'EggrollStrategy',
    'OpenESStrategy', 
    'BaseStrategy',
    'Perturbation',
    'PerturbationContext',
    'ESModule',
    'LowRankLinear',
]
