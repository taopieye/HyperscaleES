"""
ESModule: Wrapper for ES-compatible models.

This module provides a lightweight wrapper that makes any nn.Module
compatible with EGGROLL's evolution strategy.
"""
import torch
import torch.nn as nn
from typing import Iterator, Set, Optional


class ESModule(nn.Module):
    """
    Wrapper for ES-compatible models.
    
    ESModule wraps any nn.Module and provides:
    - Identification of which parameters should be evolved
    - Ability to freeze specific parameters from evolution
    - Clean interface for ES strategy integration
    
    Usage:
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        es_model = ESModule(model)
        
        # Optionally freeze some parameters
        es_model.freeze_parameter("0.bias")
        
        strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=4)
        strategy.setup(es_model)
    """
    
    def __init__(self, module: nn.Module):
        """
        Wrap a module for ES optimization.
        
        Args:
            module: The neural network to wrap
        """
        super().__init__()
        self.module = module
        self._frozen_params: Set[str] = set()
    
    def forward(self, *args, **kwargs):
        """Forward pass delegates to wrapped module."""
        return self.module(*args, **kwargs)
    
    @property
    def es_parameters(self) -> Iterator[nn.Parameter]:
        """
        Iterate over parameters that will be evolved.
        
        This excludes frozen parameters.
        
        Yields:
            Parameters that should be evolved
        """
        for name, param in self.module.named_parameters():
            if name not in self._frozen_params and param.requires_grad:
                yield param
    
    @property
    def es_named_parameters(self) -> Iterator[tuple]:
        """
        Iterate over (name, parameter) pairs for evolved parameters.
        
        Yields:
            (name, parameter) tuples for parameters that should be evolved
        """
        for name, param in self.module.named_parameters():
            if name not in self._frozen_params and param.requires_grad:
                yield name, param
    
    def freeze_parameter(self, name: str) -> None:
        """
        Exclude a parameter from evolution.
        
        The parameter will still be used in forward passes, but it won't
        receive perturbations or ES updates.
        
        Args:
            name: Name of the parameter to freeze (e.g., "0.bias", "layer1.weight")
            
        Raises:
            KeyError: If parameter name not found
        """
        # Validate the parameter exists
        found = False
        for param_name, _ in self.module.named_parameters():
            if param_name == name:
                found = True
                break
        
        if not found:
            available = [n for n, _ in self.module.named_parameters()]
            raise KeyError(
                f"Parameter '{name}' not found. Available parameters: {available}"
            )
        
        self._frozen_params.add(name)
    
    def unfreeze_parameter(self, name: str) -> None:
        """
        Re-include a parameter in evolution.
        
        Args:
            name: Name of the parameter to unfreeze
        """
        self._frozen_params.discard(name)
    
    def is_frozen(self, name: str) -> bool:
        """Check if a parameter is frozen."""
        return name in self._frozen_params
    
    @property
    def frozen_parameters(self) -> Set[str]:
        """Get the set of frozen parameter names."""
        return self._frozen_params.copy()
    
    def num_es_parameters(self) -> int:
        """Count total number of parameters being evolved."""
        return sum(p.numel() for p in self.es_parameters)
    
    def named_modules(self, *args, **kwargs):
        """Delegate to wrapped module."""
        return self.module.named_modules(*args, **kwargs)
    
    def named_parameters(self, *args, **kwargs):
        """Delegate to wrapped module."""
        return self.module.named_parameters(*args, **kwargs)
    
    def parameters(self, *args, **kwargs):
        """Delegate to wrapped module."""
        return self.module.parameters(*args, **kwargs)
    
    def to(self, *args, **kwargs):
        """Move module to device/dtype."""
        self.module = self.module.to(*args, **kwargs)
        return self
    
    def cuda(self, *args, **kwargs):
        """Move module to CUDA."""
        self.module = self.module.cuda(*args, **kwargs)
        return self
    
    def cpu(self):
        """Move module to CPU."""
        self.module = self.module.cpu()
        return self
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.module.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.module.eval()
        return self
