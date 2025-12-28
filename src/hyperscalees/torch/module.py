"""
ES-compatible module wrappers and specialized layers.

Contains:
- ESModule: Wrapper for ES-compatible models
- LowRankLinear: Linear layer optimized for low-rank perturbations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator, Optional, Dict, Any, Set


class ESModule(nn.Module):
    """
    Wrapper for ES-compatible models.
    
    This wrapper provides additional functionality for evolution strategies:
    - Parameter discovery and categorization
    - Freezing specific parameters from evolution
    - ES-specific parameter access
    
    Usage:
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        es_model = ESModule(model)
        
        # Freeze specific parameters
        es_model.freeze_parameter("0.weight")
        
        # Access only ES parameters
        for p in es_model.es_parameters():
            print(p.shape)
    """
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._frozen_params: Set[str] = set()
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to wrapped module."""
        return self.module(*args, **kwargs)
    
    @property
    def es_parameters(self) -> Iterator[nn.Parameter]:
        """
        Iterate over parameters that will be evolved.
        
        Excludes frozen parameters.
        """
        for name, param in self.module.named_parameters():
            if name not in self._frozen_params and param.requires_grad:
                yield param
    
    def freeze_parameter(self, name: str) -> None:
        """
        Exclude a parameter from evolution.
        
        Args:
            name: Full parameter name (e.g., "0.weight", "fc1.bias")
        """
        # Verify parameter exists
        found = False
        for n, _ in self.module.named_parameters():
            if n == name:
                found = True
                break
        
        if not found:
            raise ValueError(f"Parameter '{name}' not found in module")
        
        self._frozen_params.add(name)
    
    def unfreeze_parameter(self, name: str) -> None:
        """
        Include a previously frozen parameter in evolution.
        
        Args:
            name: Full parameter name
        """
        self._frozen_params.discard(name)
    
    def parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all parameters.
        
        Returns:
            Dict mapping parameter names to info dicts with:
                - type: "weight_matrix", "bias", or "other"
                - shape: Parameter shape
                - numel: Number of elements
                - frozen: Whether parameter is frozen from evolution
        """
        info = {}
        
        for name, param in self.module.named_parameters():
            param_type = "other"
            
            if "weight" in name:
                if param.dim() >= 2:
                    param_type = "weight_matrix"
                else:
                    param_type = "weight_vector"
            elif "bias" in name:
                param_type = "bias"
            elif param.dim() >= 2:
                param_type = "weight_matrix"
            
            info[name] = {
                "type": param_type,
                "shape": tuple(param.shape),
                "numel": param.numel(),
                "frozen": name in self._frozen_params,
                "requires_grad": param.requires_grad,
            }
        
        return info
    
    def evolved_parameters(self) -> Iterator[nn.Parameter]:
        """Alias for es_parameters for backward compatibility."""
        return self.es_parameters
    
    def weight_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over weight matrix parameters only."""
        info = self.parameter_info()
        for name, param in self.module.named_parameters():
            if info[name]["type"] == "weight_matrix" and name not in self._frozen_params:
                yield param
    
    def bias_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over bias parameters only."""
        info = self.parameter_info()
        for name, param in self.module.named_parameters():
            if info[name]["type"] == "bias" and name not in self._frozen_params:
                yield param
