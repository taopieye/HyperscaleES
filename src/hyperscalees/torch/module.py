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


class LowRankLinear(nn.Module):
    """
    Linear layer optimized for low-rank perturbations.
    
    This is a drop-in replacement for nn.Linear that stores the weight
    in a factored form suitable for efficient EGGROLL perturbations.
    
    The weight W is stored as U @ V.T where U ∈ R^{out×r}, V ∈ R^{in×r}.
    This allows perturbations to be added directly to the factors.
    
    Note: For standard ES use, regular nn.Linear works fine. This layer
    is for advanced use cases where you want persistent low-rank structure.
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        rank: Rank of the weight matrix factorization
        bias: If True, adds a learnable bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: Optional[int] = None,
        bias: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Default rank is min(in_features, out_features)
        if rank is None:
            self.rank = min(in_features, out_features)
        else:
            self.rank = min(rank, in_features, out_features)
        
        # Factored weight: W = U @ V.T
        self.U = nn.Parameter(torch.empty(out_features, self.rank))
        self.V = nn.Parameter(torch.empty(in_features, self.rank))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters using scaled initialization."""
        # Initialize U and V such that U @ V.T has appropriate scale
        # Using He initialization scaled by sqrt(2 / fan_in)
        std = (2.0 / self.in_features) ** 0.5 / (self.rank ** 0.5)
        
        nn.init.normal_(self.U, std=std)
        nn.init.normal_(self.V, std=std)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    @property
    def weight(self) -> torch.Tensor:
        """
        Reconstruct full weight matrix.
        
        Note: This materializes the full matrix. For efficient computation,
        use forward() which keeps the factored form.
        """
        return self.U @ self.V.T
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using factored computation.
        
        Computes x @ V @ U.T instead of x @ W.T for efficiency.
        """
        # x @ (U @ V.T).T = x @ V @ U.T
        out = x @ self.V @ self.U.T
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, bias={self.bias is not None}'
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: Optional[int] = None) -> 'LowRankLinear':
        """
        Create LowRankLinear from existing nn.Linear using SVD.
        
        Args:
            linear: Source nn.Linear layer
            rank: Target rank (defaults to full rank)
        
        Returns:
            LowRankLinear with factored weights
        """
        W = linear.weight.data  # (out, in)
        
        if rank is None:
            rank = min(W.shape)
        else:
            rank = min(rank, W.shape[0], W.shape[1])
        
        # SVD decomposition
        U_full, S, Vh_full = torch.linalg.svd(W, full_matrices=False)
        
        # Truncate to desired rank
        U = U_full[:, :rank]  # (out, rank)
        S = S[:rank]
        Vh = Vh_full[:rank, :]  # (rank, in)
        
        # Distribute singular values
        sqrt_S = torch.sqrt(S)
        U_scaled = U * sqrt_S.unsqueeze(0)  # (out, rank)
        V_scaled = (Vh.T * sqrt_S.unsqueeze(0))  # (in, rank)
        
        # Create layer
        layer = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            bias=linear.bias is not None
        )
        
        # Set weights
        layer.U.data = U_scaled
        layer.V.data = V_scaled
        
        if linear.bias is not None:
            layer.bias.data = linear.bias.data.clone()
        
        return layer
