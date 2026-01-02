"""
TRM Core - Functional implementation of Tiny Recursive Model.

This module implements TRM as a pure functional model (no nn.Module)
for vmap compatibility with EGGROLL batched perturbations.

Key differences from standard PyTorch TRM:
1. Parameters stored as flat dict, not in nn.Module
2. Forward pass takes params explicitly
3. Designed for batched perturbation injection
"""

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


def init_trm_params(
    input_dim: int = 64,
    hidden_dim: int = 64,
    output_dim: int = 5,
    expansion: int = 4,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Tensor]:
    """
    Initialize TRM parameters as a flat dict.
    
    Architecture:
    1. Input projection: input_dim -> hidden_dim
    2. Tiny network (shared, applied recursively):
       - Up projection: hidden_dim -> hidden_dim * expansion
       - Down projection: hidden_dim * expansion -> hidden_dim
    3. y update network:
       - Combine: hidden_dim * 2 -> hidden_dim
       - Transform: hidden_dim -> hidden_dim
    4. Output head: hidden_dim -> output_dim
    
    Following TRM paper: 2-layer tiny network is optimal.
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Hidden dimension throughout
        output_dim: Output dimension
        expansion: FFN expansion factor
        device: Device for tensors
        dtype: Data type
        
    Returns:
        Dict mapping parameter names to tensors
    """
    params = {}
    ffn_dim = hidden_dim * expansion
    
    def xavier_init(shape: Tuple[int, ...]) -> Tensor:
        fan_in = shape[-1] if len(shape) >= 2 else shape[0]
        fan_out = shape[-2] if len(shape) >= 2 else shape[0]
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return torch.randn(*shape, device=device, dtype=dtype) * std
    
    def zeros(shape: Tuple[int, ...]) -> Tensor:
        return torch.zeros(*shape, device=device, dtype=dtype)
    
    # 1. Input projection
    params['input_proj.weight'] = xavier_init((hidden_dim, input_dim))
    params['input_proj.bias'] = zeros((hidden_dim,))
    
    # 2. Tiny network (the recursive block)
    # Up: hidden_dim -> ffn_dim
    params['tiny.up.weight'] = xavier_init((ffn_dim, hidden_dim))
    params['tiny.up.bias'] = zeros((ffn_dim,))
    
    # Down: ffn_dim -> hidden_dim
    params['tiny.down.weight'] = xavier_init((hidden_dim, ffn_dim))
    params['tiny.down.bias'] = zeros((hidden_dim,))
    
    # LayerNorm (as scale and bias)
    params['tiny.norm.weight'] = torch.ones(hidden_dim, device=device, dtype=dtype)
    params['tiny.norm.bias'] = zeros((hidden_dim,))
    
    # 3. y update network
    # Combine z and y: hidden_dim * 2 -> hidden_dim
    params['y_update.combine.weight'] = xavier_init((hidden_dim, hidden_dim * 2))
    params['y_update.combine.bias'] = zeros((hidden_dim,))
    
    # Transform: hidden_dim -> hidden_dim
    params['y_update.transform.weight'] = xavier_init((hidden_dim, hidden_dim))
    params['y_update.transform.bias'] = zeros((hidden_dim,))
    
    # 4. Learnable initial y
    params['y_init'] = torch.randn(1, hidden_dim, device=device, dtype=dtype) * 0.02
    
    # 5. Output head
    params['output.weight'] = xavier_init((output_dim, hidden_dim))
    params['output.bias'] = zeros((output_dim,))
    
    return params


def get_param_shapes(params: Dict[str, Tensor]) -> Dict[str, Tuple[int, int]]:
    """
    Get shapes of 2D weight parameters for perturbation generation.
    
    Returns:
        Dict mapping weight names to (out_features, in_features)
    """
    shapes = {}
    for name, tensor in params.items():
        if tensor.dim() == 2 and 'weight' in name:
            shapes[name] = (tensor.shape[0], tensor.shape[1])
    return shapes


def count_parameters(params: Dict[str, Tensor]) -> int:
    """Count total number of parameters."""
    return sum(p.numel() for p in params.values())


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    """Functional layer norm."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * weight + bias


def tiny_network_forward(
    z: Tensor,  # (batch, hidden_dim)
    params: Dict[str, Tensor],
) -> Tensor:
    """
    Forward through the tiny network (one recursive step).
    
    Architecture: LayerNorm(z + FFN(z))
    Where FFN is: Down(GELU(Up(z)))
    
    This is the network that gets applied L_cycles times.
    """
    # FFN: Up -> GELU -> Down
    h = F.linear(z, params['tiny.up.weight'], params['tiny.up.bias'])
    h = F.gelu(h)
    h = F.linear(h, params['tiny.down.weight'], params['tiny.down.bias'])
    
    # Residual + LayerNorm
    z = layer_norm(
        z + h,
        params['tiny.norm.weight'],
        params['tiny.norm.bias'],
    )
    
    return z


def y_update_forward(
    z: Tensor,  # (batch, hidden_dim)
    y: Tensor,  # (batch, hidden_dim)
    params: Dict[str, Tensor],
) -> Tensor:
    """
    Update y given refined z.
    
    Architecture: y = y + Transform(GELU(Combine([z, y])))
    """
    # Combine z and y
    combined = torch.cat([z, y], dim=-1)  # (batch, hidden_dim * 2)
    h = F.linear(combined, params['y_update.combine.weight'], params['y_update.combine.bias'])
    h = F.gelu(h)
    h = F.linear(h, params['y_update.transform.weight'], params['y_update.transform.bias'])
    
    # Residual update
    return y + h


def trm_forward(
    params: Dict[str, Tensor],
    x: Tensor,  # (batch, input_dim)
    L_cycles: int = 6,  # Latent refinement steps
    H_cycles: int = 3,  # Supervision steps
) -> Tensor:  # (batch, output_dim)
    """
    TRM forward pass (non-batched, for reference).
    
    Algorithm:
        z = input_proj(x)
        y = y_init
        for h in range(H_cycles):
            for l in range(L_cycles):
                z = tiny_network(z)
            y = y_update(z, y)
        return output_head(y)
    
    Args:
        params: Parameter dict from init_trm_params
        x: Input tensor (batch, input_dim)
        L_cycles: Number of latent refinement steps per supervision
        H_cycles: Number of supervision steps
        
    Returns:
        Output tensor (batch, output_dim)
    """
    batch_size = x.shape[0]
    
    # Input projection
    z = F.linear(x, params['input_proj.weight'], params['input_proj.bias'])
    
    # Initialize y
    y = params['y_init'].expand(batch_size, -1)
    
    # TRM recursion
    for h in range(H_cycles):
        # Refine z L_cycles times
        for l in range(L_cycles):
            z = tiny_network_forward(z, params)
        
        # Update y
        y = y_update_forward(z, y, params)
    
    # Output
    output = F.linear(y, params['output.weight'], params['output.bias'])
    
    return output
