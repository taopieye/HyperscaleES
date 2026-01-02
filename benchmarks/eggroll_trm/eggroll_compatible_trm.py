"""
EGGROLL-Compatible TRM (Tiny Recursive Model).

Key design principle: NO ACTIVATION FUNCTIONS.
Activations break the low-rank structure of perturbations (EGGROLL paper Section 6.3).
"""

import torch
import torch.nn as nn
from torch import Tensor


class EGGTinyBlock(nn.Module):
    """2-layer linear FFN with no activations: h = x + down(up(x))"""
    
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.up = nn.Linear(dim, dim * expansion)
        self.down = nn.Linear(dim * expansion, dim)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        
    def forward(self, x: Tensor) -> Tensor:
        return x + self.down(self.up(x))


class EGGTRM(nn.Module):
    """
    EGGROLL-compatible TRM. No activations, pure linear.
    
    Architecture:
        z = input_proj(x)
        for H_cycles:
            for L_cycles: z = tiny_block(z)
            y = y + transform(concat(z, y))
        return head(y)
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 5,
        L_cycles: int = 6,
        H_cycles: int = 3,
        expansion: int = 4,
    ):
        super().__init__()
        self.L_cycles = L_cycles
        self.H_cycles = H_cycles
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.y_init = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.tiny_block = EGGTinyBlock(hidden_dim, expansion)
        self.y_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.y_transform = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)
        
        nn.init.normal_(self.y_transform.weight, std=0.02)
        nn.init.zeros_(self.y_transform.bias)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        z = self.input_proj(x)
        y = self.y_init.expand(batch_size, -1)
        
        for _ in range(self.H_cycles):
            for _ in range(self.L_cycles):
                z = self.tiny_block(z)
            combined = torch.cat([z, y], dim=-1)
            y = y + self.y_transform(self.y_combine(combined))
        
        return self.head(y)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
