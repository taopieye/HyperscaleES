"""
Memory-Efficient Batched Forward for EGGROLL TRM.

Key insight: We can evaluate ALL population members efficiently by:
1. Computing base forward once (shared across population)
2. Adding per-member perturbation terms efficiently
3. Using einsum for batched outer products

This avoids replicating the model or data per population member.

Memory complexity:
- Naive: O(pop_size × batch × hidden × L_cycles × H_cycles)
- Ours: O(pop_size × batch × hidden) + O(batch × hidden) for base
"""

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

from .perturbation import generate_layer_perturbations


def perturbed_linear_population(
    x: Tensor,               # (pop, batch, in_dim)
    weight: Tensor,          # (out_dim, in_dim) - SHARED
    bias: Optional[Tensor],  # (out_dim,) - SHARED
    A: Tensor,               # (pop, out_dim)
    B: Tensor,               # (pop, in_dim)
    sigma: float,
) -> Tensor:                 # (pop, batch, out_dim)
    """
    Batched perturbed linear for entire population.
    
    Computes: y_p = x_p @ W.T + b + σ * (x_p @ B_p) * A_p
    
    Key optimization: The base term (x @ W.T + b) can use standard
    F.linear which broadcasts efficiently. Only the perturbation
    term needs per-population computation.
    
    Args:
        x: Input (pop, batch, in_dim)
        weight: Shared weight matrix (out_dim, in_dim)
        bias: Shared bias (out_dim,) or None
        A: Per-population perturbation (pop, out_dim)
        B: Per-population perturbation (pop, in_dim)
        sigma: Perturbation scale
        
    Returns:
        Output (pop, batch, out_dim)
    """
    # Base linear - F.linear broadcasts weight across leading dims
    base = F.linear(x, weight, bias)  # (pop, batch, out_dim)
    
    if sigma == 0.0:
        return base
    
    # Perturbation: (x @ B) * A for each population member
    # x: (pop, batch, in), B: (pop, in) -> xB: (pop, batch)
    xB = torch.einsum('pbi,pi->pb', x, B)  # (pop, batch)
    
    # xB: (pop, batch), A: (pop, out) -> pert: (pop, batch, out)
    perturbation = xB.unsqueeze(-1) * A.unsqueeze(1)  # (pop, batch, out)
    
    return base + sigma * perturbation


def layer_norm_batched(
    x: Tensor,      # (pop, batch, dim)
    weight: Tensor, # (dim,)
    bias: Tensor,   # (dim,)
    eps: float = 1e-5,
) -> Tensor:
    """Functional layer norm for batched input."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * weight + bias


def forward_population_batched(
    params: Dict[str, Tensor],
    x: Tensor,                  # (batch, input_dim)
    pop_size: int,
    epoch: int,
    sigma: float,
    L_cycles: int = 6,
    H_cycles: int = 3,
    base_seed: int = 42,
) -> Tensor:                    # (pop, batch, output_dim)
    """
    Forward pass for entire population with memory-efficient batching.
    
    Memory optimization strategy:
    1. Expand input to (pop, batch, dim) only at the start
    2. Reuse weight matrices (shared across population)
    3. Generate perturbations on-the-fly for each layer
    4. Intermediate activations: (pop, batch, hidden_dim) - manageable
    
    For pop=16K, batch=64, hidden=64:
    - Intermediate activations: 16K × 64 × 64 × 4 bytes = 256 MB
    - This is feasible for modern GPUs!
    
    Args:
        params: Model parameters (on CUDA)
        x: Input data (batch, input_dim) - will be expanded
        pop_size: Number of population members
        epoch: Current epoch for perturbation generation
        sigma: Perturbation scale
        L_cycles: Latent refinement steps per supervision
        H_cycles: Supervision steps
        base_seed: Random seed
        
    Returns:
        Outputs for all population members (pop, batch, output_dim)
    """
    device = x.device
    dtype = x.dtype
    batch = x.shape[0]
    hidden_dim = params['input_proj.weight'].shape[0]
    
    # Thread IDs for perturbation generation
    thread_ids = torch.arange(pop_size, device=device)
    
    def get_perts(name: str) -> Tuple[Tensor, Tensor]:
        shape = (params[name].shape[0], params[name].shape[1])
        return generate_layer_perturbations(
            name, shape, epoch, thread_ids, base_seed, device, dtype
        )
    
    # === 1. Input projection ===
    # x: (batch, input_dim) -> (pop, batch, hidden_dim)
    
    # First, expand x to (pop, batch, input_dim)
    x_pop = x.unsqueeze(0).expand(pop_size, -1, -1)  # (pop, batch, input_dim)
    
    # Perturbed input projection
    A_inp, B_inp = get_perts('input_proj.weight')
    z = perturbed_linear_population(
        x_pop,
        params['input_proj.weight'],
        params['input_proj.bias'],
        A_inp, B_inp,
        sigma,
    )  # (pop, batch, hidden_dim)
    
    del x_pop  # Free expanded input
    
    # === 2. Initialize y ===
    # y_init: (1, hidden_dim) -> (pop, batch, hidden_dim)
    y = params['y_init'].unsqueeze(0).expand(pop_size, batch, -1).clone()
    
    # === 3. TRM Recursion ===
    for h_cycle in range(H_cycles):
        # Refine z L_cycles times through tiny network
        for l_cycle in range(L_cycles):
            # Tiny network: LayerNorm(z + FFN(z))
            
            # FFN Up: hidden -> ffn
            A_up, B_up = get_perts('tiny.up.weight')
            h = perturbed_linear_population(
                z, params['tiny.up.weight'], params['tiny.up.bias'],
                A_up, B_up, sigma,
            )
            h = F.gelu(h)
            
            # FFN Down: ffn -> hidden
            A_down, B_down = get_perts('tiny.down.weight')
            h = perturbed_linear_population(
                h, params['tiny.down.weight'], params['tiny.down.bias'],
                A_down, B_down, sigma,
            )
            
            # Residual + LayerNorm
            z = layer_norm_batched(
                z + h,
                params['tiny.norm.weight'],
                params['tiny.norm.bias'],
            )
        
        # Update y: y = y + Transform(GELU(Combine([z, y])))
        combined = torch.cat([z, y], dim=-1)  # (pop, batch, hidden * 2)
        
        A_comb, B_comb = get_perts('y_update.combine.weight')
        h = perturbed_linear_population(
            combined, params['y_update.combine.weight'], params['y_update.combine.bias'],
            A_comb, B_comb, sigma,
        )
        h = F.gelu(h)
        
        A_trans, B_trans = get_perts('y_update.transform.weight')
        h = perturbed_linear_population(
            h, params['y_update.transform.weight'], params['y_update.transform.bias'],
            A_trans, B_trans, sigma,
        )
        
        y = y + h
    
    # === 4. Output head ===
    A_out, B_out = get_perts('output.weight')
    output = perturbed_linear_population(
        y, params['output.weight'], params['output.bias'],
        A_out, B_out, sigma,
    )  # (pop, batch, output_dim)
    
    return output


def forward_population_memory_efficient(
    params: Dict[str, Tensor],
    x: Tensor,                  # (batch, input_dim)
    pop_size: int,
    epoch: int,
    sigma: float,
    L_cycles: int = 6,
    H_cycles: int = 3,
    base_seed: int = 42,
    pop_batch_size: int = 1024,  # Process population in chunks
) -> Tensor:                    # (pop, batch, output_dim)
    """
    Ultra memory-efficient forward by chunking population dimension.
    
    When pop_size is very large, we process population in chunks to
    avoid OOM while still leveraging batched computation within each chunk.
    
    Args:
        ... same as forward_population_batched ...
        pop_batch_size: Number of population members per chunk
        
    Returns:
        Outputs for all population members (pop, batch, output_dim)
    """
    if pop_size <= pop_batch_size:
        return forward_population_batched(
            params, x, pop_size, epoch, sigma, L_cycles, H_cycles, base_seed
        )
    
    # Process in chunks
    outputs = []
    for start in range(0, pop_size, pop_batch_size):
        end = min(start + pop_batch_size, pop_size)
        chunk_size = end - start
        
        # Adjust base_seed to maintain deterministic perturbations
        chunk_output = forward_population_batched(
            params, x, chunk_size, epoch, sigma, L_cycles, H_cycles,
            base_seed=base_seed + start,  # Offset seed for different members
        )
        outputs.append(chunk_output)
        
        # Free memory
        torch.cuda.empty_cache()
    
    return torch.cat(outputs, dim=0)
