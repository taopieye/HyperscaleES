"""
Perturbation classes for EGGROLL.

This module provides:
- Perturbation: A single low-rank perturbation (A @ B.T)
- PerturbationContext: Context manager for batched perturbation evaluation
"""
import torch
import torch.nn as nn
from typing import Tuple, Iterator, Dict, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
from contextlib import contextmanager

if TYPE_CHECKING:
    from .strategy import EggrollStrategy

from .triton_kernels import (
    generate_lowrank_factors_torch,
    batched_perturbed_linear_torch,
)


@dataclass
class Perturbation:
    """
    A single low-rank perturbation for a parameter.
    
    The perturbation is stored in factored form: ΔW = A @ B.T
    This representation has O(r * (m + n)) memory instead of O(m * n).
    
    Attributes:
        A: Factor matrix (out_features, rank), scaled by sigma
        B: Factor matrix (in_features, rank)
        sigma: The noise scale used to generate A
    """
    A: torch.Tensor  # (out_features, rank)
    B: torch.Tensor  # (in_features, rank)
    sigma: float
    
    @property
    def factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the low-rank factors (A, B)."""
        return self.A, self.B
    
    def as_matrix(self) -> torch.Tensor:
        """
        Reconstruct the full perturbation matrix.
        
        WARNING: This materializes O(m * n) memory. Use only for testing/debugging.
        In production, use the factored form via batched_forward.
        """
        return self.A @ self.B.t()
    
    @property
    def rank(self) -> int:
        """Get the rank of this perturbation."""
        return self.A.shape[1]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the full perturbation matrix (without materializing it)."""
        return (self.A.shape[0], self.B.shape[0])


class PerturbationContext:
    """
    Context manager for applying perturbations during evaluation.
    
    This class is the main interface for EGGROLL's batched forward pass.
    It manages noise generation and provides efficient batched computation.
    
    Usage:
        with strategy.perturb(population_size=64, epoch=0) as pop:
            # Batched forward - ONE kernel evaluates ALL population members
            outputs = pop.batched_forward(model, x)
            
            # Or iterate for debugging (less efficient)
            for member_id in pop.iterate():
                output = model(x[member_id])
    """
    
    def __init__(
        self,
        strategy: "EggrollStrategy",
        population_size: int,
        epoch: int,
    ):
        """
        Initialize perturbation context.
        
        Args:
            strategy: The parent EggrollStrategy
            population_size: Number of population members to evaluate
            epoch: Current epoch (for noise scheduling)
        """
        self._strategy = strategy
        self._population_size = population_size
        self._epoch = epoch
        self._active = False
        self._cached_factors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._current_member_id: Optional[int] = None
        
        # Original forward hooks (to restore after context)
        self._original_forwards: Dict[str, Any] = {}
    
    @property
    def population_size(self) -> int:
        """Number of population members."""
        return self._population_size
    
    @property
    def epoch(self) -> int:
        """Current epoch."""
        return self._epoch
    
    def __enter__(self) -> "PerturbationContext":
        """Enter the perturbation context."""
        self._active = True
        self._generate_all_factors()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the perturbation context."""
        self._active = False
        self._cached_factors.clear()
        self._current_member_id = None
        self._restore_original_forwards()
        return False
    
    def _generate_all_factors(self) -> None:
        """
        Pre-generate all low-rank factors for all population members.
        
        This is done once when entering the context. The factors are stored
        in a compact form: (population_size, dim1, rank) and (population_size, dim2, rank).
        
        For true on-the-fly generation (Triton), this would instead store
        just the seeds and generate during batched_forward.
        """
        if self._strategy.model is None:
            raise RuntimeError("Strategy not set up. Call strategy.setup(model) first.")
        
        model = self._strategy.model
        device = next(model.parameters()).device
        
        for param_id, (name, param) in enumerate(self._strategy._es_params.items()):
            if param.ndim != 2:
                # Only apply low-rank to 2D params (weights)
                continue
            
            out_features, in_features = param.shape
            member_ids = torch.arange(self._population_size, device=device)
            
            A, B = generate_lowrank_factors_torch(
                out_features=out_features,
                in_features=in_features,
                rank=self._strategy.rank,
                seed=self._strategy._seed,
                epoch=self._epoch,
                member_ids=member_ids,
                param_id=param_id,
                sigma=self._strategy.sigma,
                noise_reuse=self._strategy.noise_reuse,
                device=device,
                dtype=param.dtype,
            )
            
            self._cached_factors[name] = (A, B)
    
    def _restore_original_forwards(self) -> None:
        """Restore original forward methods after iteration."""
        for name, forward in self._original_forwards.items():
            module = dict(self._strategy.model.named_modules())[name]
            module.forward = forward
        self._original_forwards.clear()
    
    def get_factors(
        self, 
        member_id: int, 
        param_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get low-rank factors (A, B) for a specific member and parameter.
        
        Args:
            member_id: Population member index
            param_name: Name of the parameter (e.g., "0.weight")
            
        Returns:
            A: (out_features, rank) factor matrix
            B: (in_features, rank) factor matrix
        """
        if not self._active:
            raise RuntimeError("get_factors() must be called within perturb() context")
        
        if param_name not in self._cached_factors:
            raise KeyError(f"Parameter '{param_name}' not found. Available: {list(self._cached_factors.keys())}")
        
        A_all, B_all = self._cached_factors[param_name]
        return A_all[member_id], B_all[member_id]
    
    def batched_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        member_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batched forward pass with per-sample perturbations.
        
        This is the main API for efficient evaluation. It computes the output
        for all population members in a single batched operation.
        
        Args:
            model: The neural network
            x: Input tensor (batch_size, *input_dims)
            member_ids: Which population member for each sample.
                       Shape (batch_size,). Defaults to [0, 1, 2, ..., batch_size-1].
                       
        Returns:
            Output tensor (batch_size, *output_dims)
            
        The computation for each linear layer:
            output[i] = x[i] @ W.T + x[i] @ B[member_ids[i]] @ A[member_ids[i]].T + bias
        """
        if not self._active:
            raise RuntimeError("batched_forward() must be called within perturb() context")
        
        # Validate inputs
        device = x.device
        if not device.type == 'cuda':
            raise RuntimeError(
                f"Input tensor must be on CUDA device, got {device}. "
                "EGGROLL requires GPU for batched perturbations."
            )
        
        # Check model is on GPU
        model_device = next(model.parameters()).device
        if not model_device.type == 'cuda':
            raise RuntimeError(
                f"Model must be on CUDA device, got {model_device}. "
                "Move your model to GPU with model.cuda() before calling batched_forward."
            )
        
        # Default member_ids: one-to-one mapping
        if member_ids is None:
            member_ids = torch.arange(x.shape[0], device=device)
        
        # Validate member_ids
        if member_ids.max() >= self._population_size:
            raise ValueError(
                f"member_ids contains index {member_ids.max().item()} but "
                f"population_size is {self._population_size}"
            )
        
        # Apply perturbed forward pass through the model
        return self._apply_perturbed_forward(model, x, member_ids)
    
    def _apply_perturbed_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        member_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the perturbed forward pass through the model.
        
        This walks through the model's modules and applies low-rank perturbations
        to each Linear layer.
        """
        # Handle Sequential models
        if isinstance(model, nn.Sequential):
            out = x
            for idx, module in enumerate(model):
                if isinstance(module, nn.Linear):
                    param_name = f"{idx}.weight"
                    out = self._perturbed_linear(module, out, member_ids, param_name)
                else:
                    out = module(out)
            return out
        
        # Handle single Linear layer
        if isinstance(model, nn.Linear):
            return self._perturbed_linear(model, x, member_ids, "weight")
        
        # For more complex models, we need to hook into each Linear layer
        # This is a simplified version - full implementation would use hooks
        raise NotImplementedError(
            f"batched_forward not yet implemented for model type {type(model)}. "
            "Currently supports nn.Sequential and nn.Linear."
        )
    
    def _perturbed_linear(
        self,
        linear: nn.Linear,
        x: torch.Tensor,
        member_ids: torch.Tensor,
        param_name: str,
    ) -> torch.Tensor:
        """
        Apply a single perturbed linear layer.
        
        Computes: out[i] = x[i] @ W.T + x[i] @ B[member_ids[i]] @ A[member_ids[i]].T + bias
        """
        if param_name not in self._cached_factors:
            # This parameter isn't being perturbed (e.g., bias)
            return linear(x)
        
        A_all, B_all = self._cached_factors[param_name]
        
        return batched_perturbed_linear_torch(
            x=x,
            weight=linear.weight,
            bias=linear.bias,
            A=A_all,
            B=B_all,
            member_ids=member_ids,
        )
    
    def iterate(self) -> Iterator[int]:
        """
        Iterate through population members for sequential evaluation.
        
        This is less efficient than batched_forward but useful for:
        - Debugging
        - Environments that can't be batched
        - When you need fine-grained control
        
        Usage:
            with strategy.perturb(population_size=64, epoch=0) as pop:
                for member_id in pop.iterate():
                    # Model now uses perturbation for this member
                    output = model(x)
        
        Yields:
            member_id: Index of the current population member
        """
        if not self._active:
            raise RuntimeError("iterate() must be called within perturb() context")
        
        for member_id in range(self._population_size):
            self._current_member_id = member_id
            self._install_perturbed_forwards(member_id)
            yield member_id
        
        self._current_member_id = None
        self._restore_original_forwards()
    
    def _install_perturbed_forwards(self, member_id: int) -> None:
        """
        Install perturbed forward methods for sequential evaluation.
        
        This temporarily replaces each Linear layer's forward method to apply
        the perturbation for the given member_id.
        """
        model = self._strategy.model
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                param_name = f"{name}.weight" if name else "weight"
                
                # Skip if no perturbation for this param
                if param_name not in self._cached_factors:
                    continue
                
                # Save original forward
                if name not in self._original_forwards:
                    self._original_forwards[name] = module.forward
                
                # Get factors for this member
                A, B = self.get_factors(member_id, param_name)
                
                # Create perturbed forward
                def make_perturbed_forward(orig_module, A_factor, B_factor):
                    def perturbed_forward(x):
                        # Compute: x @ W.T + x @ B @ A.T + bias
                        base = x @ orig_module.weight.t()
                        pert = x @ B_factor @ A_factor.t()
                        out = base + pert
                        if orig_module.bias is not None:
                            out = out + orig_module.bias
                        return out
                    return perturbed_forward
                
                module.forward = make_perturbed_forward(module, A, B)
