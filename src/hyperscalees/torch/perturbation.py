"""
Perturbation classes for EGGROLL PyTorch implementation.

Contains:
- Perturbation: Represents a single low-rank perturbation (A @ B.T)
- PerturbationContext: Context manager for applying perturbations during forward pass
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional, Iterator, Any
from dataclasses import dataclass


@dataclass
class Perturbation:
    """
    Represents a low-rank perturbation as A @ B.T.
    
    The perturbation is stored in factored form to save memory.
    For a weight matrix W of shape (m, n), the perturbation is:
        ΔW = A @ B.T where A: (m, r), B: (n, r)
    
    This reduces storage from O(mn) to O(r(m+n)).
    
    Attributes:
        A: Left factor, shape (m, r) where m is output dimension
        B: Right factor, shape (n, r) where n is input dimension
        member_id: Population member index this perturbation belongs to
        epoch: Epoch when this perturbation was generated
        param_name: Name of the parameter this perturbation applies to
    """
    A: torch.Tensor
    B: torch.Tensor
    member_id: Optional[int] = None
    epoch: Optional[int] = None
    param_name: Optional[str] = None
    
    @property
    def factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the low-rank factors (A, B)."""
        return self.A, self.B
    
    @property
    def rank(self) -> int:
        """Get the rank of the perturbation (number of columns in A and B)."""
        if self.A.dim() < 2:
            return 1  # 1D tensors have rank 1
        return self.A.shape[-1]
    
    def as_matrix(self) -> torch.Tensor:
        """
        Reconstruct the full perturbation matrix A @ B.T.
        
        WARNING: This materializes the full matrix and should only be
        used for testing/debugging. In production, use the factored form.
        """
        # Handle 1D tensors (bias perturbations)
        if self.A.dim() == 1:
            # For 1D, the perturbation is A * B (broadcast)
            # Result should match A's shape
            result = self.A * self.B
            # Ensure result has the same shape as A (the primary shape)
            if result.shape != self.A.shape:
                result = result.expand_as(self.A)
            return result
        return self.A @ self.B.T
    
    def storage_stats(self) -> Dict[str, Any]:
        """
        Return storage statistics comparing low-rank vs full-rank.
        
        Returns:
            Dict with keys:
                - full_rank_elements: Elements needed for full matrix
                - low_rank_elements: Elements in factored form
                - savings_ratio: How much smaller factored form is
        """
        # Handle 1D tensors
        if self.A.dim() == 1:
            m = self.A.shape[0]
            n = self.B.shape[0] if self.B.dim() >= 1 else 1
            r = 1
        else:
            m, r = self.A.shape
            n = self.B.shape[0]
        
        full_rank_elements = m * n
        low_rank_elements = r * (m + n)
        
        return {
            'full_rank_elements': full_rank_elements,
            'low_rank_elements': low_rank_elements,
            'savings_ratio': full_rank_elements / low_rank_elements if low_rank_elements > 0 else float('inf')
        }


class PerturbationContext:
    """
    Context manager for applying perturbations during evaluation.
    
    This is the main interface for perturbed forward passes. It supports
    both batched evaluation (efficient) and sequential iteration (for debugging).
    
    Usage:
        with strategy.perturb(population_size=64, epoch=0) as pop:
            # Batched evaluation (recommended)
            outputs = pop.batched_forward(model, x_batch)
            
            # Or sequential (for debugging)
            for member_id in pop.iterate():
                output = model(x)  # Uses member_id's perturbation
    """
    
    def __init__(
        self,
        strategy: 'BaseStrategy',
        population_size: int,
        epoch: int,
    ):
        """
        Initialize perturbation context.
        
        Args:
            strategy: The ES strategy managing perturbations
            population_size: Number of population members
            epoch: Current epoch (used for deterministic noise)
        """
        self._strategy = strategy
        self._population_size = population_size
        self._epoch = epoch
        self._active_member_id: Optional[int] = None
        self._original_params: Dict[str, torch.Tensor] = {}
        self._perturbations_cache: Dict[str, List[Perturbation]] = {}
        self._entered = False
        
    @property
    def population_size(self) -> int:
        """Number of population members."""
        return self._population_size
    
    def __enter__(self) -> 'PerturbationContext':
        """Enter context - save original parameters."""
        if self._strategy._in_perturbation_context:
            raise RuntimeError(
                "Cannot nest perturbation contexts. Exit the current context before "
                "entering a new one."
            )
        
        self._strategy._in_perturbation_context = True
        self._entered = True
        
        # Save original parameters for restoration
        if self._strategy.model is not None:
            for name, param in self._strategy.model.named_parameters():
                if param.requires_grad:
                    self._original_params[name] = param.data.clone()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original parameters."""
        self._strategy._in_perturbation_context = False
        self._entered = False
        
        # Restore original parameters
        if self._strategy.model is not None:
            for name, param in self._strategy.model.named_parameters():
                if name in self._original_params:
                    param.data.copy_(self._original_params[name])
        
        self._original_params.clear()
        self._perturbations_cache.clear()
        self._active_member_id = None
        
        return False  # Don't suppress exceptions
    
    def batched_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        member_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Batched forward pass with per-sample perturbations.
        
        This is the efficient API - evaluates all population members in one call.
        
        Args:
            model: The neural network
            x: Input tensor, shape (batch_size, *input_dims)
            member_ids: Which population member to use for each batch element.
                       Defaults to member_ids[i] = i (one-to-one mapping).
                       Shape: (batch_size,)
        
        Returns:
            Output tensor, shape (batch_size, *output_dims)
        
        Raises:
            RuntimeError: If model is not on CUDA or x is not on CUDA
        """
        # Check GPU requirement
        if not x.is_cuda:
            raise RuntimeError(
                "EGGROLL requires CUDA tensors. Input tensor is on CPU.\n"
                "Please move your input to GPU: x = x.cuda()"
            )
        
        # Check model is on GPU
        model_device = next(model.parameters()).device
        if not model_device.type == 'cuda':
            raise RuntimeError(
                "EGGROLL requires CUDA. Model is on CPU.\n"
                "Please move your model to GPU: model = model.cuda()"
            )
        
        batch_size = x.shape[0]
        
        if member_ids is None:
            member_ids = torch.arange(batch_size, device=x.device)
        
        # Generate perturbations for all members needed
        unique_members = torch.unique(member_ids)
        
        # Use the strategy's batched forward implementation
        return self._strategy._batched_forward_impl(
            model, x, member_ids, self._epoch, self._population_size
        )
    
    def iterate(self) -> Iterator[int]:
        """
        Iterate through population members for sequential evaluation.
        
        This is slower than batched_forward but useful for debugging
        or when working with non-batched environments.
        
        Yields:
            member_id: The index of the current population member
        
        Usage:
            with strategy.perturb(population_size=64, epoch=0) as pop:
                for member_id in pop.iterate():
                    output = model(x)  # Uses this member's perturbation
        """
        for member_id in range(self._population_size):
            self._active_member_id = member_id
            self._apply_perturbation(member_id)
            yield member_id
        
        self._active_member_id = None
        # Restore original parameters after iteration
        if self._strategy.model is not None:
            for name, param in self._strategy.model.named_parameters():
                if name in self._original_params:
                    param.data.copy_(self._original_params[name])
    
    def _apply_perturbation(self, member_id: int):
        """Apply perturbation for a specific member to the model."""
        if self._strategy.model is None:
            return
        
        for name, param in self._strategy.model.named_parameters():
            if name in self._original_params:
                # Get perturbation for this parameter
                pert = self._strategy._sample_perturbation(
                    param=self._original_params[name],
                    member_id=member_id,
                    epoch=self._epoch,
                    param_name=name
                )
                # Apply perturbation
                param.data.copy_(self._original_params[name] + pert.as_matrix())
    
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
            Tuple (A, B) where perturbation = A @ B.T
        """
        if self._strategy.model is None:
            raise RuntimeError("Strategy not set up with a model")
        
        # Find the parameter
        param = None
        for name, p in self._strategy.model.named_parameters():
            if name == param_name:
                param = self._original_params.get(name, p)
                break
        
        if param is None:
            raise ValueError(f"Parameter '{param_name}' not found in model")
        
        pert = self._strategy._sample_perturbation(
            param=param,
            member_id=member_id,
            epoch=self._epoch,
            param_name=param_name
        )
        
        return pert.factors
