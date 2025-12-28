"""
EggrollStrategy: Main interface for EGGROLL evolution strategy in PyTorch.

This module provides the high-level API for using EGGROLL to evolve neural networks.
It handles:
- Parameter discovery and tracking
- Perturbation context management
- ES gradient estimation and parameter updates
- Checkpointing and state management
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Iterator, ContextManager
from dataclasses import dataclass
from contextlib import contextmanager
import warnings

from .perturbation import PerturbationContext, Perturbation
from .triton_kernels import (
    generate_lowrank_factors_torch,
    compute_es_gradient_torch,
)


@dataclass
class EggrollConfig:
    """Configuration for EggrollStrategy."""
    sigma: float = 0.1
    lr: float = 0.01
    rank: int = 4
    antithetic: bool = True
    noise_reuse: int = 0
    optimizer: str = "sgd"
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    seed: int = 42


class EggrollStrategy:
    """
    Low-rank evolution strategy with the EGGROLL algorithm.
    
    EGGROLL generates perturbations as low-rank matrices (A @ B.T) which enables:
    - Memory-efficient perturbations: O(r * (m+n)) instead of O(m*n)
    - Batched evaluation of all population members
    - On-the-fly noise regeneration (no need to store perturbations)
    
    Usage:
        strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=4)
        strategy.setup(model)
        
        for epoch in range(100):
            x = get_batch()
            
            with strategy.perturb(population_size=64, epoch=epoch) as pop:
                outputs = pop.batched_forward(model, x)
                fitnesses = compute_fitnesses(outputs)
            
            metrics = strategy.step(fitnesses)
    
    Args:
        sigma: Noise scale for perturbations
        lr: Learning rate for ES updates
        rank: Rank of low-rank perturbations (higher = more expressive)
        optimizer: Optimizer name ("sgd", "adam", etc.)
        optimizer_kwargs: Additional optimizer arguments
        antithetic: Use mirrored sampling for variance reduction
        noise_reuse: Number of epochs to reuse noise (0 = same noise always)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        sigma: float = 0.1,
        lr: float = 0.01,
        rank: int = 4,
        optimizer: str = "sgd",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        antithetic: bool = True,
        noise_reuse: int = 0,
        seed: Optional[int] = None,
    ):
        self.sigma = sigma
        self.lr = lr
        self.rank = rank
        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.antithetic = antithetic
        self.noise_reuse = noise_reuse
        self._seed = seed if seed is not None else 42
        
        # These are set during setup()
        self.model: Optional[nn.Module] = None
        self._es_params: Dict[str, nn.Parameter] = {}
        self._param_ids: Dict[str, int] = {}
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._current_context: Optional[PerturbationContext] = None
        self._step_count: int = 0
    
    @classmethod
    def from_config(cls, config: EggrollConfig) -> "EggrollStrategy":
        """Create strategy from config dataclass."""
        return cls(
            sigma=config.sigma,
            lr=config.lr,
            rank=config.rank,
            optimizer=config.optimizer,
            optimizer_kwargs=config.optimizer_kwargs,
            antithetic=config.antithetic,
            noise_reuse=config.noise_reuse,
            seed=config.seed,
        )
    
    def setup(self, model: nn.Module) -> None:
        """
        Attach strategy to a model and discover parameters.
        
        This should be called once before training. It:
        - Validates the model is on GPU
        - Identifies parameters to evolve
        - Initializes the optimizer
        
        Args:
            model: The neural network to evolve
            
        Raises:
            RuntimeError: If model is not on CUDA device
        """
        # Check GPU
        try:
            device = next(model.parameters()).device
        except StopIteration:
            raise RuntimeError("Model has no parameters")
        
        if device.type != 'cuda':
            raise RuntimeError(
                "\n" + "=" * 70 + "\n"
                "EGGROLL needs a CUDA GPU\n" 
                "=" * 70 + "\n\n"
                "EGGROLL-Torch is designed for GPU-accelerated batched perturbations.\n"
                "On CPU, you'd lose the speed advantage that makes it worth using.\n\n"
                "A few options:\n"
                "  • Use a machine with an NVIDIA GPU\n"
                "  • Try Google Colab (free GPU tier)\n"
                "  • For CPU-only work, check out OpenAI's ES or other CPU-friendly libraries\n\n"
                f"Your model is currently on: {device}\n"
                "Move it with: model = model.cuda()\n"
                + "=" * 70
            )
        
        self.model = model
        self._discover_parameters()
        self._setup_optimizer()
    
    def _discover_parameters(self) -> None:
        """
        Discover and index parameters for evolution.
        
        By default, all 2D parameters (weight matrices) are evolved with low-rank
        perturbations. 1D parameters (biases) are evolved with full perturbations.
        """
        self._es_params = {}
        self._param_ids = {}
        
        param_id = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Store with a normalized name for Sequential models
                # e.g., "0.weight" for model[0].weight
                self._es_params[name] = param
                self._param_ids[name] = param_id
                param_id += 1
    
    def _setup_optimizer(self) -> None:
        """Initialize the optimizer for ES updates."""
        params = list(self._es_params.values())
        
        if self.optimizer_name.lower() == "sgd":
            self._optimizer = torch.optim.SGD(params, lr=self.lr, **self.optimizer_kwargs)
        elif self.optimizer_name.lower() == "adam":
            self._optimizer = torch.optim.Adam(params, lr=self.lr, **self.optimizer_kwargs)
        elif self.optimizer_name.lower() == "adamw":
            self._optimizer = torch.optim.AdamW(params, lr=self.lr, **self.optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def perturb(
        self, 
        population_size: int, 
        epoch: int = 0
    ) -> PerturbationContext:
        """
        Context manager for perturbed evaluation.
        
        Args:
            population_size: Number of population members to evaluate
            epoch: Current epoch (for noise scheduling)
            
        Returns:
            PerturbationContext for batched or sequential evaluation
            
        Usage:
            with strategy.perturb(population_size=64, epoch=0) as pop:
                outputs = pop.batched_forward(model, x)
        """
        if self.model is None:
            raise RuntimeError("Strategy not set up. Call strategy.setup(model) first.")
        
        # Validate population size for antithetic sampling
        if self.antithetic and population_size % 2 != 0:
            raise ValueError(
                f"With antithetic=True, population_size must be even. Got {population_size}"
            )
        
        return PerturbationContext(
            strategy=self,
            population_size=population_size,
            epoch=epoch,
        )
    
    def _sample_perturbation(
        self,
        param: nn.Parameter,
        member_id: int,
        epoch: int,
    ) -> Perturbation:
        """
        Sample a single perturbation for a parameter.
        
        This is mainly for testing/debugging. In production, use batched_forward
        which generates perturbations on-the-fly.
        
        Args:
            param: The parameter to perturb
            member_id: Population member index
            epoch: Current epoch
            
        Returns:
            Perturbation object with low-rank factors
        """
        if param.ndim != 2:
            raise ValueError(
                f"Low-rank perturbations only support 2D parameters, got {param.ndim}D"
            )
        
        # Find param_id
        param_id = None
        for name, p in self._es_params.items():
            if p is param:
                param_id = self._param_ids[name]
                break
        
        if param_id is None:
            raise ValueError("Parameter not found in strategy's tracked parameters")
        
        out_features, in_features = param.shape
        member_ids = torch.tensor([member_id], device=param.device)
        
        A, B = generate_lowrank_factors_torch(
            out_features=out_features,
            in_features=in_features,
            rank=self.rank,
            seed=self._seed,
            epoch=epoch,
            member_ids=member_ids,
            param_id=param_id,
            sigma=self.sigma,
            noise_reuse=self.noise_reuse,
            device=param.device,
            dtype=param.dtype,
        )
        
        return Perturbation(
            A=A[0],  # Remove batch dim
            B=B[0],
            sigma=self.sigma,
        )
    
    def step(self, fitnesses: torch.Tensor) -> Dict[str, Any]:
        """
        Update parameters based on fitnesses.
        
        This computes the ES gradient estimate and applies an optimizer step.
        
        Args:
            fitnesses: Fitness values for each population member (population_size,)
            
        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            raise RuntimeError("Strategy not set up. Call strategy.setup(model) first.")
        
        # Normalize fitnesses
        normalized_fitnesses = self._normalize_fitnesses(fitnesses)
        
        # Get the last perturbation context (we need the factors)
        # In production, we'd regenerate them using the same seed
        # For now, we store them in the context
        
        # Compute ES gradients and apply updates
        self._optimizer.zero_grad()
        
        # For each parameter, compute ES gradient
        device = fitnesses.device
        population_size = fitnesses.shape[0]
        
        for name, param in self._es_params.items():
            if param.ndim != 2:
                # Skip non-2D params for now (biases)
                continue
            
            param_id = self._param_ids[name]
            out_features, in_features = param.shape
            
            # Regenerate factors (same seed = same factors)
            member_ids = torch.arange(population_size, device=device)
            A, B = generate_lowrank_factors_torch(
                out_features=out_features,
                in_features=in_features,
                rank=self.rank,
                seed=self._seed,
                epoch=0,  # TODO: track current epoch
                member_ids=member_ids,
                param_id=param_id,
                sigma=self.sigma,
                noise_reuse=self.noise_reuse,
                device=device,
                dtype=param.dtype,
            )
            
            # Compute ES gradient
            grad = compute_es_gradient_torch(normalized_fitnesses, A, B, self.sigma)
            
            # Scale by sqrt(population_size) as in original EGGROLL
            grad = -grad * (population_size ** 0.5)
            
            # Assign gradient to parameter
            param.grad = grad.to(param.dtype)
        
        # Apply optimizer step
        self._optimizer.step()
        self._step_count += 1
        
        return {
            "step": self._step_count,
            "fitness_mean": fitnesses.mean().item(),
            "fitness_std": fitnesses.std().item(),
            "fitness_max": fitnesses.max().item(),
            "fitness_min": fitnesses.min().item(),
        }
    
    def _normalize_fitnesses(self, fitnesses: torch.Tensor) -> torch.Tensor:
        """
        Normalize fitnesses for ES update.
        
        Uses z-score normalization: (fitness - mean) / std
        """
        mean = fitnesses.mean()
        std = fitnesses.std() + 1e-8
        return (fitnesses - mean) / std
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize state for checkpointing.
        
        Returns:
            Dictionary containing all state needed to resume training
        """
        state = {
            "sigma": self.sigma,
            "lr": self.lr,
            "rank": self.rank,
            "antithetic": self.antithetic,
            "noise_reuse": self.noise_reuse,
            "seed": self._seed,
            "step_count": self._step_count,
        }
        
        if self._optimizer is not None:
            state["optimizer_state"] = self._optimizer.state_dict()
        
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Restore from checkpoint.
        
        Args:
            state: State dictionary from state_dict()
        """
        self.sigma = state["sigma"]
        self.lr = state["lr"]
        self.rank = state["rank"]
        self.antithetic = state["antithetic"]
        self.noise_reuse = state["noise_reuse"]
        self._seed = state["seed"]
        self._step_count = state["step_count"]
        
        if "optimizer_state" in state and self._optimizer is not None:
            self._optimizer.load_state_dict(state["optimizer_state"])
