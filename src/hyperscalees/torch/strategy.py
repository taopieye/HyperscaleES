"""
Evolution Strategy implementations for PyTorch.

Contains:
- BaseStrategy: Abstract base class for all ES strategies
- EggrollStrategy: Low-rank evolution strategy (EGGROLL algorithm)
- OpenESStrategy: Standard OpenAI Evolution Strategy
"""

import torch
import torch.nn as nn
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterator, List, Tuple, Union, ContextManager, Set
from dataclasses import dataclass, field

from .perturbation import Perturbation, PerturbationContext


@dataclass
class EggrollConfig:
    """Configuration for Eggroll strategy."""
    sigma: float = 0.1
    lr: float = 0.01
    rank: int = 4
    antithetic: bool = True
    noise_reuse: int = 0
    optimizer: str = "sgd"
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None


@dataclass
class OpenESConfig:
    """Configuration for OpenES strategy."""
    sigma: float = 0.1
    lr: float = 0.01
    antithetic: bool = True
    noise_reuse: int = 0
    optimizer: str = "sgd"
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None


class BaseStrategy(ABC):
    """
    Abstract base class defining the Evolution Strategy interface.
    
    All ES implementations must conform to this interface.
    """
    
    def __init__(
        self,
        sigma: float = 0.1,
        lr: float = 0.01,
        antithetic: bool = True,
        noise_reuse: int = 0,
        optimizer: str = "sgd",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        self._sigma = sigma
        self._lr = lr
        self._antithetic = antithetic
        self._noise_reuse = noise_reuse
        self._optimizer_type = optimizer
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._seed = seed or 42
        
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._device: Optional[torch.device] = None
        self._in_perturbation_context: bool = False
        self._last_perturbations: Dict[str, List[Perturbation]] = {}
        self._last_epoch: int = 0
        self._last_population_size: int = 0
        self._param_keys: Dict[str, int] = {}  # Maps param name to unique key
        self._included_params: Optional[Set[str]] = None
        self._excluded_params: Set[str] = set()
    
    @classmethod
    def from_config(cls, config: Union[EggrollConfig, OpenESConfig]) -> 'BaseStrategy':
        """Create strategy from config object."""
        return cls(**config.__dict__)
    
    @property
    def model(self) -> Optional[nn.Module]:
        """The attached model."""
        return self._model
    
    @property
    def sigma(self) -> float:
        """Current noise scale."""
        return self._sigma
    
    @sigma.setter
    def sigma(self, value: float) -> None:
        """Set noise scale."""
        self._sigma = value
    
    @property
    def lr(self) -> float:
        """Current learning rate."""
        return self._lr
    
    @lr.setter
    def lr(self, value: float) -> None:
        """Set learning rate."""
        self._lr = value
        # Update optimizer learning rate if it exists
        if self._optimizer is not None:
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = value
    
    def setup(
        self, 
        model: nn.Module,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None
    ) -> None:
        """
        Attach strategy to a model and discover parameters.
        
        This should:
        - Store reference to model
        - Verify model is on GPU
        - Identify which parameters to evolve
        - Initialize optimizer state
        
        Args:
            model: The neural network to optimize
            include: If provided, only evolve these parameters (by name)
            exclude: If provided, exclude these parameters from evolution
        """
        # Check GPU requirement
        try:
            device = next(model.parameters()).device
        except StopIteration:
            raise RuntimeError("Model has no parameters")
        
        if device.type != 'cuda':
            raise RuntimeError(
                "\n" + "="*70 + "\n"
                "EGGROLL needs a CUDA GPU\n"
                "="*70 + "\n\n"
                "EGGROLL-Torch is designed for GPU-accelerated batched perturbations.\n"
                "On CPU, you'd lose the speed advantage that makes it worth using.\n\n"
                "A few options:\n"
                "  • Use a machine with an NVIDIA GPU\n"
                "  • Try Google Colab (free GPU tier)\n"
                "  • For CPU-only work, check out OpenAI's ES or other CPU-friendly libraries\n\n"
                "Current device: " + str(device) + "\n"
                + "="*70
            )
        
        self._model = model
        self._device = device
        
        # Handle include/exclude
        if include is not None:
            self._included_params = set(include)
        else:
            self._included_params = None
        
        if exclude is not None:
            self._excluded_params = set(exclude)
        else:
            self._excluded_params = set()
        
        # Assign unique keys to each parameter for deterministic noise generation
        for i, (name, _) in enumerate(model.named_parameters()):
            self._param_keys[name] = i
        
        # Initialize optimizer
        self._init_optimizer()
    
    def _init_optimizer(self):
        """Initialize the optimizer for ES updates."""
        if self._model is None:
            return
        
        params = list(self.parameters())
        if len(params) == 0:
            return
        
        if self._optimizer_type == "sgd":
            self._optimizer = torch.optim.SGD(params, lr=self._lr, **self._optimizer_kwargs)
        elif self._optimizer_type == "adam":
            self._optimizer = torch.optim.Adam(params, lr=self._lr, **self._optimizer_kwargs)
        elif self._optimizer_type == "adamw":
            self._optimizer = torch.optim.AdamW(params, lr=self._lr, **self._optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {self._optimizer_type}")
    
    def _should_evolve_param(self, name: str, param: torch.Tensor) -> bool:
        """Check if a parameter should be evolved."""
        if not param.requires_grad:
            return False
        
        if self._included_params is not None:
            return name in self._included_params
        
        if name in self._excluded_params:
            return False
        
        return True
    
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """
        Iterate over parameters that will be evolved.
        
        Respects include/exclude settings.
        """
        if self._model is None:
            return iter([])
        
        for name, p in self._model.named_parameters():
            if self._should_evolve_param(name, p):
                yield p
    
    def named_parameters(self) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """
        Iterate over named parameters that will be evolved.
        
        Respects include/exclude settings.
        """
        if self._model is None:
            return iter([])
        
        for name, p in self._model.named_parameters():
            if self._should_evolve_param(name, p):
                yield name, p
    
    def weight_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Iterate over 2D weight matrix parameters."""
        for name, p in self.named_parameters():
            if p.dim() >= 2 and 'weight' in name:
                yield p
    
    def bias_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Iterate over 1D bias parameters."""
        for name, p in self.named_parameters():
            if p.dim() == 1 and 'bias' in name:
                yield p
    
    def perturb(self, population_size: int, epoch: int = 0) -> PerturbationContext:
        """
        Context manager for applying perturbations.
        
        Primary usage (batched):
            with strategy.perturb(population_size=64, epoch=0) as pop:
                outputs = pop.batched_forward(model, x_batch)
        
        Args:
            population_size: Number of population members
            epoch: Current epoch (for deterministic noise)
        
        Returns:
            PerturbationContext for use in with statement
        """
        return PerturbationContext(self, population_size, epoch)
    
    def eval(self) -> ContextManager:
        """
        Context manager for explicit no-perturbation mode.
        
        Usage:
            with strategy.eval():
                output = model(x)  # Guaranteed no perturbation
        """
        class NoOpContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        return NoOpContext()
    
    @abstractmethod
    def _sample_perturbation(
        self,
        param: torch.Tensor,
        member_id: int,
        epoch: int,
        param_name: str = ""
    ) -> Perturbation:
        """
        Sample a perturbation for a parameter.
        
        Args:
            param: The parameter tensor
            member_id: Population member index
            epoch: Current epoch
            param_name: Name of the parameter (for key derivation)
        
        Returns:
            Perturbation object with low-rank factors
        """
        pass
    
    @abstractmethod
    def sample_perturbations(
        self,
        param: torch.Tensor,
        population_size: int,
        epoch: int,
        param_name: str = ""
    ) -> List[Perturbation]:
        """
        Sample perturbations for all population members.
        
        Args:
            param: The parameter tensor
            population_size: Number of population members
            epoch: Current epoch
            param_name: Name of the parameter
        
        Returns:
            List of Perturbation objects
        """
        pass
    
    @abstractmethod
    def _batched_forward_impl(
        self,
        model: nn.Module,
        x: torch.Tensor,
        member_ids: torch.Tensor,
        epoch: int,
        population_size: int
    ) -> torch.Tensor:
        """
        Implementation of batched forward pass.
        
        Args:
            model: Neural network
            x: Input tensor (batch_size, *input_dims)
            member_ids: Population member for each batch element
            epoch: Current epoch
            population_size: Total population size
        
        Returns:
            Output tensor (batch_size, *output_dims)
        """
        pass
    
    def normalize_fitnesses(self, fitnesses: torch.Tensor) -> torch.Tensor:
        """
        Normalize fitness scores to have zero mean and unit variance.
        
        Args:
            fitnesses: Raw fitness scores, shape (population_size,)
        
        Returns:
            Normalized fitness scores
        """
        mean = fitnesses.mean()
        var = fitnesses.var(unbiased=False)
        
        # Add epsilon to prevent division by zero
        std = torch.sqrt(var + 1e-8)
        
        # If variance is essentially zero, return zeros
        if std < 1e-10:
            return torch.zeros_like(fitnesses)
        
        return (fitnesses - mean) / std
    
    @abstractmethod
    def step(self, fitnesses: torch.Tensor) -> Dict[str, Any]:
        """
        Update model parameters based on fitness scores.
        
        Args:
            fitnesses: Fitness scores for each population member
        
        Returns:
            Dict with update metrics (e.g., gradient norm, param delta)
        """
        pass
    
    def _get_perturbation(
        self,
        param_name: str,
        member_id: int,
        epoch: int
    ) -> Perturbation:
        """
        Get a perturbation for a specific parameter and member.
        
        This is used during the update step to reconstruct perturbations.
        """
        if self._model is None:
            raise RuntimeError("Strategy not set up with a model")
        
        for name, param in self._model.named_parameters():
            if name == param_name:
                return self._sample_perturbation(param, member_id, epoch, param_name)
        
        raise ValueError(f"Parameter '{param_name}' not found")
    
    def state_dict(self) -> Dict[str, Any]:
        """Get serializable state for checkpointing."""
        state = {
            'sigma': self._sigma,
            'lr': self._lr,
            'antithetic': self._antithetic,
            'noise_reuse': self._noise_reuse,
            'optimizer_type': self._optimizer_type,
            'optimizer_kwargs': self._optimizer_kwargs,
            'seed': self._seed,
            'param_keys': self._param_keys,
        }
        
        if self._optimizer is not None:
            state['optimizer_state'] = self._optimizer.state_dict()
        
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self._sigma = state.get('sigma', self._sigma)
        self._lr = state.get('lr', self._lr)
        self._antithetic = state.get('antithetic', self._antithetic)
        self._noise_reuse = state.get('noise_reuse', self._noise_reuse)
        self._optimizer_type = state.get('optimizer_type', self._optimizer_type)
        self._optimizer_kwargs = state.get('optimizer_kwargs', self._optimizer_kwargs)
        self._seed = state.get('seed', self._seed)
        self._param_keys = state.get('param_keys', self._param_keys)
        
        if self._optimizer is not None and 'optimizer_state' in state:
            self._optimizer.load_state_dict(state['optimizer_state'])


class EggrollStrategy(BaseStrategy):
    """
    Low-rank evolution strategy with the EGGROLL algorithm.
    
    EGGROLL generates perturbations as AB^T where A ∈ R^{m×r}, B ∈ R^{n×r}
    with r << min(m,n). This reduces auxiliary storage from mn to r(m+n).
    
    The key insight is that the forward pass can be computed efficiently:
        x @ (W + AB^T)^T = x @ W^T + x @ B @ A^T
    
    This avoids materializing the full perturbation matrix.
    """
    
    def __init__(
        self,
        sigma: float = 0.1,
        lr: float = 0.01,
        rank: int = 4,
        antithetic: bool = True,
        noise_reuse: int = 0,
        optimizer: str = "sgd",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            sigma=sigma,
            lr=lr,
            antithetic=antithetic,
            noise_reuse=noise_reuse,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            seed=seed,
            **kwargs
        )
        self._rank = rank
    
    @property
    def rank(self) -> int:
        """Perturbation rank."""
        return self._rank
    
    def _get_generator_for_perturbation(
        self,
        member_id: int,
        epoch: int,
        param_key: int
    ) -> torch.Generator:
        """
        Get a deterministic generator for a specific perturbation.
        
        The generator is seeded based on:
        - Base seed
        - Epoch (considering noise_reuse)
        - Member ID (considering antithetic pairing)
        - Parameter key
        """
        gen = torch.Generator(device=self._device)
        
        # Compute effective epoch (for noise reuse)
        if self._noise_reuse == 0:
            effective_epoch = 0
        else:
            effective_epoch = epoch // self._noise_reuse
        
        # Compute effective member ID (for antithetic sampling)
        if self._antithetic:
            effective_member = member_id // 2
        else:
            effective_member = member_id
        
        # Combine into a single seed
        combined_seed = (
            self._seed * 1000003 + 
            effective_epoch * 1009 + 
            effective_member * 10007 + 
            param_key
        )
        
        gen.manual_seed(combined_seed % (2**31 - 1))
        return gen
    
    def _sample_perturbation(
        self,
        param: torch.Tensor,
        member_id: int,
        epoch: int,
        param_name: str = ""
    ) -> Perturbation:
        """
        Sample a low-rank perturbation for a parameter.
        
        For antithetic sampling:
        - Even member_id (0, 2, 4, ...): +ε
        - Odd member_id (1, 3, 5, ...): -ε
        """
        if param.dim() < 2:
            # For 1D parameters (biases), treat as (n, 1) matrix
            m, n = param.shape[0], 1
            is_1d = True
        else:
            # For 2D+ parameters, use first two dimensions
            m, n = param.shape[0], param.shape[1]
            is_1d = False
        
        r = min(self._rank, m, n)  # Rank can't exceed matrix dimensions
        
        # Get deterministic generator
        param_key = self._param_keys.get(param_name, hash(param_name) % 10000)
        gen = self._get_generator_for_perturbation(member_id, epoch, param_key)
        
        # Generate low-rank factors
        # Combined tensor for both A and B, then split
        combined = torch.randn(
            (m + n, r), 
            device=param.device, 
            dtype=param.dtype,
            generator=gen
        )
        
        A = combined[:m]  # (m, r)
        B = combined[m:]  # (n, r)
        
        # Apply sigma scaling (normalized by sqrt(rank) for consistent magnitude)
        sigma_scaled = self._sigma / math.sqrt(r)
        
        # Apply antithetic sign
        if self._antithetic and member_id % 2 == 1:
            sign = -1.0
        else:
            sign = 1.0
        
        A = A * sigma_scaled * sign
        # B keeps its original sign (perturbation is A @ B.T)
        
        if is_1d:
            # Squeeze back to 1D
            A = A.squeeze(-1)
            B = B.squeeze(-1)
        
        return Perturbation(A=A, B=B)
    
    def sample_perturbations(
        self,
        param: torch.Tensor,
        population_size: int,
        epoch: int,
        param_name: str = ""
    ) -> List[Perturbation]:
        """Sample perturbations for all population members."""
        return [
            self._sample_perturbation(param, i, epoch, param_name)
            for i in range(population_size)
        ]
    
    def _batched_forward_impl(
        self,
        model: nn.Module,
        x: torch.Tensor,
        member_ids: torch.Tensor,
        epoch: int,
        population_size: int
    ) -> torch.Tensor:
        """
        Efficient batched forward pass with low-rank perturbations.
        
        For each linear layer, computes:
            output[i] = x[i] @ W.T + bias + x[i] @ B[member_ids[i]] @ A[member_ids[i]].T
        """
        batch_size = x.shape[0]
        current_input = x
        
        # Process each layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Get base weight and bias
                W = module.weight  # (out_features, in_features)
                bias = module.bias  # (out_features,) or None
                
                # Compute base output: x @ W.T
                base_output = current_input @ W.T
                if bias is not None:
                    base_output = base_output + bias
                
                # Compute perturbation output for each batch element
                # We need to apply different perturbations based on member_ids
                
                # Get all unique perturbations needed
                unique_members = torch.unique(member_ids)
                
                # Pre-compute perturbation factors for all unique members
                perturbations = {}
                param_name = None
                for n, p in model.named_parameters():
                    if p is W:
                        param_name = n
                        break
                
                if param_name is None:
                    # Fallback for modules not in direct named_parameters
                    param_name = name + ".weight"
                
                for m in unique_members.tolist():
                    pert = self._sample_perturbation(W, int(m), epoch, param_name)
                    perturbations[int(m)] = pert
                
                # Compute perturbation contributions
                # For each sample, add x @ B @ A.T
                pert_output = torch.zeros_like(base_output)
                
                for m in unique_members.tolist():
                    mask = (member_ids == m)
                    if mask.any():
                        A, B = perturbations[int(m)].factors
                        # x @ B @ A.T for samples with this member_id
                        x_subset = current_input[mask]  # (subset_size, in_features)
                        # Efficient computation: (x @ B) @ A.T
                        pert_contrib = (x_subset @ B) @ A.T  # (subset_size, out_features)
                        pert_output[mask] = pert_contrib
                
                current_input = base_output + pert_output
                
            elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.GELU)):
                # Activation functions - apply directly
                current_input = module(current_input)
            elif isinstance(module, nn.Sequential):
                # Skip sequential containers - we process their children
                pass
            elif isinstance(module, nn.Dropout):
                # Apply dropout (typically in eval mode, so no-op)
                current_input = module(current_input)
            elif isinstance(module, nn.BatchNorm1d):
                current_input = module(current_input)
            elif isinstance(module, nn.LayerNorm):
                current_input = module(current_input)
        
        return current_input
    
    def step(self, fitnesses: torch.Tensor) -> Dict[str, Any]:
        """
        Update parameters based on fitness scores.
        
        Computes ES gradient estimate:
            ∇̂ = (1/N) Σᵢ f_normalized[i] * perturbation[i]
        
        For low-rank perturbations:
            ∇̂ = (1/N) Σᵢ f_normalized[i] * A[i] @ B[i].T
        """
        if self._model is None:
            raise RuntimeError("Strategy not set up. Call setup() first.")
        
        fitnesses = fitnesses.to(self._device)
        population_size = fitnesses.shape[0]
        
        # Normalize fitnesses
        normalized = self.normalize_fitnesses(fitnesses)
        
        # Compute gradients for each parameter
        metrics = {}
        total_grad_norm = 0.0
        total_param_delta = 0.0
        
        # Store gradients
        gradients = {}
        
        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Accumulate weighted perturbations
            if param.dim() >= 2:
                # Matrix parameters - use low-rank accumulation
                m, n = param.shape[0], param.shape[1]
                grad_accum = torch.zeros_like(param)
                
                for i in range(population_size):
                    pert = self._sample_perturbation(param, i, self._last_epoch, name)
                    A, B = pert.factors
                    # Weighted contribution: f_i * A_i @ B_i.T
                    # Use einsum for efficiency: (r,) @ (m, r).T @ (n, r) -> (m, n)
                    weight = normalized[i].item()
                    grad_accum += weight * (A @ B.T)
                
                # Average over population
                grad = -grad_accum / population_size * math.sqrt(population_size)
            else:
                # 1D parameters (biases) - simpler handling
                grad_accum = torch.zeros_like(param)
                
                for i in range(population_size):
                    pert = self._sample_perturbation(param, i, self._last_epoch, name)
                    A, B = pert.factors
                    weight = normalized[i].item()
                    grad_accum += weight * A * B
                
                grad = -grad_accum / population_size * math.sqrt(population_size)
            
            gradients[name] = grad
            total_grad_norm += grad.norm().item() ** 2
        
        total_grad_norm = math.sqrt(total_grad_norm)
        
        # Apply gradients via optimizer
        if self._optimizer is not None:
            self._optimizer.zero_grad()
            
            for name, param in self._model.named_parameters():
                if name in gradients:
                    param.grad = gradients[name]
            
            # Record param values before step
            params_before = {n: p.clone() for n, p in self._model.named_parameters()}
            
            self._optimizer.step()
            
            # Compute param delta
            for n, p in self._model.named_parameters():
                if n in params_before:
                    total_param_delta += (p - params_before[n]).norm().item() ** 2
            total_param_delta = math.sqrt(total_param_delta)
        
        self._last_epoch += 1
        self._last_population_size = population_size
        
        metrics['grad_norm'] = total_grad_norm
        metrics['param_delta'] = total_param_delta
        metrics['fitness_mean'] = fitnesses.mean().item()
        metrics['fitness_std'] = fitnesses.std().item()
        
        return metrics


class OpenESStrategy(BaseStrategy):
    """
    Standard OpenAI Evolution Strategy (full-rank perturbations).
    
    This is the classic ES approach where perturbations are full-rank
    Gaussian noise. Less memory-efficient than EGGROLL but simpler.
    """
    
    def _get_generator_for_perturbation(
        self,
        member_id: int,
        epoch: int,
        param_key: int
    ) -> torch.Generator:
        """Get deterministic generator for a specific perturbation."""
        gen = torch.Generator(device=self._device)
        
        if self._noise_reuse == 0:
            effective_epoch = 0
        else:
            effective_epoch = epoch // self._noise_reuse
        
        if self._antithetic:
            effective_member = member_id // 2
        else:
            effective_member = member_id
        
        combined_seed = (
            self._seed * 1000003 + 
            effective_epoch * 1009 + 
            effective_member * 10007 + 
            param_key
        )
        
        gen.manual_seed(combined_seed % (2**31 - 1))
        return gen
    
    def _sample_perturbation(
        self,
        param: torch.Tensor,
        member_id: int,
        epoch: int,
        param_name: str = ""
    ) -> Perturbation:
        """
        Sample a full-rank perturbation (stored as low-rank for API consistency).
        
        For OpenES, we generate full noise but still wrap in Perturbation
        for API compatibility. This means A is the full noise and B is identity-ish.
        """
        param_key = self._param_keys.get(param_name, hash(param_name) % 10000)
        gen = self._get_generator_for_perturbation(member_id, epoch, param_key)
        
        # Generate full perturbation
        noise = torch.randn(
            param.shape,
            device=param.device,
            dtype=param.dtype,
            generator=gen
        )
        
        # Apply sigma and antithetic sign
        if self._antithetic and member_id % 2 == 1:
            sign = -1.0
        else:
            sign = 1.0
        
        noise = noise * self._sigma * sign
        
        # For 1D tensors, return simple perturbation
        if param.dim() == 1:
            # Store as A=noise, B=ones (so A * B = noise)
            A = noise
            B = torch.ones(1, device=param.device, dtype=param.dtype)
            return Perturbation(A=A, B=B)
        
        # For 2D+, we need to represent full matrix as "low-rank"
        # A = noise, B = identity-like
        # But this defeats the purpose - for OpenES, we just store full noise
        # Use rank = min(m, n) essentially
        m, n = param.shape[0], param.shape[1]
        r = min(m, n)
        
        # SVD decomposition to get low-rank factors (expensive but correct)
        # For OpenES, we typically just use full noise directly
        # For API compatibility, store A as (m, n) reshaped and B as identity
        A = noise  # (m, n) - treated as (m, n) @ I_n.T = (m, n)
        B = torch.eye(n, device=param.device, dtype=param.dtype)  # (n, n)
        
        # Actually, let's be smarter - return A and B such that A @ B.T = noise
        # Simplest: A = noise, B = I[:, :n] but that's still full rank storage
        # For OpenES we accept full storage
        return Perturbation(A=noise, B=torch.eye(n, device=param.device, dtype=param.dtype))
    
    def sample_perturbations(
        self,
        param: torch.Tensor,
        population_size: int,
        epoch: int,
        param_name: str = ""
    ) -> List[Perturbation]:
        """Sample perturbations for all population members."""
        return [
            self._sample_perturbation(param, i, epoch, param_name)
            for i in range(population_size)
        ]
    
    def _batched_forward_impl(
        self,
        model: nn.Module,
        x: torch.Tensor,
        member_ids: torch.Tensor,
        epoch: int,
        population_size: int
    ) -> torch.Tensor:
        """Batched forward with full-rank perturbations."""
        batch_size = x.shape[0]
        current_input = x
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight
                bias = module.bias
                
                base_output = current_input @ W.T
                if bias is not None:
                    base_output = base_output + bias
                
                unique_members = torch.unique(member_ids)
                
                param_name = None
                for n, p in model.named_parameters():
                    if p is W:
                        param_name = n
                        break
                if param_name is None:
                    param_name = name + ".weight"
                
                perturbations = {}
                for m in unique_members.tolist():
                    pert = self._sample_perturbation(W, int(m), epoch, param_name)
                    perturbations[int(m)] = pert
                
                pert_output = torch.zeros_like(base_output)
                
                for m in unique_members.tolist():
                    mask = (member_ids == m)
                    if mask.any():
                        # For OpenES, A is the full perturbation
                        A, B = perturbations[int(m)].factors
                        delta_W = A @ B.T if B.dim() == 2 else A * B
                        x_subset = current_input[mask]
                        pert_contrib = x_subset @ delta_W.T
                        pert_output[mask] = pert_contrib
                
                current_input = base_output + pert_output
                
            elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.GELU)):
                current_input = module(current_input)
            elif isinstance(module, nn.Sequential):
                pass
            elif isinstance(module, nn.Dropout):
                current_input = module(current_input)
            elif isinstance(module, nn.BatchNorm1d):
                current_input = module(current_input)
            elif isinstance(module, nn.LayerNorm):
                current_input = module(current_input)
        
        return current_input
    
    def step(self, fitnesses: torch.Tensor) -> Dict[str, Any]:
        """Update parameters based on fitness scores."""
        if self._model is None:
            raise RuntimeError("Strategy not set up. Call setup() first.")
        
        fitnesses = fitnesses.to(self._device)
        population_size = fitnesses.shape[0]
        
        normalized = self.normalize_fitnesses(fitnesses)
        
        metrics = {}
        total_grad_norm = 0.0
        total_param_delta = 0.0
        
        gradients = {}
        
        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue
            
            grad_accum = torch.zeros_like(param)
            
            for i in range(population_size):
                pert = self._sample_perturbation(param, i, self._last_epoch, name)
                A, B = pert.factors
                if param.dim() >= 2:
                    delta = A @ B.T if B.dim() == 2 else A * B
                else:
                    delta = A * B if B.dim() == 1 else A
                weight = normalized[i].item()
                grad_accum += weight * delta
            
            grad = -grad_accum / population_size * math.sqrt(population_size)
            gradients[name] = grad
            total_grad_norm += grad.norm().item() ** 2
        
        total_grad_norm = math.sqrt(total_grad_norm)
        
        if self._optimizer is not None:
            self._optimizer.zero_grad()
            
            for name, param in self._model.named_parameters():
                if name in gradients:
                    param.grad = gradients[name]
            
            params_before = {n: p.clone() for n, p in self._model.named_parameters()}
            
            self._optimizer.step()
            
            for n, p in self._model.named_parameters():
                if n in params_before:
                    total_param_delta += (p - params_before[n]).norm().item() ** 2
            total_param_delta = math.sqrt(total_param_delta)
        
        self._last_epoch += 1
        self._last_population_size = population_size
        
        metrics['grad_norm'] = total_grad_norm
        metrics['param_delta'] = total_param_delta
        metrics['fitness_mean'] = fitnesses.mean().item()
        metrics['fitness_std'] = fitnesses.std().item()
        
        return metrics
