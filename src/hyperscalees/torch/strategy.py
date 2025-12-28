"""
Evolution Strategy implementations for PyTorch.

Contains:
- BaseStrategy: Abstract base class for all ES strategies
- EggrollStrategy: Low-rank evolution strategy (EGGROLL algorithm)
- OpenESStrategy: Standard OpenAI Evolution Strategy

ALL random number generation uses Triton kernels. No PyTorch RNG.
"""

import torch
import torch.nn as nn
import math
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterator, List, Tuple, Union, ContextManager, Set, Callable
from dataclasses import dataclass, field

from .perturbation import Perturbation, PerturbationContext
from .triton_kernels import (
    fused_perturbed_forward,
    generate_factors,
    compute_layer_seeds,
)


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
    
    # Valid string values for fitness_transform
    VALID_FITNESS_TRANSFORMS = {"rank", "centered_rank", None}
    
    def __init__(
        self,
        sigma: float = 0.1,
        lr: float = 0.01,
        antithetic: bool = True,
        noise_reuse: int = 0,
        optimizer: str = "sgd",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        fitness_transform: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]] = "centered_rank",
        evolve_params: Optional[List[str]] = None,
        exclude_params: Optional[List[str]] = None,
    ):
        """
        Initialize the Evolution Strategy.
        
        Args:
            sigma: Noise standard deviation for perturbations.
            lr: Learning rate for parameter updates.
            antithetic: Whether to use antithetic sampling (pairs of opposite perturbations).
            noise_reuse: Number of epochs to reuse the same noise. 0 means new noise each epoch.
            optimizer: Optimizer type ("sgd", "adam", "rmsprop").
            optimizer_kwargs: Additional kwargs for the optimizer.
            seed: Random seed for reproducibility.
            fitness_transform: Transform for fitness normalization. Can be:
                - "rank": Rank-based transform
                - "centered_rank": Centered rank-based transform (default)
                - None: No transform (raw fitnesses)
                - A callable: Custom transform function
            evolve_params: List of parameter name patterns to evolve. If None, all params evolved.
            exclude_params: List of parameter name patterns to exclude from evolution.
        """
        self._sigma = sigma
        self._lr = lr
        self._antithetic = antithetic
        self._noise_reuse = noise_reuse
        self._optimizer_type = optimizer
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._seed = seed if seed is not None else torch.randint(0, 2**31, (1,)).item()
        
        # Validate and set fitness transform
        if isinstance(fitness_transform, str):
            if fitness_transform not in self.VALID_FITNESS_TRANSFORMS:
                raise ValueError(
                    f"Invalid fitness_transform: {fitness_transform}. "
                    f"Must be one of {self.VALID_FITNESS_TRANSFORMS} or a callable."
                )
        elif fitness_transform is not None and not callable(fitness_transform):
            raise ValueError(
                f"fitness_transform must be a string, callable, or None. Got {type(fitness_transform)}"
            )
        self._fitness_transform = fitness_transform
        
        # Parameter selection patterns
        self._evolve_params = evolve_params
        self._exclude_params = exclude_params or []
        
        # Will be set in setup()
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._device: Optional[torch.device] = None
        self._param_keys: Dict[str, int] = {}
        self._current_epoch: int = 0
        self._last_population_size: Optional[int] = None
        self._layer_cache: Optional[List] = None  # For batched forward
        self._in_perturbation_context: bool = False  # Track if in perturb() context
    
    @property
    def model(self) -> Optional[nn.Module]:
        """The model being optimized."""
        return self._model
    
    @property
    def sigma(self) -> float:
        """Perturbation standard deviation."""
        return self._sigma
    
    @property
    def lr(self) -> float:
        """Learning rate."""
        return self._lr
    
    @property
    def seed(self) -> int:
        """Random seed."""
        return self._seed
    
    @property
    def antithetic(self) -> bool:
        """Whether using antithetic sampling."""
        return self._antithetic
    
    @property
    def noise_reuse(self) -> int:
        """Noise reuse period."""
        return self._noise_reuse
    
    @property
    def current_epoch(self) -> int:
        """Current epoch counter."""
        return self._current_epoch
    
    @property
    def device(self) -> Optional[torch.device]:
        """Device of the model."""
        return self._device
    
    def _should_evolve_param(self, param_name: str, param: torch.Tensor) -> bool:
        """
        Determine if a parameter should be evolved based on selection patterns.
        
        Returns True if:
        - evolve_params is None (all params evolved by default)
        - OR param_name matches any pattern in evolve_params
        
        AND:
        - param_name does NOT match any pattern in exclude_params
        - param requires_grad (for parameters that are trainable)
        """
        # Check if parameter is trainable
        if not param.requires_grad:
            return False
        
        # Check exclusion patterns first
        for pattern in self._exclude_params:
            if pattern in param_name:
                return False
        
        # If no explicit include patterns, evolve all (that aren't excluded)
        if self._evolve_params is None:
            return True
        
        # Check inclusion patterns
        for pattern in self._evolve_params:
            if pattern in param_name:
                return True
        
        return False
    
    def setup(self, model: nn.Module) -> "BaseStrategy":
        """
        Initialize the strategy for a model.
        
        Args:
            model: The neural network model to optimize.
        
        Returns:
            self for method chaining.
        """
        self._model = model
        self._device = next(model.parameters()).device
        
        # Create unique keys for each parameter
        self._param_keys = {
            name: i for i, (name, _) in enumerate(model.named_parameters())
        }
        
        # Create optimizer for the model parameters
        if self._optimizer_type.lower() == "sgd":
            self._optimizer = torch.optim.SGD(
                model.parameters(), lr=self._lr, **self._optimizer_kwargs
            )
        elif self._optimizer_type.lower() == "adam":
            self._optimizer = torch.optim.Adam(
                model.parameters(), lr=self._lr, **self._optimizer_kwargs
            )
        elif self._optimizer_type.lower() == "rmsprop":
            self._optimizer = torch.optim.RMSprop(
                model.parameters(), lr=self._lr, **self._optimizer_kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer: {self._optimizer_type}")
        
        # Build layer cache for batched forward
        self._build_layer_cache(model)
        
        return self
    
    def _build_layer_cache(self, model: nn.Module):
        """
        Pre-compute layer specs for fast batched forward.
        
        Caches: (op_type, layer_data) for each layer in forward order.
        """
        self._layer_cache = []
        param_names_map = {id(p): n for n, p in model.named_parameters()}
        
        layer_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight
                bias = module.bias
                param_name = param_names_map.get(id(W), f"{name}.weight" if name else "weight")
                evolve = self._should_evolve_param(param_name, W)
                
                # Store: (type, W, bias, layer_idx, m, n, evolve)
                self._layer_cache.append((
                    'linear', W, bias, layer_idx, W.shape[0], W.shape[1], evolve
                ))
                layer_idx += 1
                
            elif isinstance(module, nn.ReLU):
                self._layer_cache.append(('relu',))
            elif isinstance(module, nn.Tanh):
                self._layer_cache.append(('tanh',))
            elif isinstance(module, nn.Sigmoid):
                self._layer_cache.append(('sigmoid',))
            elif isinstance(module, nn.GELU):
                self._layer_cache.append(('gelu',))
            elif isinstance(module, nn.Dropout):
                self._layer_cache.append(('dropout', module.p, module.training))
            elif isinstance(module, nn.LayerNorm):
                self._layer_cache.append((
                    'layernorm', module.normalized_shape, module.weight, module.bias, module.eps
                ))
            elif isinstance(module, nn.BatchNorm1d):
                self._layer_cache.append((
                    'batchnorm', module.running_mean, module.running_var,
                    module.weight, module.bias, module.eps
                ))
    
    def _apply_fitness_transform(self, fitnesses: torch.Tensor) -> torch.Tensor:
        """
        Apply fitness transformation for variance reduction.
        
        Args:
            fitnesses: Raw fitness values.
        
        Returns:
            Transformed fitness values (zero-mean).
        """
        if self._fitness_transform is None:
            return fitnesses
        
        if callable(self._fitness_transform) and not isinstance(self._fitness_transform, str):
            return self._fitness_transform(fitnesses)
        
        if self._fitness_transform == "rank":
            # Rank-based transform: convert to ranks, then normalize to [-0.5, 0.5]
            ranks = torch.argsort(torch.argsort(fitnesses)).float()
            n = len(fitnesses)
            return ranks / (n - 1) - 0.5
        
        elif self._fitness_transform == "centered_rank":
            # Centered rank-based transform (OpenAI ES style)
            n = len(fitnesses)
            ranks = torch.argsort(torch.argsort(fitnesses)).float()
            # Map ranks to centered values
            centered = (ranks - (n - 1) / 2) / (n / 2)
            return centered
        
        return fitnesses
    
    def batched_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        population_size: int,
        epoch: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute forward pass for entire population in one call.
        
        Args:
            model: The model to evaluate.
            x: Input tensor of shape (population_size, *input_dims).
            population_size: Number of population members.
            epoch: Epoch number (uses current_epoch if None).
        
        Returns:
            Output tensor of shape (population_size, *output_dims).
        """
        if epoch is None:
            epoch = self._current_epoch
        
        # Store for step()
        self._last_population_size = population_size
        
        # Generate member IDs
        member_ids = torch.arange(population_size, device=x.device, dtype=torch.int64)
        
        return self._batched_forward_impl(model, x, member_ids, epoch, population_size)
    
    @abstractmethod
    def _batched_forward_impl(
        self,
        model: nn.Module,
        x: torch.Tensor,
        member_ids: torch.Tensor,
        epoch: int,
        population_size: int
    ) -> torch.Tensor:
        """Implementation of batched forward pass."""
        pass
    
    @abstractmethod
    def _sample_perturbation(
        self,
        param: torch.Tensor,
        member_id: int,
        epoch: int,
        param_name: str = ""
    ) -> Perturbation:
        """Sample a perturbation for a parameter."""
        pass
    
    @abstractmethod
    def step(self, fitnesses: torch.Tensor, prenormalized: bool = False) -> Dict[str, Any]:
        """Update parameters based on fitness scores."""
        pass
    
    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the strategy."""
        state = {
            'sigma': self._sigma,
            'lr': self._lr,
            'antithetic': self._antithetic,
            'noise_reuse': self._noise_reuse,
            'seed': self._seed,
            'current_epoch': self._current_epoch,
            'param_keys': self._param_keys,
        }
        if self._optimizer is not None:
            state['optimizer_state'] = self._optimizer.state_dict()
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load strategy state."""
        self._sigma = state['sigma']
        self._lr = state['lr']
        self._antithetic = state['antithetic']
        self._noise_reuse = state['noise_reuse']
        self._seed = state['seed']
        self._current_epoch = state['current_epoch']
        self._param_keys = state['param_keys']
        if self._optimizer is not None and 'optimizer_state' in state:
            self._optimizer.load_state_dict(state['optimizer_state'])
    
    def perturb(self, population_size: int, epoch: Optional[int] = None) -> PerturbationContext:
        """
        Create a perturbation context for evaluating population members.
        
        Args:
            population_size: Number of population members to evaluate.
            epoch: Epoch number (uses current_epoch if None).
        
        Returns:
            PerturbationContext that can be used as a context manager.
        
        Example:
            with strategy.perturb(population_size=64, epoch=0) as pop:
                outputs = pop.batched_forward(model, x)
                fitnesses = compute_fitness(outputs)
        """
        if self._model is None:
            raise RuntimeError("Strategy not set up. Call setup() first.")
        
        if epoch is None:
            epoch = self._current_epoch
        
        return PerturbationContext(self, population_size, epoch)


class EggrollStrategy(BaseStrategy):
    """
    Low-rank evolution strategy with the EGGROLL algorithm.
    
    EGGROLL generates perturbations as AB^T where A ∈ R^{m×r}, B ∈ R^{n×r}
    with r << min(m,n). This reduces auxiliary storage from mn to r(m+n).
    
    The key insight is that the forward pass can be computed efficiently:
        x @ (W + AB^T)^T = x @ W^T + x @ B @ A^T
    
    This avoids materializing the full perturbation matrix.
    
    ALL random number generation uses Triton kernels with tl.rand().
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
    
    def _sample_perturbation(
        self,
        param: torch.Tensor,
        member_id: int,
        epoch: int,
        param_name: str = ""
    ) -> Perturbation:
        """
        Sample a low-rank perturbation for a parameter using Triton.
        
        Uses generate_factors() which uses the SAME RNG as the fused forward kernel.
        """
        if param.dim() < 2:
            m, n = param.shape[0], 1
            is_1d = True
        else:
            m, n = param.shape[0], param.shape[1]
            is_1d = False
        
        r = min(self._rank, m, n)
        layer_idx = self._param_keys.get(param_name, hash(param_name) % 10000)
        
        # Create single-element tensors for the Triton interface
        member_ids = torch.tensor([member_id], device=param.device, dtype=torch.int64)
        seeds, signs = compute_layer_seeds(
            self._seed, epoch, member_ids, layer_idx,
            self._antithetic, self._noise_reuse
        )
        
        # Generate factors using Triton (same RNG as forward)
        A, B = generate_factors(
            seeds, m, n, r, self._sigma,
            param.dtype, param.device
        )
        
        # Apply antithetic sign to A
        A = A * signs.unsqueeze(-1).unsqueeze(-1)
        
        # Remove batch dimension
        A = A.squeeze(0)  # (m, r)
        B = B.squeeze(0)  # (n, r)
        
        if is_1d:
            A = A.squeeze(-1)
            B = B.squeeze(-1)
        
        return Perturbation(A=A, B=B, member_id=member_id, epoch=epoch, param_name=param_name)
    
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
    
    def _generate_all_perturbations(
        self,
        param: torch.Tensor,
        population_size: int,
        epoch: int,
        param_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate all low-rank perturbation factors for all population members.
        
        Uses Triton generate_factors() for consistency with forward pass.
        
        Returns:
            all_A: (population_size, m_out, r) - already scaled by sigma and antithetic sign
            all_B: (population_size, n_in, r)
        """
        if param.dim() < 2:
            m_out, n_in = param.shape[0], 1
        else:
            m_out, n_in = param.shape[0], param.shape[1]
        
        layer_idx = self._param_keys.get(param_name, hash(param_name) % 10000)
        member_ids = torch.arange(population_size, device=param.device, dtype=torch.int64)
        
        seeds, signs = compute_layer_seeds(
            self._seed, epoch, member_ids, layer_idx,
            self._antithetic, self._noise_reuse
        )
        
        A, B = generate_factors(
            seeds, m_out, n_in, self._rank, self._sigma,
            param.dtype, param.device
        )
        
        # Apply antithetic signs to A
        A = A * signs.unsqueeze(-1).unsqueeze(-1)
        
        return A, B
    
    def _batched_forward_impl(
        self,
        model: nn.Module,
        x: torch.Tensor,
        member_ids: torch.Tensor,
        epoch: int,
        population_size: int
    ) -> torch.Tensor:
        """
        Fused batched forward pass using Triton kernels.
        
        Uses fused_perturbed_forward() which generates A, B on-the-fly
        with the SAME RNG as generate_factors().
        """
        current = x  # (N, input_dim)
        
        for layer_spec in self._layer_cache:
            op_type = layer_spec[0]
            
            if op_type == 'linear':
                _, W, bias, layer_idx, m_out, n_in, evolve = layer_spec
                
                if evolve:
                    # Compute seeds for this layer
                    seeds, signs = compute_layer_seeds(
                        self._seed, epoch, member_ids, layer_idx,
                        self._antithetic, self._noise_reuse
                    )
                    
                    # Fused Triton kernel: matmul + RNG + perturbation
                    current = fused_perturbed_forward(
                        current, W, bias,
                        seeds, signs,  # RNG seeds and antithetic signs
                        self._sigma,
                        self._rank,
                    )
                else:
                    # Non-evolved: standard matmul
                    current = current @ W.T
                    if bias is not None:
                        current = current + bias
            
            elif op_type == 'relu':
                current = torch.relu(current)
            elif op_type == 'tanh':
                current = torch.tanh(current)
            elif op_type == 'sigmoid':
                current = torch.sigmoid(current)
            elif op_type == 'gelu':
                current = torch.nn.functional.gelu(current)
            elif op_type == 'dropout':
                _, p, training = layer_spec
                if training:
                    current = torch.nn.functional.dropout(current, p=p, training=True)
            elif op_type == 'layernorm':
                _, normalized_shape, weight, ln_bias, eps = layer_spec
                current = torch.nn.functional.layer_norm(current, normalized_shape, weight, ln_bias, eps)
            elif op_type == 'batchnorm':
                _, running_mean, running_var, weight, bn_bias, eps = layer_spec
                current = torch.nn.functional.batch_norm(
                    current, running_mean, running_var, weight, bn_bias, False, 0.0, eps
                )
        
        return current
    
    def step(self, fitnesses: torch.Tensor, prenormalized: bool = False) -> Dict[str, Any]:
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
        
        if self._last_population_size is not None and population_size != self._last_population_size:
            raise ValueError(
                f"Fitness size {population_size} does not match expected population size {self._last_population_size}"
            )
        
        original_fitnesses = fitnesses.clone()
        
        # Apply fitness transform
        if not prenormalized:
            fitnesses = self._apply_fitness_transform(fitnesses)
        
        # Zero gradients
        self._optimizer.zero_grad()
        
        # Compute ES gradient for each parameter
        for name, param in self._model.named_parameters():
            if not self._should_evolve_param(name, param):
                continue
            
            # Generate all perturbations for this parameter
            all_A, all_B = self._generate_all_perturbations(
                param, population_size, self._current_epoch, name
            )
            
            # Compute ES gradient: (1/N) Σ f[i] * A[i] @ B[i].T
            # Expand fitnesses for broadcasting: (N, 1, 1)
            f_expanded = fitnesses.view(population_size, 1, 1)
            
            # Weighted sum: (N, m, r) * (N, 1, 1) -> sum -> (m, r)
            weighted_A = (all_A * f_expanded).mean(dim=0)  # (m, r)
            
            # ES gradient = weighted_A @ B.T averaged
            # But B varies per member, so:
            # grad = (1/N) Σ f[i] * A[i] @ B[i].T
            
            # Compute full gradient
            # (N, m, r) @ (N, r, n) -> (N, m, n) -> weighted mean
            weighted_grad = torch.bmm(
                all_A * f_expanded,  # (N, m, r)
                all_B.transpose(1, 2)  # (N, r, n)
            ).mean(dim=0)  # (m, n)
            
            # Handle 1D case
            if param.dim() < 2:
                weighted_grad = weighted_grad.squeeze(-1)
            
            # Set gradient (ES maximizes, so negate for gradient descent)
            param.grad = -weighted_grad
        
        # Apply optimizer step
        self._optimizer.step()
        
        # Increment epoch
        self._current_epoch += 1
        
        return {
            'epoch': self._current_epoch - 1,
            'fitness_mean': original_fitnesses.mean().item(),
            'fitness_std': original_fitnesses.std().item(),
            'fitness_max': original_fitnesses.max().item(),
            'fitness_min': original_fitnesses.min().item(),
        }


class OpenESStrategy(BaseStrategy):
    """
    Standard OpenAI Evolution Strategy.
    
    Uses full-rank Gaussian perturbations for each parameter.
    More memory intensive than EGGROLL but may work better for small models.
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
    
    def _sample_perturbation(
        self,
        param: torch.Tensor,
        member_id: int,
        epoch: int,
        param_name: str = ""
    ) -> Perturbation:
        """
        Sample a full-rank perturbation for a parameter.
        
        For OpenES, we use A as the perturbation and B as identity-like.
        """
        # For OpenES: A is the noise, B is effectively identity
        # Use rank-1 factorization: noise = A @ B.T where B = identity column
        shape = param.shape
        layer_idx = self._param_keys.get(param_name, hash(param_name) % 10000)
        
        member_ids = torch.tensor([member_id], device=param.device, dtype=torch.int64)
        seeds, signs = compute_layer_seeds(
            self._seed, epoch, member_ids, layer_idx,
            self._antithetic, self._noise_reuse
        )
        
        # Generate noise using Triton
        numel = param.numel()
        A, _ = generate_factors(
            seeds, numel, 1, 1, self._sigma,
            param.dtype, param.device
        )
        
        A = A.squeeze(0).view(shape) * signs[0].item()
        B = torch.ones(1, device=param.device, dtype=param.dtype)
        
        return Perturbation(A=A, B=B, member_id=member_id, epoch=epoch, param_name=param_name)
    
    def _batched_forward_impl(
        self,
        model: nn.Module,
        x: torch.Tensor,
        member_ids: torch.Tensor,
        epoch: int,
        population_size: int
    ) -> torch.Tensor:
        """
        Batched forward for OpenES - apply full perturbations.
        
        Less optimized than EGGROLL since we can't fuse the full perturbation.
        """
        outputs = []
        
        for i, member_id in enumerate(member_ids):
            # Apply perturbations to model
            original_params = {}
            for name, param in model.named_parameters():
                if self._should_evolve_param(name, param):
                    original_params[name] = param.data.clone()
                    pert = self._sample_perturbation(param, member_id.item(), epoch, name)
                    param.data = param.data + pert.A
            
            # Forward pass
            out = model(x[i:i+1])
            outputs.append(out)
            
            # Restore parameters
            for name, param in model.named_parameters():
                if name in original_params:
                    param.data = original_params[name]
        
        return torch.cat(outputs, dim=0)
    
    def step(self, fitnesses: torch.Tensor, prenormalized: bool = False) -> Dict[str, Any]:
        """Update parameters based on fitness scores."""
        if self._model is None:
            raise RuntimeError("Strategy not set up. Call setup() first.")
        
        fitnesses = fitnesses.to(self._device)
        population_size = fitnesses.shape[0]
        
        original_fitnesses = fitnesses.clone()
        
        if not prenormalized:
            fitnesses = self._apply_fitness_transform(fitnesses)
        
        self._optimizer.zero_grad()
        
        for name, param in self._model.named_parameters():
            if not self._should_evolve_param(name, param):
                continue
            
            # Generate all perturbations
            all_noise = []
            for i in range(population_size):
                pert = self._sample_perturbation(param, i, self._current_epoch, name)
                all_noise.append(pert.A)
            
            all_noise = torch.stack(all_noise)  # (N, *param_shape)
            
            # ES gradient
            f_expanded = fitnesses.view(population_size, *([1] * param.dim()))
            weighted_grad = (all_noise * f_expanded).mean(dim=0)
            
            param.grad = -weighted_grad
        
        self._optimizer.step()
        self._current_epoch += 1
        
        return {
            'epoch': self._current_epoch - 1,
            'fitness_mean': original_fitnesses.mean().item(),
            'fitness_std': original_fitnesses.std().item(),
            'fitness_max': original_fitnesses.max().item(),
            'fitness_min': original_fitnesses.min().item(),
        }
