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
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterator, List, Tuple, Union, ContextManager, Set, Callable
from dataclasses import dataclass, field

from .perturbation import Perturbation, PerturbationContext
from .module import LowRankLinear


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
        optimizer: Union[str, type] = "sgd",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        fitness_transform: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
        grad_clip: Optional[float] = None,
        max_grad_norm: Optional[float] = None,  # Alias for grad_clip
        freeze_bias: bool = False,
        evolve_bias: bool = True,  # Alias: evolve_bias=False is same as freeze_bias=True
        **kwargs
    ):
        self._sigma = sigma
        self._lr = lr
        self._antithetic = antithetic
        self._noise_reuse = noise_reuse
        self._optimizer_type = optimizer
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._seed = seed or 42
        # Handle both grad_clip and max_grad_norm parameter names
        self._grad_clip = grad_clip or max_grad_norm
        # Handle both freeze_bias=True and evolve_bias=False
        self._freeze_bias = freeze_bias or (not evolve_bias)
        
        # Validate and store fitness transform
        if fitness_transform is not None:
            if isinstance(fitness_transform, str):
                if fitness_transform not in {"rank", "centered_rank"}:
                    raise ValueError(f"Unknown fitness_transform: {fitness_transform}. "
                                   f"Valid string values: 'rank', 'centered_rank'")
            elif not callable(fitness_transform):
                raise TypeError(f"fitness_transform must be a string, callable, or None, "
                              f"got {type(fitness_transform)}")
        self._fitness_transform = fitness_transform
        
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._device: Optional[torch.device] = None
        self._in_perturbation_context: bool = False
        self._last_perturbations: Dict[str, List[Perturbation]] = {}
        self._last_epoch: int = 0
        self._epoch: int = 0
        self._total_steps: int = 0
        self._last_population_size: Optional[int] = None
        self._param_keys: Dict[str, int] = {}  # Maps param name to unique key
        self._included_params: Optional[Set[str]] = None
        self._excluded_params: Set[str] = set()
        self._callbacks: Dict[str, List[Callable]] = {
            'on_step': [],
            'on_perturb': [],
        }
    
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
    
    @property
    def epoch(self) -> int:
        """Current epoch counter."""
        return self._epoch
    
    @property
    def total_steps(self) -> int:
        """Total number of update steps performed."""
        return self._total_steps
    
    @property
    def optimizer(self) -> Optional[torch.optim.Optimizer]:
        """The optimizer used for parameter updates."""
        return self._optimizer
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for a specific event.
        
        Args:
            event: Event name ('on_step', 'on_perturb')
            callback: Callable to invoke when event occurs
        """
        if event not in self._callbacks:
            raise ValueError(f"Unknown event: {event}. Valid events: {list(self._callbacks.keys())}")
        self._callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, **kwargs) -> None:
        """Trigger all callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            # Different events have different expected signatures
            if event == 'on_step':
                # on_step callbacks expect metrics as positional arg
                try:
                    callback(kwargs.get('metrics', {}))
                except TypeError:
                    # Fall back to no args
                    try:
                        callback()
                    except TypeError:
                        # Try with kwargs
                        callback(**kwargs)
            elif event == 'on_perturb':
                # on_perturb callbacks expect (population_size, epoch)
                try:
                    callback(kwargs.get('population_size'), kwargs.get('epoch'))
                except TypeError:
                    try:
                        callback(**kwargs)
                    except TypeError:
                        callback()
            else:
                # Generic: try kwargs first, fall back to no args
                try:
                    callback(**kwargs)
                except TypeError:
                    callback()
    
    def get_antithetic_partner(self, member_id: int) -> int:
        """
        Get the antithetic partner index for a population member.
        
        Args:
            member_id: Population member index
        
        Returns:
            Partner index (member_id ^ 1 for antithetic pairs)
        """
        if not self._antithetic:
            return member_id
        # Even members pair with odd (0↔1, 2↔3, etc.)
        return member_id ^ 1
    
    def is_positive_perturbation(self, member_id: int) -> bool:
        """
        Check if a member uses positive (+ε) or negative (-ε) perturbation.
        
        Args:
            member_id: Population member index
        
        Returns:
            True if positive perturbation, False if negative
        """
        if not self._antithetic:
            return True
        return member_id % 2 == 0

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
        
        # Handle both string and class optimizer types
        if isinstance(self._optimizer_type, str):
            if self._optimizer_type == "sgd":
                self._optimizer = torch.optim.SGD(params, lr=self._lr, **self._optimizer_kwargs)
            elif self._optimizer_type == "adam":
                self._optimizer = torch.optim.Adam(params, lr=self._lr, **self._optimizer_kwargs)
            elif self._optimizer_type == "adamw":
                self._optimizer = torch.optim.AdamW(params, lr=self._lr, **self._optimizer_kwargs)
            else:
                raise ValueError(f"Unknown optimizer: {self._optimizer_type}")
        elif isinstance(self._optimizer_type, type) and issubclass(self._optimizer_type, torch.optim.Optimizer):
            # Custom optimizer class passed
            self._optimizer = self._optimizer_type(params, lr=self._lr, **self._optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {self._optimizer_type}")
    
    def _should_evolve_param(self, name: str, param: torch.Tensor) -> bool:
        """Check if a parameter should be evolved."""
        if not param.requires_grad:
            return False
        
        # Check freeze_bias option
        if self._freeze_bias and 'bias' in name:
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
        
        Raises:
            RuntimeError: If setup() has not been called
            ValueError: If population_size <= 0
        """
        # Validate setup
        if self._model is None:
            raise RuntimeError(
                "Strategy not set up. Call setup(model) before perturb()."
            )
        
        # Validate population size
        if population_size <= 0:
            raise ValueError(f"population_size must be positive, got {population_size}")
        
        # Warn about odd population with antithetic
        if self._antithetic and population_size % 2 != 0:
            warnings.warn(
                f"Using antithetic sampling with odd population_size={population_size}. "
                "The last member will not have a pair. Consider using an even population size.",
                UserWarning
            )
        
        self._trigger_callbacks('on_perturb', population_size=population_size, epoch=epoch)
        self._last_population_size = population_size
        
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
        Normalize fitness scores based on the configured transform.
        
        If fitness_transform is:
        - None: Standard normalization (zero mean, unit variance)
        - "rank": Rank-based transformation
        - "centered_rank": Centered rank transformation (symmetric around 0)
        - callable: Apply the custom function directly (no further normalization)
        
        Args:
            fitnesses: Raw fitness scores, shape (population_size,)
        
        Returns:
            Transformed/normalized fitness scores
        """
        # Handle string transforms
        if isinstance(self._fitness_transform, str):
            if self._fitness_transform == "rank":
                # Rank transformation: convert to ranks, then normalize
                ranks = torch.argsort(torch.argsort(fitnesses)).float()
                # Normalize ranks to [0, 1] then center
                n = len(fitnesses)
                normalized_ranks = ranks / (n - 1) if n > 1 else torch.zeros_like(ranks)
                return normalized_ranks - normalized_ranks.mean()
            elif self._fitness_transform == "centered_rank":
                # Centered rank: ranks centered around 0, bounded
                ranks = torch.argsort(torch.argsort(fitnesses)).float()
                n = len(fitnesses)
                # Map to [-0.5, 0.5] range
                centered = (ranks - (n - 1) / 2) / n
                return centered
        
        # Handle custom callable - apply directly, no further normalization
        if callable(self._fitness_transform):
            return self._fitness_transform(fitnesses)
        
        # Default: standard normalization (zero mean, unit variance)
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
            'optimizer_type': self._optimizer_type if isinstance(self._optimizer_type, str) else str(self._optimizer_type),
            'optimizer_kwargs': self._optimizer_kwargs,
            'seed': self._seed,
            'param_keys': self._param_keys,
            'epoch': self._epoch,
            'total_steps': self._total_steps,
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
        self._epoch = state.get('epoch', self._epoch)
        self._total_steps = state.get('total_steps', self._total_steps)
        
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
    ) -> Tuple[torch.Generator, int]:
        """
        Get a deterministic generator and seed for a specific perturbation.
        
        The generator is seeded based on:
        - Base seed
        - Epoch (considering noise_reuse)
        - Member ID (considering antithetic pairing)
        - Parameter key
        
        Returns:
            Tuple of (CPU generator, seed) for deterministic noise generation.
            We use CPU generator to avoid CUDA generator issues.
        """
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
        
        seed = combined_seed % (2**31 - 1)
        gen = torch.Generator()  # CPU generator
        gen.manual_seed(seed)
        return gen, seed
    
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
        
        # Get deterministic generator (on CPU)
        param_key = self._param_keys.get(param_name, hash(param_name) % 10000)
        gen, _ = self._get_generator_for_perturbation(member_id, epoch, param_key)
        
        # Generate low-rank factors on CPU then move to device
        # Combined tensor for both A and B, then split
        combined = torch.randn(
            (m + n, r), 
            dtype=param.dtype,
            generator=gen
        ).to(param.device)
        
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
                
                # Find the parameter name for this weight
                param_name = None
                for n, p in model.named_parameters():
                    if p is W:
                        param_name = n
                        break
                
                if param_name is None:
                    # Fallback for modules not in direct named_parameters
                    param_name = name + ".weight"
                
                # Check if this parameter should be evolved
                if not self._should_evolve_param(param_name, W):
                    # Skip perturbation for frozen/excluded parameters
                    current_input = base_output
                    continue
                
                # Compute perturbation output for each batch element
                # We need to apply different perturbations based on member_ids
                
                # Get all unique perturbations needed
                unique_members = torch.unique(member_ids)
                
                # Pre-compute perturbation factors for all unique members
                perturbations = {}
                
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
            
            elif isinstance(module, LowRankLinear):
                # LowRankLinear: W = U @ V.T, forward is x @ V @ U.T + bias
                U = module.U  # (out_features, rank)
                V = module.V  # (in_features, rank)
                bias = module.bias
                
                # Compute base output: x @ V @ U.T
                base_output = (current_input @ V) @ U.T
                if bias is not None:
                    base_output = base_output + bias
                
                # For now, LowRankLinear doesn't get additional perturbations
                # (it's already low-rank by design)
                # TODO: Could add perturbations to U and V factors
                current_input = base_output
                
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
    
    def step(self, fitnesses: torch.Tensor, prenormalized: bool = False) -> Dict[str, Any]:
        """
        Update parameters based on fitness scores.
        
        Computes ES gradient estimate:
            ∇̂ = (1/N) Σᵢ f_normalized[i] * perturbation[i]
        
        For low-rank perturbations:
            ∇̂ = (1/N) Σᵢ f_normalized[i] * A[i] @ B[i].T
        
        Args:
            fitnesses: Fitness scores for each population member.
            prenormalized: If True, skip internal normalization (fitnesses are already normalized).
        """
        if self._model is None:
            raise RuntimeError("Strategy not set up. Call setup() first.")
        
        fitnesses = fitnesses.to(self._device)
        population_size = fitnesses.shape[0]
        
        # Check fitness size matches expected population size
        if self._last_population_size is not None and population_size != self._last_population_size:
            raise ValueError(
                f"Fitness size {population_size} does not match expected population size {self._last_population_size}"
            )
        
        # Store original fitnesses for metrics before any transforms
        original_fitnesses = fitnesses.clone()
        
        # Apply custom fitness transform if provided
        if self._fitness_transform is not None:
            fitnesses = self._fitness_transform(fitnesses)
        
        # Normalize fitnesses (skip if prenormalized)
        if prenormalized:
            normalized = fitnesses
        else:
            normalized = self.normalize_fitnesses(fitnesses)
        
        # Compute gradients for each parameter
        metrics = {}
        total_grad_norm = 0.0
        total_param_delta = 0.0
        
        # Store gradients
        gradients = {}
        
        for name, param in self._model.named_parameters():
            if not self._should_evolve_param(name, param):
                continue
            
            # Accumulate weighted perturbations
            if param.dim() >= 2:
                # Matrix parameters - use low-rank accumulation
                m, n = param.shape[0], param.shape[1]
                grad_accum = torch.zeros_like(param)
                
                for i in range(population_size):
                    pert = self._sample_perturbation(param, i, self._epoch, name)
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
                    pert = self._sample_perturbation(param, i, self._epoch, name)
                    A, B = pert.factors
                    weight = normalized[i].item()
                    grad_accum += weight * A * B
                
                grad = -grad_accum / population_size * math.sqrt(population_size)
            
            gradients[name] = grad
            total_grad_norm += grad.norm().item() ** 2
        
        total_grad_norm = math.sqrt(total_grad_norm)
        
        # Apply gradient clipping if configured
        if self._grad_clip is not None and total_grad_norm > self._grad_clip:
            clip_coef = self._grad_clip / (total_grad_norm + 1e-8)
            for name in gradients:
                gradients[name] = gradients[name] * clip_coef
            total_grad_norm = self._grad_clip
        
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
        
        self._epoch += 1
        self._total_steps += 1
        self._last_epoch = self._epoch
        self._last_population_size = population_size
        
        metrics['grad_norm'] = total_grad_norm
        metrics['param_delta'] = total_param_delta
        metrics['param_delta_norm'] = total_param_delta
        metrics['fitness_mean'] = original_fitnesses.mean().item()
        metrics['fitness_std'] = original_fitnesses.std().item()
        metrics['fitness_max'] = original_fitnesses.max().item()
        metrics['fitness_min'] = original_fitnesses.min().item()
        
        self._trigger_callbacks('on_step', metrics=metrics, epoch=self._epoch)
        
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
    ) -> Tuple[torch.Generator, int]:
        """Get deterministic generator and seed for a specific perturbation."""
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
        
        seed = combined_seed % (2**31 - 1)
        gen = torch.Generator()  # CPU generator
        gen.manual_seed(seed)
        return gen, seed
    
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
        gen, _ = self._get_generator_for_perturbation(member_id, epoch, param_key)
        
        # Generate full perturbation on CPU then move to device
        noise = torch.randn(
            param.shape,
            dtype=param.dtype,
            generator=gen
        ).to(param.device)
        
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
            return Perturbation(A=A, B=B, member_id=member_id, epoch=epoch, param_name=param_name)
        
        # For 2D+, store full noise with identity
        m, n = param.shape[0], param.shape[1]
        A = noise  # (m, n)
        B = torch.eye(n, device=param.device, dtype=param.dtype)  # (n, n)
        
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
                
                # Find the parameter name for this weight
                param_name = None
                for n, p in model.named_parameters():
                    if p is W:
                        param_name = n
                        break
                if param_name is None:
                    param_name = name + ".weight"
                
                # Check if this parameter should be evolved
                if not self._should_evolve_param(param_name, W):
                    # Skip perturbation for frozen/excluded parameters
                    current_input = base_output
                    continue
                
                unique_members = torch.unique(member_ids)
                
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
            
            elif isinstance(module, LowRankLinear):
                # LowRankLinear: W = U @ V.T, forward is x @ V @ U.T + bias
                U = module.U
                V = module.V
                bias = module.bias
                
                base_output = (current_input @ V) @ U.T
                if bias is not None:
                    base_output = base_output + bias
                
                current_input = base_output
                
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
    
    def step(self, fitnesses: torch.Tensor, prenormalized: bool = False) -> Dict[str, Any]:
        """Update parameters based on fitness scores.
        
        Args:
            fitnesses: Fitness scores for each population member.
            prenormalized: If True, skip internal normalization (fitnesses are already normalized).
        """
        if self._model is None:
            raise RuntimeError("Strategy not set up. Call setup() first.")
        
        fitnesses = fitnesses.to(self._device)
        population_size = fitnesses.shape[0]
        
        # Check fitness size matches expected population size
        if self._last_population_size is not None and population_size != self._last_population_size:
            raise ValueError(
                f"Fitness size {population_size} does not match expected population size {self._last_population_size}"
            )
        
        # Store original fitnesses for metrics before any transforms
        original_fitnesses = fitnesses.clone()
        
        # Apply custom fitness transform if provided
        if self._fitness_transform is not None:
            fitnesses = self._fitness_transform(fitnesses)
        
        # Normalize fitnesses (skip if prenormalized)
        if prenormalized:
            normalized = fitnesses
        else:
            normalized = self.normalize_fitnesses(fitnesses)
        
        metrics = {}
        total_grad_norm = 0.0
        total_param_delta = 0.0
        
        gradients = {}
        
        for name, param in self._model.named_parameters():
            if not self._should_evolve_param(name, param):
                continue
            
            grad_accum = torch.zeros_like(param)
            
            for i in range(population_size):
                pert = self._sample_perturbation(param, i, self._epoch, name)
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
        
        # Apply gradient clipping if configured
        if self._grad_clip is not None and total_grad_norm > self._grad_clip:
            clip_coef = self._grad_clip / (total_grad_norm + 1e-8)
            for name in gradients:
                gradients[name] = gradients[name] * clip_coef
            total_grad_norm = self._grad_clip
        
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
        
        self._epoch += 1
        self._total_steps += 1
        self._last_epoch = self._epoch
        self._last_population_size = population_size
        
        metrics['grad_norm'] = total_grad_norm
        metrics['param_delta'] = total_param_delta
        metrics['param_delta_norm'] = total_param_delta
        metrics['fitness_mean'] = original_fitnesses.mean().item()
        metrics['fitness_std'] = original_fitnesses.std().item()
        metrics['fitness_max'] = original_fitnesses.max().item()
        metrics['fitness_min'] = original_fitnesses.min().item()
        
        self._trigger_callbacks('on_step', metrics=metrics, epoch=self._epoch)
        
        return metrics
