"""
EGGROLL Core - PyTorch implementation of low-rank Evolution Strategies.

CORE CONCEPT:
    perturbed_linear(x, W, b, A, B) = x @ W.T + b + x @ B @ A.T
    
    This is a low-rank perturbation of the weight matrix, enabling efficient
    ES gradient estimation with O(r(m+n)) memory instead of O(m×n) per layer.

This module contains the core EGGROLL primitives and Dict-based API.
For experiments and examples, see the README.
"""

import torch
import math
from dataclasses import dataclass, field


__all__ = [
    # Configuration
    "EggrollConfig",
    # Dict-Based API
    "get_params_dict",
    "get_weight_shapes",
    "generate_perturbations",
    "compute_gradients",
    "update_params",
    "eggroll_step",
    "perturbed_forward",
    # Forward generation
    "make_perturbed_forward_fn",
    # Raw Primitives API
    "generate_lowrank_perturbations",
    "perturbed_linear",
    "apply_lowrank_perturbation",
    "compute_weight_perturbation",
    "compute_es_gradient",
    "normalize_fitnesses",
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EggrollConfig:
    """Configuration for EGGROLL training."""
    population_size: int = 2048
    rank: int = 4
    sigma: float = 0.1
    lr: float = 0.1
    lr_decay: float = 1.0
    sigma_decay: float = 0.999
    max_epochs: int = 100
    batch_size: int = 256
    seed: int = 42
    dtype: torch.dtype = field(default=torch.float32)


# =============================================================================
# Dict-Based API
# =============================================================================

#########
# SETUP #
#########
def get_params_dict(
    module: torch.nn.Module,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """
    Extract parameters from a PyTorch module into EGGROLL's dict format.
    
    Copies all parameters to the specified device and optionally casts dtype.
    Parameter names follow PyTorch's naming convention (e.g., '0.weight', 'fc.bias').
    
    Args:
        module: PyTorch module to extract parameters from
        device: Device to place parameters on (default: "cuda")
        dtype: Optional dtype to cast parameters to. If None, keeps original dtype.
        
    Returns:
        Dict mapping parameter names to tensors (detached, on specified device)
        
    Example:
        model = nn.Sequential(
            nn.Linear(4, 256),
            nn.Linear(256, 2),
        )
        params = get_params_dict(model)
        # {'0.weight': tensor(...), '0.bias': tensor(...), 
        #  '1.weight': tensor(...), '1.bias': tensor(...)}
        
        # Or with named modules:
        model = nn.Sequential(OrderedDict([
            ('hidden', nn.Linear(4, 256)),
            ('output', nn.Linear(256, 2)),
        ]))
        params = get_params_dict(model)
        # {'hidden.weight': ..., 'hidden.bias': ..., 'output.weight': ..., ...}
    """
    params = {}
    for name, param in module.named_parameters():
        tensor = param.detach().to(device)
        if dtype is not None:
            tensor = tensor.to(dtype)
        params[name] = tensor
    return params


def get_weight_shapes(params: dict[str, torch.Tensor]) -> dict[str, tuple[int, int]]:
    """
    Extract shapes of weight tensors for perturbation generation.
    
    Supports:
    - 2D weights (Linear): (out_dim, in_dim)
    - 4D weights (Conv2d): (out_channels, in_channels*k*k) - flattened
    
    Args:
        params: Dict of model parameters
        
    Returns:
        Dict mapping weight names to (out_dim, in_dim) shapes
        
    Example:
        params = {'fc.weight': W_fc, 'conv.weight': W_conv, ...}
        shapes = get_weight_shapes(params)
        # {'fc.weight': (10, 256), 'conv.weight': (32, 144)}
    """
    shapes: dict[str, tuple[int, int]] = {}
    for name, tensor in params.items():
        if not isinstance(tensor, torch.Tensor) or 'weight' not in name:
            continue
        if tensor.dim() == 2:
            shapes[name] = (tensor.shape[0], tensor.shape[1])
        elif tensor.dim() == 4:
            out_ch, in_ch, k1, k2 = tensor.shape
            shapes[name] = (out_ch, in_ch * k1 * k2)
    return shapes

##########
# UPDATE #
##########
def generate_perturbations(
    shapes: dict[str, tuple[int, int]],
    population_size: int,
    rank: int,
    sigma: float,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate perturbations for all weight matrices.
    
    Args:
        shapes: Dict from get_weight_shapes()
        population_size: Number of population members
        rank: Low-rank perturbation rank
        sigma: Perturbation scale
        generator: Torch random generator
        dtype: Data type
        
    Returns:
        Dict mapping weight names to (A_scaled, B) tuples
    """
    perts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, (out_dim, in_dim) in shapes.items():
        A_scaled, _, B = generate_lowrank_perturbations(
            population_size, out_dim, in_dim, rank, sigma, generator, dtype
        )
        perts[name] = (A_scaled, B)
    return perts


def compute_gradients(
    fitnesses: torch.Tensor,
    perts: dict[str, tuple[torch.Tensor, torch.Tensor]],
    population_size: int,
) -> dict[str, torch.Tensor]:
    """
    Compute ES gradients for all weight matrices.
    
    Args:
        fitnesses: (population,) normalized fitness scores
        perts: Dict from generate_perturbations()
        population_size: Number of population members
        
    Returns:
        Dict mapping weight names to gradient tensors
    """
    grads: dict[str, torch.Tensor] = {}
    for name, (A_scaled, B) in perts.items():
        grads[name] = compute_es_gradient(fitnesses, A_scaled, B, population_size)
    return grads


def update_params(
    params: dict[str, torch.Tensor],
    grads: dict[str, torch.Tensor],
    lr: float,
) -> dict[str, torch.Tensor]:
    """
    Update parameters with gradients.
    
    Args:
        params: Dict of model parameters
        grads: Dict from compute_gradients()
        lr: Learning rate
        
    Returns:
        Updated params dict (in-place modification)
    """
    for name, grad in grads.items():
        param = params[name]
        if grad.shape != param.shape:
            grad = grad.reshape(param.shape)
        params[name] = param + lr * grad
    return params


def eggroll_step(
    params: dict[str, torch.Tensor],
    fitnesses: torch.Tensor,
    perts: dict[str, tuple[torch.Tensor, torch.Tensor]],
    lr: float,
    sigma: float,
    config: EggrollConfig | None = None,
) -> tuple[float, float]:
    """
    Perform one EGGROLL gradient step: normalize -> compute gradients -> update params.
    
    This is the recommended high-level API for the EGGROLL training loop.
    It combines normalize_fitnesses, compute_gradients, and update_params into
    a single call, and applies learning rate and sigma decay from config.
    
    Args:
        params: Parameter dict to update in-place
        fitnesses: Raw fitness values (will be normalized internally)
        perts: Perturbations dict from generate_perturbations()
        lr: Current learning rate
        sigma: Current sigma (noise scale)
        config: EggrollConfig with lr_decay and sigma_decay (optional, default no decay)
        
    Returns:
        (new_lr, new_sigma) tuple with decayed values
        
    Example:
        config = EggrollConfig(population_size=2048, rank=4, sigma=0.1, lr=0.1,
                               lr_decay=0.999, sigma_decay=0.999)
        current_lr, current_sigma = config.lr, config.sigma
        
        for epoch in range(max_epochs):
            perts = generate_perturbations(shapes, config.population_size, config.rank, 
                                           current_sigma, gen, config.dtype)
            logits = forward(x, params, perts)
            fitnesses = compute_fitness(logits)
            current_lr, current_sigma = eggroll_step(
                params, fitnesses, perts, current_lr, current_sigma, config
            )
    """
    # Infer population size from perturbations
    first_pert = next(iter(perts.values()))
    population_size = first_pert[0].shape[0]
    
    # Core EGGROLL update
    normalized = normalize_fitnesses(fitnesses)
    grads = compute_gradients(normalized, perts, population_size)
    update_params(params, grads, lr)
    
    # Apply decay from config (default: no decay)
    lr_decay = config.lr_decay if config is not None else 1.0
    sigma_decay = config.sigma_decay if config is not None else 1.0
    return lr * lr_decay, sigma * sigma_decay

###########
# FORWARD #
###########

def perturbed_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    perts: dict[str, tuple[torch.Tensor, torch.Tensor]],
    weight_name: str,
) -> torch.Tensor:
    """
    Apply perturbed linear using perturbations from dict.
    
    Args:
        x: Input tensor (pop, *batch, in_dim)
        weight: Weight matrix (out_dim, in_dim)
        bias: Bias vector (out_dim,) or None
        perts: Perturbations dict from generate_perturbations()
        weight_name: Key in perts dict (e.g., 'layer1.weight')
        
    Returns:
        Output tensor (pop, *batch, out_dim)
    """
    if weight_name in perts:
        A_scaled, B = perts[weight_name]
        return perturbed_linear(x, weight, bias, A_scaled, B)
    else:
        return x @ weight.T + (bias if bias is not None else 0)


def make_perturbed_forward_fn(
    module: torch.nn.Module,
    example_input: torch.Tensor | None = None,
) -> tuple[callable, callable]:
    """
    Auto-generate perturbed forward and eval forward functions from a PyTorch module.
    
    Uses torch.fx to trace the module and replace nn.Linear layers with perturbed_forward.
    Works for Sequential models and simple feedforward architectures. For complex models
    with dynamic control flow, you may need to write the forward functions manually.
    
    Args:
        module: PyTorch module to trace. Must be traceable by torch.fx.
        example_input: Optional example input for tracing. Required for some modules.
        
    Returns:
        (forward_fn, forward_eval_fn) tuple:
        - forward_fn(x, params, perts) -> output with perturbations applied
        - forward_eval_fn(x, params) -> output without perturbations (for inference)
        
    Example:
        model = nn.Sequential(
            nn.Linear(4, 256),
            nn.Tanh(),
            nn.Linear(256, 2),
        )
        forward, forward_eval = make_perturbed_forward_fn(model)
        
        # Training
        params = get_params_dict(model)
        shapes = get_weight_shapes(params)
        perts = generate_perturbations(shapes, pop, rank, sigma, gen, dtype)
        output = forward(x, params, perts)
        
        # Inference  
        output = forward_eval(x, params)
        
    Limitations:
        - Only supports nn.Linear layers (not Conv2d - use perturbed_conv2d manually)
        - Module must be traceable by torch.fx (no dynamic control flow)
        - Activation functions must be in torch.nn or torch.nn.functional
    """
    import torch.fx as fx
    
    # Get the structure: list of (name, module_type, weight_name, bias_name)
    linear_layers = []
    for name, submodule in module.named_modules():
        if isinstance(submodule, torch.nn.Linear):
            weight_name = f"{name}.weight" if name else "weight"
            bias_name = f"{name}.bias" if name else "bias"
            has_bias = submodule.bias is not None
            linear_layers.append((name, weight_name, bias_name, has_bias))
    
    # Trace the module to get computation graph
    try:
        traced = fx.symbolic_trace(module)
    except Exception as e:
        raise RuntimeError(
            f"Failed to trace module with torch.fx: {e}\n"
            "For modules with dynamic control flow, write forward functions manually."
        )
    
    # Build the forward functions by interpreting the traced graph
    def forward_fn(x: torch.Tensor, params: dict, perts: dict) -> torch.Tensor:
        """Forward with ES perturbations applied."""
        env = {'x': x}
        
        for node in traced.graph.nodes:
            if node.op == 'placeholder':
                # Input node - already in env as 'x'
                env[node.name] = x
            elif node.op == 'get_attr':
                # Skip - we use params dict instead
                pass
            elif node.op == 'call_module':
                # Get the submodule
                submodule = traced.get_submodule(node.target)
                inp = env[node.args[0].name]
                
                if isinstance(submodule, torch.nn.Linear):
                    # Replace with perturbed_forward
                    weight_name = f"{node.target}.weight"
                    bias_name = f"{node.target}.bias"
                    weight = params[weight_name]
                    bias = params.get(bias_name)
                    env[node.name] = perturbed_forward(inp, weight, bias, perts, weight_name)
                else:
                    # Non-linear modules (activations, etc.) - apply directly
                    env[node.name] = submodule(inp)
            elif node.op == 'call_function':
                # Functions like torch.relu, F.tanh, etc.
                args = tuple(env.get(a.name, a) if isinstance(a, fx.Node) else a for a in node.args)
                kwargs = {k: env.get(v.name, v) if isinstance(v, fx.Node) else v for k, v in node.kwargs.items()}
                env[node.name] = node.target(*args, **kwargs)
            elif node.op == 'call_method':
                # Methods like .view(), .reshape(), etc.
                self_arg = env[node.args[0].name]
                args = tuple(env.get(a.name, a) if isinstance(a, fx.Node) else a for a in node.args[1:])
                kwargs = {k: env.get(v.name, v) if isinstance(v, fx.Node) else v for k, v in node.kwargs.items()}
                env[node.name] = getattr(self_arg, node.target)(*args, **kwargs)
            elif node.op == 'output':
                # Return node
                if isinstance(node.args[0], fx.Node):
                    return env[node.args[0].name]
                return node.args[0]
        
        raise RuntimeError("No output node found in traced graph")
    
    def forward_eval_fn(x: torch.Tensor, params: dict) -> torch.Tensor:
        """Forward without perturbations (for inference/evaluation)."""
        env = {'x': x}
        
        for node in traced.graph.nodes:
            if node.op == 'placeholder':
                env[node.name] = x
            elif node.op == 'get_attr':
                pass
            elif node.op == 'call_module':
                submodule = traced.get_submodule(node.target)
                inp = env[node.args[0].name]
                
                if isinstance(submodule, torch.nn.Linear):
                    weight_name = f"{node.target}.weight"
                    bias_name = f"{node.target}.bias"
                    weight = params[weight_name]
                    bias = params.get(bias_name)
                    # Standard linear: x @ W.T + b
                    out = inp @ weight.T
                    if bias is not None:
                        out = out + bias
                    env[node.name] = out
                else:
                    env[node.name] = submodule(inp)
            elif node.op == 'call_function':
                args = tuple(env.get(a.name, a) if isinstance(a, fx.Node) else a for a in node.args)
                kwargs = {k: env.get(v.name, v) if isinstance(v, fx.Node) else v for k, v in node.kwargs.items()}
                env[node.name] = node.target(*args, **kwargs)
            elif node.op == 'call_method':
                self_arg = env[node.args[0].name]
                args = tuple(env.get(a.name, a) if isinstance(a, fx.Node) else a for a in node.args[1:])
                kwargs = {k: env.get(v.name, v) if isinstance(v, fx.Node) else v for k, v in node.kwargs.items()}
                env[node.name] = getattr(self_arg, node.target)(*args, **kwargs)
            elif node.op == 'output':
                if isinstance(node.args[0], fx.Node):
                    return env[node.args[0].name]
                return node.args[0]
        
        raise RuntimeError("No output node found in traced graph")
    
    return forward_fn, forward_eval_fn


# =============================================================================
# Low-Rank Primitives (@torch.compile for zero overhead)
# =============================================================================

def generate_lowrank_perturbations(
    population_size: int,
    out_dim: int,
    in_dim: int,
    rank: int,
    sigma: float,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate antithetic low-rank perturbations for EGGROLL on CUDA.
    
    Memory: O(r(m+n)) per layer where m=out_dim, n=in_dim, r=rank
    
    Returns:
        A_scaled: (population_size, out_dim, rank) - scaled A factors
        A: (population_size, out_dim, rank) - unscaled A factors
        B: (population_size, in_dim, rank) - B factors (unscaled)
        
    The perturbation to weight W is: A_scaled @ B.T (low-rank approximation)
    Antithetic: first half and second half are negatives of each other.
    """
    half_pop = population_size // 2
    scale = sigma / (rank ** 0.5)
    
    # Generate positive half
    A_pos = torch.randn(half_pop, out_dim, rank, device="cuda", dtype=dtype, generator=generator)
    B_pos = torch.randn(half_pop, in_dim, rank, device="cuda", dtype=dtype, generator=generator)
    
    # Antithetic pairs: negate A, keep B
    A = torch.cat([A_pos, -A_pos], dim=0)
    B = torch.cat([B_pos, B_pos], dim=0)
    
    # Scale A by sigma/sqrt(rank)
    A_scaled = A * scale
    
    return A_scaled, A, B


@torch.compile
def apply_lowrank_perturbation(x, B, A_scaled):
    """
    Apply low-rank perturbation to input tensor.
    
    Computes: x @ B @ A.T  (never materializes m×n matrix)
    
    Supports any number of batch dimensions:
        - (pop, in_dim) -> (pop, out_dim)
        - (pop, batch, in_dim) -> (pop, batch, out_dim)
        - (pop, b1, b2, in_dim) -> (pop, b1, b2, out_dim)
    
    Args:
        x: (population, *batch_dims, in_dim) - input tensor
        B: (population, in_dim, rank) - B factors
        A_scaled: (population, out_dim, rank) - scaled A factors
        
    Returns:
        perturbation: (population, *batch_dims, out_dim)
    """
    pop_size = x.shape[0]
    in_dim = x.shape[-1]
    out_dim = A_scaled.shape[1]
    batch_shape = x.shape[1:-1]
    
    x_flat = x.reshape(pop_size, -1, in_dim)
    pert_flat = torch.einsum('pbi,pir,pjr->pbj', x_flat, B, A_scaled)
    return pert_flat.reshape(pop_size, *batch_shape, out_dim)


@torch.compile
def perturbed_linear(x, W, b, A_scaled, B):
    """
    Perturbed linear layer: base linear + low-rank perturbation.
    
    Computes: x @ W.T + b + x @ B @ A.T
    
    THIS IS THE FUNDAMENTAL EGGROLL PRIMITIVE.
    
    The key insight from JAX EGGROLL: we compute x @ B @ A.T (two rank-r matmuls)
    instead of x @ (A @ B.T), which would materialize the full m×n perturbation.
    
    Args:
        x: (population, *batch_dims, in_dim) - input tensor
        W: (out_dim, in_dim) - weight matrix (shared across population)
        b: (out_dim,) - bias vector (shared across population)
        A_scaled: (population, out_dim, rank) - scaled A perturbation factors
        B: (population, in_dim, rank) - B perturbation factors
        
    Returns:
        output: (population, *batch_dims, out_dim)
    """
    base = x @ W.T + b
    pert = apply_lowrank_perturbation(x, B, A_scaled)
    return base + pert


@torch.compile
def normalize_fitnesses(fitnesses, eps=1e-8):
    """
    Normalize fitness scores to zero mean, unit variance.
    
    Args:
        fitnesses: (population,) - raw fitness scores
        eps: small constant for numerical stability
        
    Returns:
        normalized: (population,) - normalized fitness scores
    """
    return (fitnesses - fitnesses.mean()) / (fitnesses.std() + eps)


@torch.compile
def compute_es_gradient(fitnesses, A_scaled, B, population_size):
    """
    Compute ES gradient from fitnesses and perturbation factors.
    
    Args:
        fitnesses: (population,) - normalized fitness scores
        A_scaled: (population, out_dim, rank) - scaled A factors
        B: (population, in_dim, rank) - B factors
        population_size: int
        
    Returns:
        gradient: (out_dim, in_dim) - gradient estimate for weight matrix
    """
    sqrt_N = math.sqrt(population_size)
    f = fitnesses[:, None, None]
    return torch.einsum('nir,njr->ij', f * A_scaled, B) / sqrt_N
