# PyTorch EGGROLL Test Suite

> **Test-Driven Development** for the PyTorch port of EGGROLL — a cleaner, more intuitive API inspired by Keras, PyTorch Lightning, and EvoTorch.

This test suite defines the **target API** for the PyTorch implementation. Tests are written first to guide implementation.

---

## ⚠️ CRITICAL IMPLEMENTATION REQUIREMENTS ⚠️

### Understanding the Core JAX Algorithm

The reference EGGROLL implementation in JAX has two key algorithmic insights that **MUST** be preserved in the PyTorch port:

#### 1. Low-Rank Perturbations

Instead of materializing full perturbation matrices, EGGROLL keeps perturbations in factored form:

```python
# JAX reference (from noiser/eggroll.py):
def get_lora_update_params(frozen_noiser_params, base_sigma, iterinfo, param, key):
    a, b = param.shape
    lora_params = jax.random.normal(key, (a+b, rank), dtype=param.dtype)
    B = lora_params[:b]  # b x r
    A = lora_params[b:]  # a x r
    return A * sigma, B  # Never compute A @ B.T explicitly!
```

The forward pass computes `x @ W.T + x @ B @ A.T` — same result as `x @ (W + A @ B.T).T`, but with **O(r × (m + n))** memory instead of **O(m × n)**.

#### 2. On-the-Fly PRNG with vmap

This is the key to EGGROLL's parallelism. The JAX implementation uses **on-the-fly, deterministic PRNG** via `jax.random.fold_in`:

```python
# JAX reference: noise is generated PER-THREAD, ON-THE-FLY
key = jax.random.fold_in(jax.random.fold_in(base_key, epoch), thread_id)
noise = jax.random.normal(key, shape)

# Then vmapped across all population members:
A, B = jax.vmap(get_lora_update_params, in_axes=(None, 0, None, None))(
    sigma, iterinfo, param, key
)
```

**Why this matters:** Each thread/population-member generates its own noise independently. There's no central noise buffer. This means:
- Memory usage is **O(rank)** per thread, not **O(population × rank)**
- Noise generation is **embarrassingly parallel** — every GPU thread can compute its own noise
- The same noise can be **regenerated** during the update step without storing it

### PyTorch Implementation Requirements

The PyTorch implementation **MUST** replicate this pattern. Standard PyTorch operations won't cut it because `torch.randn()` is sequential and uses a global RNG state.

**Required approach — choose ONE:**

#### Option A: Triton Kernels (STRONGLY PREFERRED)

Write custom Triton kernels that:
1. Accept a base seed + (epoch, member_id) as inputs
2. Use a fast, parallelizable PRNG (Philox, Threefry, or similar) **per-thread**
3. Generate low-rank factors A and B on-the-fly during the forward pass
4. Regenerate the same noise during the backward/update pass using the same seed

```python
# Pseudocode for the Triton kernel pattern:
@triton.jit
def perturbed_matmul_kernel(
    x_ptr, W_ptr, out_ptr,
    seed, epoch, member_id,  # Deterministic noise params
    rank, sigma,
    ...
):
    # Each thread computes its portion of:
    # out = x @ W.T + x @ B @ A.T
    # where A, B are generated on-the-fly from (seed, epoch, member_id)
    
    pid = tl.program_id(0)
    
    # Generate noise for this thread's portion of A and B
    # using Philox PRNG seeded with (seed ^ epoch ^ member_id ^ pid)
    local_key = philox_seed(seed, epoch, member_id, pid)
    noise = philox_randn(local_key, ...)
    
    # Compute perturbed output
    ...
```

#### Option B: Custom CUDA C++ Kernels (Fallback)

If Triton doesn't provide enough control, write CUDA C++ kernels with the same pattern:
- Per-thread PRNG state (cuRAND's Philox works well)
- On-the-fly noise generation
- No global noise buffers

### What NOT To Do

❌ **DO NOT** pre-generate all noise into a tensor:
```python
# BAD: This defeats the purpose of EGGROLL
noise = torch.randn(population_size, param.numel())  # O(pop × params) memory!
```

❌ **DO NOT** use `torch.Generator` with sequential sampling:
```python
# BAD: Sequential, can't parallelize across population
for member in range(population_size):
    noise[member] = torch.randn(shape, generator=gen)
```

❌ **DO NOT** materialize the full perturbation matrix:
```python
# BAD: O(m × n) memory per population member
perturbed_W = W + A @ B.T  
```

### Implementation Checklist

Before considering the PyTorch port complete, verify:

- [ ] Noise is generated **per-thread** using a parallelizable PRNG (Philox/Threefry)
- [ ] The same (seed, epoch, member_id) always produces the same noise (determinism)
- [ ] Memory scales as **O(rank × (m + n))**, not O(m × n) per population member
- [ ] Forward pass computes `x @ W.T + x @ B @ A.T`, never `x @ (W + A @ B.T).T`
- [ ] Update step can regenerate noise without storing it
- [ ] Batched forward evaluates **all population members in one kernel launch**

---

## Quick Start

### Supervised Learning (with DataLoader)

Evolve a model to minimize loss on a dataset:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from hyperscalees.torch import EggrollStrategy

# Some dummy data
X = torch.randn(1000, 8, device='cuda')
y = torch.randn(1000, 2, device='cuda')
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

# Your model
model = nn.Sequential(
    nn.Linear(8, 32), nn.ReLU(),
    nn.Linear(32, 2)
).cuda()

# Set up EGGROLL
strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=4)
strategy.setup(model)

population_size = 32

for epoch in range(100):
    for x_batch, y_batch in loader:
        # Stack the batch for each population member: (pop_size, batch_size, features)
        # Then reshape to (pop_size * batch_size, features) for batched_forward
        x_expanded = x_batch.unsqueeze(0).expand(population_size, -1, -1)
        x_flat = x_expanded.reshape(-1, x_batch.shape[-1])
        
        # Map each row to its population member
        member_ids = torch.arange(population_size, device='cuda').repeat_interleave(len(x_batch))
        
        with strategy.perturb(population_size=population_size, epoch=epoch) as pop:
            # One call evaluates all 32 perturbed models on all 64 samples
            preds_flat = pop.batched_forward(model, x_flat, member_ids=member_ids)
            preds = preds_flat.reshape(population_size, len(x_batch), -1)
            
            # Compute fitness (negative loss) for each population member
            y_expanded = y_batch.unsqueeze(0).expand(population_size, -1, -1)
            losses = ((preds - y_expanded) ** 2).mean(dim=(1, 2))  # MSE per member
            fitnesses = -losses  # Higher is better
        
        strategy.step(fitnesses)
    
    print(f"Epoch {epoch}: best_fitness={fitnesses.max():.4f}")
```

### Reinforcement Learning (with Gym)

Evolve a policy to maximize episode returns:

```python
import gymnasium as gym
import torch
import torch.nn as nn
from hyperscalees.torch import EggrollStrategy

# Vectorized environment: one env per population member
population_size = 64
envs = gym.vector.make("CartPole-v1", num_envs=population_size)

# Policy network
policy = nn.Sequential(
    nn.Linear(4, 32), nn.Tanh(),
    nn.Linear(32, 2)
).cuda()

strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=4)
strategy.setup(policy)

for epoch in range(100):
    obs, _ = envs.reset()
    episode_returns = torch.zeros(population_size, device='cuda')
    dones = torch.zeros(population_size, dtype=torch.bool, device='cuda')
    
    with strategy.perturb(population_size=population_size, epoch=epoch) as pop:
        while not dones.all():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device='cuda')
            
            with torch.no_grad():
                # All 64 policies evaluated in one call
                action_logits = pop.batched_forward(policy, obs_tensor)
            
            actions = action_logits.argmax(dim=-1).cpu().numpy()
            obs, rewards, terminated, truncated, _ = envs.step(actions)
            
            episode_returns += torch.as_tensor(rewards, device='cuda') * (~dones).float()
            dones = dones | torch.as_tensor(terminated | truncated, device='cuda')
    
    strategy.step(episode_returns)
    print(f"Epoch {epoch}: mean={episode_returns.mean():.1f}, max={episode_returns.max():.1f}")
```

---

## Why EGGROLL?

Evolution strategies are embarrassingly parallel — each population member can be evaluated independently. But most ES implementations still evaluate members one at a time, leaving GPU utilization on the table.

EGGROLL fixes this with **low-rank perturbations** that can be applied efficiently in a single batched operation:

```python
# Instead of 64 separate forward passes...
for i in range(64):
    output = model(x[i])  # Slow: 64 kernel launches

# ...one batched call with per-sample perturbations
outputs = pop.batched_forward(model, x_batch)  # Fast: 1 kernel launch
```

The magic is in the math: instead of materializing a full perturbation matrix, we keep it factored as `A @ B.T` and compute `x @ W.T + x @ B @ A.T`. Same result, way less memory and compute.

---

## GPU Requirement

EGGROLL-Torch needs a CUDA GPU to deliver on its promise. The whole point is GPU-accelerated batched perturbations — on CPU, you'd lose the speed advantage that makes EGGROLL worth using.

```python
# The implementation checks for this
strategy.setup(model.cpu())  # → RuntimeError explaining why GPU is needed
```

If you're working on a machine without a GPU, no worries — there are great CPU-friendly ES libraries out there (like OpenAI's original ES implementation). EGGROLL is specifically for when you want to squeeze maximum throughput out of GPU hardware.

---

## Design Philosophy

### From JAX to PyTorch

The original JAX implementation is powerful but has a steep learning curve. The PyTorch port aims to feel natural if you already know PyTorch:

| Original JAX API | New PyTorch API |
|-----------------|-----------------|
| `EggRoll.do_mm(frozen_noiser_params, noiser_params, param, key, iterinfo, x)` | `pop.batched_forward(model, x)` |
| `NOISER.init_noiser(params, sigma, lr, solver=optax.adamw, ...)` | `EggrollStrategy(sigma=0.1, lr=0.01)` |
| Manual parameter tree management | Automatic parameter discovery |

### Core Principles

1. **GPU-accelerated**: Designed for CUDA from the ground up
2. **Pythonic**: If you know PyTorch, you know this API
3. **Stateful**: Classes manage their own state — no threading params everywhere
4. **Composable**: Swap ES algorithms without changing model code
5. **Type-safe**: Full type hints and dataclasses
6. **RL-friendly**: Works great with Gym, Brax, and custom environments

---

## The Key Insight: Batched Forward

The most important thing to understand about EGGROLL is `batched_forward`. This single method is what makes everything fast:

```python
with strategy.perturb(population_size=64, epoch=0) as pop:
    # This evaluates ALL 64 population members in ONE call
    outputs = pop.batched_forward(model, x_batch)
```

Under the hood, it's computing:
```
output[i] = x[i] @ W.T + x[i] @ B_i @ A_i.T
```

Where `A_i` and `B_i` are the low-rank perturbation factors for population member `i`. The base weights `W` are shared, perturbations are applied per-sample, and it all happens in a single batched matmul.

### Why Not Materialize the Perturbation?

You might be tempted to compute `W + A @ B.T` directly. Don't! That defeats the purpose:

```python
# Tempting but inefficient
perturbed_W = W + A @ B.T  # O(m*n) memory
output = x @ perturbed_W.T  # O(m*n) compute

# What EGGROLL actually does
output = x @ W.T + x @ B @ A.T  # O(r*(m+n)) — much smaller when r << min(m,n)
```

---

## Reinforcement Learning Integration

EGGROLL works great with RL environments. The key is matching your environment's batching capability to EGGROLL's batched evaluation.

### Environment Compatibility

| Environment Type | Recommended Approach | Notes |
|-----------------|---------------------|-------|
| **Custom GPU envs** | `batched_forward` | Best performance — everything on GPU |
| **Brax / MJX** | `batched_forward` | JAX physics + PyTorch policy works great |
| **Vectorized Gym** | `batched_forward` | Use `gym.vector.make()` |
| **Single Gym env** | `iterate()` | Works, but you're leaving performance on the table |

### Example: Vectorized Gym (Recommended)

This is probably what most people want — standard Gym benchmarks with full batching:

```python
import gymnasium as gym
import torch
from hyperscalees.torch import EggrollStrategy

population_size = 64
envs = gym.vector.make("HalfCheetah-v4", num_envs=population_size)

policy = nn.Sequential(
    nn.Linear(17, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 6)
).cuda()

strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=4)
strategy.setup(policy)

for epoch in range(1000):
    obs, _ = envs.reset()
    episode_returns = torch.zeros(population_size, device='cuda')
    dones = torch.zeros(population_size, dtype=torch.bool, device='cuda')
    
    with strategy.perturb(population_size=population_size, epoch=epoch) as pop:
        while not dones.all():
            obs_tensor = torch.as_tensor(obs, device='cuda', dtype=torch.float32)
            
            with torch.no_grad():
                # One call, all 64 policies evaluated
                actions = pop.batched_forward(policy, obs_tensor)
            
            obs, rewards, terminated, truncated, _ = envs.step(actions.cpu().numpy())
            episode_returns += torch.as_tensor(rewards, device='cuda') * (~dones).float()
            dones = dones | torch.as_tensor(terminated | truncated, device='cuda')
    
    metrics = strategy.step(episode_returns)
    print(f"Epoch {epoch}: mean={episode_returns.mean():.1f}")
```

### Example: Custom Batched Environment

For maximum performance, design your environment to be natively batched:

```python
class MyBatchedEnv:
    """Environment that processes all agents in parallel."""
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
    
    def reset(self) -> torch.Tensor:
        return torch.randn(self.batch_size, 16, device='cuda')
    
    def step(self, actions: torch.Tensor):
        # actions: (batch_size, action_dim)
        obs = torch.randn(self.batch_size, 16, device='cuda')
        rewards = self.compute_rewards(actions)  # Batched reward computation
        dones = torch.rand(self.batch_size, device='cuda') < 0.01
        return obs, rewards, dones

# Everything stays on GPU — no CPU↔GPU transfers in the inner loop
env = MyBatchedEnv(batch_size=256)
```

### Example: Brax / MJX

EGGROLL plays nicely with JAX-based physics simulators:

```python
import jax
from brax import envs as brax_envs
from brax.io import torch as brax_torch

population_size = 512
env = brax_envs.create("ant", batch_size=population_size)

policy = nn.Sequential(
    nn.Linear(87, 256), nn.Tanh(),
    nn.Linear(256, 256), nn.Tanh(),
    nn.Linear(256, 8)
).cuda()

strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=8)
strategy.setup(policy)

for epoch in range(1000):
    state = env.reset(jax.random.PRNGKey(epoch))
    episode_returns = torch.zeros(population_size, device='cuda')
    
    with strategy.perturb(population_size=population_size, epoch=epoch) as pop:
        for _ in range(1000):
            obs_torch = brax_torch.jax_to_torch(state.obs).cuda()
            
            with torch.no_grad():
                actions_torch = pop.batched_forward(policy, obs_torch)
            
            state = env.step(state, brax_torch.torch_to_jax(actions_torch.cpu()))
            episode_returns += brax_torch.jax_to_torch(state.reward).cuda()
            
            if state.done.all():
                break
    
    metrics = strategy.step(episode_returns)
```

### Sequential Evaluation (When You Really Need It)

Sometimes you're stuck with an environment that can't be batched — maybe it's a legacy simulator or has complex internal state. EGGROLL still works, you'll just be trading off some performance:

```python
import gymnasium as gym

env = gym.make("CartPole-v1")  # Single environment
policy = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2)).cuda()

strategy = EggrollStrategy(sigma=0.1, lr=0.01, rank=4)
strategy.setup(policy)

for epoch in range(100):
    fitnesses = []
    
    with strategy.perturb(population_size=64, epoch=epoch) as pop:
        for member_id in pop.iterate():
            # Each iteration configures the model for this population member
            episode_return = 0.0
            obs, _ = env.reset()
            done = False
            
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, device='cuda', dtype=torch.float32)
                    action = policy(obs_tensor).argmax().item()
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                done = terminated or truncated
            
            fitnesses.append(episode_return)
    
    metrics = strategy.step(torch.tensor(fitnesses, device='cuda'))
```

This works fine for prototyping or when your environment truly can't be parallelized. But if you find yourself here in production, it's worth asking whether you can:
- Use `gym.vector.make()` to run multiple env instances
- Refactor your custom env to accept batched actions
- Use a batched simulator like Brax or Isaac Gym

---

## The `batched_forward` API

Here's what's happening under the hood:

```python
class PerturbationContext:
    def batched_forward(
        self, 
        model: nn.Module, 
        x: Tensor,
        member_ids: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass applying different perturbations to each batch element.
        
        Args:
            model: The neural network
            x: Input tensor, shape (batch_size, *input_dims)
            member_ids: Which population member to use for each batch element.
                       Defaults to member_ids[i] = i (one-to-one mapping).
        
        Returns:
            Output tensor, shape (batch_size, *output_dims)
        
        The computation for each linear layer:
            output[i] = x[i] @ W.T + x[i] @ B[member_ids[i]] @ A[member_ids[i]].T
        """
```

The `member_ids` parameter is useful when your batch size doesn't match your population size — for example, if you're evaluating multiple timesteps per member.

---

## Model Integration

### Wrapping Existing Models

Any `nn.Module` works out of the box:

```python
from hyperscalees.torch import ESModule

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
es_model = ESModule(model)  # Ready for ES optimization
```

### Accessing Perturbation Factors (Advanced)

For debugging or analysis, you can inspect the low-rank factors directly:

```python
with strategy.perturb(population_size=64, epoch=0) as pop:
    # Get the factors for a specific member and parameter
    A, B = pop.get_factors(member_id=0, param_name="0.weight")
    # A: (out_features, rank), B: (in_features, rank)
    # The perturbation is A @ B.T, but we never compute that directly
```

---

## Test Structure

Each test file covers a core concept:

```
tests/torch_tests/
├── README.md                           # This file
├── conftest.py                         # Shared fixtures
├── test_strategy_api.py                # Strategy interface contract
├── test_low_rank_perturbations.py      # Low-rank structure verification
├── test_forward_equivalence.py         # Efficient forward pass correctness
├── test_antithetic_sampling.py         # Variance reduction via mirrored sampling
├── test_deterministic_noise.py         # Reproducibility guarantees
├── test_gradient_accumulation.py       # High-rank from accumulated low-rank
├── test_fitness_shaping.py             # Fitness normalization
├── test_parameter_updates.py           # ES gradient estimation
├── test_model_integration.py           # nn.Module compatibility
├── test_rl_integration.py              # RL environment patterns
└── test_distributed.py                 # Multi-GPU support
```

**Note:** All tests above are REQUIRED. Do not add extra test files (e.g., `test_efficiency_correctness.py`, `test_triton_kernels.py`) — efficiency and kernel correctness should be verified within the relevant domain tests above.

---

## Running Tests

```bash
# Run all PyTorch tests
pytest tests/torch_tests/ -v

# Run a specific concept
pytest tests/torch_tests/test_low_rank_perturbations.py -v

# Skip unimplemented features
pytest tests/torch_tests/ -v -m "not unimplemented"
```

---

## Key Fixtures

| Fixture | Description |
|---------|-------------|
| `device` | CUDA device (GPU required) |
| `base_generator` | Seeded torch.Generator for reproducibility |
| `small_tensor` | 8×4 tensor for detailed inspection |
| `medium_tensor` | 64×32 tensor for rank tests |
| `simple_mlp` | Basic nn.Sequential for integration tests |
| `eggroll_strategy` | Pre-configured EggrollStrategy |

---

## Implementation Checklist

Tests are written first — check them off as you implement:

- [ ] `test_strategy_api.py` — Core strategy interface
- [ ] `test_low_rank_perturbations.py` — Perturbation generation
- [ ] `test_forward_equivalence.py` — Efficient forward pass
- [ ] `test_antithetic_sampling.py` — Mirrored sampling
- [ ] `test_deterministic_noise.py` — Reproducible noise
- [ ] `test_gradient_accumulation.py` — Update accumulation
- [ ] `test_fitness_shaping.py` — Fitness normalization
- [ ] `test_parameter_updates.py` — ES updates
- [ ] `test_model_integration.py` — PyTorch integration
- [ ] `test_rl_integration.py` — RL environment compatibility
- [ ] `test_distributed.py` — Multi-GPU support

**Do not add additional test files.** If you need to test implementation details (e.g., Triton kernel correctness), add those tests to the appropriate domain file above.

---

## API Reference

### `EggrollStrategy`

```python
class EggrollStrategy:
    """Low-rank evolution strategy with the EGGROLL algorithm."""
    
    def __init__(
        self,
        sigma: float = 0.1,        # Noise scale
        lr: float = 0.01,          # Learning rate
        rank: int = 4,             # Perturbation rank
        optimizer: str = "adam",   # Optimizer for ES updates
        optimizer_kwargs: dict = None,
        antithetic: bool = True,   # Use mirrored sampling
        noise_reuse: int = 0,      # Epochs to reuse noise
    ): ...
    
    def setup(self, model: nn.Module) -> None:
        """Attach to model and discover parameters."""
        
    def perturb(self, population_size: int, epoch: int = 0) -> PerturbationContext:
        """Context manager for perturbed evaluation."""
        
    def step(self, fitnesses: Tensor) -> dict:
        """Update parameters. Returns metrics dict."""
        
    def state_dict(self) -> dict:
        """Serialize state for checkpointing."""
        
    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint."""
```

### `PerturbationContext`

```python
class PerturbationContext:
    """Manages perturbations during evaluation."""
    
    def batched_forward(self, model: nn.Module, x: Tensor, 
                        member_ids: Tensor = None) -> Tensor:
        """Batched forward with per-sample perturbations."""
    
    def iterate(self) -> Iterator[int]:
        """Iterate through population members (for sequential evaluation)."""
    
    def get_factors(self, member_id: int, param_name: str) -> Tuple[Tensor, Tensor]:
        """Get low-rank factors A, B for inspection."""
    
    @property
    def population_size(self) -> int: ...
```

### `ESModule`

```python
class ESModule(nn.Module):
    """Wrapper for ES-compatible models."""
    
    def __init__(self, module: nn.Module): ...
    
    @property
    def es_parameters(self) -> Iterator[Tensor]:
        """Parameters that will be evolved."""
        
    def freeze_parameter(self, name: str) -> None:
        """Exclude a parameter from evolution."""
```
