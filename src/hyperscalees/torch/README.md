# Functional EGGROLL (Torch)

> **Train neural networks without backpropagation.** EGGROLL (**E**volution **G**uided **G**eneral **O**ptimization via **L**ow-rank **L**earning) uses evolution strategies with low-rank perturbations to estimate gradients.

## Understanding EGGROLL

**New to the codebase?** Peep the [test suite](tests/test_core.py).

```bash
# Run tests with documentation output
pytest src/hyperscalees/torch/tests/ -v -s
```

---

## Why EGGROLL?

EGGROLL is an evolution strategies (ES) algorithm designed to scale backprop-free optimization to **large population sizes** for modern neural networks. The key innovation: instead of sampling full-rank perturbation matrices $E \in \mathbb{R}^{m \times n}$, EGGROLL samples low-rank factors $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{n \times r}$ (with $r \ll \min(m, n)$) and forms $E = \frac{1}{\sqrt{r}} A B^\top$.

| Problem with Backprop | EGGROLL Solution |
|-----------------------|------------------|
| Requires differentiable loss | Works with **any** fitness function (reward, accuracy, custom metric) |
| Sequential layer-by-layer computation | **Embarrassingly parallel** across population |
| Memory-hungry for large models | **O(r(m+n))** auxiliary storage per layer vs O(m×n) for full ES |
| Gradients vanish/explode in long-horizon settings | Population-based smoothing tolerates discontinuities |

**Important:** While each perturbation is low-rank, the **overall EGGROLL update is high-rank**—it's rank $\min(Nr, m, n)$ since it averages across $N$ population members. This means you get the memory benefits of low-rank without sacrificing expressiveness.

---

## Quick Start (5 minutes)

### Prerequisites
```bash
pip install torch gymnasium torchvision rich numpy
```

> **Note:** EGGROLL is **GPU-only**. Requires CUDA.

### Run Your First Experiment

```bash
# Solve CartPole
python -m hyperscalees.torch.recipes --experiment cartpole

# Train MNIST MLP classifier 
python -m hyperscalees.torch.recipes --experiment mnist

# Train MNIST CNN with perturbed conv layers
python -m hyperscalees.torch.recipes --experiment mnist_cnn
```
---

## The Core Idea

Instead of backprop, EGGROLL:

1. **Generates N slightly different versions** of your network (via low-rank noise: $\frac{\sigma}{\sqrt{r}} A_i B_i^\top$ per member)
2. **Evaluates all of them in parallel** on your task
3. **Computes a gradient estimate** from which versions did best
4. **Updates weights** in the direction of better performance

The magic: **low-rank perturbations** make this 100x more memory-efficient than naive ES, while the aggregate update remains high-rank.

---

## Quick Start: From PyTorch Module to EGGROLL

The fastest way to get started is to define your model with standard PyTorch, then auto-convert:

```python
import torch
import torch.nn as nn
from hyperscalees.torch.core import (
    EggrollConfig, get_params_dict, get_weight_shapes, make_perturbed_forward_fn,
    generate_perturbations, eggroll_step
)

# 1. Define your model with standard PyTorch
model = nn.Sequential(
    nn.Linear(4, 256),
    nn.Tanh(),
    nn.Linear(256, 2),
)

# 2. Auto-convert to EGGROLL format
params = get_params_dict(model)                      # Extract params dict
shapes = get_weight_shapes(params)                   # Get shapes for perturbation gen
forward, forward_eval = make_perturbed_forward_fn(model)  # Auto-gen forward functions!

# 3. Training loop
config = EggrollConfig(population_size=2048, rank=4, sigma=0.1, lr=0.1,
                       lr_decay=0.999, sigma_decay=0.999)
current_lr, current_sigma = config.lr, config.sigma

for epoch in range(100):
    gen = torch.Generator(device="cuda").manual_seed(42 + epoch)
    perts = generate_perturbations(shapes, config.population_size, config.rank, 
                                   current_sigma, gen, config.dtype)
    
    # Forward pass (perturbations applied to all Linear layers)
    output = forward(x, params, perts)
    
    # One-liner EGGROLL step: normalize -> gradients -> update -> decay
    current_lr, current_sigma = eggroll_step(
        params, your_fitness_fn(output), perts, current_lr, current_sigma, config
    )

# 4. Inference (no perturbations)
output = forward_eval(x, params)
```

**Note:** `make_perturbed_forward_fn` uses `torch.fx` to trace your model. It works great for Sequential models and simple feedforward nets. For complex architectures with dynamic control flow, you'll need to write the forward functions manually (see recipes below).

---

## Complete Working Recipes

For more context, see the [recipes](recipies.py) file.

### Recipe 1: Reinforcement Learning (CartPole)

**Goal:** Train a policy to balance a pole for 500 timesteps.

```python
import torch
import torch.nn as nn
import gymnasium as gym
from hyperscalees.torch.core import (
    EggrollConfig, get_params_dict, get_weight_shapes, make_perturbed_forward_fn,
    generate_perturbations, eggroll_step
)

# ============ CONFIG ============
config = EggrollConfig(
    population_size=2048,  # Number of parallel policy variants
    rank=4,                # Low-rank perturbation rank
    sigma=0.2,             # Noise scale (will decay)
    lr=0.1,                # Learning rate
    sigma_decay=0.999,     # Decay sigma each epoch
    lr_decay=0.9995,       # Decay LR each epoch
    max_epochs=300,        # Max training epochs
    seed=42,
)

# ============ MODEL DEFINITION ============
model = nn.Sequential(
    nn.Linear(4, 256),
    nn.Tanh(),
    nn.Linear(256, 2),
)
# Initialize with small weights
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.1)
        nn.init.zeros_(m.bias)

# ============ CONVERT TO EGGROLL FORMAT ============
params = get_params_dict(model)                       # Extract params to dict
shapes = get_weight_shapes(params)                    # Get shapes for perturbations
forward, forward_eval = make_perturbed_forward_fn(model)  # Auto-gen forward functions

# ============ TRAINING LOOP ============
envs = gym.make_vec("CartPole-v1", num_envs=config.population_size)
torch.manual_seed(config.seed)
current_lr, current_sigma = config.lr, config.sigma

for epoch in range(config.max_epochs):
    # 1. Generate perturbations for ALL weights at once
    gen = torch.Generator(device="cuda").manual_seed(config.seed + epoch * 1000)
    perts = generate_perturbations(shapes, config.population_size, config.rank, 
                                   current_sigma, gen, config.dtype)
    
    # 2. Run episodes (all 2048 policies in parallel)
    obs, _ = envs.reset(seed=epoch)
    episode_returns = torch.zeros(config.population_size, device="cuda")
    dones = torch.zeros(config.population_size, dtype=torch.bool, device="cuda")
    
    for step in range(500):
        obs_t = torch.as_tensor(obs, device="cuda", dtype=torch.float32)
        logits = forward(obs_t, params, perts)
        actions = logits.argmax(dim=-1).cpu().numpy()
        
        obs, rewards, terminated, truncated, _ = envs.step(actions)
        episode_returns += torch.as_tensor(rewards, device="cuda") * (~dones).float()
        dones = dones | torch.as_tensor(terminated | truncated, device="cuda")
        
        if dones.all():
            break
    
    # 3. ES update
    current_lr, current_sigma = eggroll_step(
        params, episode_returns, perts, current_lr, current_sigma, config
    )
    
    # 4. Check if solved
    mean_ret = episode_returns.mean().item()
    if mean_ret >= 475:
        print(f"Solved at epoch {epoch}!")
        break

envs.close()
```

### Recipe 2: Supervised Learning (MNIST MLP)

**Goal:** Train a classifier on MNIST digits.

```python
import torch
import torch.nn as nn
import numpy as np
from hyperscalees.torch.core import (
    EggrollConfig, get_params_dict, get_weight_shapes, make_perturbed_forward_fn,
    generate_perturbations, eggroll_step,
)
from hyperscalees.torch.recipes import (
    load_mnist_flat, compute_classification_fitness
)

# ============ CONFIG ============
config = EggrollConfig(
    population_size=4096,  # Larger population for SL
    rank=4,
    sigma=0.15,
    lr=0.1,
    sigma_decay=0.999,
    lr_decay=1.0,          # No LR decay for MNIST
    max_epochs=100,
    batch_size=256,        # Mini-batch size per ES step
    seed=42,
)

# ============ DATA LOADING ============
train_imgs, train_labels, test_imgs, test_labels = load_mnist_flat(config.dtype)
# train_imgs: (60000, 784), test_imgs: (10000, 784)

# ============ MODEL DEFINITION ============
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.Tanh(),
    nn.Linear(256, 10),
)
# Initialize with small weights
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.1)
        nn.init.zeros_(m.bias)

# ============ CONVERT TO EGGROLL FORMAT ============
params = get_params_dict(model)
shapes = get_weight_shapes(params)
forward, forward_eval = make_perturbed_forward_fn(model)

# ============ TRAINING LOOP ============
torch.manual_seed(config.seed)
rng = np.random.default_rng(config.seed)
current_lr, current_sigma = config.lr, config.sigma

for epoch in range(config.max_epochs):
    # 1. Sample random mini-batch
    idx = torch.tensor(rng.integers(0, len(train_imgs), size=config.batch_size), device="cuda")
    batch_imgs = train_imgs[idx]      # (batch, 784)
    batch_labels = train_labels[idx]  # (batch,)
    
    # 2. Generate perturbations
    gen = torch.Generator(device="cuda").manual_seed(config.seed + epoch * 1000)
    perts = generate_perturbations(shapes, config.population_size, config.rank, 
                                   current_sigma, gen, config.dtype)
    
    # 3. Forward: expand batch for population
    x = batch_imgs.unsqueeze(0).expand(config.population_size, -1, -1)  # (pop, batch, 784)
    logits = forward(x, params, perts)  # (pop, batch, 10)
    
    # 4. Compute fitness (negative cross-entropy loss)
    fitnesses = compute_classification_fitness(logits, batch_labels)
    
    # 5. Step
    current_lr, current_sigma = eggroll_step(
        params, fitnesses, perts, current_lr, current_sigma, config
    )
    
    # 6. Evaluate on full test set
    with torch.no_grad():
        test_logits = forward_eval(test_imgs, params)
        test_acc = (test_logits.argmax(dim=-1) == test_labels).float().mean().item() * 100
    
    print(f"Epoch {epoch:3d} | test_acc={test_acc:5.1f}%")
```

### Recipe 3: CNN with Perturbed Convolutions

**Goal:** Train a CNN where even conv layers are perturbed.

```python
import torch
import math
from hyperscalees.torch.core import (
    get_weight_shapes, generate_perturbations, compute_gradients,
    update_params, perturbed_forward, normalize_fitnesses,
)
from hyperscalees.torch.recipes import (
    perturbed_conv2d, load_mnist_2d, compute_classification_fitness
)

# ============ HYPERPARAMETERS ============
population_size = 2048
rank = 4
sigma = 0.1
lr = 0.1
lr_decay = 0.998
sigma_decay = 0.999
max_epochs = 100
batch_size = 64
seed = 42

# ============ DATA LOADING ============
train_imgs, train_labels, test_imgs, test_labels = load_mnist_2d(torch.float32)
# train_imgs: (60000, 1, 28, 28)

# ============ MODEL DEFINITION ============
# CNN: Conv1(1->16, 3x3) -> ReLU -> Pool -> Conv2(16->32, 3x3) -> ReLU -> Pool -> FC(1568->10)
fc_in = 32 * 7 * 7  # After 2x2 pooling twice: 28 -> 14 -> 7

params = {
    'conv1.weight': torch.randn(16, 1, 3, 3, device="cuda") * 0.1,
    'conv2.weight': torch.randn(32, 16, 3, 3, device="cuda") * 0.1,
    'fc.weight': torch.randn(10, fc_in, device="cuda") / math.sqrt(fc_in),
    'fc.bias': torch.zeros(10, device="cuda"),
}
shapes = get_weight_shapes(params)  # Handles both 2D and 4D weights

# ============ FORWARD FUNCTIONS ============
def forward(x, params, perts):
    """x: (batch, 1, 28, 28) -> (pop, batch, 10) — training with perturbations"""
    # Conv1 + ReLU + Pool
    x = perturbed_conv2d(x, params['conv1.weight'], perts, 'conv1.weight', padding=1)
    x = torch.relu(x)
    pop_size, batch_size = x.shape[0], x.shape[1]
    x = x.reshape(pop_size * batch_size, *x.shape[2:])
    x = torch.nn.functional.max_pool2d(x, 2)
    x = x.reshape(pop_size, batch_size, *x.shape[1:])
    
    # Conv2 + ReLU + Pool
    x = perturbed_conv2d(x, params['conv2.weight'], perts, 'conv2.weight', padding=1)
    x = torch.relu(x)
    x = x.reshape(pop_size * batch_size, *x.shape[2:])
    x = torch.nn.functional.max_pool2d(x, 2)
    x = x.reshape(pop_size, batch_size, *x.shape[1:])
    
    # Flatten + FC
    x = x.reshape(pop_size, batch_size, -1)
    return perturbed_forward(x, params['fc.weight'], params['fc.bias'], perts, 'fc.weight')

def forward_eval(x, params):
    """x: (batch, 1, 28, 28) -> (batch, 10) — evaluation/inference"""
    x = torch.nn.functional.conv2d(x, params['conv1.weight'], padding=1)
    x = torch.relu(x)
    x = torch.nn.functional.max_pool2d(x, 2)
    x = torch.nn.functional.conv2d(x, params['conv2.weight'], padding=1)
    x = torch.relu(x)
    x = torch.nn.functional.max_pool2d(x, 2)
    x = x.flatten(1)
    return x @ params['fc.weight'].T + params['fc.bias']

# ============ TRAINING LOOP ============
# (Same pattern as Recipe 2, but using CNN forward functions)
```

---

## API Reference

### Module Conversion

Convert standard PyTorch modules to EGGROLL format automatically.

| Function | Purpose | Example |
|----------|---------|---------|
| `get_params_dict(module, device, dtype)` | Extract params from nn.Module | `params = get_params_dict(model)` |
| `make_perturbed_forward_fn(module)` | Auto-generate forward functions | `fwd, fwd_eval = make_perturbed_forward_fn(model)` |

**`get_params_dict`** copies all parameters to the specified device (default: CUDA) and returns them as a flat dict with PyTorch's naming convention (`'0.weight'`, `'fc.bias'`, etc.).

**`make_perturbed_forward_fn`** uses `torch.fx` to trace your module and automatically replaces `nn.Linear` layers with `perturbed_forward` calls. Returns `(forward_fn, forward_eval_fn)`. Limitations:
- Only handles `nn.Linear` (not Conv2d—use `perturbed_conv2d` manually)
- Module must be traceable by `torch.fx` (no dynamic control flow)

### Dict-Based API (Recommended)

Manages perturbations and gradients across multiple weight matrices via string keys.

| Function | Purpose | Example |
|----------|---------|---------|
| `get_weight_shapes(params)` | Auto-detect which params need perturbations | `shapes = get_weight_shapes(params)` |
| `generate_perturbations(shapes, pop, rank, sigma, gen, dtype)` | Generate all perturbations in one call | `perts = generate_perturbations(shapes, 2048, 4, 0.1, gen, torch.float32)` |
| `perturbed_forward(x, W, b, perts, key)` | Perturbed linear layer | `h = perturbed_forward(x, W, b, perts, 'layer1.weight')` |
| `perturbed_conv2d(x, W, perts, key, padding)` | Perturbed conv layer (in recipes.py) | `h = perturbed_conv2d(x, W, perts, 'conv1.weight', padding=1)` |
| `eggroll_step(params, fitnesses, perts, lr, sigma, config)` | **One-liner ES update** with decay from config | `lr, sigma = eggroll_step(params, f, perts, lr, sigma, config)` |
| `compute_gradients(fitnesses, perts, pop)` | Compute all gradients at once | `grads = compute_gradients(f, perts, pop)` |
| `update_params(params, grads, lr)` | Update all params in-place | `update_params(params, grads, 0.1)` |
| `normalize_fitnesses(fitnesses)` | Zero-mean, unit-variance normalization | `f = normalize_fitnesses(episode_returns)` |

### Raw Primitives API (Maximum Control)

For when you need fine-grained control over each layer.

| Function | Purpose |
|----------|---------|
| `generate_lowrank_perturbations(pop, out_dim, in_dim, rank, sigma, gen, dtype)` | Generate antithetic A, B for one weight matrix |
| `perturbed_linear(x, W, b, A_scaled, B)` | Apply perturbed linear: `x @ W.T + b + x @ B @ A.T` |
| `apply_lowrank_perturbation(x, B, A_scaled)` | Just the perturbation term: `x @ B @ A.T` |
| `compute_es_gradient(fitnesses, A_scaled, B, pop)` | Gradient for one weight via `einsum` |

**Note:** `A_scaled` already includes the $\sigma / \sqrt{r}$ factor, so you don't need to scale it yourself.

---

### Hyperparameter Guide

### What Each Hyperparameter Does

| Param | Default | Effect |
|-------|---------|--------|
| `population_size` | 2048 | More = smoother gradients, slower per-step. The paper uses up to 262,144 for pretraining! |
| `rank` | 4 | Low-rank dimension. Higher = more expressive perturbations. Theory shows $O(1/r)$ convergence. |
| `sigma` | 0.1-0.2 | Perturbation scale (before $1/\sqrt{r}$ normalization). Too high = chaotic, too low = stuck. |
| `lr` | 0.1 | Learning rate. ES is relatively robust to LR—0.1 is a good default. |
| `sigma_decay` | 0.999 | Decay sigma each epoch for fine-tuning toward local optima. |
| `lr_decay` | 1.0 | Optional LR decay. Usually not needed for ES. |
| `batch_size` | 256 | Mini-batch size for supervised learning. 64-256 works well. |

---

## The Math (Optional)

### Low-Rank Perturbation

For weight matrix $W \in \mathbb{R}^{m \times n}$, the perturbation is:

$$\Delta W = \frac{1}{\sqrt{r}} A \cdot B^\top$$

where $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{n \times r}$ with rank $r \ll \min(m, n)$.

The $\frac{1}{\sqrt{r}}$ scaling ensures the variance of perturbations remains bounded as rank increases. In practice, we absorb this into the sigma: `A_scaled = A * (sigma / sqrt(r))`.

### ES Gradient Estimate (EGGROLL Update)

$$\nabla_W \approx \frac{1}{\sqrt{N}} \sum_{i=1}^{N} f_i \cdot A_i \cdot B_i^\top$$

where:
- $f_i$ = normalized fitness for population member $i$
- $N$ = population size
- $A_i, B_i$ = perturbation factors for member $i$ (scaled by $\sigma / \sqrt{r}$)

This matches the paper's Equation 8. Note: the gradient is computed **without materializing** $A_i B_i^\top$ — we use `einsum('nir,njr->ij', f * A, B)` to go directly from factors to gradient.

### Antithetic Sampling

Population members are paired: members $2k$ and $2k+1$ use opposite-sign perturbations (+A, -A). This variance reduction technique is standard in ES and halves the effective noise.

### Why Low-Rank Works

The EGGROLL paper proves that the low-rank gradient estimate converges to the full-rank ES gradient at rate $O(1/r)$ — much faster than the typical $O(1/\sqrt{r})$ from the central limit theorem. This fast rate comes from the symmetry of the noise distribution (odd moments vanish).

| Full ES | EGGROLL |
|---------|---------|
| Store $N \times m \times n$ perturbations | Store $N \times (m + n) \times r$ factors |
| For 256×784 weight, N=2048: **~400 MB/layer** | Same setting, rank=4: **~17 MB/layer** |

The update is still high-rank: averaging $N$ rank-$r$ matrices gives rank $\min(Nr, m, n)$.

---

## Troubleshooting

### "CUDA out of memory"
- Reduce `population_size` (try 1024)
- Reduce `batch_size` for supervised learning
- Check if other processes are using GPU

### "Training doesn't converge"
- Increase `sigma` (try 0.2-0.3) — you need enough exploration
- Increase `population_size` — more samples = better gradient estimate
- Check your fitness function: **higher values must mean better** (ES maximizes)

### "Results are noisy/unstable"
- Increase `population_size` — the gradient variance scales as $1/N$
- Add `sigma_decay` (0.999) to settle into local optima
- For supervised learning: increase `batch_size` to reduce fitness variance

---

## Performance

- **torch.compile:** All forward passes are JIT-compiled for performance
- **Throughput:** 1M+ steps/sec on CartPole (RTX 4090)
- **Memory:** Population 4096 + batch 256 uses ~1-2GB VRAM for MNIST
- **TF32 enabled:** Automatic TensorFloat32 for faster matmul on Ampere+ GPUs

### Caveat: Perturbation Generation

The `generate_perturbations()` function uses `torch.Generator` which is **not `torch.compile` compatible**. This means perturbation generation runs eagerly (not fused into the compiled graph).

**In practice, this doesn't matter for models <10M params:**
- Perturbation generation is only **~5% of total step time**
- The forward pass dominates (95%) and IS fully compiled
- For pop=2048, dim=256, rank=8: generation takes ~0.06ms vs ~1.3ms for forward

The JAX reference implementation uses counter-based Philox PRNG (via `jax.random.fold_in`) which enables deterministic noise reconstruction without storage. This allows regenerating perturbations on-demand during the update step. The PyTorch implementation stores perturbations between forward and update for simplicity, which is fine for sub-10M param models where memory isn't the bottleneck.

---

## Running All Experiments

```bash
# Individual experiments
python -m hyperscalees.torch.recipes --experiment cartpole
python -m hyperscalees.torch.recipes --experiment mnist_mlp
python -m hyperscalees.torch.recipes --experiment mnist_cnn

# Benchmarking
python -m hyperscalees.torch.recipes --experiment hyperscale

# Run everything
python -m hyperscalees.torch.recipes --experiment all
```


---

## File Structure

```
core.py                        # Core EGGROLL implementation
├── EggrollConfig              # Dataclass for hyperparameters
├── Module Conversion
│   ├── get_params_dict        # Extract params from nn.Module
│   └── make_perturbed_forward_fn  # Auto-gen forward functions via torch.fx
├── Dict-Based API
│   ├── get_weight_shapes      # Auto-detect perturbed weights (2D and 4D)
│   ├── generate_perturbations # Generate A, B factors for all weights
│   ├── perturbed_forward      # x @ W.T + b + x @ B @ A.T (never materializes A @ B.T)
│   ├── compute_gradients      # einsum('nir,njr->ij', f*A, B) / sqrt(N)
│   └── update_params          # Apply gradients to weights
└── Raw Primitives
    ├── generate_lowrank_perturbations  # Antithetic A, B factors
    ├── perturbed_linear                # Core: base + low-rank perturbation
    ├── apply_lowrank_perturbation      # Just x @ B @ A.T
    ├── compute_weight_perturbation     # Materialize A @ B.T (for conv only!)
    ├── compute_es_gradient             # Single-weight ES gradient
    └── normalize_fitnesses             # Zero-mean, unit-variance

recipes.py                     # Experiments & recipes
├── GPU Utilities
│   ├── get_gpu_stats
│   ├── print_gpu_stats
│   └── reset_gpu_stats
├── Data Loading
│   ├── load_mnist_flat
│   └── load_mnist_2d
├── Experiment Helpers
│   ├── compute_classification_fitness  # -cross_entropy (higher = better)
│   └── perturbed_conv2d                # Grouped conv for batched CNN
└── Experiments
    ├── cartpole()
    ├── mnist_mlp()
    └── mnist_cnn()
```

---

## References

- **Paper:** Sarkar et al., "Evolution Strategies at the Hyperscale"
- **JAX Reference Implementation:** `src/hyperscalees/noiser/eggroll.py`
- **Project Website:** https://eshyperscale.github.io/
