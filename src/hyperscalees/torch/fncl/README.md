# Functional EGGROLL (Torch)

> **Train neural networks without backpropagation.** EGGROLL uses evolution strategies with low-rank perturbations to estimate gradients—achieving **1.3M+ steps/sec** on CartPole and competitive accuracy on MNIST in seconds.

## Why EGGROLL?

| Problem with Backprop | EGGROLL Solution |
|-----------------------|------------------|
| Requires differentiable loss | Works with **any** fitness function (reward, accuracy, custom metric) |
| Sequential layer-by-layer computation | **Embarrassingly parallel** across population |
| Memory-hungry for large models | **O(r(m+n))** memory per layer vs O(m×n) for full ES |
| Hard to debug gradients | **Black-box**: just evaluate fitness, get gradients |

**Real-world wins:**
- ✅ **RL without reward shaping**: Direct policy optimization from sparse rewards
- ✅ **Non-differentiable objectives**: Optimize BLEU, accuracy, latency—anything measurable
- ✅ **Hyperparameter-free architecture search**: No learning rate scheduling needed
- ✅ **Hardware failures? No problem**: Population-based = natural fault tolerance

---

## Quick Start (5 minutes)

### Prerequisites
```bash
pip install torch gymnasium torchvision rich numpy
```

> **Note:** EGGROLL is **GPU-only**. Requires CUDA.

### Run Your First Experiment

```bash
# Solve CartPole in ~12 epochs (~9 seconds)
python -m hyperscalees.torch.fncl.recipes --experiment cartpole

# Train MNIST MLP classifier to ~88% accuracy (~3 seconds)  
python -m hyperscalees.torch.fncl.recipes --experiment mnist

# Train MNIST CNN with perturbed conv layers (~13 seconds)
python -m hyperscalees.torch.fncl.recipes --experiment mnist_cnn
```

**Expected output (CartPole):**
```
Torch EGGROLL - CartPole-v1
Population: 2048, Rank: 4, Sigma: 0.2, LR: 0.1
Network: 4 -> 256 -> 2 (1,794 params)

Epoch   0 | mean=  80.0 max= 500.0 min=   8.0 |   55,593 steps/s
Epoch  10 | mean= 470.7 max= 500.0 min=   8.0 |  991,203 steps/s
Epoch  12 | mean= 477.8 max= 500.0 min=   9.0 | 1,087,968 steps/s
Solved at epoch 12!
```

---

## The Core Idea (30-second version)

Instead of backprop, EGGROLL:

1. **Generates 2048 slightly different versions** of your network (via low-rank noise)
2. **Evaluates all of them in parallel** on your task
3. **Computes a gradient estimate** from which versions did best
4. **Updates weights** in the direction of better performance

The magic: **low-rank perturbations** make this 100x more memory-efficient than naive ES.

```python
# The fundamental operation (what you need to know):
perturbed_output = perturbed_linear(x, W, b, A_scaled, B)
# Equivalent to: x @ (W + A @ B.T).T + b
# But computed efficiently with O(rank) overhead instead of O(params)
```

---

## Complete Working Recipes

These are **copy-paste ready** examples that match the proven implementations in `recipes.py`.

### Recipe 1: Reinforcement Learning (CartPole)

**Goal:** Train a policy to balance a pole for 500 timesteps.

```python
import torch
import gymnasium as gym
from hyperscalees.torch.fncl.core import (
    get_weight_shapes, generate_perturbations, compute_gradients,
    update_params, perturbed_forward, normalize_fitnesses
)

# ============ HYPERPARAMETERS (proven to work) ============
population_size = 2048   # Number of parallel policy variants
rank = 4                 # Low-rank perturbation rank
sigma = 0.2              # Noise scale (will decay)
lr = 0.1                 # Learning rate
sigma_decay = 0.999      # Decay sigma each epoch
lr_decay = 0.9995        # Decay LR each epoch
max_epochs = 300         # Max training epochs
seed = 42

# ============ MODEL DEFINITION ============
# 2-layer MLP: 4 (obs) -> 256 (hidden, tanh) -> 2 (actions)
params = {
    'layer1.weight': torch.randn(256, 4, device="cuda") * 0.1,
    'layer1.bias': torch.zeros(256, device="cuda"),
    'layer2.weight': torch.randn(2, 256, device="cuda") * 0.1,
    'layer2.bias': torch.zeros(2, device="cuda"),
}
shapes = get_weight_shapes(params)  # Auto-detect which params to perturb

# ============ FORWARD FUNCTIONS ============
def forward(obs, params, perts):
    """Forward pass with ES perturbations applied (training)."""
    h = torch.tanh(perturbed_forward(obs, params['layer1.weight'], 
                                     params['layer1.bias'], perts, 'layer1.weight'))
    return perturbed_forward(h, params['layer2.weight'], 
                             params['layer2.bias'], perts, 'layer2.weight')

def forward_eval(obs, params):
    """Forward pass without perturbations (evaluation/inference)."""
    h = torch.tanh(obs @ params['layer1.weight'].T + params['layer1.bias'])
    return h @ params['layer2.weight'].T + params['layer2.bias']

# ============ TRAINING LOOP ============
envs = gym.make_vec("CartPole-v1", num_envs=population_size)
torch.manual_seed(seed)

for epoch in range(max_epochs):
    # 1. Generate perturbations for ALL weights at once
    gen = torch.Generator(device="cuda").manual_seed(seed + epoch * 1000)
    perts = generate_perturbations(shapes, population_size, rank, sigma, gen, torch.float32)
    
    # 2. Run episodes (all 2048 policies in parallel!)
    obs, _ = envs.reset(seed=epoch)
    episode_returns = torch.zeros(population_size, device="cuda")
    dones = torch.zeros(population_size, dtype=torch.bool, device="cuda")
    
    for step in range(500):
        obs_t = torch.as_tensor(obs, device="cuda", dtype=torch.float32)
        logits = forward(obs_t, params, perts)
        actions = logits.argmax(dim=-1).cpu().numpy()
        
        obs, rewards, terminated, truncated, _ = envs.step(actions)
        episode_returns += torch.as_tensor(rewards, device="cuda") * (~dones).float()
        dones = dones | torch.as_tensor(terminated | truncated, device="cuda")
        
        if dones.all():
            break
    
    # 3. ES update: compute gradient from fitnesses
    fitnesses = normalize_fitnesses(episode_returns)
    grads = compute_gradients(fitnesses, perts, population_size)
    update_params(params, grads, lr)
    
    # 4. Decay hyperparameters
    lr *= lr_decay
    sigma *= sigma_decay
    
    # 5. Check if solved
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
import numpy as np
from hyperscalees.torch.fncl.core import (
    get_weight_shapes, generate_perturbations, compute_gradients,
    update_params, perturbed_forward, normalize_fitnesses,
)
from hyperscalees.torch.fncl.recipes import (
    load_mnist_flat, compute_classification_fitness
)

# ============ HYPERPARAMETERS (proven to work) ============
population_size = 4096   # Larger population for SL
rank = 4
sigma = 0.15
lr = 0.1
sigma_decay = 0.999
max_epochs = 100
batch_size = 256         # Mini-batch size per ES step
seed = 42

# ============ DATA LOADING ============
train_imgs, train_labels, test_imgs, test_labels = load_mnist_flat(torch.float32)
# train_imgs: (60000, 784), test_imgs: (10000, 784)

# ============ MODEL DEFINITION ============
# 2-layer MLP: 784 -> 256 (tanh) -> 10
params = {
    'layer1.weight': torch.randn(256, 784, device="cuda") * 0.1,
    'layer1.bias': torch.zeros(256, device="cuda"),
    'layer2.weight': torch.randn(10, 256, device="cuda") * 0.1,
    'layer2.bias': torch.zeros(10, device="cuda"),
}
shapes = get_weight_shapes(params)

# ============ FORWARD FUNCTIONS ============
def forward(x, params, perts):
    """x: (pop, batch, 784) -> (pop, batch, 10) — training with perturbations"""
    h = torch.tanh(perturbed_forward(x, params['layer1.weight'], 
                                     params['layer1.bias'], perts, 'layer1.weight'))
    return perturbed_forward(h, params['layer2.weight'], 
                             params['layer2.bias'], perts, 'layer2.weight')

def forward_eval(x, params):
    """x: (batch, 784) -> (batch, 10) — evaluation/inference"""
    h = torch.tanh(x @ params['layer1.weight'].T + params['layer1.bias'])
    return h @ params['layer2.weight'].T + params['layer2.bias']

# ============ TRAINING LOOP ============
torch.manual_seed(seed)
rng = np.random.default_rng(seed)

for epoch in range(max_epochs):
    # 1. Sample random mini-batch
    idx = torch.tensor(rng.integers(0, len(train_imgs), size=batch_size), device="cuda")
    batch_imgs = train_imgs[idx]      # (batch, 784)
    batch_labels = train_labels[idx]  # (batch,)
    
    # 2. Generate perturbations
    gen = torch.Generator(device="cuda").manual_seed(seed + epoch * 1000)
    perts = generate_perturbations(shapes, population_size, rank, sigma, gen, torch.float32)
    
    # 3. Forward: expand batch for population
    x = batch_imgs.unsqueeze(0).expand(population_size, -1, -1)  # (pop, batch, 784)
    logits = forward(x, params, perts)  # (pop, batch, 10)
    
    # 4. Compute fitness (negative cross-entropy loss)
    fitnesses = compute_classification_fitness(logits, batch_labels)
    fitnesses = normalize_fitnesses(fitnesses)
    
    # 5. ES update
    grads = compute_gradients(fitnesses, perts, population_size)
    update_params(params, grads, lr)
    
    sigma *= sigma_decay
    
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
from hyperscalees.torch.fncl.core import (
    get_weight_shapes, generate_perturbations, compute_gradients,
    update_params, perturbed_forward, normalize_fitnesses,
)
from hyperscalees.torch.fncl.recipes import (
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
shapes = get_weight_shapes(params)  # Handles both 2D and 4D weights!

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

### Dict-Based API (Recommended)

Manages perturbations and gradients across multiple weight matrices via string keys.

| Function | Purpose | Example |
|----------|---------|---------|
| `get_weight_shapes(params)` | Auto-detect which params need perturbations | `shapes = get_weight_shapes(params)` |
| `generate_perturbations(shapes, pop, rank, sigma, gen, dtype)` | Generate all perturbations in one call | `perts = generate_perturbations(shapes, 2048, 4, 0.1, gen, torch.float32)` |
| `perturbed_forward(x, W, b, perts, key)` | Perturbed linear layer | `h = perturbed_forward(x, W, b, perts, 'layer1.weight')` |
| `perturbed_conv2d(x, W, perts, key, padding)` | Perturbed conv layer (in recipes.py) | `h = perturbed_conv2d(x, W, perts, 'conv1.weight', padding=1)` |
| `compute_gradients(fitnesses, perts, pop)` | Compute all gradients at once | `grads = compute_gradients(f, perts, pop)` |
| `update_params(params, grads, lr)` | Update all params in-place | `update_params(params, grads, 0.1)` |
| `normalize_fitnesses(fitnesses)` | Zero-mean, unit-variance normalization | `f = normalize_fitnesses(episode_returns)` |

### Raw Primitives API (Maximum Control)

For when you need fine-grained control over each layer.

| Function | Purpose |
|----------|---------|
| `generate_lowrank_perturbations(pop, out_dim, in_dim, rank, sigma, gen, dtype)` | Generate A, B for one weight matrix |
| `perturbed_linear(x, W, b, A_scaled, B)` | Apply perturbed linear to input |
| `apply_lowrank_perturbation(x, B, A_scaled)` | Just the perturbation (no base W) |
| `compute_es_gradient(fitnesses, A_scaled, B, pop)` | Gradient for one weight matrix |

---

## Hyperparameter Guide

### What Each Hyperparameter Does

| Param | Default | Effect |
|-------|---------|--------|
| `population_size` | 2048 | More = smoother gradients, slower per-step. Use 2048-4096 for most tasks. |
| `rank` | 4 | Low-rank dimension. Higher = more expressive noise, more memory. 4 works well. |
| `sigma` | 0.1-0.2 | Noise scale. Too high = chaotic, too low = stuck. Start 0.1-0.2, decay. |
| `lr` | 0.1 | Learning rate. ES is robust to LR—0.1 is usually fine. |
| `sigma_decay` | 0.999 | Decay sigma each epoch for fine-tuning. |
| `lr_decay` | 1.0 | Optional LR decay. Usually not needed. |
| `batch_size` | 256 | Mini-batch size for SL. 64-256 works well. |

### Recommended Settings by Task

| Task | population | rank | sigma | lr | Notes |
|------|-----------|------|-------|-----|-------|
| CartPole | 2048 | 4 | 0.2 | 0.1 | Solves in ~12 epochs, 1M+ steps/s |
| MNIST MLP | 4096 | 4 | 0.15 | 0.1 | ~88% acc in 100 epochs, 0.02s/epoch |
| MNIST CNN | 2048 | 4 | 0.1 | 0.1 | ~75% acc, 0.13s/epoch (grouped conv) |
| Custom RL | 2048-4096 | 4 | 0.1-0.3 | 0.05-0.1 | Start high sigma, decay |

---

## The Math (Optional)

### Low-Rank Perturbation

For weight matrix $W \in \mathbb{R}^{m \times n}$, the perturbation is:

$$\Delta W = A \cdot B^T$$

where $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{n \times r}$ with rank $r \ll \min(m, n)$.

### ES Gradient Estimate

$$\nabla_W \approx \frac{1}{\sqrt{N}} \sum_{i=1}^{N} f_i \cdot A_i \cdot B_i^T$$

where:
- $f_i$ = normalized fitness for population member $i$
- $N$ = population size
- $A_i, B_i$ = perturbation factors for member $i$

### Why Low-Rank?

| Full ES | EGGROLL |
|---------|---------|
| Store $N \times m \times n$ perturbations | Store $N \times (m + n) \times r$ factors |
| For 256×784 weight, N=2048: **3.2 GB** | Same setting, rank=4: **16 MB** |

That's **200x memory reduction** with minimal loss in gradient quality.

---

## Troubleshooting

### "CUDA out of memory"
- Reduce `population_size` (try 1024)
- Reduce `batch_size` for SL
- Check if other processes are using GPU

### "Training doesn't converge"
- Increase `sigma` (try 0.2-0.3)
- Increase `population_size`
- Check your fitness function (higher = better)

### "Results are noisy/unstable"
- Increase `population_size`
- Add `sigma_decay` (0.999)
- For SL: increase `batch_size`

### "CartPole doesn't solve"
- Make sure you're using `gymnasium` not `gym`
- Check that envs are vectorized: `gym.make_vec(...)`
- Verify episode returns are summed correctly

---

## Performance

- **torch.compile:** All forward passes are JIT-compiled for JAX-parity speed
- **Throughput:** 1M+ steps/sec on CartPole (RTX 3090)
- **Grouped conv:** `perturbed_conv2d` uses grouped convolution for efficient batched CNN training
- **Memory:** Population 4096 + batch 256 uses ~1-2GB VRAM for MNIST
- **TF32 enabled:** Automatic TensorFloat32 for faster matmul on Ampere+ GPUs

### Caveat: Perturbation Generation

The `generate_perturbations()` function uses `torch.Generator` which is **not `torch.compile` compatible**. This means perturbation generation runs eagerly (not fused into the compiled graph).

**In practice, this doesn't matter for models <10M params:**
- Perturbation generation is only **~5% of total step time**
- The forward pass dominates (95%) and IS fully compiled
- For pop=2048, dim=256, rank=8: generation takes ~0.06ms vs ~1.3ms for forward

JAX's approach (counter-based Philox PRNG) would allow fusing generation into the compiled graph and enable on-the-fly regeneration (saving memory). However, benchmarks show this adds ~40% overhead due to breaking fusion opportunities. Not worth it for sub-10M param models where memory isn't the bottleneck.

---

## Running All Experiments

```bash
# Individual experiments
python -m hyperscalees.torch.fncl.recipes --experiment cartpole
python -m hyperscalees.torch.fncl.recipes --experiment mnist
python -m hyperscalees.torch.fncl.recipes --experiment mnist_cnn

# Benchmarking
python -m hyperscalees.torch.fncl.recipes --experiment hyperscale
python -m hyperscalees.torch.fncl.recipes --experiment hyperscale

# Run everything
python -m hyperscalees.torch.fncl.recipes --experiment all
```


---

## File Structure

```
core.py                        # Core EGGROLL implementation
├── EggrollConfig              # Dataclass for hyperparameters
├── Dict-Based API
│   ├── get_weight_shapes      # Auto-detect perturbed weights
│   ├── generate_perturbations # Generate all perturbations
│   ├── perturbed_forward      # Perturbed linear layer
│   ├── compute_gradients      # Compute all ES gradients
│   └── update_params          # Update weights
└── Raw Primitives
    ├── generate_lowrank_perturbations
    ├── perturbed_linear
    ├── apply_lowrank_perturbation
    ├── compute_weight_perturbation
    ├── compute_es_gradient
    └── normalize_fitnesses

recipes.py                     # Experiments & recipes
├── GPU Utilities
│   ├── get_gpu_stats
│   ├── print_gpu_stats
│   └── reset_gpu_stats
├── Data Loading
│   ├── load_mnist_flat
│   └── load_mnist_2d
├── Experiment Helpers
│   ├── compute_classification_fitness
│   └── perturbed_conv2d       # Perturbed conv layer (grouped conv)
└── Experiments
    ├── main()                 # CartPole
    ├── main_mnist()           # MNIST MLP
    └── main_mnist_cnn()       # MNIST CNN
```
