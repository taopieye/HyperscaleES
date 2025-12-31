# CartPole-v1 Benchmark Results

## Overview

This benchmark replicates the **CartPole-v1** experiments from the EGGROLL paper
([Section G.1, Figure 9](https://arxiv.org/abs/...)). We compare:

- **EGGROLL** - Low-rank Evolution Strategies with efficient gradient estimation
- **PPO** - Proximal Policy Optimization (gradient-based RL baseline)
- **OpenES** - Full-rank Evolution Strategies (ES baseline)

### Why This Matters

Evolution Strategies (ES) are an attractive alternative to gradient-based RL because they:
- Don't require backpropagation through time
- Are embarrassingly parallel across population members
- Can handle non-differentiable objectives and sparse rewards

However, standard ES (like OpenES) scales poorly with parameter count due to full-rank
perturbations. **EGGROLL** addresses this by using low-rank perturbations, achieving
similar sample efficiency with dramatically reduced compute.

---

## Experiment Configuration

| Setting | Value |
|---------|-------|
| Environment | `CartPole-v1` |
| Max Steps | 50,000 |
| Seeds | 2 per method |
| Date | 2025-12-28 19:53:45 |

## Learning Curves

![Learning Curves](learning_curves.png)

*Mean return over environment steps. Shaded regions show ±1 standard deviation across seeds.*
*Dashed line at 475 indicates the 'solved' threshold for CartPole-v1.*

## Summary Table

| Method | Final Return | Best Return | Wall Time (s) | Steps to Solve | Speedup vs OpenES |
|--------|--------------|-------------|---------------|----------------|-------------------|
| OpenES (JAX) | 23.1 ± 2.8 | 478.5 ± 21.5 | 18.4 ± 3.9 | — | 1.00x (baseline) |
| PPO (Rejax) | 108.3 ± 10.6 | 108.3 ± 10.6 | 7.1 ± 0.9 | — | **2.59x** |
| EGGROLL (Torch) | 19.7 ± 0.5 | 371.5 ± 51.5 | 8.5 ± 2.2 | — | **2.16x** |

## Training Time Comparison

![Training Time](training_time.png)

## Speedup vs OpenES

![Speedup](speedup.png)

*EGGROLL's low-rank perturbations require less compute than OpenES's full-rank perturbations,*
*while PPO benefits from efficient GPU parallelization of gradient computation.*

## Final Return Distribution

![Final Returns](final_returns.png)

## Key Findings

1. **Best Final Return:** PPO (Rejax)
2. **Fastest Training:** PPO (Rejax)
3. **Solved Environment:** None reached the 475 threshold

### EGGROLL vs OpenES

- EGGROLL is **2.16x faster** than OpenES
- OpenES achieves **3.4 higher** final return

## Hyperparameters

All hyperparameters match those in Tables 3 and 19 of the EGGROLL paper.

<details>
<summary><b>OpenES (JAX)</b></summary>

| Parameter | Value |
|-----------|-------|
| `activation` | `pqn` |
| `deterministic_policy` | `True` |
| `layer_size` | `256` |
| `learning_rate` | `0.1` |
| `lr_decay` | `0.9995` |
| `n_layers` | `3` |
| `n_parallel_evaluations` | `4` |
| `optimizer` | `adamw` |
| `pop_size` | `512` |
| `rank_transform` | `True` |
| `sigma` | `0.5` |
| `sigma_decay` | `0.9995` |

</details>

<details>
<summary><b>PPO (Rejax)</b></summary>

| Parameter | Value |
|-----------|-------|
| `clip_eps` | `0.2` |
| `ent_coef` | `0.0001` |
| `gae_lambda` | `0.9` |
| `gamma` | `0.995` |
| `layer_size` | `256` |
| `learning_rate` | `0.0003` |
| `max_grad_norm` | `0.5` |
| `n_layers` | `3` |
| `normalize_obs` | `True` |
| `normalize_rew` | `False` |
| `num_envs` | `256` |
| `num_epochs` | `4` |
| `num_minibatches` | `32` |
| `num_steps` | `128` |
| `vf_coef` | `0.5` |

</details>

<details>
<summary><b>EGGROLL (Torch)</b></summary>

| Parameter | Value |
|-----------|-------|
| `activation` | `pqn` |
| `deterministic_policy` | `False` |
| `layer_size` | `256` |
| `learning_rate` | `0.1` |
| `lr_decay` | `0.9995` |
| `n_layers` | `3` |
| `n_parallel_evaluations` | `1` |
| `optimizer` | `sgd` |
| `pop_size` | `2048` |
| `rank` | `4` |
| `rank_transform` | `False` |
| `sigma` | `0.2` |
| `sigma_decay` | `0.999` |

</details>

## Individual Run Details

<details>
<summary>Click to expand per-seed results</summary>

| Method | Seed | Final Return | Best Return | Wall Time (s) | Steps to Solve |
|--------|------|--------------|-------------|---------------|----------------|
| OpenES (JAX) | 42 | 25.9 | 500.0 | 22.3 | — |
| OpenES (JAX) | 142 | 20.3 | 457.0 | 14.4 | — |
| PPO (Rejax) | 42 | 119.0 | 119.0 | 8.0 | — |
| PPO (Rejax) | 142 | 97.7 | 97.7 | 6.1 | — |
| EGGROLL (Torch) | 42 | 20.2 | 320.0 | 6.3 | — |
| EGGROLL (Torch) | 142 | 19.3 | 423.0 | 10.7 | — |

</details>

---

## Reproducing These Results

```bash
# Install dependencies
uv sync

# Run the benchmark
uv run python benchmarks/benchmark_cartpole.py \
    --methods openes ppo torch_eggroll \
    --num-seeds 2 \
    --max-steps 50000
```
