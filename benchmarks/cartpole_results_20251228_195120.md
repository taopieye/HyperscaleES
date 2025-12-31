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
| Max Steps | 30,000 |
| Seeds | 2 per method |
| Date | 2025-12-28 19:51:21 |

## Learning Curves

![Learning Curves](learning_curves.png)

*Mean return over environment steps. Shaded regions show ±1 standard deviation across seeds.*
*Dashed line at 475 indicates the 'solved' threshold for CartPole-v1.*

## Summary Table

| Method | Final Return | Best Return | Wall Time (s) | Steps to Solve | Speedup vs OpenES |
|--------|--------------|-------------|---------------|----------------|-------------------|
| PPO (Rejax) | 73.1 ± 0.1 | 73.1 ± 0.1 | 6.9 ± 0.9 | — | — |

## Training Time Comparison

![Training Time](training_time.png)

## Final Return Distribution

![Final Returns](final_returns.png)

## Key Findings

1. **Best Final Return:** PPO (Rejax)
2. **Fastest Training:** PPO (Rejax)
3. **Solved Environment:** None reached the 475 threshold

## Hyperparameters

All hyperparameters match those in Tables 3 and 19 of the EGGROLL paper.

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

## Individual Run Details

<details>
<summary>Click to expand per-seed results</summary>

| Method | Seed | Final Return | Best Return | Wall Time (s) | Steps to Solve |
|--------|------|--------------|-------------|---------------|----------------|
| PPO (Rejax) | 42 | 73.2 | 73.2 | 7.8 | — |
| PPO (Rejax) | 142 | 73.1 | 73.1 | 6.0 | — |

</details>

---

## Reproducing These Results

```bash
# Install dependencies
uv sync

# Run the benchmark
uv run python benchmarks/benchmark_cartpole.py \
    --methods ppo \
    --num-seeds 2 \
    --max-steps 30000
```
