# CartPole-v1 Benchmark Results

**Date:** 2025-12-28 19:46:44
**Environment:** CartPole-v1
**Max Steps:** 50,000
**Seeds:** 2 per method

## Summary Table

| Method | Final Return | Best Return | Wall Time (s) | Steps to Solve | Speedup vs OpenES |
|--------|--------------|-------------|---------------|----------------|-------------------|
| OpenES (JAX) | 23.1 ± 2.8 | 478.5 ± 21.5 | 18.2 ± 1.9 | — | 1.00x (baseline) |
| PPO (Rejax) | 107.7 ± 10.6 | 107.7 ± 10.6 | 8.0 ± 0.3 | — | 2.28x |

## Performance Analysis

- **Best Final Return:** PPO (Rejax)
- **Fastest Training:** PPO (Rejax)
- **Solved Environment:** None (threshold: 475 mean return)

## Training Time Comparison

```
OpenES (JAX)         |████████████████████████████████████████| 18.2s
PPO (Rejax)          |█████████████████░░░░░░░░░░░░░░░░░░░░░░░| 8.0s
```

## Learning Curves (Mean Return vs Steps)

Data points for plotting:

### OpenES (JAX)

| Steps | Mean Return | Std |
|-------|-------------|-----|
| 9,117 | 18.0 | 0.2 |
| 18,155 | 18.2 | 0.6 |
| 27,490 | 18.1 | 0.1 |
| 37,881 | 19.9 | 0.4 |
| 49,741 | 20.9 | 2.2 |
| 63,015 | 23.1 | 2.8 |

### PPO (Rejax)

| Steps | Mean Return | Std |
|-------|-------------|-----|
| 10,000 | 24.7 | 1.8 |
| 20,000 | 34.8 | 2.8 |
| 30,000 | 54.0 | 2.2 |
| 40,000 | 72.4 | 0.6 |
| 50,000 | 93.9 | 3.3 |
| 60,000 | 107.7 | 10.6 |

## Hyperparameters

### OpenES (JAX)

| Parameter | Value |
|-----------|-------|
| activation | pqn |
| deterministic_policy | True |
| layer_size | 256 |
| learning_rate | 0.1 |
| lr_decay | 0.9995 |
| n_layers | 3 |
| n_parallel_evaluations | 4 |
| optimizer | adamw |
| pop_size | 512 |
| rank_transform | True |
| sigma | 0.5 |
| sigma_decay | 0.9995 |

### PPO (Rejax)

| Parameter | Value |
|-----------|-------|
| clip_eps | 0.2 |
| ent_coef | 0.0001 |
| gae_lambda | 0.9 |
| gamma | 0.995 |
| layer_size | 256 |
| learning_rate | 0.0003 |
| max_grad_norm | 0.5 |
| n_layers | 3 |
| normalize_obs | True |
| normalize_rew | False |
| num_envs | 256 |
| num_epochs | 4 |
| num_minibatches | 32 |
| num_steps | 128 |
| vf_coef | 0.5 |

## Individual Run Details

| Method | Seed | Final Return | Best Return | Wall Time (s) | Steps to Solve |
|--------|------|--------------|-------------|---------------|----------------|
| OpenES (JAX) | 42 | 25.9 | 500.0 | 20.1 | — |
| OpenES (JAX) | 142 | 20.3 | 457.0 | 16.3 | — |
| PPO (Rejax) | 42 | 118.3 | 118.3 | 8.3 | — |
| PPO (Rejax) | 142 | 97.1 | 97.1 | 7.7 | — |
