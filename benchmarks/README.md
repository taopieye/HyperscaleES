# EGGROLL Benchmarks

This directory contains benchmarking scripts for comparing different implementations of EGGROLL.

## Benchmark Scripts

### `benchmark_batched_forward.py`

Compares the batched forward pass performance between JAX and PyTorch implementations.

**Usage:**
```bash
# Run all benchmarks with default settings
python benchmark_batched_forward.py

# Run only PyTorch benchmarks
python benchmark_batched_forward.py --frameworks torch

# Run only JAX benchmarks
python benchmark_batched_forward.py --frameworks jax

# Custom population sizes
python benchmark_batched_forward.py --pop-sizes 32 64 128 256

# Custom model configuration
python benchmark_batched_forward.py --input-dim 768 --hidden-dim 3072 --num-layers 4

# Full options
python benchmark_batched_forward.py \
    --frameworks jax torch \
    --input-dim 512 \
    --hidden-dim 2048 \
    --output-dim 512 \
    --num-layers 3 \
    --rank 4 \
    --sigma 0.1 \
    --warmup 10 \
    --iters 50 \
    --pop-sizes 32 64 128 256 512 1024 2048 \
    --output results.json
```

**What it measures:**
- Forward pass time for evaluating all population members
- Throughput (samples per second)
- Time variance across iterations

**Default population sizes tested:** 32, 64, 128, 256, 512, 1024, 2048

## Requirements

- **JAX benchmarks:** JAX with GPU support
- **PyTorch benchmarks:** PyTorch with CUDA

## Output

Results are printed to console in a summary table and saved to a JSON file.

Example output:
```
BENCHMARK SUMMARY
================================================================================
  Pop Size |     JAX (ms)    |   JAX (samp/s)  | PyTorch (ms)    | PyTorch (samp/s)
--------------------------------------------------------------------------------
        32 |        1.234 ms |        25929/s  |        0.987 ms |        32421/s
        64 |        1.456 ms |        43956/s  |        1.123 ms |        56990/s
       ...

SPEEDUP (JAX time / PyTorch time)
----------------------------------------
  Pop    32: PyTorch is 1.25x faster
  Pop    64: PyTorch is 1.30x faster
```
