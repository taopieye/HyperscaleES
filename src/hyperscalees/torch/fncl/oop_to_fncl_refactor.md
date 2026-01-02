# PyTorch EGGROLL Functional Refactor

> **Design Document** for refactoring PyTorch EGGROLL from OOP (context managers + hooks) to a pure functional implementation.

**Source of Truth:** The JAX implementation in `src/hyperscalees/noiser/eggroll.py`

**Reference:** EGGROLL paper at `HyperscaleES/paper.pdf`

**Note:** The quilt project's TinyPerceiver is a domain-specific GNN implementation. It serves as an *example* of functional EGGROLL patterns, but our goal is parity with the JAX reference for general-purpose ML/optimization.

---

## ⚠️ CRITICAL IMPLEMENTATION REQUIREMENTS ⚠️

### The Problem with the OOP Implementation

The current `EggrollStrategy` uses context managers and forward hooks:

```python
with strategy.perturb(population_size=64, epoch=epoch) as ctx:
    outputs = ctx.batched_forward(model, x)
```

**Measured overhead: 75-100x slower than functional.**

| Population | OOP (hooks + ctx mgr) | Functional | Speedup |
|------------|----------------------|------------|---------|
| 1,024 | 10,894 evals/s | 1,086,548 evals/s | **100x** |
| 4,096 | 11,768 evals/s | 1,097,679 evals/s | **93x** |
| 16,384 | 11,917 evals/s | 1,112,636 evals/s | **93x** |
| 65,536 | 11,385 evals/s | 145,066 evals/s | **13x** |

The overhead comes from:
1. `__enter__`/`__exit__` calls on the context manager
2. `register_forward_hook` on every Linear layer
3. Dict lookups to find perturbations during hook execution
4. Python interpreter overhead in the hot path

### What NOT To Do

❌ **DO NOT** iterate through population members:
```python
# BAD: Serial evaluation defeats the entire purpose
for member_id in range(population_size):
    out[member_id] = model(x[member_id])
```

❌ **DO NOT** iterate through parameters:
```python
# BAD: Serial loop in Python
for name, param in model.named_parameters():
    perturbed[name] = param + perturbation[name]
```

❌ **DO NOT** materialize full perturbation matrices:
```python
# BAD: O(m × n) memory per population member
perturbed_W = W + A @ B.T  
```

---

## Approach: Single-File, Minimal, Emergent Design

Instead of designing upfront, we build a **single-file implementation** that solves CartPole-v1. No OOP for the sake of it. Gross code is fine. Let the design emerge from what actually works.

```
src/hyperscalees/torch/fncl/
├── oop_to_fncl_refactor.md   # This document
└── eggroll_fncl.py           # THE implementation (single file, runnable)
```

### Constraints

- **Imports:** `torch`, `rich`, and whatever comes up naturally (gymnasium for CartPole)
- **Entry point:** `python -m hyperscalees.torch.fncl.eggroll_fncl` runs CartPole experiment
- **No premature abstraction:** Write the code that works first, refactor later
- **GPU only:** No CPU fallbacks, no apologies

### The Core Insight (from JAX)

The JAX `do_mm` is the atomic operation:

```python
# JAX reference:
def do_mm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
    base_ans = x @ param.T
    if iterinfo is None:
        return base_ans
    A, B = get_lora_update_params(...)
    return base_ans + x @ B @ A.T
```

That's it. `x @ W.T + x @ B @ A.T`. Never materialize `W + A @ B.T`.

---

## Implementation Plan

### Step 1: Bare Bones CartPole

Write the minimum code to:
1. Initialize a 2-layer MLP as flat tensors (no nn.Module)
2. Generate low-rank perturbations for a population
3. Run batched forward: `x @ W.T + x @ B @ A.T`
4. Compute fitness (episode returns)
5. ES update: `grad = einsum('n,nir,njr->ij', fitnesses, A, B) / N`
6. Repeat until solved

### Step 2: Verify Against JAX

Once CartPole works, compare outputs against the JAX implementation with the same seed. This is a sanity check, not a formal test suite.

### Step 3: Extract Patterns

After we have working code, identify what wants to be a function vs what's just noise. Refactor only what's clearly reusable.

---

## Non-Goals (For Now)

- Multi-GPU support
- Kernel fusion / compilation optimizations
- Formal test suite
- Clean API design
- Backwards compatibility with OOP implementation

We optimize for **one GPU, maximum throughput, working code**.

---

## References

- JAX EGGROLL: `src/hyperscalees/noiser/eggroll.py`
- EGGROLL paper: `HyperscaleES/paper.pdf`
- Overhead benchmark: `benchmarks/eggroll_trm/benchmark_overhead.py`
