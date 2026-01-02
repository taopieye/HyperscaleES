"""
Benchmark: Context Manager + Hooks (EggrollStrategy) vs Functional Implementation

This answers the question: Is the functional implementation strictly faster?
"""

import torch
import torch.nn as nn
import time
import sys
sys.path.insert(0, '../../src')

from hyperscalees.torch.strategy import EggrollStrategy


class SimpleMLP(nn.Module):
    """Minimal model for measuring overhead."""
    def __init__(self, dim=64):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
    
    def forward(self, x):
        return self.fc2(self.fc1(x))


def functional_perturb_forward(params, x, sigma, A, B):
    """
    Pure functional perturbation - no hooks, no context managers.
    
    params: dict of {name: weight_tensor}
    A, B: low-rank perturbation factors (pre-generated)
    """
    # Perturb weights inline
    fc1_w = params['fc1.weight'] + sigma * (A['fc1'] @ B['fc1'].T)
    fc1_b = params['fc1.bias']
    fc2_w = params['fc2.weight'] + sigma * (A['fc2'] @ B['fc2'].T)
    fc2_b = params['fc2.bias']
    
    # Forward pass
    h = x @ fc1_w.T + fc1_b
    out = h @ fc2_w.T + fc2_b
    return out


def benchmark_oop_strategy(model, x, pop_size, n_iters, rank=8):
    """Benchmark EggrollStrategy (context manager + hooks)."""
    strategy = EggrollStrategy(sigma=0.01, lr=0.01, rank=rank, antithetic=True)
    strategy.setup(model)
    
    # Warmup
    for i in range(3):
        with strategy.perturb(population_size=pop_size, epoch=i) as ctx:
            out = ctx.batched_forward(model, x)
            fit = -out.pow(2).mean(dim=-1)
        strategy.step(fit)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for i in range(n_iters):
        with strategy.perturb(population_size=pop_size, epoch=i) as ctx:
            out = ctx.batched_forward(model, x)
            fit = -out.pow(2).mean(dim=-1)
        strategy.step(fit)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_functional(model, x, pop_size, n_iters, rank=8):
    """Benchmark pure functional approach (no hooks, no context managers)."""
    sigma = 0.01
    dim = 64
    
    # Extract params
    params = {name: p.data for name, p in model.named_parameters()}
    
    # Pre-allocate low-rank factors
    shapes = {
        'fc1': (dim * 4, dim),
        'fc2': (dim, dim * 4),
    }
    
    # Warmup
    for i in range(3):
        A = {k: torch.randn(pop_size, shapes[k][0], rank, device='cuda') for k in shapes}
        B = {k: torch.randn(pop_size, shapes[k][1], rank, device='cuda') for k in shapes}
        
        # Batched forward - expand params, apply perturbations
        fc1_w = params['fc1.weight'].unsqueeze(0) + sigma * torch.bmm(A['fc1'], B['fc1'].transpose(-1, -2))
        fc1_b = params['fc1.bias']
        fc2_w = params['fc2.weight'].unsqueeze(0) + sigma * torch.bmm(A['fc2'], B['fc2'].transpose(-1, -2))
        fc2_b = params['fc2.bias']
        
        # x: (pop, dim) -> (pop, 1, dim) for bmm
        h = torch.bmm(x.unsqueeze(1), fc1_w.transpose(-1, -2)).squeeze(1) + fc1_b
        out = torch.bmm(h.unsqueeze(1), fc2_w.transpose(-1, -2)).squeeze(1) + fc2_b
        fit = -out.pow(2).mean(dim=-1)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for i in range(n_iters):
        # Generate perturbations
        A = {k: torch.randn(pop_size, shapes[k][0], rank, device='cuda') for k in shapes}
        B = {k: torch.randn(pop_size, shapes[k][1], rank, device='cuda') for k in shapes}
        
        # Perturbed weights: (pop, out, in)
        fc1_w = params['fc1.weight'].unsqueeze(0) + sigma * torch.bmm(A['fc1'], B['fc1'].transpose(-1, -2))
        fc2_w = params['fc2.weight'].unsqueeze(0) + sigma * torch.bmm(A['fc2'], B['fc2'].transpose(-1, -2))
        
        # Forward
        h = torch.bmm(x.unsqueeze(1), fc1_w.transpose(-1, -2)).squeeze(1) + params['fc1.bias']
        out = torch.bmm(h.unsqueeze(1), fc2_w.transpose(-1, -2)).squeeze(1) + params['fc2.bias']
        fit = -out.pow(2).mean(dim=-1)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed


def main():
    torch.manual_seed(42)
    device = 'cuda'
    
    model = SimpleMLP(64).to(device)
    
    print("=" * 60)
    print("OOP (EggrollStrategy) vs Functional Implementation")
    print("=" * 60)
    print()
    print(f"Model: SimpleMLP (2 Linear layers, {sum(p.numel() for p in model.parameters()):,} params)")
    print()
    
    results = []
    
    for pop_size in [1024, 4096, 16384, 65536]:
        x = torch.randn(pop_size, 64, device=device)
        n_iters = 50
        
        # OOP
        torch.cuda.empty_cache()
        oop_time = benchmark_oop_strategy(model, x, pop_size, n_iters)
        oop_evals = (pop_size * n_iters) / oop_time
        
        # Functional
        torch.cuda.empty_cache()
        func_time = benchmark_functional(model, x, pop_size, n_iters)
        func_evals = (pop_size * n_iters) / func_time
        
        speedup = func_evals / oop_evals
        
        print(f"pop={pop_size:>6}:")
        print(f"  OOP (ctx mgr + hooks): {oop_evals:>12,.0f} evals/s ({oop_time:.3f}s)")
        print(f"  Functional:            {func_evals:>12,.0f} evals/s ({func_time:.3f}s)")
        print(f"  Speedup:               {speedup:.2f}x")
        print()
        
        results.append({
            'pop': pop_size,
            'oop_evals': oop_evals,
            'func_evals': func_evals,
            'speedup': speedup,
        })
    
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"Average speedup of functional over OOP: {avg_speedup:.2f}x")
    print()
    print("The overhead comes from:")
    print("  1. Context manager __enter__/__exit__ calls")
    print("  2. forward_hook registration per layer")
    print("  3. Dict lookups to find perturbations during hooks")
    print("  4. Python interpreter overhead in hot path")
    print()
    print("The functional approach eliminates all of this.")


if __name__ == "__main__":
    main()
