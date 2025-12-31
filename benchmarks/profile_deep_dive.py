#!/usr/bin/env python3
"""
Deep Dive Profiler - Find where all the time goes in the full forward pass.

The anatomy profiler showed individual ops sum to ~0.5ms but full forward is 1.4ms.
This script finds the missing time.
"""
import torch
import torch.nn as nn
import time
from contextlib import contextmanager


class CUDATimer:
    """High-precision CUDA timer."""
    
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    @contextmanager
    def time(self):
        self.start_event.record()
        yield
        self.end_event.record()
        torch.cuda.synchronize()
    
    def elapsed_ms(self) -> float:
        return self.start_event.elapsed_time(self.end_event)


def main():
    print("="*70)
    print("Deep Dive: Where Does the Time Go?")
    print("="*70)
    
    from hyperscalees.torch import EggrollStrategy
    from hyperscalees.torch.triton_kernels import (
        generate_lowrank_factors_torch,
        batched_perturbed_linear_torch,
    )
    
    device = torch.device('cuda')
    dtype = torch.float32
    pop_size = 2048
    obs_dim = 4
    act_dim = 2
    layer_size = 256
    n_layers = 3
    rank = 4
    
    # Build model
    layers = []
    in_dim = obs_dim
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(in_dim, layer_size))
        layers.append(nn.Tanh())
        in_dim = layer_size
    layers.append(nn.Linear(in_dim, act_dim))
    model = nn.Sequential(*layers).cuda()
    
    strategy = EggrollStrategy(
        sigma=0.2,
        rank=rank,
        lr=0.1,
        seed=42,
        antithetic=True,
    )
    strategy.setup(model)
    
    obs_batch = torch.randn(pop_size, obs_dim, device=device, dtype=dtype)
    member_ids = torch.arange(pop_size, device=device)
    
    timer = CUDATimer()
    num_iterations = 100
    
    # Warmup
    for warmup_epoch in range(10):
        with strategy.perturb(population_size=pop_size, epoch=warmup_epoch) as pop:
            with torch.no_grad():
                _ = pop.batched_forward(model, obs_batch)
    torch.cuda.synchronize()
    
    # Profile full forward with breakdown
    print("\n--- Full Forward Pass Breakdown ---")
    
    # Time the context manager overhead
    context_times = []
    for epoch in range(num_iterations):
        with timer.time():
            with strategy.perturb(population_size=pop_size, epoch=epoch) as pop:
                pass  # Just the context manager
        context_times.append(timer.elapsed_ms())
    
    print(f"Context manager overhead: {sum(context_times)/len(context_times):.3f}ms")
    
    # Time each layer manually
    print("\n--- Per-Layer Timing ---")
    
    # Get param info
    param_ids = {}
    for idx, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name:
            param_ids[name] = idx
    
    # Time layer 0: Linear(4, 256)
    linear0 = model[0]
    layer0_noise_times = []
    layer0_perturb_times = []
    
    for epoch in range(num_iterations):
        # Time noise generation
        with timer.time():
            A0, B0 = generate_lowrank_factors_torch(
                out_features=256, in_features=4, rank=rank,
                seed=42, epoch=epoch, member_ids=member_ids, param_id=0,
                sigma=0.2, noise_reuse=0, antithetic=True,
                device=device, dtype=dtype,
            )
        layer0_noise_times.append(timer.elapsed_ms())
        
        # Time perturbed linear
        with timer.time():
            out0 = batched_perturbed_linear_torch(
                x=obs_batch, weight=linear0.weight, bias=linear0.bias,
                A=A0, B=B0, member_ids=member_ids,
            )
        layer0_perturb_times.append(timer.elapsed_ms())
    
    print(f"Layer 0 (4->256):")
    print(f"  Noise gen: {sum(layer0_noise_times)/len(layer0_noise_times):.3f}ms")
    print(f"  Perturbed linear: {sum(layer0_perturb_times)/len(layer0_perturb_times):.3f}ms")
    
    # Time layer 2: Linear(256, 256)
    linear2 = model[2]
    layer2_noise_times = []
    layer2_perturb_times = []
    x_hidden = torch.randn(pop_size, 256, device=device, dtype=dtype)
    
    for epoch in range(num_iterations):
        with timer.time():
            A2, B2 = generate_lowrank_factors_torch(
                out_features=256, in_features=256, rank=rank,
                seed=42, epoch=epoch, member_ids=member_ids, param_id=1,
                sigma=0.2, noise_reuse=0, antithetic=True,
                device=device, dtype=dtype,
            )
        layer2_noise_times.append(timer.elapsed_ms())
        
        with timer.time():
            out2 = batched_perturbed_linear_torch(
                x=x_hidden, weight=linear2.weight, bias=linear2.bias,
                A=A2, B=B2, member_ids=member_ids,
            )
        layer2_perturb_times.append(timer.elapsed_ms())
    
    print(f"Layer 2 (256->256):")
    print(f"  Noise gen: {sum(layer2_noise_times)/len(layer2_noise_times):.3f}ms")
    print(f"  Perturbed linear: {sum(layer2_perturb_times)/len(layer2_perturb_times):.3f}ms")
    
    # Time layer 4: Linear(256, 2)
    linear4 = model[4]
    layer4_noise_times = []
    layer4_perturb_times = []
    
    for epoch in range(num_iterations):
        with timer.time():
            A4, B4 = generate_lowrank_factors_torch(
                out_features=2, in_features=256, rank=rank,
                seed=42, epoch=epoch, member_ids=member_ids, param_id=2,
                sigma=0.2, noise_reuse=0, antithetic=True,
                device=device, dtype=dtype,
            )
        layer4_noise_times.append(timer.elapsed_ms())
        
        with timer.time():
            out4 = batched_perturbed_linear_torch(
                x=x_hidden, weight=linear4.weight, bias=linear4.bias,
                A=A4, B=B4, member_ids=member_ids,
            )
        layer4_perturb_times.append(timer.elapsed_ms())
    
    print(f"Layer 4 (256->2):")
    print(f"  Noise gen: {sum(layer4_noise_times)/len(layer4_noise_times):.3f}ms")
    print(f"  Perturbed linear: {sum(layer4_perturb_times)/len(layer4_perturb_times):.3f}ms")
    
    # Time activations
    activation_times = []
    for _ in range(num_iterations):
        with timer.time():
            _ = torch.tanh(x_hidden)
            _ = torch.tanh(x_hidden)
        activation_times.append(timer.elapsed_ms())
    
    print(f"\nActivations (2x tanh): {sum(activation_times)/len(activation_times):.3f}ms")
    
    # Full forward for comparison
    full_times = []
    for epoch in range(num_iterations):
        with timer.time():
            with strategy.perturb(population_size=pop_size, epoch=epoch) as pop:
                with torch.no_grad():
                    _ = pop.batched_forward(model, obs_batch)
        full_times.append(timer.elapsed_ms())
    
    print(f"\nFull forward pass: {sum(full_times)/len(full_times):.3f}ms")
    
    # Sum up expected time
    expected_time = (
        sum(context_times)/len(context_times) +
        sum(layer0_noise_times)/len(layer0_noise_times) +
        sum(layer0_perturb_times)/len(layer0_perturb_times) +
        sum(layer2_noise_times)/len(layer2_noise_times) +
        sum(layer2_perturb_times)/len(layer2_perturb_times) +
        sum(layer4_noise_times)/len(layer4_noise_times) +
        sum(layer4_perturb_times)/len(layer4_perturb_times) +
        sum(activation_times)/len(activation_times)
    )
    
    print(f"\nExpected (sum of parts): {expected_time:.3f}ms")
    print(f"Actual (full forward):   {sum(full_times)/len(full_times):.3f}ms")
    print(f"Unaccounted time:        {sum(full_times)/len(full_times) - expected_time:.3f}ms")
    
    # Profile the hook mechanism
    print("\n--- Hook vs Direct Execution ---")
    
    # Direct execution (no hooks, no context manager)
    direct_times = []
    for epoch in range(num_iterations):
        with timer.time():
            with torch.no_grad():
                # Layer 0
                A0, B0 = generate_lowrank_factors_torch(
                    out_features=256, in_features=4, rank=rank,
                    seed=42, epoch=epoch, member_ids=member_ids, param_id=0,
                    sigma=0.2, noise_reuse=0, antithetic=True,
                    device=device, dtype=dtype,
                )
                x = batched_perturbed_linear_torch(obs_batch, linear0.weight, linear0.bias, A0, B0, member_ids)
                x = torch.tanh(x)
                
                # Layer 2
                A2, B2 = generate_lowrank_factors_torch(
                    out_features=256, in_features=256, rank=rank,
                    seed=42, epoch=epoch, member_ids=member_ids, param_id=1,
                    sigma=0.2, noise_reuse=0, antithetic=True,
                    device=device, dtype=dtype,
                )
                x = batched_perturbed_linear_torch(x, linear2.weight, linear2.bias, A2, B2, member_ids)
                x = torch.tanh(x)
                
                # Layer 4
                A4, B4 = generate_lowrank_factors_torch(
                    out_features=2, in_features=256, rank=rank,
                    seed=42, epoch=epoch, member_ids=member_ids, param_id=2,
                    sigma=0.2, noise_reuse=0, antithetic=True,
                    device=device, dtype=dtype,
                )
                x = batched_perturbed_linear_torch(x, linear4.weight, linear4.bias, A4, B4, member_ids)
        direct_times.append(timer.elapsed_ms())
    
    print(f"Direct execution (no hooks): {sum(direct_times)/len(direct_times):.3f}ms")
    print(f"Full forward (with hooks):   {sum(full_times)/len(full_times):.3f}ms")
    print(f"Hook overhead:               {sum(full_times)/len(full_times) - sum(direct_times)/len(direct_times):.3f}ms")
    
    # === CUDA Profiling ===
    print("\n--- CUDA Kernel Analysis ---")
    print("Running torch.profiler to see kernel-level details...")
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for epoch in range(10):
            with strategy.perturb(population_size=pop_size, epoch=epoch) as pop:
                with torch.no_grad():
                    _ = pop.batched_forward(model, obs_batch)
    
    # Print top CUDA kernels
    print("\nTop CUDA kernels by time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
