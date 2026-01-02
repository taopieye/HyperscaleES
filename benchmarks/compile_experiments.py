"""
Compilation experiments comparing JAX vs Torch EGGROLL performance.

Results (RTX 4090, Pop=4096, MNIST MLP 784->256->10):
- JAX:             inf=4.54ms  full=8.58ms  eff=52.9%
- Torch (eager):   inf=5.90ms  full=11.32ms eff=52.1%  (0.76x JAX)
- Torch (compile): inf=4.53ms  full=8.71ms  eff=52.0%  (0.99x JAX)

Key finding: torch.compile closes the performance gap with JAX.
"""

import torch
import jax
import jax.numpy as jnp
import time
import math
import optax
from hyperscalees.noiser.eggroll import EggRoll
from hyperscalees.models.common import MLP, simple_es_tree_key


def benchmark_jax_vs_torch_compiled(pop_size: int = 4096):
    """
    Head-to-head comparison: JAX vs Torch (eager and compiled).
    """
    torch.set_float32_matmul_precision('high')
    
    print('JAX vs Torch (Compiled) - Final Comparison')
    print('=' * 60)
    print()
    
    in_dim, hidden_dim, out_dim = 784, 256, 10
    batch_size = 128
    rank = 4
    sigma = 0.1
    seed = 42
    
    print(f'Population: {pop_size}')
    
    # === JAX ===
    jax.clear_caches()
    key = jax.random.key(seed)
    model_key = jax.random.fold_in(key, 0)
    es_key = jax.random.fold_in(key, 1)
    
    frozen_params, params, scan_map, es_map = MLP.rand_init(
        model_key, in_dim=in_dim, out_dim=out_dim, hidden_dims=[hidden_dim],
        use_bias=True, activation='relu', dtype='float32',
    )
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, sigma=sigma, lr=0.1, solver=optax.sgd, rank=rank,
    )
    
    x_jax = jnp.ones((batch_size, in_dim))
    x_pop_jax = jnp.broadcast_to(x_jax, (pop_size, batch_size, in_dim))
    
    @jax.jit
    def jax_inference(params, x):
        def fwd(x):
            h = jax.nn.relu(x @ params['0']['weight'].T + params['0']['bias'])
            return h @ params['1']['weight'].T + params['1']['bias']
        return jax.vmap(fwd)(x)
    
    def forward_noisy(noiser_params, params, iterinfo, x):
        return MLP.forward(EggRoll, frozen_noiser_params, noiser_params, frozen_params, params, es_tree_key, iterinfo, x)
    
    jit_forward = jax.jit(jax.vmap(lambda n, p, i, x: forward_noisy(n, p, i, x), in_axes=(None, None, 0, 0)))
    
    @jax.jit
    def jax_do_update(noiser_params, params, fitnesses, iterinfos):
        return EggRoll.do_updates(frozen_noiser_params, noiser_params, params, es_tree_key, fitnesses, iterinfos, es_map)
    
    def jax_full_step():
        iterinfo = (jnp.zeros(pop_size, dtype=jnp.int32), jnp.arange(pop_size))
        logits = jit_forward(noiser_params, params, iterinfo, x_pop_jax)
        fitnesses = logits.mean(axis=(1, 2))
        fitnesses = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
        return jax_do_update(noiser_params, params, fitnesses, iterinfo)
    
    # Warmup JAX
    _ = jax_inference(params, x_pop_jax).block_until_ready()
    _ = jax_full_step()
    for _ in range(20):
        _ = jax_inference(params, x_pop_jax).block_until_ready()
        _ = jax_full_step()
    
    # Time JAX
    t0 = time.perf_counter()
    for _ in range(100):
        _ = jax_inference(params, x_pop_jax).block_until_ready()
    jax_inf = (time.perf_counter() - t0) / 100 * 1000
    
    t0 = time.perf_counter()
    for _ in range(100):
        _ = jax_full_step()
    jax_full = (time.perf_counter() - t0) / 100 * 1000
    
    jax_eff = (jax_inf / jax_full) * 100
    
    print(f'  JAX:            inf={jax_inf:.2f}ms  full={jax_full:.2f}ms  eff={jax_eff:.1f}%')
    
    # === Torch ===
    torch.cuda.empty_cache()
    device = 'cuda'
    dtype = torch.float32
    
    W1 = torch.randn(hidden_dim, in_dim, device=device, dtype=dtype)
    W2 = torch.randn(out_dim, hidden_dim, device=device, dtype=dtype)
    b1 = torch.zeros(hidden_dim, device=device, dtype=dtype)
    b2 = torch.zeros(out_dim, device=device, dtype=dtype)
    
    x = torch.ones(pop_size, batch_size, in_dim, device=device, dtype=dtype)
    
    gen = torch.Generator(device=device).manual_seed(seed)
    half_pop = pop_size // 2
    scale = sigma / (rank ** 0.5)
    
    A1_pos = torch.randn(half_pop, hidden_dim, rank, device=device, dtype=dtype, generator=gen)
    B1_pos = torch.randn(half_pop, in_dim, rank, device=device, dtype=dtype, generator=gen)
    A1 = torch.cat([A1_pos, -A1_pos], dim=0)
    B1 = torch.cat([B1_pos, B1_pos], dim=0)
    A1_scaled = A1 * scale
    
    A2_pos = torch.randn(half_pop, out_dim, rank, device=device, dtype=dtype, generator=gen)
    B2_pos = torch.randn(half_pop, hidden_dim, rank, device=device, dtype=dtype, generator=gen)
    A2 = torch.cat([A2_pos, -A2_pos], dim=0)
    B2 = torch.cat([B2_pos, B2_pos], dim=0)
    A2_scaled = A2 * scale
    
    def torch_inference():
        h = torch.relu(x @ W1.T + b1)
        return h @ W2.T + b2
    
    def torch_full_step():
        base1 = x @ W1.T + b1
        pert1 = torch.einsum('pbi,pir,pjr->pbj', x, B1, A1_scaled)
        h = torch.relu(base1 + pert1)
        base2 = h @ W2.T + b2
        pert2 = torch.einsum('pbi,pir,pjr->pbj', h, B2, A2_scaled)
        logits = base2 + pert2
        
        fitnesses = logits.mean(dim=(1, 2))
        fitnesses = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
        sqrt_N = math.sqrt(pop_size)
        f = fitnesses[:, None, None]
        grad1 = torch.einsum('nir,njr->ij', f * A1, B1) / sqrt_N
        grad2 = torch.einsum('nir,njr->ij', f * A2, B2) / sqrt_N
        return grad1, grad2
    
    # Warmup eager
    for _ in range(20):
        _ = torch_inference()
        _ = torch_full_step()
    torch.cuda.synchronize()
    
    # Time eager
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = torch_inference()
    torch.cuda.synchronize()
    eager_inf = (time.perf_counter() - t0) / 100 * 1000
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = torch_full_step()
    torch.cuda.synchronize()
    eager_full = (time.perf_counter() - t0) / 100 * 1000
    
    eager_eff = (eager_inf / eager_full) * 100
    print(f'  Torch (eager):  inf={eager_inf:.2f}ms  full={eager_full:.2f}ms  eff={eager_eff:.1f}%')
    
    # Compile
    print('  Compiling Torch...')
    compiled_inf = torch.compile(torch_inference)
    compiled_full = torch.compile(torch_full_step)
    
    for _ in range(3):
        _ = compiled_inf()
        _ = compiled_full()
    torch.cuda.synchronize()
    
    for _ in range(20):
        _ = compiled_inf()
        _ = compiled_full()
    torch.cuda.synchronize()
    
    # Time compiled
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = compiled_inf()
    torch.cuda.synchronize()
    compiled_inf_t = (time.perf_counter() - t0) / 100 * 1000
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = compiled_full()
    torch.cuda.synchronize()
    compiled_full_t = (time.perf_counter() - t0) / 100 * 1000
    
    compiled_eff = (compiled_inf_t / compiled_full_t) * 100
    print(f'  Torch (compile):inf={compiled_inf_t:.2f}ms  full={compiled_full_t:.2f}ms  eff={compiled_eff:.1f}%')
    
    print()
    print('Summary:')
    print(f'  Eager vs JAX:    {jax_full/eager_full:.2f}x (JAX is faster)')
    print(f'  Compiled vs JAX: {jax_full/compiled_full_t:.2f}x', end='')
    if compiled_full_t < jax_full:
        print(' (Torch is FASTER!)')
    elif compiled_full_t < jax_full * 1.1:
        print(' (roughly equal)')
    else:
        print(' (JAX is faster)')
    
    return {
        'pop_size': pop_size,
        'jax_inf_ms': jax_inf,
        'jax_full_ms': jax_full,
        'jax_eff': jax_eff,
        'torch_eager_inf_ms': eager_inf,
        'torch_eager_full_ms': eager_full,
        'torch_eager_eff': eager_eff,
        'torch_compiled_inf_ms': compiled_inf_t,
        'torch_compiled_full_ms': compiled_full_t,
        'torch_compiled_eff': compiled_eff,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pop-size', type=int, default=4096)
    args = parser.parse_args()
    
    benchmark_jax_vs_torch_compiled(args.pop_size)
