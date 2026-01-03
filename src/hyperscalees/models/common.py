import jax
import jax.numpy as jnp

from .base_model import Model, CommonInit, CommonParams

PARAM = 0
MM_PARAM = 1
EMB_PARAM = 2
EXCLUDED=3

def layer_norm(x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    std = jnp.sqrt(var + eps)
    return (x - mean) / std

ACTIVATIONS = {
    'relu': jax.nn.relu,
    'silu': jax.nn.silu,
    'pqn': lambda x: jax.nn.relu(layer_norm(x))
}

def recursive_scan_split(param, base_key, scan_tuple):
    # scan_tuple = tuple() implies no split
    if len(scan_tuple) == 0:
        return base_key
    # otherwise, it is (0, 1, ...)
    split_keys = jax.random.split(base_key, param.shape[scan_tuple[0]])
    return jax.vmap(recursive_scan_split, in_axes=(None, 0, None))(param, split_keys, scan_tuple[1:])

def simple_es_tree_key(params, base_key, scan_map):
    vals, treedef = jax.tree.flatten(params)
    all_keys = jax.random.split(base_key, len(vals))
    partial_key_tree = jax.tree.unflatten(treedef, all_keys)
    return jax.tree.map(recursive_scan_split, params, partial_key_tree, scan_map)

def merge_inits(**kwargs):
    params = {}
    frozen_params = {}
    scan_map = {}
    es_map = {}
    for k in kwargs:
        params[k] = kwargs[k].params #k_params
        scan_map[k] = kwargs[k].scan_map #k_scan_map
        es_map[k] = kwargs[k].es_map #k_es_map
        if kwargs[k].frozen_params is not None:
            frozen_params[k] = kwargs[k].frozen_params
    if not frozen_params:
        frozen_params = None

    return CommonInit(frozen_params, params, scan_map, es_map)

def merge_frozen(common, **kwargs):
    new_frozen_params = common.frozen_params or {}
    new_frozen_params = new_frozen_params | kwargs
    return common._replace(frozen_params=new_frozen_params)

def call_submodule(cls, name, common_params, *args, **kwargs):
    sub_common_params = common_params._replace(
        frozen_params=common_params.frozen_params[name] if common_params.frozen_params and name in common_params.frozen_params else None,
        params=common_params.params[name],
        es_tree_key=common_params.es_tree_key[name]
    )
    return cls._forward(sub_common_params, *args, **kwargs)

class Parameter(Model):
    @classmethod
    def rand_init(cls, key, shape, scale, raw_value, dtype, *args, **kwargs):
        if raw_value is not None:
            params = raw_value.astype(dtype=dtype)
        else:
            params = (jax.random.normal(key, shape) * scale).astype(dtype=dtype)
        
        frozen_params = None
        scan_map = ()
        es_map = PARAM
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, *args, **kwargs):
        return common_params.noiser.get_noisy_standard(common_params.frozen_noiser_params, common_params.noiser_params, common_params.params, common_params.es_tree_key, common_params.iterinfo)

class MM(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, dtype, *args, **kwargs):
        scale = 1 / jnp.sqrt(in_dim)
        params = (jax.random.normal(key, (out_dim, in_dim)) * scale).astype(dtype=dtype)
        frozen_params = None
        scan_map = ()
        es_map = MM_PARAM
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        return common_params.noiser.do_mm(common_params.frozen_noiser_params, common_params.noiser_params, common_params.params, common_params.es_tree_key, common_params.iterinfo, x)

class TMM(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, dtype, *args, **kwargs):
        scale = 1 / jnp.sqrt(in_dim)
        params = jax.random.normal(key, (in_dim, out_dim), dtype=dtype) * scale
        frozen_params = None
        scan_map = ()
        es_map = MM_PARAM
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        return common_params.noiser.do_Tmm(common_params.frozen_noiser_params, common_params.noiser_params, common_params.params, common_params.es_tree_key, common_params.iterinfo, x)

class Embedding(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, dtype, *args, **kwargs):
        scale = 1 / jnp.sqrt(in_dim)
        params = jax.random.normal(key, (in_dim, out_dim), dtype=dtype) * scale
        frozen_params = None
        scan_map = ()
        es_map = EMB_PARAM
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        return common_params.noiser.do_emb(common_params.frozen_noiser_params, common_params.noiser_params, common_params.params, common_params.es_tree_key, common_params.iterinfo, x)

class Linear(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, use_bias, dtype, *args, **kwargs):
        if use_bias:
            return merge_inits(
                weight=MM.rand_init(key, in_dim, out_dim, dtype),
                bias=Parameter.rand_init(key, None, None, jnp.zeros(out_dim, dtype=dtype), dtype)
            )
        else:
            return merge_inits(
                weight=MM.rand_init(key, in_dim, out_dim, dtype),
            )

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        ans = call_submodule(MM, 'weight', common_params, x)
        if "bias" in common_params.params:
            ans += call_submodule(Parameter, 'bias', common_params)
        return ans
            
class MLP(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, hidden_dims, use_bias, activation, dtype, *args, **kwargs):
        input_dims = [in_dim] + list(hidden_dims)
        output_dims = list(hidden_dims) + [out_dim]

        all_keys = jax.random.split(key, len(input_dims))

        merged_params = merge_inits(**{str(t): Linear.rand_init(all_keys[t], input_dims[t], output_dims[t], use_bias, dtype) for t in range(len(input_dims))})
        return merge_frozen(merged_params, activation=activation)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        num_blocks = len(common_params.params)
        for t in range(num_blocks):
            x = call_submodule(Linear, str(t), common_params, x)
            if t != num_blocks - 1:
                x = ACTIVATIONS[common_params.frozen_params['activation']](x)
        return x


# =============================================================================
# CNN Components for MNIST
# =============================================================================

class Conv2d(Model):
    """
    2D Convolution layer.
    
    Uses EXCLUDED es_map to skip perturbation - conv layers are shared across population.
    This is more memory efficient and matches typical EGGROLL usage where only
    large weight matrices (FC layers) benefit from low-rank perturbation.
    """
    @classmethod
    def rand_init(cls, key, in_channels, out_channels, kernel_size, dtype, *args, **kwargs):
        # He initialization
        fan_in = in_channels * kernel_size * kernel_size
        scale = jnp.sqrt(2.0 / fan_in)
        params = (jax.random.normal(key, (out_channels, in_channels, kernel_size, kernel_size)) * scale).astype(dtype=dtype)
        frozen_params = None
        scan_map = ()
        es_map = EXCLUDED  # No perturbation for conv layers (shared across population)
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        """
        x: (batch, channels, height, width) in NCHW format
        """
        from jax import lax
        
        # Get weights (unperturbed since es_map=EXCLUDED)
        weight = common_params.noiser.get_noisy_standard(
            common_params.frozen_noiser_params, 
            common_params.noiser_params, 
            common_params.params, 
            common_params.es_tree_key, 
            common_params.iterinfo
        )
        
        # Convolution with SAME padding
        dn = lax.conv_dimension_numbers(x.shape, weight.shape, ('NCHW', 'OIHW', 'NCHW'))
        return lax.conv_general_dilated(x, weight, (1, 1), 'SAME', dimension_numbers=dn)


class MaxPool2d(Model):
    """2D Max Pooling - no learnable parameters."""
    @classmethod
    def rand_init(cls, key, kernel_size, dtype, *args, **kwargs):
        # No learnable params, just store config in frozen_params
        frozen_params = {'kernel_size': kernel_size}
        params = {}  # Empty dict
        scan_map = {}
        es_map = {}
        return CommonInit(frozen_params, params, scan_map, es_map)
    
    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        from jax import lax
        k = common_params.frozen_params['kernel_size']
        # NCHW format: pool over H and W
        return lax.reduce_window(x, -jnp.inf, lax.max, (1, 1, k, k), (1, 1, k, k), 'VALID')


class Flatten(Model):
    """Flatten spatial dimensions - no learnable parameters."""
    @classmethod
    def rand_init(cls, key, dtype, *args, **kwargs):
        frozen_params = None
        params = {}
        scan_map = {}
        es_map = {}
        return CommonInit(frozen_params, params, scan_map, es_map)
    
    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        # x: (batch, channels, height, width) -> (batch, channels * height * width)
        return x.reshape(x.shape[0], -1)


class SimpleCNN(Model):
    """
    Simple CNN for MNIST classification with EGGROLL-efficient forward pass.
    
    Architecture:
    - Conv1: 1 -> 16 channels, 3x3
    - ReLU + MaxPool 2x2
    - Conv2: 16 -> 32 channels, 3x3
    - ReLU + MaxPool 2x2
    - Flatten: 32 * 7 * 7 = 1568
    - FC: 1568 -> num_classes
    
    For EGGROLL efficiency:
    - Conv layers are NOT perturbed (frozen during forward)
    - Only FC layer uses low-rank (MM) perturbation
    - Conv forward is done ONCE and shared across population
    """
    @classmethod
    def rand_init(cls, key, in_channels, num_classes, dtype, *args, **kwargs):
        keys = jax.random.split(key, 5)
        
        # Conv layers - use EXCLUDED so they don't get perturbed
        conv1 = Conv2d.rand_init(keys[0], in_channels, 16, 3, dtype)
        conv2 = Conv2d.rand_init(keys[1], 16, 32, 3, dtype)
        
        # Pooling (no params)
        pool = MaxPool2d.rand_init(keys[2], 2, dtype)
        
        # Flatten (no params)
        flatten = Flatten.rand_init(keys[3], dtype)
        
        # FC layer - uses low-rank perturbation via MM
        # After 2x maxpool on 28x28: 28 -> 14 -> 7, so 32 * 7 * 7 = 1568
        fc = Linear.rand_init(keys[4], 1568, num_classes, use_bias=True, dtype=dtype)
        
        return merge_inits(
            conv1=conv1,
            conv2=conv2,
            pool=pool,
            flatten=flatten,
            fc=fc,
        )

    @classmethod
    def conv_forward(cls, params, x):
        """
        Run conv layers without perturbation.
        This can be called once and result shared across population.
        
        x: (batch, 1, 28, 28) for MNIST
        returns: (batch, 1568) flattened features
        """
        from jax import lax
        
        # Conv1 + ReLU + Pool
        conv1_w = params['conv1']
        dn = lax.conv_dimension_numbers(x.shape, conv1_w.shape, ('NCHW', 'OIHW', 'NCHW'))
        x = lax.conv_general_dilated(x, conv1_w, (1, 1), 'SAME', dimension_numbers=dn)
        x = jax.nn.relu(x)
        x = lax.reduce_window(x, -jnp.inf, lax.max, (1, 1, 2, 2), (1, 1, 2, 2), 'VALID')
        
        # Conv2 + ReLU + Pool
        conv2_w = params['conv2']
        dn = lax.conv_dimension_numbers(x.shape, conv2_w.shape, ('NCHW', 'OIHW', 'NCHW'))
        x = lax.conv_general_dilated(x, conv2_w, (1, 1), 'SAME', dimension_numbers=dn)
        x = jax.nn.relu(x)
        x = lax.reduce_window(x, -jnp.inf, lax.max, (1, 1, 2, 2), (1, 1, 2, 2), 'VALID')
        
        # Flatten
        return x.reshape(x.shape[0], -1)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        """
        Full forward pass.
        
        x: (batch, 1, 28, 28) for MNIST
        returns: (batch, num_classes)
        
        Note: For EGGROLL efficiency, consider using conv_forward() once
        and then calling fc_forward() for each population member via vmap.
        """
        # Conv layers (unperturbed)
        x = call_submodule(Conv2d, 'conv1', common_params, x)
        x = jax.nn.relu(x)
        x = call_submodule(MaxPool2d, 'pool', common_params, x)
        
        x = call_submodule(Conv2d, 'conv2', common_params, x)
        x = jax.nn.relu(x)
        x = call_submodule(MaxPool2d, 'pool', common_params, x)
        
        # Flatten
        x = call_submodule(Flatten, 'flatten', common_params, x)
        
        # FC (perturbed via low-rank)
        x = call_submodule(Linear, 'fc', common_params, x)
        
        return x
    
    @classmethod
    def fc_forward(cls, noiser, frozen_noiser_params, noiser_params, frozen_params, 
                   params, es_tree_key, iterinfo, x):
        """
        Forward pass for FC layer only (with perturbation).
        Use this with vmap over population after running conv_forward once.
        
        x: (batch, 1568) flattened features
        returns: (batch, num_classes)
        """
        fc_params = CommonParams(
            noiser, frozen_noiser_params, noiser_params,
            frozen_params['fc'] if frozen_params and 'fc' in frozen_params else None,
            params['fc'],
            es_tree_key['fc'],
            iterinfo
        )
        return Linear._forward(fc_params, x)
