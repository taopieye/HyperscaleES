"""
Test: PyTorch nn.Module integration for the ES strategy.

TARGET API: The strategy should integrate seamlessly with standard PyTorch
nn.Module classes, requiring minimal changes to existing model code.

Design principles:
1. Any nn.Module should work (no special base class required)
2. Automatic parameter discovery
3. Clean separation between model and strategy
4. Support for common layer types (Linear, Conv2d, etc.)
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conftest import (
    EggrollConfig,
    assert_tensors_close,
    count_parameters,
    make_fitnesses,
    unimplemented
)


# ============================================================================
# Basic Module Integration Tests
# ============================================================================

class TestBasicModuleIntegration:
    """Verify basic integration with nn.Module."""

    def test_setup_with_sequential(self, simple_mlp, eggroll_config):
        """
        Should work with nn.Sequential models.
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        assert strategy.model is simple_mlp, \
            f"strategy.model should reference the passed model"
        
        # Should be able to run a forward pass
        device = simple_mlp[0].weight.device
        x = torch.randn(8, 8, device=device)
        
        with strategy.perturb(population_size=8, epoch=0) as pop:
            outputs = pop.batched_forward(simple_mlp, x)
        
        assert outputs.shape[0] == 8, \
            f"Output batch size should match population size, got {outputs.shape[0]}"

    def test_setup_with_custom_module(self, device, eggroll_config):
        """
        Should work with custom nn.Module subclasses.
        """
        from hyperscalees.torch import EggrollStrategy
        
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 16)
                self.fc2 = nn.Linear(16, 2)
            
            def forward(self, x):
                return self.fc2(F.relu(self.fc1(x)))
        
        model = MyModel().to(device)
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(model)
        
        assert strategy.model is model, \
            "Strategy should hold reference to custom module"
        
        # Verify it works
        x = torch.randn(8, 8, device=device)
        with strategy.perturb(population_size=8, epoch=0) as pop:
            outputs = pop.batched_forward(model, x)
        
        assert outputs.shape == (8, 2), \
            f"Expected output shape (8, 2), got {outputs.shape}"

    def test_setup_with_nested_modules(self, device, eggroll_config):
        """
        Should handle nested module hierarchies.
        """
        from hyperscalees.torch import EggrollStrategy
        
        class Block(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc = nn.Linear(dim, dim)
            
            def forward(self, x):
                return F.relu(self.fc(x))
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block(16) for _ in range(3)])
                self.input_proj = nn.Linear(8, 16)
                self.output = nn.Linear(16, 2)
            
            def forward(self, x):
                x = self.input_proj(x)
                for block in self.blocks:
                    x = block(x)
                return self.output(x)
        
        model = Model().to(device)
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(model)
        
        # Should find all 5 Linear layers (3 in blocks + input_proj + output)
        weight_count = sum(1 for n, _ in strategy.named_parameters() if 'weight' in n)
        assert weight_count == 5, \
            f"Should find 5 weight matrices in nested model, found {weight_count}"


# ============================================================================
# Parameter Discovery Tests
# ============================================================================

class TestParameterDiscovery:
    """Verify automatic parameter discovery."""

    def test_finds_all_linear_weights(self, simple_mlp, eggroll_config):
        """
        Should automatically find all Linear layer weights.
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        weight_count = len(list(strategy.weight_parameters()))
        
        # simple_mlp has 3 Linear layers (but bias=False, so just weights)
        # Count actual weight params in model
        expected = sum(1 for n, p in simple_mlp.named_parameters() if 'weight' in n and p.dim() >= 2)
        
        assert weight_count == expected, \
            f"Expected {expected} weight matrices, found {weight_count}"

    def test_finds_all_biases(self, mlp_with_bias, eggroll_config):
        """
        Should automatically find all bias parameters.
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(mlp_with_bias)
        
        bias_count = len(list(strategy.bias_parameters()))
        expected = sum(1 for n, _ in mlp_with_bias.named_parameters() if 'bias' in n)
        
        assert bias_count == expected, \
            f"Expected {expected} biases, found {bias_count}"

    def test_respects_requires_grad(self, device, eggroll_config):
        """
        Should only evolve parameters with requires_grad=True.
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Create a fresh model to avoid any state issues
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        ).to(device)
        
        # Freeze first layer
        model[0].weight.requires_grad = False
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(model)
        
        evolved_params = list(strategy.parameters())
        
        # Use identity comparison to avoid tensor broadcasting errors
        frozen_weight_evolved = any(p is model[0].weight for p in evolved_params)
        assert not frozen_weight_evolved, \
            "Frozen parameter (requires_grad=False) should not be evolved"


# ============================================================================
# Layer Type Support Tests
# ============================================================================

class TestLayerTypeSupport:
    """Verify support for various layer types."""

    def test_linear_layer_support(self, device, eggroll_config):
        """
        nn.Linear should be fully supported.
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(8, 4).to(device)
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(model)
        
        x = torch.randn(8, 8, device=device)
        with strategy.perturb(population_size=8, epoch=0) as pop:
            outputs = pop.batched_forward(model, x)
        
        assert outputs.shape == (8, 4), \
            f"Linear layer output shape should be (8, 4), got {outputs.shape}"

    def test_embedding_layer_support(self, device, eggroll_config):
        """
        nn.Embedding should be supported.
        """
        from hyperscalees.torch import EggrollStrategy
        
        class EmbeddingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.fc = nn.Linear(64, 10)
            
            def forward(self, x):
                return self.fc(self.embed(x).mean(dim=1))
        
        model = EmbeddingModel().to(device)
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(model)
        
        # Verify embedding is discovered
        param_names = [n for n, _ in strategy.named_parameters()]
        assert any('embed' in n for n in param_names), \
            f"Embedding parameters should be discovered, got: {param_names}"


# ============================================================================
# Model State Tests
# ============================================================================

class TestModelState:
    """Verify model state handling."""

    def test_train_mode_preserved(self, simple_mlp, eggroll_config):
        """
        Model train/eval mode should be preserved during perturbation.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        simple_mlp.train()
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        x = torch.randn(8, 8, device=device)
        
        with strategy.perturb(population_size=8, epoch=0) as pop:
            assert simple_mlp.training == True, \
                "Model should remain in train mode inside perturb context"
            pop.batched_forward(simple_mlp, x)
        
        assert simple_mlp.training == True, \
            "Model should remain in train mode after perturb context"

    def test_eval_mode_preserved(self, simple_mlp, eggroll_config):
        """
        Eval mode should be preserved during perturbation.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        simple_mlp.eval()
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        x = torch.randn(8, 8, device=device)
        
        with strategy.perturb(population_size=8, epoch=0) as pop:
            assert simple_mlp.training == False, \
                "Model should remain in eval mode inside perturb context"
            pop.batched_forward(simple_mlp, x)
        
        assert simple_mlp.training == False, \
            "Model should remain in eval mode after perturb context"

    def test_parameters_restored_after_context(
        self, simple_mlp, batch_input_small, eggroll_config
    ):
        """
        Original parameters should be restored after perturb() context.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        original = {n: p.clone() for n, p in simple_mlp.named_parameters()}
        
        with strategy.perturb(population_size=8, epoch=0) as pop:
            x = torch.randn(8, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        # Parameters should be restored after context exit
        for name, param in simple_mlp.named_parameters():
            assert torch.equal(param, original[name]), \
                f"Parameter {name} not restored after perturb() context"


# ============================================================================
# Device Handling Tests
# ============================================================================

class TestDeviceHandling:
    """Verify correct device handling (GPU required)."""

    def test_cuda_model_works(self, device, eggroll_config):
        """
        Should work with CUDA models.
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(8, 4).to(device)
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(model)
        
        x = torch.randn(8, 8, device=device)
        
        with strategy.perturb(population_size=8, epoch=0) as pop:
            output = pop.batched_forward(model, x)
        
        assert output.device.type == "cuda", \
            f"Output should be on CUDA, got {output.device}"

    def test_rejects_cpu_model(self, eggroll_config):
        """
        Should reject CPU models with a helpful error message.
        """
        from hyperscalees.torch import EggrollStrategy
        
        model = nn.Linear(8, 4).cpu()
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        
        with pytest.raises(RuntimeError, match="CUDA|GPU"):
            strategy.setup(model)


# ============================================================================
# Serialization Tests
# ============================================================================

class TestModelSerialization:
    """Test model + strategy serialization."""

    def test_strategy_state_dict_complete(self, simple_mlp, eggroll_config):
        """
        Strategy state_dict should include all necessary state.
        """
        from hyperscalees.torch import EggrollStrategy
        
        device = simple_mlp[0].weight.device
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(simple_mlp)
        
        # Do a step to populate state
        population_size = 8
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, 8, device=device)
            pop.batched_forward(simple_mlp, x)
        
        fitnesses = make_fitnesses(population_size, device=device)
        strategy.step(fitnesses)
        
        state = strategy.state_dict()
        
        # Check required keys
        assert "sigma" in state, \
            f"state_dict should contain 'sigma', got keys: {list(state.keys())}"
        assert "lr" in state, \
            f"state_dict should contain 'lr', got keys: {list(state.keys())}"
        assert "epoch" in state, \
            f"state_dict should contain 'epoch', got keys: {list(state.keys())}"
        
        # Should have either seed or generator state
        has_seed_info = "seed" in state or "generator_state" in state
        assert has_seed_info, \
            f"state_dict should contain 'seed' or 'generator_state', got keys: {list(state.keys())}"


# ============================================================================
# LowRankLinear Layer Tests
# ============================================================================

class TestLowRankLinear:
    """Test the LowRankLinear layer for ES-optimized models."""

    def test_lowrank_linear_basic_forward(self, device):
        """
        LowRankLinear should work as a drop-in for nn.Linear.
        """
        from hyperscalees.torch import LowRankLinear
        
        model = nn.Sequential(
            LowRankLinear(8, 16),
            nn.ReLU(),
            LowRankLinear(16, 2)
        ).to(device)
        
        x = torch.randn(4, 8, device=device)
        outputs = model(x)
        
        assert outputs.shape == (4, 2), \
            f"Output shape should be (4, 2), got {outputs.shape}"

    def test_lowrank_linear_with_batched_forward(self, device, eggroll_config):
        """
        LowRankLinear models work with batched_forward API.
        """
        from hyperscalees.torch import LowRankLinear, EggrollStrategy
        
        model = nn.Sequential(
            LowRankLinear(8, 16),
            nn.ReLU(),
            LowRankLinear(16, 2)
        ).to(device)
        
        strategy = EggrollStrategy(**eggroll_config.__dict__)
        strategy.setup(model)
        
        x = torch.randn(4, 8, device=device)
        
        with strategy.perturb(population_size=4, epoch=0) as pop:
            outputs = pop.batched_forward(model, x)
        
        assert outputs.shape == (4, 2), \
            f"Output shape should be (4, 2), got {outputs.shape}"

    def test_lowrank_linear_rank_parameter(self, device):
        """
        LowRankLinear should respect the rank parameter.
        """
        from hyperscalees.torch import LowRankLinear
        
        layer = LowRankLinear(64, 32, rank=8).to(device)
        
        assert layer.rank == 8, \
            f"Layer rank should be 8, got {layer.rank}"
        assert layer.U.shape == (32, 8), \
            f"U factor should be (32, 8), got {layer.U.shape}"
        assert layer.V.shape == (64, 8), \
            f"V factor should be (64, 8), got {layer.V.shape}"

    def test_lowrank_linear_forward_correctness(self, device):
        """
        LowRankLinear forward should produce correct results.
        """
        from hyperscalees.torch import LowRankLinear
        
        layer = LowRankLinear(8, 4, rank=4).to(device)
        x = torch.randn(2, 8, device=device)
        
        # Forward pass
        y = layer(x)
        
        # Manual computation: x @ V @ U.T + bias
        expected = x @ layer.V @ layer.U.T
        if layer.bias is not None:
            expected = expected + layer.bias
        
        assert torch.allclose(y, expected, atol=1e-5), \
            "Forward pass should equal x @ V @ U.T + bias"

    def test_lowrank_linear_from_linear(self, device):
        """
        LowRankLinear.from_linear should approximate original nn.Linear.
        """
        from hyperscalees.torch import LowRankLinear
        
        # Create a full-rank linear
        linear = nn.Linear(16, 8).to(device)
        
        # Convert to low-rank (full rank means exact reconstruction)
        low_rank = LowRankLinear.from_linear(linear)
        
        x = torch.randn(4, 16, device=device)
        
        y_original = linear(x)
        y_lowrank = low_rank(x)
        
        assert torch.allclose(y_original, y_lowrank, atol=1e-4), \
            "Full-rank LowRankLinear should closely match original nn.Linear"

    def test_lowrank_linear_weight_property(self, device):
        """
        weight property should return U @ V.T
        """
        from hyperscalees.torch import LowRankLinear
        
        layer = LowRankLinear(8, 4, rank=4).to(device)
        
        W_reconstructed = layer.weight
        W_expected = layer.U @ layer.V.T
        
        assert torch.allclose(W_reconstructed, W_expected), \
            "weight property should return U @ V.T"
        assert W_reconstructed.shape == (4, 8), \
            f"weight shape should be (out, in) = (4, 8), got {W_reconstructed.shape}"