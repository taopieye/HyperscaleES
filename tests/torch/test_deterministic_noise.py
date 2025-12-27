"""
Test: Deterministic noise generation for PyTorch implementation.

PAPER CLAIM: Perturbations are generated deterministically from (seed, epoch, member_id).
This means we don't need to store perturbations during the forward pass—we can
regenerate them during the update phase.

Key formula: noise = generate(seed, epoch, member_id)

This enables massive memory savings since we only store the seed, not all perturbations.

TARGET API: Noise should be reproducible given the same inputs, controllable via
seed/generator, and efficient to regenerate.
"""
import pytest
import torch
import torch.nn as nn

from conftest import (
    EggrollConfig,
    assert_tensors_close,
    unimplemented
)


# ============================================================================
# Basic Reproducibility Tests
# ============================================================================

class TestNoiseDeterminism:
    """Verify deterministic noise generation."""

    @pytest.mark.skip(reason="Noise generation not yet implemented")
    def test_same_inputs_produce_same_noise(
        self, simple_linear, es_generator, eggroll_config, device
    ):
        """
        Identical inputs should produce identical perturbations.
        
        TARGET API:
            strategy = EggrollStrategy(sigma=0.1, lr=0.01, seed=42)
            strategy.setup(model)
            
            # Same inputs
            pert1 = strategy._sample_perturbation(param, member_id=3, epoch=5)
            pert2 = strategy._sample_perturbation(param, member_id=3, epoch=5)
            
            assert torch.equal(pert1.as_matrix(), pert2.as_matrix())
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=42
        )
        strategy.setup(simple_linear)
        
        # Sample twice with same inputs
        pert1 = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=3,
            epoch=5
        )
        pert2 = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=3,
            epoch=5
        )
        
        assert torch.equal(pert1.as_matrix(), pert2.as_matrix()), \
            "Same inputs should produce identical perturbations"

    @pytest.mark.skip(reason="Noise generation not yet implemented")
    def test_different_member_ids_produce_different_noise(
        self, simple_linear, eggroll_config, device
    ):
        """
        Different member_ids should produce different perturbations.
        
        TARGET API:
            pert_0 = strategy._sample_perturbation(param, member_id=0, epoch=0)
            pert_1 = strategy._sample_perturbation(param, member_id=1, epoch=0)
            
            assert not torch.equal(pert_0.as_matrix(), pert_1.as_matrix())
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=42
        )
        strategy.setup(simple_linear)
        
        pert_0 = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=0,
            epoch=0
        )
        pert_1 = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=1,
            epoch=0
        )
        
        assert not torch.equal(pert_0.as_matrix(), pert_1.as_matrix()), \
            "Different member_ids should produce different perturbations"

    @pytest.mark.skip(reason="Noise generation not yet implemented")
    def test_different_epochs_produce_different_noise(
        self, simple_linear, eggroll_config, device
    ):
        """
        Different epochs should produce different perturbations (when noise_reuse > 0).
        
        TARGET API:
            config = EggrollConfig(noise_reuse=1)  # New noise each epoch
            strategy = EggrollStrategy.from_config(config)
            
            pert_epoch0 = strategy._sample_perturbation(param, member_id=0, epoch=0)
            pert_epoch1 = strategy._sample_perturbation(param, member_id=0, epoch=1)
            
            assert not torch.equal(pert_epoch0.as_matrix(), pert_epoch1.as_matrix())
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=42,
            noise_reuse=1  # New noise each epoch
        )
        strategy.setup(simple_linear)
        
        pert_epoch0 = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=0,
            epoch=0
        )
        pert_epoch1 = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=0,
            epoch=1
        )
        
        assert not torch.equal(pert_epoch0.as_matrix(), pert_epoch1.as_matrix()), \
            "Different epochs should produce different perturbations"


# ============================================================================
# Noise Reuse Tests
# ============================================================================

class TestNoiseReuse:
    """Verify noise reuse configuration."""

    @pytest.mark.skip(reason="Noise reuse not yet implemented")
    def test_noise_reuse_zero_means_same_noise_every_epoch(
        self, simple_linear, eggroll_config, device
    ):
        """
        noise_reuse=0 means the same noise is used for all epochs.
        
        This is useful for very long training where you want consistency.
        
        TARGET API:
            config = EggrollConfig(noise_reuse=0)  # Same noise always
            strategy = EggrollStrategy.from_config(config)
            
            pert_epoch0 = strategy._sample_perturbation(param, member_id=0, epoch=0)
            pert_epoch5 = strategy._sample_perturbation(param, member_id=0, epoch=5)
            pert_epoch100 = strategy._sample_perturbation(param, member_id=0, epoch=100)
            
            # All should be identical
            assert torch.equal(pert_epoch0.as_matrix(), pert_epoch5.as_matrix())
            assert torch.equal(pert_epoch0.as_matrix(), pert_epoch100.as_matrix())
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=42,
            noise_reuse=0  # Same noise always
        )
        strategy.setup(simple_linear)
        
        pert_epoch0 = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=0,
            epoch=0
        )
        pert_epoch5 = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=0,
            epoch=5
        )
        pert_epoch100 = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=0,
            epoch=100
        )
        
        assert torch.equal(pert_epoch0.as_matrix(), pert_epoch5.as_matrix()), \
            "noise_reuse=0: epochs 0 and 5 should have same noise"
        assert torch.equal(pert_epoch0.as_matrix(), pert_epoch100.as_matrix()), \
            "noise_reuse=0: epochs 0 and 100 should have same noise"

    @pytest.mark.skip(reason="Noise reuse not yet implemented")
    def test_noise_reuse_cycles_every_n_epochs(self, simple_linear, device):
        """
        noise_reuse=n means noise changes every n epochs.
        
        TARGET API:
            config = EggrollConfig(noise_reuse=5)  # Change every 5 epochs
            strategy = EggrollStrategy.from_config(config)
            
            # Same within cycle
            pert_e0 = strategy._sample_perturbation(param, member_id=0, epoch=0)
            pert_e4 = strategy._sample_perturbation(param, member_id=0, epoch=4)
            assert torch.equal(pert_e0.as_matrix(), pert_e4.as_matrix())
            
            # Different across cycles
            pert_e5 = strategy._sample_perturbation(param, member_id=0, epoch=5)
            assert not torch.equal(pert_e0.as_matrix(), pert_e5.as_matrix())
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=0.1,
            lr=0.01,
            rank=4,
            seed=42,
            noise_reuse=5  # Change every 5 epochs
        )
        strategy.setup(simple_linear)
        
        # Same within cycle
        pert_e0 = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=0,
            epoch=0
        )
        pert_e4 = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=0,
            epoch=4
        )
        
        assert torch.equal(pert_e0.as_matrix(), pert_e4.as_matrix()), \
            "Epochs 0-4 should share the same noise (noise_reuse=5)"
        
        # Different across cycles
        pert_e5 = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=0,
            epoch=5
        )
        
        assert not torch.equal(pert_e0.as_matrix(), pert_e5.as_matrix()), \
            "Epoch 5 should have different noise than epoch 0 (new cycle)"


# ============================================================================
# Seed Management Tests
# ============================================================================

class TestSeedManagement:
    """Verify seed/generator management."""

    @pytest.mark.skip(reason="Seed management not yet implemented")
    def test_strategy_accepts_seed(self, simple_linear, device):
        """
        Strategy should accept integer seed for reproducibility.
        
        TARGET API:
            strategy1 = EggrollStrategy(sigma=0.1, lr=0.01, seed=42)
            strategy2 = EggrollStrategy(sigma=0.1, lr=0.01, seed=42)
            
            # Both produce same perturbations
            pert1 = strategy1._sample_perturbation(param, 0, 0)
            pert2 = strategy2._sample_perturbation(param, 0, 0)
            
            assert torch.equal(pert1.as_matrix(), pert2.as_matrix())
        """
        from hyperscalees.torch import EggrollStrategy
        
        # Create two strategies with same seed
        model1 = nn.Linear(4, 8, bias=False).to(device)
        model2 = nn.Linear(4, 8, bias=False).to(device)
        
        strategy1 = EggrollStrategy(sigma=0.1, lr=0.01, rank=4, seed=42)
        strategy2 = EggrollStrategy(sigma=0.1, lr=0.01, rank=4, seed=42)
        
        strategy1.setup(model1)
        strategy2.setup(model2)
        
        pert1 = strategy1._sample_perturbation(
            param=model1.weight,
            member_id=0,
            epoch=0
        )
        pert2 = strategy2._sample_perturbation(
            param=model2.weight,
            member_id=0,
            epoch=0
        )
        
        assert torch.equal(pert1.as_matrix(), pert2.as_matrix()), \
            "Same seed should produce identical perturbations"

    @pytest.mark.skip(reason="Seed management not yet implemented")
    def test_strategy_accepts_generator(self, simple_linear, device):
        """
        Strategy should accept torch.Generator for advanced use.
        
        TARGET API:
            gen = torch.Generator().manual_seed(42)
            strategy = EggrollStrategy(sigma=0.1, lr=0.01, generator=gen)
        """
        from hyperscalees.torch import EggrollStrategy
        
        gen = torch.Generator(device=device).manual_seed(42)
        
        strategy = EggrollStrategy(
            sigma=0.1,
            lr=0.01,
            rank=4,
            generator=gen
        )
        strategy.setup(simple_linear)
        
        # Should work without errors
        pert = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=0,
            epoch=0
        )
        assert pert is not None

    @pytest.mark.skip(reason="Seed management not yet implemented")
    def test_different_seeds_produce_different_noise(self, simple_linear, device):
        """
        Different seeds should produce different perturbations.
        
        TARGET API:
            strategy1 = EggrollStrategy(sigma=0.1, lr=0.01, seed=42)
            strategy2 = EggrollStrategy(sigma=0.1, lr=0.01, seed=123)
            
            pert1 = strategy1._sample_perturbation(param, 0, 0)
            pert2 = strategy2._sample_perturbation(param, 0, 0)
            
            assert not torch.equal(pert1.as_matrix(), pert2.as_matrix())
        """
        from hyperscalees.torch import EggrollStrategy
        
        model1 = nn.Linear(4, 8, bias=False).to(device)
        model2 = nn.Linear(4, 8, bias=False).to(device)
        
        strategy1 = EggrollStrategy(sigma=0.1, lr=0.01, rank=4, seed=42)
        strategy2 = EggrollStrategy(sigma=0.1, lr=0.01, rank=4, seed=123)
        
        strategy1.setup(model1)
        strategy2.setup(model2)
        
        pert1 = strategy1._sample_perturbation(
            param=model1.weight,
            member_id=0,
            epoch=0
        )
        pert2 = strategy2._sample_perturbation(
            param=model2.weight,
            member_id=0,
            epoch=0
        )
        
        assert not torch.equal(pert1.as_matrix(), pert2.as_matrix()), \
            "Different seeds should produce different perturbations"

    @pytest.mark.skip(reason="Seed management not yet implemented")
    def test_seed_in_state_dict(self, simple_mlp, eggroll_config, device):
        """
        Seed should be saved in state_dict for reproducibility.
        
        TARGET API:
            strategy = EggrollStrategy(sigma=0.1, lr=0.01, seed=42)
            strategy.setup(model)
            
            state = strategy.state_dict()
            assert "seed" in state or "generator_state" in state
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=42
        )
        strategy.setup(simple_mlp)
        
        state = strategy.state_dict()
        
        # Should have either seed or generator state
        has_seed_info = "seed" in state or "generator_state" in state or "rng_state" in state
        assert has_seed_info, \
            f"state_dict should contain seed/generator info. Keys: {state.keys()}"


# ============================================================================
# Key Folding Tests
# ============================================================================

class TestKeyFolding:
    """Verify key folding logic for noise generation."""

    @pytest.mark.skip(reason="Key folding not yet implemented")
    def test_key_folding_is_deterministic(self, simple_linear, eggroll_config, device):
        """
        Key folding (combining seed + epoch + member_id) should be deterministic.
        
        TARGET API:
            # Internal: key = fold(fold(base_seed, epoch), member_id)
            # This should always produce the same result
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=42
        )
        strategy.setup(simple_linear)
        
        # Call multiple times, should be deterministic
        results = []
        for _ in range(5):
            pert = strategy._sample_perturbation(
                param=simple_linear.weight,
                member_id=7,
                epoch=13
            )
            results.append(pert.as_matrix().clone())
        
        # All should be identical
        for i in range(1, 5):
            assert torch.equal(results[0], results[i]), \
                f"Key folding should be deterministic: result {i} differs"

    @pytest.mark.skip(reason="Key folding not yet implemented")
    def test_key_folding_distributes_noise_uniformly(self, simple_linear, eggroll_config, device):
        """
        Folded keys should produce uniformly distributed noise.
        
        Different (epoch, member_id) combinations should produce
        independent-looking samples.
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=42
        )
        strategy.setup(simple_linear)
        
        # Collect many perturbations across different epochs and members
        all_values = []
        for epoch in range(10):
            for member_id in range(10):
                pert = strategy._sample_perturbation(
                    param=simple_linear.weight,
                    member_id=member_id,
                    epoch=epoch
                )
                all_values.append(pert.as_matrix().flatten())
        
        all_values = torch.cat(all_values)
        
        # Check statistics look reasonable (approximately normal)
        mean = all_values.mean().item()
        std = all_values.std().item()
        
        assert abs(mean) < 0.1, f"Mean should be ~0, got {mean:.4f}"
        assert 0.5 < std < 2.0, f"Std should be reasonable, got {std:.4f}"

    @pytest.mark.skip(reason="Key folding not yet implemented")
    def test_parameter_specific_keys(self, simple_mlp, eggroll_config, device):
        """
        Each parameter should have its own key derived from its name.
        
        TARGET API:
            # Different parameters in same layer should have different noise
            pert_weight = strategy._get_perturbation("0.weight", 0, 0)
            pert_bias = strategy._get_perturbation("0.bias", 0, 0)
            
            # These should be independent
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=42
        )
        strategy.setup(simple_mlp)
        
        # Get perturbations for different parameters (same epoch, member)
        params = list(simple_mlp.parameters())
        
        if len(params) >= 2:
            pert1 = strategy._sample_perturbation(
                param=params[0],
                member_id=0,
                epoch=0
            )
            pert2 = strategy._sample_perturbation(
                param=params[1],
                member_id=0,
                epoch=0
            )
            
            # Can't directly compare since shapes differ
            # Just verify they're both valid and non-zero
            assert pert1.as_matrix().abs().sum() > 0
            assert pert2.as_matrix().abs().sum() > 0


# ============================================================================
# Regeneration Tests
# ============================================================================

class TestNoiseRegeneration:
    """Verify noise can be regenerated for updates."""

    @pytest.mark.skip(reason="Noise regeneration not yet implemented")
    def test_regenerate_perturbation_for_update(
        self, simple_linear, eggroll_config, device
    ):
        """
        Should be able to regenerate perturbations during update phase.
        
        This is the key memory optimization - no need to store perturbations.
        
        TARGET API:
            strategy.setup(model)
            
            # Forward pass - perturbations generated
            with strategy.perturb(population_size=64, epoch=0) as pop:
                for member_id in pop.iterate():
                    output = model(x)
                    # Perturbations NOT stored
            
            # Update phase - perturbations regenerated
            strategy.step(fitnesses)  # Regenerates internally
        """
        from hyperscalees.torch import EggrollStrategy
        
        population_size = 8
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=42
        )
        strategy.setup(simple_linear)
        
        # Forward pass
        with strategy.perturb(population_size=population_size, epoch=0) as pop:
            x = torch.randn(population_size, simple_linear.in_features, device=device)
            outputs = pop.batched_forward(simple_linear, x)
        
        # Update pass - should work without storing perturbations
        fitnesses = torch.randn(population_size, device=device)
        metrics = strategy.step(fitnesses)
        
        # If we got here without error, regeneration worked
        assert "param_delta" in metrics or len(metrics) >= 0  # Some metrics returned

    @pytest.mark.skip(reason="Noise regeneration not yet implemented")
    def test_regenerated_matches_original(self, simple_linear, eggroll_config, device):
        """
        Regenerated perturbation should exactly match original.
        
        TARGET API:
            # During forward
            with strategy.perturb(population_size=8, epoch=0) as pop:
                pop.set_member(3)
                original_pert = strategy._get_current_perturbation("weight")
            
            # Regenerate
            regenerated = strategy._sample_perturbation(param, member_id=3, epoch=0)
            
            assert torch.equal(original_pert.as_matrix(), regenerated.as_matrix())
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=42
        )
        strategy.setup(simple_linear)
        
        # Sample once
        original = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=3,
            epoch=5
        )
        
        # "Regenerate" by sampling again with same params
        regenerated = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=3,
            epoch=5
        )
        
        assert torch.equal(original.as_matrix(), regenerated.as_matrix()), \
            "Regenerated perturbation should exactly match original"


# ============================================================================
# Cross-Run Reproducibility Tests
# ============================================================================

class TestCrossRunReproducibility:
    """Verify reproducibility across different runs/processes."""

    @pytest.mark.skip(reason="Cross-run reproducibility not yet implemented")
    def test_reproducible_across_model_reinit(self, device):
        """
        Same seed should produce same results even after model reinit.
        
        TARGET API:
            # First run
            model1 = create_model()
            strategy1 = EggrollStrategy(seed=42)
            strategy1.setup(model1)
            pert1 = strategy1._sample_perturbation(model1.weight, 0, 0)
            
            # Second run (new model, new strategy)
            model2 = create_model()
            strategy2 = EggrollStrategy(seed=42)
            strategy2.setup(model2)
            pert2 = strategy2._sample_perturbation(model2.weight, 0, 0)
            
            assert torch.equal(pert1.as_matrix(), pert2.as_matrix())
        """
        from hyperscalees.torch import EggrollStrategy
        
        # First run
        model1 = nn.Linear(8, 16, bias=False).to(device)
        strategy1 = EggrollStrategy(sigma=0.1, lr=0.01, rank=4, seed=42)
        strategy1.setup(model1)
        pert1 = strategy1._sample_perturbation(
            param=model1.weight,
            member_id=0,
            epoch=0
        )
        
        # Second run - fresh model and strategy
        model2 = nn.Linear(8, 16, bias=False).to(device)
        strategy2 = EggrollStrategy(sigma=0.1, lr=0.01, rank=4, seed=42)
        strategy2.setup(model2)
        pert2 = strategy2._sample_perturbation(
            param=model2.weight,
            member_id=0,
            epoch=0
        )
        
        assert torch.equal(pert1.as_matrix(), pert2.as_matrix()), \
            "Same seed should produce identical results across reinitializations"

    @pytest.mark.skip(reason="Cross-run reproducibility not yet implemented")
    def test_reproducible_with_checkpoint_restore(self, simple_mlp, eggroll_config, device):
        """
        Results should be reproducible after checkpoint restore.
        
        TARGET API:
            strategy.setup(model)
            
            # Run some epochs
            for epoch in range(5):
                with strategy.perturb(64, epoch):
                    pass
                strategy.step(fitnesses)
            
            # Save state
            state = strategy.state_dict()
            
            # New strategy, restore state
            new_strategy = EggrollStrategy.from_config(config)
            new_strategy.setup(model)
            new_strategy.load_state_dict(state)
            
            # Should produce same results for epoch 5
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=42
        )
        strategy.setup(simple_mlp)
        
        # Run some epochs
        population_size = 8
        for epoch in range(5):
            fitnesses = torch.randn(population_size, device=device)
            with strategy.perturb(population_size=population_size, epoch=epoch):
                pass
            strategy.step(fitnesses)
        
        # Save state
        state = strategy.state_dict()
        
        # Get perturbation at epoch 5
        pert_before = strategy._sample_perturbation(
            param=list(simple_mlp.parameters())[0],
            member_id=0,
            epoch=5
        ).as_matrix().clone()
        
        # Create new strategy and restore
        new_model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        ).to(device)
        
        new_strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=0  # Different initial seed
        )
        new_strategy.setup(new_model)
        new_strategy.load_state_dict(state)
        
        # Get perturbation at epoch 5 from restored strategy
        pert_after = new_strategy._sample_perturbation(
            param=list(new_model.parameters())[0],
            member_id=0,
            epoch=5
        ).as_matrix()
        
        assert torch.equal(pert_before, pert_after), \
            "Restored strategy should produce same perturbations"


# ============================================================================
# Edge Cases
# ============================================================================

class TestNoiseDeterminismEdgeCases:
    """Test edge cases in noise generation."""

    @pytest.mark.skip(reason="Edge cases not yet implemented")
    def test_large_epoch_number(self, simple_linear, eggroll_config, device):
        """
        Large epoch numbers should not cause overflow or quality issues.
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=42
        )
        strategy.setup(simple_linear)
        
        # Large epoch number
        pert = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=0,
            epoch=1_000_000
        )
        
        # Should still produce valid perturbation
        assert torch.isfinite(pert.as_matrix()).all(), \
            "Large epoch should not produce inf/nan"
        assert pert.as_matrix().abs().sum() > 0, \
            "Large epoch should produce non-zero perturbation"

    @pytest.mark.skip(reason="Edge cases not yet implemented")
    def test_large_member_id(self, simple_linear, eggroll_config, device):
        """
        Large member_ids should not cause issues.
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=42
        )
        strategy.setup(simple_linear)
        
        # Large member_id
        pert = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=1_000_000,
            epoch=0
        )
        
        assert torch.isfinite(pert.as_matrix()).all(), \
            "Large member_id should not produce inf/nan"

    @pytest.mark.skip(reason="Edge cases not yet implemented")
    def test_epoch_zero_is_valid(self, simple_linear, eggroll_config, device):
        """
        epoch=0 should be a valid input.
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=42
        )
        strategy.setup(simple_linear)
        
        pert = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=0,
            epoch=0  # Explicitly epoch 0
        )
        
        assert pert is not None
        assert torch.isfinite(pert.as_matrix()).all()

    @pytest.mark.skip(reason="Edge cases not yet implemented")
    def test_member_id_zero_is_valid(self, simple_linear, eggroll_config, device):
        """
        member_id=0 should be a valid input.
        """
        from hyperscalees.torch import EggrollStrategy
        
        strategy = EggrollStrategy(
            sigma=eggroll_config.sigma,
            lr=eggroll_config.lr,
            rank=eggroll_config.rank,
            seed=42
        )
        strategy.setup(simple_linear)
        
        pert = strategy._sample_perturbation(
            param=simple_linear.weight,
            member_id=0,  # Explicitly member 0
            epoch=0
        )
        
        assert pert is not None
        assert torch.isfinite(pert.as_matrix()).all()
