"""Tests for production training components."""

import pytest
import torch
import numpy as np
import random
from pathlib import Path

from app.llm.training.trainer import (
    ReproducibilityManager,
    NumericalStabilityMonitor,
    SafeCheckpointManager,
)


class TestReproducibilityManager:
    """Test reproducibility manager."""
    
    def test_set_global_seed(self):
        """Test global seed setting."""
        manager = ReproducibilityManager(seed=42)
        manager.set_global_seed()
        
        # Check seeds are set
        r1 = random.random()
        n1 = np.random.random()
        t1 = torch.rand(1).item()
        
        # Reset and check reproducibility
        manager.set_global_seed()
        r2 = random.random()
        n2 = np.random.random()
        t2 = torch.rand(1).item()
        
        assert r1 == r2
        assert n1 == n2
        assert t1 == t2
    
    def test_save_load_rng_states(self):
        """Test RNG state saving and loading."""
        manager = ReproducibilityManager(seed=42)
        manager.set_global_seed()
        
        # Generate some random numbers
        random.random()
        np.random.random()
        torch.rand(1)
        
        # Save states
        states = manager.save_rng_states()
        assert "python" in states
        assert "numpy" in states
        assert "torch" in states
        
        # Generate more random numbers
        r1 = random.random()
        n1 = np.random.random()
        t1 = torch.rand(1).item()
        
        # Load states and verify we get same numbers
        manager.load_rng_states(states)
        r2 = random.random()
        n2 = np.random.random()
        t2 = torch.rand(1).item()
        
        assert r1 == r2
        assert n1 == n2
        assert t1 == t2
    
    def test_capture_environment(self, tmp_path):
        """Test environment capture."""
        manager = ReproducibilityManager(seed=42)
        env_info = manager.capture_environment(tmp_path)
        
        assert "timestamp" in env_info
        assert "python_version" in env_info
        assert "torch_version" in env_info
        assert "seed" in env_info
        assert env_info["seed"] == 42
        
        # Check file was created
        env_file = tmp_path / "environment.json"
        assert env_file.exists()


class TestNumericalStabilityMonitor:
    """Test numerical stability monitor."""
    
    def test_check_tensor_valid(self):
        """Test checking valid tensor."""
        monitor = NumericalStabilityMonitor()
        tensor = torch.randn(10, 10)
        
        assert monitor.check_tensor(tensor, "test") is True
        assert monitor.nan_count == 0
        assert monitor.inf_count == 0
    
    def test_check_tensor_nan(self):
        """Test detecting NaN."""
        monitor = NumericalStabilityMonitor()
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        
        assert monitor.check_tensor(tensor, "test") is False
        assert monitor.nan_count == 1
    
    def test_check_tensor_inf(self):
        """Test detecting Inf."""
        monitor = NumericalStabilityMonitor()
        tensor = torch.tensor([1.0, float('inf'), 3.0])
        
        assert monitor.check_tensor(tensor, "test") is False
        assert monitor.inf_count == 1
    
    def test_check_gradients(self):
        """Test gradient checking."""
        monitor = NumericalStabilityMonitor()
        
        # Create simple model
        model = torch.nn.Linear(10, 10)
        
        # Forward and backward
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Check gradients
        stats = monitor.check_gradients(model)
        
        assert stats["stable"] is True
        assert stats["total_norm"] > 0
        assert stats["num_params"] > 0
    
    def test_get_stats(self):
        """Test getting statistics."""
        monitor = NumericalStabilityMonitor()
        
        # Trigger some issues
        monitor.check_tensor(torch.tensor([float('nan')]), "test1")
        monitor.check_tensor(torch.tensor([float('inf')]), "test2")
        
        stats = monitor.get_stats()
        assert stats["nan_count"] == 1
        assert stats["inf_count"] == 1


class TestSafeCheckpointManager:
    """Test safe checkpoint manager."""
    
    def test_save_checkpoint(self, tmp_path):
        """Test checkpoint saving."""
        manager = SafeCheckpointManager(tmp_path)
        
        # Create dummy model and optimizer
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            step=100,
            metrics={"loss": 0.5},
        )
        
        assert checkpoint_path.exists()
        assert (checkpoint_path / "training_state.pt").exists()
        assert (checkpoint_path / "metadata.json").exists()
    
    def test_load_checkpoint(self, tmp_path):
        """Test checkpoint loading."""
        manager = SafeCheckpointManager(tmp_path)
        
        # Create and save checkpoint
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            step=100,
            metrics={"loss": 0.5},
        )
        
        # Load checkpoint
        loaded = manager.load_checkpoint(checkpoint_path)
        
        assert "metadata" in loaded
        assert "training_state" in loaded
        assert loaded["metadata"]["epoch"] == 1
        assert loaded["metadata"]["step"] == 100
        assert loaded["metadata"]["metrics"]["loss"] == 0.5
    
    def test_load_nonexistent_checkpoint(self, tmp_path):
        """Test loading nonexistent checkpoint."""
        manager = SafeCheckpointManager(tmp_path)
        
        with pytest.raises(FileNotFoundError):
            manager.load_checkpoint(tmp_path / "nonexistent")

