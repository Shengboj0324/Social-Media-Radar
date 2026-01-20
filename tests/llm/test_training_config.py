"""Tests for production training configuration system."""

import pytest
from pathlib import Path
import tempfile
import yaml

from app.llm.training.config import (
    ProductionTrainingConfig,
    ModelConfig,
    LoRAConfig,
    TrainingConfig,
    DataConfig,
    CheckpointConfig,
    LoggingConfig,
    ReproducibilityConfig,
    DistributedConfig,
    OptimizerType,
    SchedulerType,
    PrecisionType,
)


class TestModelConfig:
    """Test model configuration."""
    
    def test_valid_config(self):
        """Test valid model configuration."""
        config = ModelConfig(
            base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            model_max_length=4096,
        )
        assert config.base_model == "meta-llama/Meta-Llama-3.1-8B-Instruct"
        assert config.model_max_length == 4096
    
    def test_invalid_max_length(self):
        """Test invalid max length."""
        with pytest.raises(ValueError):
            ModelConfig(model_max_length=64)  # Too small
    
    def test_empty_model_name(self):
        """Test empty model name."""
        with pytest.raises(ValueError):
            ModelConfig(base_model="")


class TestLoRAConfig:
    """Test LoRA configuration."""
    
    def test_valid_config(self):
        """Test valid LoRA configuration."""
        config = LoRAConfig(
            r=16,
            alpha=32,
            dropout=0.05,
        )
        assert config.r == 16
        assert config.alpha == 32
    
    def test_invalid_rank(self):
        """Test invalid rank."""
        with pytest.raises(ValueError):
            LoRAConfig(r=0)  # Too small
    
    def test_both_quantization(self):
        """Test both 4-bit and 8-bit quantization."""
        with pytest.raises(ValueError):
            LoRAConfig(use_4bit=True, use_8bit=True)


class TestTrainingConfig:
    """Test training configuration."""
    
    def test_valid_config(self):
        """Test valid training configuration."""
        config = TrainingConfig(
            num_epochs=3,
            learning_rate=2e-4,
            optimizer=OptimizerType.PAGED_ADAMW_32BIT,
        )
        assert config.num_epochs == 3
        assert config.learning_rate == 2e-4
    
    def test_invalid_learning_rate(self):
        """Test invalid learning rate."""
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=0.0)  # Must be > 0
    
    def test_both_warmup_settings(self):
        """Test both warmup ratio and steps."""
        with pytest.raises(ValueError):
            TrainingConfig(warmup_ratio=0.1, warmup_steps=100)


class TestDataConfig:
    """Test data configuration."""
    
    def test_valid_config(self, tmp_path):
        """Test valid data configuration."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"messages": []}\n')
        
        config = DataConfig(train_file=train_file)
        assert config.train_file == train_file
    
    def test_invalid_splits(self, tmp_path):
        """Test invalid data splits."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"messages": []}\n')
        
        with pytest.raises(ValueError):
            DataConfig(
                train_file=train_file,
                val_split=0.6,
                test_split=0.5,  # Total > 1.0
            )


class TestProductionTrainingConfig:
    """Test complete production configuration."""
    
    def test_from_yaml(self, tmp_path):
        """Test loading from YAML."""
        # Create test data file
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"messages": []}\n')
        
        # Create config
        config_dict = {
            "config_version": "1.0.0",
            "experiment_name": "test",
            "model": {
                "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            },
            "data": {
                "train_file": str(train_file),
            },
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)
        
        # Load config
        config = ProductionTrainingConfig.from_yaml(config_file)
        assert config.experiment_name == "test"
        assert config.model.base_model == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    def test_to_yaml(self, tmp_path):
        """Test saving to YAML."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"messages": []}\n')
        
        config = ProductionTrainingConfig(
            experiment_name="test",
            data=DataConfig(train_file=train_file),
        )
        
        config_file = tmp_path / "config.yaml"
        config.to_yaml(config_file)
        
        assert config_file.exists()
        
        # Load and verify
        loaded = ProductionTrainingConfig.from_yaml(config_file)
        assert loaded.experiment_name == "test"
    
    def test_effective_batch_size(self, tmp_path):
        """Test effective batch size calculation."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"messages": []}\n')
        
        config = ProductionTrainingConfig(
            data=DataConfig(train_file=train_file),
            training=TrainingConfig(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
            ),
            distributed=DistributedConfig(world_size=2),
        )
        
        # 4 * 4 * 2 = 32
        assert config.get_effective_batch_size() == 32

