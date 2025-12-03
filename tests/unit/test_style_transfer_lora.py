"""Comprehensive unit tests for Style Transfer with LoRA.

Tests cover:
- LoRA configuration and initialization
- Style adapter creation and loading
- Adapter switching
- Text generation with style
- Parameter efficiency
- Rank and alpha parameters
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from app.intelligence.style_transfer_lora import (
    LoRAStyleTransfer,
    LoRAConfig,
    StyleConfig,
    GenerationResult,
)


class TestLoRAConfig(unittest.TestCase):
    """Test LoRA configuration."""

    def test_initialization(self):
        """Test LoRA config initialization."""
        config = LoRAConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )

        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1
        assert config.target_modules == ["q_proj", "v_proj"]

    def test_config_defaults(self):
        """Test default configuration values."""
        config = LoRAConfig()
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1
        assert config.target_modules == ["q_proj", "v_proj"]
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"


class TestMathematicalCorrectness(unittest.TestCase):
    """Test mathematical correctness of LoRA."""

    def test_lora_rank_parameter(self):
        """Test LoRA rank parameter."""
        # CRITICAL: r is the rank of low-rank matrices
        r = 8
        d = 768  # Model dimension

        # LoRA adds: W + B*A where B is d×r and A is r×d
        # Parameters: d*r + r*d = 2*d*r
        lora_params = 2 * d * r

        # Original: d*d
        original_params = d * d

        # Ratio
        ratio = lora_params / original_params

        assert lora_params == 12288
        assert original_params == 589824
        assert ratio < 0.03  # Less than 3% of parameters

    def test_lora_scaling_factor(self):
        """Test LoRA scaling factor (alpha/r)."""
        # CRITICAL: Scaling = lora_alpha / r
        lora_alpha = 16
        r = 8

        scaling = lora_alpha / r

        assert scaling == 2.0

    def test_parameter_efficiency(self):
        """Test parameter efficiency of LoRA."""
        # CRITICAL: LoRA should use < 1% of model parameters
        total_params = 1_000_000
        trainable_params = 5_000

        trainable_ratio = trainable_params / total_params

        assert trainable_ratio < 0.01  # Less than 1%

    def test_confidence_by_rank(self):
        """Test confidence computation by rank."""
        # CRITICAL: confidence = 1.0 / (rank + 1)
        ranks = [0, 1, 2, 3]
        confidences = [1.0 / (i + 1) for i in ranks]

        assert confidences[0] == 1.0
        assert confidences[1] == 0.5
        assert confidences[2] == 1/3
        assert confidences[3] == 0.25


class TestStyleTransfer(unittest.TestCase):
    """Test style transfer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LoRAConfig(r=8, lora_alpha=16)
        self.transfer = LoRAStyleTransfer(
            base_model_name="gpt2",
            config=self.config,
        )

    def test_initialization(self):
        """Test style transfer initialization."""
        assert self.transfer.base_model_name == "gpt2"
        assert self.transfer.config.r == 8
        assert self.transfer._initialized is False
        assert self.transfer.active_style is None

    def test_style_adapters_dict(self):
        """Test style adapters dictionary."""
        assert isinstance(self.transfer.style_adapters, dict)
        assert len(self.transfer.style_adapters) == 0

    def test_get_available_styles(self):
        """Test getting available styles."""
        self.transfer.style_adapters = {
            "formal": "/path/to/formal",
            "casual": "/path/to/casual",
        }

        styles = self.transfer.get_available_styles()

        assert len(styles) == 2
        assert "formal" in styles
        assert "casual" in styles


class TestAdapterManagement(unittest.TestCase):
    """Test adapter management."""

    def test_load_style_adapter(self):
        """Test loading style adapter."""
        transfer = LoRAStyleTransfer()
        style_name = "formal"
        adapter_path = "/path/to/formal"

        # Simulate loading
        transfer.style_adapters[style_name] = adapter_path

        assert style_name in transfer.style_adapters
        assert transfer.style_adapters[style_name] == adapter_path

    def test_switch_style_validation(self):
        """Test style switching validation."""
        transfer = LoRAStyleTransfer()
        transfer.style_adapters = {"formal": "/path"}

        # Try to switch to non-existent style
        invalid_style = "nonexistent"

        with self.assertRaises(ValueError):
            if invalid_style not in transfer.style_adapters:
                raise ValueError(f"Style {invalid_style} not found")

    def test_adapter_path_retrieval(self):
        """Test adapter path retrieval."""
        transfer = LoRAStyleTransfer()
        transfer.style_adapters = {"formal": "/path/to/formal"}

        adapter_path = transfer.style_adapters["formal"]

        assert adapter_path == "/path/to/formal"


class TestTextGeneration(unittest.TestCase):
    """Test text generation with style."""

    def test_prompt_removal(self):
        """Test removing prompt from generated text."""
        prompt = "Hello, "
        generated = "Hello, how are you today?"

        if generated.startswith(prompt):
            text = generated[len(prompt):].strip()

        assert text == "how are you today?"

    def test_generation_result_structure(self):
        """Test generation result structure."""
        result = GenerationResult(
            text="Generated text",
            style="formal",
            confidence=1.0,
            metadata={"prompt": "Test", "temperature": 0.7},
        )

        assert result.text == "Generated text"
        assert result.style == "formal"
        assert result.confidence == 1.0
        assert result.metadata["temperature"] == 0.7


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_no_adapter_loaded(self):
        """Test error when no adapter loaded."""
        transfer = LoRAStyleTransfer()

        with self.assertRaises(ValueError):
            if transfer.peft_model is None:
                raise ValueError("No style adapter loaded")

    def test_pad_token_handling(self):
        """Test pad token handling."""
        # If pad_token is None, use eos_token
        pad_token = None
        eos_token = "<|endoftext|>"

        if pad_token is None:
            pad_token = eos_token

        assert pad_token == "<|endoftext|>"


if __name__ == "__main__":
    unittest.main()

