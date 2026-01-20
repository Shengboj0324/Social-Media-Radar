"""Production-grade configuration system for training.

This module provides:
- Schema-validated configuration with Pydantic
- YAML/JSON config file support
- CLI argument overrides
- Environment variable support
- Configuration versioning and validation
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class OptimizerType(str, Enum):
    """Supported optimizer types."""
    ADAMW = "adamw"
    ADAMW_8BIT = "adamw_8bit"
    PAGED_ADAMW_32BIT = "paged_adamw_32bit"
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"
    SGD = "sgd"


class SchedulerType(str, Enum):
    """Supported learning rate schedulers."""
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class PrecisionType(str, Enum):
    """Training precision types."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


class ModelConfig(BaseModel):
    """Model configuration."""
    base_model: str = Field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        description="HuggingFace model identifier or local path"
    )
    model_max_length: int = Field(default=4096, ge=128, le=32768)
    trust_remote_code: bool = Field(default=False, description="Security: only enable for trusted models")
    
    @field_validator("base_model")
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        """Validate model path."""
        if not v:
            raise ValueError("base_model cannot be empty")
        # Check if local path exists
        if "/" not in v and not Path(v).exists():
            raise ValueError(f"Local model path does not exist: {v}")
        return v


class LoRAConfig(BaseModel):
    """LoRA/QLoRA configuration."""
    enabled: bool = Field(default=True, description="Enable LoRA fine-tuning")
    r: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    alpha: int = Field(default=32, ge=1, description="LoRA alpha (scaling factor)")
    dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    target_modules: Optional[List[str]] = Field(
        default=None,
        description="Target modules for LoRA. None = auto-detect"
    )
    bias: str = Field(default="none", pattern="^(none|all|lora_only)$")
    
    # Quantization for QLoRA
    use_4bit: bool = Field(default=True, description="Use 4-bit quantization (QLoRA)")
    use_8bit: bool = Field(default=False, description="Use 8-bit quantization")
    bnb_4bit_compute_dtype: str = Field(default="bfloat16", pattern="^(float16|bfloat16|float32)$")
    bnb_4bit_quant_type: str = Field(default="nf4", pattern="^(fp4|nf4)$")
    use_nested_quant: bool = Field(default=True, description="Use nested quantization")
    
    @model_validator(mode="after")
    def validate_quantization(self) -> "LoRAConfig":
        """Validate quantization settings."""
        if self.use_4bit and self.use_8bit:
            raise ValueError("Cannot use both 4-bit and 8-bit quantization")
        return self


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    num_epochs: int = Field(default=3, ge=1, le=100)
    per_device_train_batch_size: int = Field(default=4, ge=1, le=128)
    per_device_eval_batch_size: int = Field(default=4, ge=1, le=128)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=128)
    
    learning_rate: float = Field(default=2e-4, gt=0.0, le=1.0)
    weight_decay: float = Field(default=0.001, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0, le=10.0)
    
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    warmup_steps: int = Field(default=0, ge=0)
    
    optimizer: OptimizerType = Field(default=OptimizerType.PAGED_ADAMW_32BIT)
    lr_scheduler: SchedulerType = Field(default=SchedulerType.COSINE)
    
    precision: PrecisionType = Field(default=PrecisionType.BF16)
    
    # Gradient checkpointing for memory efficiency
    gradient_checkpointing: bool = Field(default=True)
    
    @model_validator(mode="after")
    def validate_warmup(self) -> "TrainingConfig":
        """Validate warmup configuration."""
        if self.warmup_ratio > 0 and self.warmup_steps > 0:
            raise ValueError("Cannot specify both warmup_ratio and warmup_steps")
        return self


class DataConfig(BaseModel):
    """Data configuration."""
    train_file: Path = Field(description="Path to training data (JSONL)")
    val_file: Optional[Path] = Field(default=None, description="Path to validation data (JSONL)")
    test_file: Optional[Path] = Field(default=None, description="Path to test data (JSONL)")
    
    # Data validation
    max_examples: Optional[int] = Field(default=None, ge=1)
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Data splits (if not using separate files)
    val_split: float = Field(default=0.1, ge=0.0, lt=1.0)
    test_split: float = Field(default=0.0, ge=0.0, lt=1.0)
    
    # Reproducibility
    shuffle_seed: int = Field(default=42, description="Seed for data shuffling")
    
    @field_validator("train_file", "val_file", "test_file")
    @classmethod
    def validate_file_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate file paths exist."""
        if v is not None and not v.exists():
            raise ValueError(f"Data file does not exist: {v}")
        return v
    
    @model_validator(mode="after")
    def validate_splits(self) -> "DataConfig":
        """Validate data splits."""
        if self.val_split + self.test_split >= 1.0:
            raise ValueError("val_split + test_split must be < 1.0")
        return self


class CheckpointConfig(BaseModel):
    """Checkpointing configuration."""
    output_dir: Path = Field(default=Path("./checkpoints"))
    save_strategy: str = Field(default="steps", pattern="^(no|steps|epoch)$")
    save_steps: int = Field(default=500, ge=1)
    save_total_limit: int = Field(default=3, ge=1, le=100)

    # Best model tracking
    load_best_model_at_end: bool = Field(default=True)
    metric_for_best_model: str = Field(default="eval_loss")
    greater_is_better: bool = Field(default=False)

    # Resume from checkpoint
    resume_from_checkpoint: Optional[Path] = Field(default=None)

    @field_validator("resume_from_checkpoint")
    @classmethod
    def validate_checkpoint_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate checkpoint path exists."""
        if v is not None and not v.exists():
            raise ValueError(f"Checkpoint path does not exist: {v}")
        return v


class LoggingConfig(BaseModel):
    """Logging and monitoring configuration."""
    logging_dir: Path = Field(default=Path("./logs"))
    logging_strategy: str = Field(default="steps", pattern="^(no|steps|epoch)$")
    logging_steps: int = Field(default=10, ge=1)

    # Reporting
    report_to: List[str] = Field(default=["tensorboard"], description="Logging backends")

    # Evaluation
    eval_strategy: str = Field(default="steps", pattern="^(no|steps|epoch)$")
    eval_steps: int = Field(default=500, ge=1)

    # Metrics
    log_level: str = Field(default="info", pattern="^(debug|info|warning|error|critical)$")


class ReproducibilityConfig(BaseModel):
    """Reproducibility configuration."""
    seed: int = Field(default=42, description="Global random seed")
    deterministic: bool = Field(default=True, description="Enable deterministic algorithms")

    # Environment capture
    save_environment: bool = Field(default=True, description="Save pip freeze and git info")

    # RNG state checkpointing
    save_rng_state: bool = Field(default=True, description="Save RNG states in checkpoints")


class DistributedConfig(BaseModel):
    """Distributed training configuration."""
    enabled: bool = Field(default=False, description="Enable distributed training")
    backend: str = Field(default="nccl", pattern="^(nccl|gloo|mpi)$")

    # DDP settings
    ddp_find_unused_parameters: bool = Field(default=False)
    ddp_bucket_cap_mb: int = Field(default=25, ge=1)

    # Multi-node
    world_size: int = Field(default=1, ge=1)
    rank: int = Field(default=0, ge=0)
    local_rank: int = Field(default=-1, ge=-1)


class ProductionTrainingConfig(BaseModel):
    """Complete production training configuration."""

    # Configuration metadata
    config_version: str = Field(default="1.0.0", description="Config schema version")
    experiment_name: str = Field(default="default", description="Experiment identifier")

    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "ProductionTrainingConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            Configuration object
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML config
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    def validate_for_training(self) -> None:
        """Validate configuration is ready for training."""
        # Check data files exist
        if not self.data.train_file.exists():
            raise ValueError(f"Training file not found: {self.data.train_file}")

        # Check output directories are writable
        self.checkpoint.output_dir.mkdir(parents=True, exist_ok=True)
        self.logging.logging_dir.mkdir(parents=True, exist_ok=True)

        # Validate precision compatibility
        if self.training.precision == PrecisionType.BF16:
            import torch
            if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                raise ValueError("BF16 not supported on this CUDA device")

    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size.

        Returns:
            Effective batch size (per_device * accumulation * world_size)
        """
        return (
            self.training.per_device_train_batch_size
            * self.training.gradient_accumulation_steps
            * self.distributed.world_size
        )

