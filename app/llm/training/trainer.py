"""Production-grade training loop with comprehensive safety and reproducibility.

This module provides:
- Reproducible training with global seeding
- Numerical stability checks (NaN/Inf detection)
- Comprehensive checkpointing with RNG states
- Early stopping and learning rate scheduling
- Gradient clipping and monitoring
- Safe checkpoint loading/saving
- Training metrics persistence
"""

import hashlib
import json
import logging
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from datasets import load_dataset
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

from .config import ProductionTrainingConfig, PrecisionType

logger = logging.getLogger(__name__)


class ReproducibilityManager:
    """Manages reproducibility across training runs."""
    
    def __init__(self, seed: int = 42, deterministic: bool = True):
        """Initialize reproducibility manager.
        
        Args:
            seed: Global random seed
            deterministic: Enable deterministic algorithms
        """
        self.seed = seed
        self.deterministic = deterministic
        self.rng_states: Dict[str, Any] = {}
    
    def set_global_seed(self) -> None:
        """Set global random seed for all libraries."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Enable deterministic algorithms (may impact performance)
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        logger.info(f"Set global seed to {self.seed} (deterministic={self.deterministic})")
    
    def save_rng_states(self) -> Dict[str, Any]:
        """Save RNG states for reproducibility.
        
        Returns:
            Dictionary of RNG states
        """
        states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        
        if torch.cuda.is_available():
            states["torch_cuda"] = torch.cuda.get_rng_state_all()
        
        self.rng_states = states
        return states
    
    def load_rng_states(self, states: Dict[str, Any]) -> None:
        """Load RNG states.
        
        Args:
            states: Dictionary of RNG states
        """
        random.setstate(states["python"])
        np.random.set_state(states["numpy"])
        torch.set_rng_state(states["torch"])
        
        if torch.cuda.is_available() and "torch_cuda" in states:
            torch.cuda.set_rng_state_all(states["torch_cuda"])
        
        logger.info("Loaded RNG states")
    
    def capture_environment(self, output_dir: Path) -> Dict[str, str]:
        """Capture environment information for reproducibility.
        
        Args:
            output_dir: Directory to save environment info
            
        Returns:
            Environment information dictionary
        """
        env_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "python_version": subprocess.check_output(
                ["python", "--version"], text=True
            ).strip(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "seed": self.seed,
            "deterministic": self.deterministic,
        }
        
        # Capture pip freeze
        try:
            pip_freeze = subprocess.check_output(
                ["pip", "freeze"], text=True
            )
            env_info["pip_freeze"] = pip_freeze
        except Exception as e:
            logger.warning(f"Failed to capture pip freeze: {e}")
        
        # Capture git info
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
            ).strip()
            env_info["git_commit"] = git_commit
            env_info["git_branch"] = git_branch
        except Exception as e:
            logger.warning(f"Failed to capture git info: {e}")
        
        # Save to file
        env_file = output_dir / "environment.json"
        with open(env_file, "w") as f:
            json.dump(env_info, f, indent=2)
        
        logger.info(f"Saved environment info to {env_file}")
        return env_info


class NumericalStabilityMonitor:
    """Monitors training for numerical instability."""

    def __init__(self):
        """Initialize stability monitor."""
        self.nan_count = 0
        self.inf_count = 0
        self.large_gradient_count = 0

    def check_tensor(self, tensor: torch.Tensor, name: str) -> bool:
        """Check tensor for NaN/Inf values.

        Args:
            tensor: Tensor to check
            name: Name for logging

        Returns:
            True if tensor is stable
        """
        if torch.isnan(tensor).any():
            self.nan_count += 1
            logger.error(f"NaN detected in {name}")
            return False

        if torch.isinf(tensor).any():
            self.inf_count += 1
            logger.error(f"Inf detected in {name}")
            return False

        return True

    def check_gradients(self, model: torch.nn.Module, threshold: float = 10.0) -> Dict[str, float]:
        """Check gradients for numerical issues.

        Args:
            model: Model to check
            threshold: Threshold for large gradients

        Returns:
            Gradient statistics
        """
        total_norm = 0.0
        max_grad = 0.0
        num_params = 0

        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_grad = max(max_grad, param_norm)
                num_params += 1

                # Check for NaN/Inf
                if not self.check_tensor(p.grad.data, f"gradient_{p.shape}"):
                    return {"stable": False, "total_norm": float('nan')}

        total_norm = total_norm ** 0.5

        if total_norm > threshold:
            self.large_gradient_count += 1
            logger.warning(f"Large gradient norm detected: {total_norm:.4f}")

        return {
            "stable": True,
            "total_norm": total_norm,
            "max_grad": max_grad,
            "num_params": num_params,
        }

    def get_stats(self) -> Dict[str, int]:
        """Get stability statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "large_gradient_count": self.large_gradient_count,
        }


class SafeCheckpointManager:
    """Manages checkpoints with atomic writes and validation."""

    def __init__(self, output_dir: Path):
        """Initialize checkpoint manager.

        Args:
            output_dir: Directory for checkpoints
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        rng_states: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save checkpoint with atomic write.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: LR scheduler state
            epoch: Current epoch
            step: Current step
            metrics: Training metrics
            rng_states: RNG states for reproducibility

        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint-epoch{epoch}-step{step}"
        checkpoint_dir = self.output_dir / checkpoint_name
        temp_dir = self.output_dir / f"{checkpoint_name}.tmp"

        try:
            # Save to temporary directory first (atomic write)
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(temp_dir)
            else:
                torch.save(model.state_dict(), temp_dir / "model.pt")

            # Save optimizer and scheduler
            torch.save({
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
                "step": step,
                "metrics": metrics,
                "rng_states": rng_states,
            }, temp_dir / "training_state.pt")

            # Save metadata
            metadata = {
                "epoch": epoch,
                "step": step,
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat(),
            }
            with open(temp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Atomic rename
            if checkpoint_dir.exists():
                import shutil
                shutil.rmtree(checkpoint_dir)
            temp_dir.rename(checkpoint_dir)

            logger.info(f"Saved checkpoint to {checkpoint_dir}")
            return checkpoint_dir

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
            raise

    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load checkpoint with validation.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Checkpoint data
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load and validate metadata
        metadata_file = checkpoint_path / "metadata.json"
        if not metadata_file.exists():
            raise ValueError(f"Invalid checkpoint: missing metadata.json")

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Load training state
        state_file = checkpoint_path / "training_state.pt"
        if state_file.exists():
            training_state = torch.load(state_file, map_location="cpu")
        else:
            training_state = {}

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return {
            "metadata": metadata,
            "training_state": training_state,
            "checkpoint_path": checkpoint_path,
        }


class ProductionTrainer:
    """Production-grade trainer with comprehensive safety and monitoring."""

    def __init__(self, config: ProductionTrainingConfig):
        """Initialize production trainer.

        Args:
            config: Training configuration
        """
        if not TRAINING_AVAILABLE:
            raise ImportError(
                "Training dependencies not available. "
                "Install with: pip install transformers peft datasets"
            )

        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

        # Initialize managers
        self.reproducibility = ReproducibilityManager(
            seed=config.reproducibility.seed,
            deterministic=config.reproducibility.deterministic,
        )
        self.stability_monitor = NumericalStabilityMonitor()
        self.checkpoint_manager = SafeCheckpointManager(config.checkpoint.output_dir)

        logger.info(f"Initialized ProductionTrainer: experiment={config.experiment_name}")

    def setup(self) -> None:
        """Setup training environment."""
        # Set global seed
        self.reproducibility.set_global_seed()

        # Validate configuration
        self.config.validate_for_training()

        # Capture environment
        if self.config.reproducibility.save_environment:
            self.reproducibility.capture_environment(self.config.checkpoint.output_dir)

        # Save configuration
        config_path = self.config.checkpoint.output_dir / "config.yaml"
        self.config.to_yaml(config_path)
        logger.info(f"Saved configuration to {config_path}")

    def load_model(self) -> None:
        """Load and prepare model for training."""
        logger.info(f"Loading model: {self.config.model.base_model}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model,
            trust_remote_code=self.config.model.trust_remote_code,
            model_max_length=self.config.model.model_max_length,
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Quantization config
        bnb_config = None
        if self.config.lora.enabled:
            if self.config.lora.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.config.lora.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=getattr(
                        torch, self.config.lora.bnb_4bit_compute_dtype
                    ),
                    bnb_4bit_use_double_quant=self.config.lora.use_nested_quant,
                )
            elif self.config.lora.use_8bit:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=self.config.model.trust_remote_code,
        )

        # Apply LoRA
        if self.config.lora.enabled:
            if self.config.lora.use_4bit or self.config.lora.use_8bit:
                self.model = prepare_model_for_kbit_training(self.model)

            # Auto-detect target modules if not specified
            target_modules = self.config.lora.target_modules
            if target_modules is None:
                # Default for Llama models
                target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ]

            lora_config = LoraConfig(
                r=self.config.lora.r,
                lora_alpha=self.config.lora.alpha,
                target_modules=target_modules,
                lora_dropout=self.config.lora.dropout,
                bias=self.config.lora.bias,
                task_type=TaskType.CAUSAL_LM,
            )

            self.model = get_peft_model(self.model, lora_config)

            # Log trainable parameters
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_percent = 100 * trainable_params / total_params

            logger.info(
                f"LoRA applied: {trainable_params:,} / {total_params:,} "
                f"trainable ({trainable_percent:.2f}%)"
            )

        # Enable gradient checkpointing
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")

    def prepare_datasets(self) -> Tuple[Any, Optional[Any]]:
        """Prepare training and validation datasets.

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        logger.info("Preparing datasets...")

        # Load datasets
        data_files = {"train": str(self.config.data.train_file)}
        if self.config.data.val_file:
            data_files["validation"] = str(self.config.data.val_file)

        dataset = load_dataset("json", data_files=data_files)

        # Tokenize function
        def tokenize_function(examples):
            prompts = []
            for messages in examples["messages"]:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                prompts.append(prompt)

            return self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.config.model.model_max_length,
                padding="max_length",
            )

        # Tokenize datasets
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing",
        )

        train_dataset = tokenized_dataset["train"]
        val_dataset = tokenized_dataset.get("validation")

        logger.info(
            f"Datasets prepared: train={len(train_dataset)}, "
            f"val={len(val_dataset) if val_dataset else 0}"
        )

        return train_dataset, val_dataset

    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset

        Returns:
            Training metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info("Starting training...")

        # Save RNG states
        if self.config.reproducibility.save_rng_state:
            self.reproducibility.save_rng_states()

        # Configure precision
        fp16 = self.config.training.precision == PrecisionType.FP16
        bf16 = self.config.training.precision == PrecisionType.BF16

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.config.checkpoint.output_dir),
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,

            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.max_grad_norm,

            warmup_ratio=self.config.training.warmup_ratio,
            warmup_steps=self.config.training.warmup_steps,

            optim=self.config.training.optimizer.value,
            lr_scheduler_type=self.config.training.lr_scheduler.value,

            fp16=fp16,
            bf16=bf16,

            logging_dir=str(self.config.logging.logging_dir),
            logging_strategy=self.config.logging.logging_strategy,
            logging_steps=self.config.logging.logging_steps,

            save_strategy=self.config.checkpoint.save_strategy,
            save_steps=self.config.checkpoint.save_steps,
            save_total_limit=self.config.checkpoint.save_total_limit,

            evaluation_strategy=self.config.logging.eval_strategy if val_dataset else "no",
            eval_steps=self.config.logging.eval_steps if val_dataset else None,

            load_best_model_at_end=self.config.checkpoint.load_best_model_at_end and val_dataset is not None,
            metric_for_best_model=self.config.checkpoint.metric_for_best_model,
            greater_is_better=self.config.checkpoint.greater_is_better,

            report_to=self.config.logging.report_to,

            # Distributed training
            ddp_find_unused_parameters=self.config.distributed.ddp_find_unused_parameters,

            # Reproducibility
            seed=self.config.reproducibility.seed,
            data_seed=self.config.data.shuffle_seed,

            # Disable tqdm for cleaner logs
            disable_tqdm=False,
        )

        # Callbacks
        callbacks = []
        if val_dataset and self.config.checkpoint.load_best_model_at_end:
            # Early stopping
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.0,
                )
            )

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks,
        )

        # Resume from checkpoint if specified
        resume_from = None
        if self.config.checkpoint.resume_from_checkpoint:
            resume_from = str(self.config.checkpoint.resume_from_checkpoint)
            logger.info(f"Resuming from checkpoint: {resume_from}")

        # Train
        try:
            train_result = self.trainer.train(resume_from_checkpoint=resume_from)

            # Save final model
            self.trainer.save_model()

            # Save metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)

            # Save final state
            self.trainer.save_state()

            logger.info(f"Training completed successfully. Metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")

            # Log stability stats
            stability_stats = self.stability_monitor.get_stats()
            logger.error(f"Stability stats: {stability_stats}")

            raise

    def evaluate(self, eval_dataset: Any) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            eval_dataset: Evaluation dataset

        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train() first.")

        logger.info("Starting evaluation...")
        metrics = self.trainer.evaluate(eval_dataset)

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

        logger.info(f"Evaluation completed. Metrics: {metrics}")
        return metrics

    def save_model(self, output_dir: Optional[Path] = None) -> None:
        """Save the trained model.

        Args:
            output_dir: Output directory (default: checkpoint output_dir)
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        if output_dir is None:
            output_dir = self.config.checkpoint.output_dir / "final_model"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        else:
            torch.save(self.model.state_dict(), output_dir / "model.pt")

        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)

        # Save config
        config_path = output_dir / "training_config.yaml"
        self.config.to_yaml(config_path)

        logger.info(f"Model saved to {output_dir}")

    @staticmethod
    def load_trained_model(
        model_path: Path,
        device_map: str = "auto",
    ) -> Tuple[Any, Any]:
        """Load a trained model.

        Args:
            model_path: Path to saved model
            device_map: Device map for loading

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading trained model from {model_path}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            trust_remote_code=False,  # Security: don't trust remote code
        )

        logger.info("Model loaded successfully")
        return model, tokenizer


