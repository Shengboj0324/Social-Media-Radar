"""LoRA/QLoRA fine-tuning for efficient model customization.

This module provides industrial-grade fine-tuning capabilities:
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- QLoRA (Quantized LoRA) for memory-efficient training
- Distributed training support
- Automatic checkpoint management
- Training metrics and logging
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Optional imports for training
try:
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    Trainer = None
    TrainingArguments = None

logger = logging.getLogger(__name__)


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA fine-tuning."""

    # Model configuration
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_max_length: int = 4096

    # LoRA configuration
    lora_r: int = 16  # Rank
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.05
    target_modules: List[str] = None  # Auto-detect if None

    # Quantization (for QLoRA)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True

    # Training configuration
    output_dir: str = "./lora_models"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # Optimization
    optim: str = "paged_adamw_32bit"
    weight_decay: float = 0.001
    fp16: bool = False
    bf16: bool = True

    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True

    # Distributed training
    ddp_find_unused_parameters: bool = False

    def __post_init__(self):
        """Set default target modules if not specified."""
        if self.target_modules is None:
            # Default for Llama models
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]


class LoRATrainer:
    """Trainer for LoRA/QLoRA fine-tuning."""

    def __init__(self, config: LoRATrainingConfig):
        """Initialize LoRA trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

        logger.info(f"Initialized LoRA trainer: base_model={config.base_model}")

    def load_model(self) -> None:
        """Load base model and apply LoRA."""
        logger.info(f"Loading base model: {self.config.base_model}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            model_max_length=self.config.model_max_length,
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config for QLoRA
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )
        else:
            bnb_config = None

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare model for k-bit training
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_percent = 100 * trainable_params / total_params

        logger.info(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({trainable_percent:.2f}%)"
        )

    def prepare_dataset(self, train_file: Path, val_file: Optional[Path] = None):
        """Prepare dataset for training.

        Args:
            train_file: Path to training data (JSONL)
            val_file: Path to validation data (JSONL)

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        from datasets import load_dataset

        # Load datasets
        data_files = {"train": str(train_file)}
        if val_file:
            data_files["validation"] = str(val_file)

        dataset = load_dataset("json", data_files=data_files)

        # Tokenize function
        def tokenize_function(examples):
            # Format messages into prompt
            prompts = []
            for messages in examples["messages"]:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                prompts.append(prompt)

            # Tokenize
            return self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.config.model_max_length,
                padding="max_length",
            )

        # Tokenize datasets
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        return tokenized_dataset["train"], tokenized_dataset.get("validation")

    def train(
        self,
        train_dataset,
        val_dataset=None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, float]:
        """Train the model with LoRA.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if val_dataset else None,
            evaluation_strategy="steps" if val_dataset else "no",
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end and val_dataset is not None,
            ddp_find_unused_parameters=self.config.ddp_find_unused_parameters,
            report_to=["tensorboard"],
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Train
        logger.info("Starting training...")
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final model
        self.trainer.save_model()

        # Save metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)

        logger.info(f"Training completed. Metrics: {metrics}")
        return metrics

    def evaluate(self, eval_dataset) -> Dict[str, float]:
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

    def save_model(self, output_dir: str) -> None:
        """Save the fine-tuned model.

        Args:
            output_dir: Output directory
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        logger.info(f"Model saved to {output_path}")

    @staticmethod
    def load_finetuned_model(
        base_model: str,
        lora_weights: str,
        device_map: str = "auto",
    ):
        """Load a fine-tuned LoRA model.

        Args:
            base_model: Base model name
            lora_weights: Path to LoRA weights
            device_map: Device map

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading fine-tuned model: base={base_model}, lora={lora_weights}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Load LoRA weights
        model = PeftModel.from_pretrained(model, lora_weights)

        # Merge LoRA weights (optional, for inference)
        model = model.merge_and_unload()

        logger.info("Model loaded successfully")
        return model, tokenizer

