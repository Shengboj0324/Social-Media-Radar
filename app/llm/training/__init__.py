"""LLM training and fine-tuning infrastructure.

This package provides industrial-grade fine-tuning capabilities:
- Training data pipeline with quality validation
- LoRA/QLoRA efficient fine-tuning
- Comprehensive evaluation framework
- Model versioning and deployment
"""

from app.llm.training.data_pipeline import (
    DatasetMetadata,
    TrainingDataPipeline,
    TrainingExample,
)
from app.llm.training.evaluator import EvaluationMetrics, ModelEvaluator
from app.llm.training.lora_trainer import LoRATrainer, LoRATrainingConfig

__all__ = [
    "TrainingExample",
    "DatasetMetadata",
    "TrainingDataPipeline",
    "LoRATrainingConfig",
    "LoRATrainer",
    "ModelEvaluator",
    "EvaluationMetrics",
]

