"""Training data pipeline for LLM fine-tuning.

This module provides industrial-grade data collection and preparation:
- Data collection from production usage
- Quality validation and filtering
- Format conversion (JSONL, Parquet, HuggingFace datasets)
- Data versioning and tracking
- Privacy-preserving data handling
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class TrainingExample(BaseModel):
    """Single training example for fine-tuning."""

    # Input/output pair
    messages: List[Dict[str, str]]  # OpenAI chat format
    
    # Metadata
    example_id: str = Field(default_factory=lambda: hashlib.sha256(str(datetime.utcnow()).encode()).hexdigest()[:16])
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source: str  # Where this example came from
    quality_score: Optional[float] = None  # 0.0-1.0
    
    # Privacy
    contains_pii: bool = False
    anonymized: bool = False

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Validate message format."""
        if not v:
            raise ValueError("Messages cannot be empty")
        
        for msg in v:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content'")
            
            if msg["role"] not in ["system", "user", "assistant"]:
                raise ValueError(f"Invalid role: {msg['role']}")
        
        return v

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI fine-tuning format.

        Returns:
            Dictionary in OpenAI JSONL format
        """
        return {"messages": self.messages}

    def to_huggingface_format(self) -> Dict[str, Any]:
        """Convert to HuggingFace format.

        Returns:
            Dictionary in HuggingFace format
        """
        # Extract prompt and completion
        prompt_messages = [m for m in self.messages if m["role"] != "assistant"]
        completion_messages = [m for m in self.messages if m["role"] == "assistant"]

        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in prompt_messages)
        completion = completion_messages[0]["content"] if completion_messages else ""

        return {
            "prompt": prompt,
            "completion": completion,
            "metadata": {
                "example_id": self.example_id,
                "source": self.source,
                "quality_score": self.quality_score,
            },
        }


class DatasetMetadata(BaseModel):
    """Metadata for training dataset."""

    dataset_id: str
    version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    num_examples: int
    sources: List[str]
    quality_threshold: float
    contains_pii: bool
    anonymized: bool
    description: Optional[str] = None


class TrainingDataPipeline:
    """Pipeline for collecting and preparing training data."""

    def __init__(
        self,
        output_dir: Path,
        quality_threshold: float = 0.7,
        max_examples: Optional[int] = None,
    ):
        """Initialize training data pipeline.

        Args:
            output_dir: Directory to save datasets
            quality_threshold: Minimum quality score (0.0-1.0)
            max_examples: Maximum examples to collect
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.quality_threshold = quality_threshold
        self.max_examples = max_examples
        
        self.examples: List[TrainingExample] = []
        
        logger.info(
            f"Initialized training data pipeline: "
            f"output_dir={output_dir}, "
            f"quality_threshold={quality_threshold}"
        )

    def add_example(self, example: TrainingExample) -> bool:
        """Add example to dataset.

        Args:
            example: Training example

        Returns:
            True if example was added
        """
        # Check quality threshold
        if example.quality_score is not None and example.quality_score < self.quality_threshold:
            logger.debug(
                f"Skipping example {example.example_id}: "
                f"quality {example.quality_score} < threshold {self.quality_threshold}"
            )
            return False

        # Check max examples
        if self.max_examples and len(self.examples) >= self.max_examples:
            logger.warning(f"Max examples ({self.max_examples}) reached")
            return False

        # Check for PII if not anonymized
        if example.contains_pii and not example.anonymized:
            logger.warning(
                f"Skipping example {example.example_id}: contains PII and not anonymized"
            )
            return False

        self.examples.append(example)
        logger.debug(f"Added example {example.example_id} (total: {len(self.examples)})")
        return True

    def add_from_production(
        self,
        messages: List[Dict[str, str]],
        source: str,
        quality_score: Optional[float] = None,
    ) -> bool:
        """Add example from production usage.

        Args:
            messages: Conversation messages
            source: Source identifier
            quality_score: Quality score

        Returns:
            True if example was added
        """
        example = TrainingExample(
            messages=messages,
            source=source,
            quality_score=quality_score,
        )
        
        return self.add_example(example)

    def validate_dataset(self) -> Dict[str, Any]:
        """Validate dataset quality.

        Returns:
            Validation report
        """
        if not self.examples:
            return {"valid": False, "error": "No examples in dataset"}

        # Check for duplicates
        example_ids = [ex.example_id for ex in self.examples]
        duplicates = len(example_ids) - len(set(example_ids))

        # Check quality distribution
        quality_scores = [ex.quality_score for ex in self.examples if ex.quality_score is not None]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None

        # Check sources
        sources = list(set(ex.source for ex in self.examples))

        # Check PII
        pii_count = sum(1 for ex in self.examples if ex.contains_pii)
        anonymized_count = sum(1 for ex in self.examples if ex.anonymized)

        report = {
            "valid": True,
            "num_examples": len(self.examples),
            "duplicates": duplicates,
            "avg_quality": avg_quality,
            "sources": sources,
            "pii_count": pii_count,
            "anonymized_count": anonymized_count,
            "quality_threshold": self.quality_threshold,
        }

        logger.info(f"Dataset validation: {report}")
        return report

    def export_openai_jsonl(self, filename: str = "training_data.jsonl") -> Path:
        """Export dataset in OpenAI JSONL format.

        Args:
            filename: Output filename

        Returns:
            Path to exported file
        """
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            for example in self.examples:
                f.write(json.dumps(example.to_openai_format()) + "\n")

        logger.info(f"Exported {len(self.examples)} examples to {output_path}")
        return output_path

    def export_huggingface(self, filename: str = "training_data.json") -> Path:
        """Export dataset in HuggingFace format.

        Args:
            filename: Output filename

        Returns:
            Path to exported file
        """
        output_path = self.output_dir / filename

        data = [example.to_huggingface_format() for example in self.examples]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(self.examples)} examples to {output_path}")
        return output_path

    def export_metadata(self, dataset_id: str, version: str) -> Path:
        """Export dataset metadata.

        Args:
            dataset_id: Dataset identifier
            version: Dataset version

        Returns:
            Path to metadata file
        """
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            version=version,
            num_examples=len(self.examples),
            sources=list(set(ex.source for ex in self.examples)),
            quality_threshold=self.quality_threshold,
            contains_pii=any(ex.contains_pii for ex in self.examples),
            anonymized=all(ex.anonymized for ex in self.examples if ex.contains_pii),
        )

        output_path = self.output_dir / f"{dataset_id}_v{version}_metadata.json"

        with open(output_path, "w") as f:
            f.write(metadata.model_dump_json(indent=2))

        logger.info(f"Exported metadata to {output_path}")
        return output_path

    def split_train_val(
        self,
        val_ratio: float = 0.1,
        shuffle: bool = True,
    ) -> tuple[List[TrainingExample], List[TrainingExample]]:
        """Split dataset into train and validation sets.

        Args:
            val_ratio: Validation set ratio (0.0-1.0)
            shuffle: Whether to shuffle before splitting

        Returns:
            Tuple of (train_examples, val_examples)
        """
        import random

        examples = self.examples.copy()

        if shuffle:
            random.shuffle(examples)

        split_idx = int(len(examples) * (1 - val_ratio))
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        logger.info(
            f"Split dataset: train={len(train_examples)}, val={len(val_examples)}"
        )

        return train_examples, val_examples

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "num_examples": len(self.examples),
            "quality_threshold": self.quality_threshold,
            "max_examples": self.max_examples,
            "sources": list(set(ex.source for ex in self.examples)),
            "avg_quality": (
                sum(ex.quality_score for ex in self.examples if ex.quality_score is not None)
                / len([ex for ex in self.examples if ex.quality_score is not None])
                if any(ex.quality_score is not None for ex in self.examples)
                else None
            ),
        }

