"""Model evaluation framework for fine-tuned models.

This module provides comprehensive evaluation capabilities:
- Perplexity calculation
- BLEU/ROUGE scores
- Human evaluation metrics
- A/B testing framework
- Quality regression detection
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Optional imports for evaluation metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    rouge_scorer = None

try:
    from sacrebleu import BLEU
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    BLEU = None

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a model."""

    # Automatic metrics
    perplexity: Optional[float] = None
    bleu_score: Optional[float] = None
    rouge_1: Optional[float] = None
    rouge_2: Optional[float] = None
    rouge_l: Optional[float] = None

    # Quality metrics
    avg_response_length: Optional[float] = None
    avg_latency_ms: Optional[float] = None

    # Human evaluation (if available)
    human_rating: Optional[float] = None  # 1-5 scale
    human_preference: Optional[float] = None  # % preferred over baseline

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            k: v for k, v in self.__dict__.items() if v is not None
        }


class ModelEvaluator:
    """Evaluator for fine-tuned models."""

    def __init__(self):
        """Initialize evaluator."""
        self.bleu = BLEU() if BLEU_AVAILABLE else None
        self.rouge = (
            rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"],
                use_stemmer=True,
            )
            if ROUGE_AVAILABLE
            else None
        )

        if not BLEU_AVAILABLE:
            logger.warning("BLEU scorer not available (sacrebleu not installed)")
        if not ROUGE_AVAILABLE:
            logger.warning("ROUGE scorer not available (rouge-score not installed)")

        logger.info("Initialized model evaluator")

    def calculate_perplexity(
        self,
        model,
        tokenizer,
        texts: List[str],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> float:
        """Calculate perplexity on a set of texts.

        Args:
            model: Language model
            tokenizer: Tokenizer
            texts: List of texts
            device: Device to use

        Returns:
            Perplexity score
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(device)

                # Forward pass
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                # Accumulate
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)

        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity

    def calculate_bleu(
        self,
        predictions: List[str],
        references: List[List[str]],
    ) -> float:
        """Calculate BLEU score.

        Args:
            predictions: List of predicted texts
            references: List of reference texts (can be multiple per prediction)

        Returns:
            BLEU score
        """
        # Format references for sacrebleu
        # sacrebleu expects references as list of lists
        formatted_refs = []
        for ref_list in references:
            if isinstance(ref_list, str):
                ref_list = [ref_list]
            formatted_refs.append(ref_list)

        # Transpose references
        transposed_refs = list(zip(*formatted_refs))

        # Calculate BLEU
        bleu_score = self.bleu.corpus_score(predictions, transposed_refs)

        logger.info(f"BLEU score: {bleu_score.score:.2f}")
        return bleu_score.score

    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Calculate ROUGE scores.

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            Dictionary of ROUGE scores
        """
        rouge_scores = {
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
        }

        for pred, ref in zip(predictions, references):
            scores = self.rouge.score(ref, pred)
            rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
            rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
            rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

        # Average scores
        avg_scores = {
            k: np.mean(v) for k, v in rouge_scores.items()
        }

        logger.info(f"ROUGE scores: {avg_scores}")
        return avg_scores

    def evaluate_model(
        self,
        model,
        tokenizer,
        test_data: List[Dict[str, str]],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> EvaluationMetrics:
        """Comprehensive model evaluation.

        Args:
            model: Language model
            tokenizer: Tokenizer
            test_data: List of test examples with 'prompt' and 'reference' keys
            device: Device to use

        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating model on {len(test_data)} examples")

        # Generate predictions
        predictions = []
        references = []
        latencies = []

        model.eval()
        model.to(device)

        for example in test_data:
            prompt = example["prompt"]
            reference = example["reference"]

            # Generate
            import time
            start_time = time.time()

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                )

            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            latency_ms = (time.time() - start_time) * 1000

            predictions.append(prediction)
            references.append(reference)
            latencies.append(latency_ms)

        # Calculate metrics
        metrics = EvaluationMetrics()

        # Perplexity
        try:
            metrics.perplexity = self.calculate_perplexity(
                model, tokenizer, [ex["reference"] for ex in test_data], device
            )
        except Exception as e:
            logger.warning(f"Failed to calculate perplexity: {e}")

        # BLEU
        try:
            metrics.bleu_score = self.calculate_bleu(predictions, [[r] for r in references])
        except Exception as e:
            logger.warning(f"Failed to calculate BLEU: {e}")

        # ROUGE
        try:
            rouge_scores = self.calculate_rouge(predictions, references)
            metrics.rouge_1 = rouge_scores["rouge1"]
            metrics.rouge_2 = rouge_scores["rouge2"]
            metrics.rouge_l = rouge_scores["rougeL"]
        except Exception as e:
            logger.warning(f"Failed to calculate ROUGE: {e}")

        # Quality metrics
        metrics.avg_response_length = np.mean([len(p) for p in predictions])
        metrics.avg_latency_ms = np.mean(latencies)

        logger.info(f"Evaluation complete: {metrics.to_dict()}")
        return metrics

    def compare_models(
        self,
        model_a,
        model_b,
        tokenizer,
        test_data: List[Dict[str, str]],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> Dict[str, Any]:
        """Compare two models on the same test data.

        Args:
            model_a: First model
            model_b: Second model
            tokenizer: Tokenizer
            test_data: Test data
            device: Device to use

        Returns:
            Comparison results
        """
        logger.info("Comparing two models...")

        # Evaluate both models
        metrics_a = self.evaluate_model(model_a, tokenizer, test_data, device)
        metrics_b = self.evaluate_model(model_b, tokenizer, test_data, device)

        # Calculate improvements
        improvements = {}
        for key in metrics_a.to_dict().keys():
            val_a = getattr(metrics_a, key)
            val_b = getattr(metrics_b, key)

            if val_a is not None and val_b is not None:
                # For perplexity, lower is better
                if key == "perplexity":
                    improvement = ((val_a - val_b) / val_a) * 100
                else:
                    improvement = ((val_b - val_a) / val_a) * 100

                improvements[key] = improvement

        comparison = {
            "model_a_metrics": metrics_a.to_dict(),
            "model_b_metrics": metrics_b.to_dict(),
            "improvements": improvements,
            "winner": "model_b" if sum(improvements.values()) > 0 else "model_a",
        }

        logger.info(f"Comparison complete: {comparison}")
        return comparison

    def detect_regression(
        self,
        baseline_metrics: EvaluationMetrics,
        new_metrics: EvaluationMetrics,
        threshold: float = 0.05,  # 5% regression threshold
    ) -> Dict[str, bool]:
        """Detect quality regression.

        Args:
            baseline_metrics: Baseline metrics
            new_metrics: New model metrics
            threshold: Regression threshold (0.0-1.0)

        Returns:
            Dictionary of regression flags
        """
        regressions = {}

        for key in baseline_metrics.to_dict().keys():
            baseline_val = getattr(baseline_metrics, key)
            new_val = getattr(new_metrics, key)

            if baseline_val is not None and new_val is not None:
                # For perplexity, higher is worse
                if key == "perplexity":
                    regression = (new_val - baseline_val) / baseline_val > threshold
                # For latency, higher is worse
                elif "latency" in key:
                    regression = (new_val - baseline_val) / baseline_val > threshold
                # For other metrics, lower is worse
                else:
                    regression = (baseline_val - new_val) / baseline_val > threshold

                regressions[key] = regression

        has_regression = any(regressions.values())

        logger.info(
            f"Regression detection: has_regression={has_regression}, "
            f"regressions={regressions}"
        )

        return {
            "has_regression": has_regression,
            "regressions": regressions,
        }

