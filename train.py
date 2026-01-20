#!/usr/bin/env python3
"""Production training entrypoint for Social Media Radar LLM fine-tuning.

Usage:
    # Train with default config
    python train.py --config configs/training/default.yaml
    
    # Train with config overrides
    python train.py --config configs/training/default.yaml \\
        --training.num_epochs 5 \\
        --training.learning_rate 0.0001
    
    # Quick test run
    python train.py --config configs/training/quick-test.yaml
    
    # Resume from checkpoint
    python train.py --config configs/training/default.yaml \\
        --checkpoint.resume_from_checkpoint ./checkpoints/checkpoint-epoch1-step500
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.llm.training.config import ProductionTrainingConfig
from app.llm.training.trainer import ProductionTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_override(override: str) -> tuple[str, Any]:
    """Parse a config override argument.
    
    Args:
        override: Override in format "key.subkey=value"
        
    Returns:
        Tuple of (key_path, value)
    """
    if "=" not in override:
        raise ValueError(f"Invalid override format: {override}")
    
    key, value = override.split("=", 1)
    
    # Try to parse value as int, float, bool, or keep as string
    if value.lower() in ("true", "false"):
        value = value.lower() == "true"
    else:
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string
    
    return key, value


def apply_overrides(config_dict: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """Apply CLI overrides to config dictionary.
    
    Args:
        config_dict: Configuration dictionary
        overrides: List of override strings
        
    Returns:
        Updated configuration dictionary
    """
    for override in overrides:
        key, value = parse_override(override)
        
        # Navigate nested dict
        keys = key.split(".")
        current = config_dict
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
        logger.info(f"Override: {key} = {value}")
    
    return config_dict


def main():
    """Main training entrypoint."""
    parser = argparse.ArgumentParser(
        description="Production LLM Fine-Tuning for Social Media Radar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training configuration YAML file",
    )
    
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values (e.g., --override training.num_epochs=5)",
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without training",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Setup and validate but don't start training",
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = ProductionTrainingConfig.from_yaml(args.config)
        
        # Apply overrides
        if args.override:
            logger.info(f"Applying {len(args.override)} overrides")
            config_dict = config.model_dump()
            config_dict = apply_overrides(config_dict, args.override)
            config = ProductionTrainingConfig(**config_dict)
        
        # Validate configuration
        logger.info("Validating configuration...")
        config.validate_for_training()
        logger.info("✓ Configuration valid")
        
        if args.validate_only:
            logger.info("Validation complete (--validate-only)")
            return 0
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = ProductionTrainer(config)
        
        # Setup environment
        logger.info("Setting up training environment...")
        trainer.setup()
        
        if args.dry_run:
            logger.info("Dry run complete (--dry-run)")
            return 0
        
        # Load model
        logger.info("Loading model...")
        trainer.load_model()
        
        # Prepare datasets
        logger.info("Preparing datasets...")
        train_dataset, val_dataset = trainer.prepare_datasets()
        
        # Train
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        logger.info(f"Experiment: {config.experiment_name}")
        logger.info(f"Model: {config.model.base_model}")
        logger.info(f"Training examples: {len(train_dataset)}")
        logger.info(f"Validation examples: {len(val_dataset) if val_dataset else 0}")
        logger.info(f"Effective batch size: {config.get_effective_batch_size()}")
        logger.info(f"Total epochs: {config.training.num_epochs}")
        logger.info("=" * 80)
        
        metrics = trainer.train(train_dataset, val_dataset)
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Final metrics: {metrics}")
        logger.info(f"Model saved to: {config.checkpoint.output_dir / 'final_model'}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

