# Production LLM Fine-Tuning Guide

Complete guide for fine-tuning the Social Media Radar LLM with production-grade safety and reproducibility.

## Quick Start

```bash
# Install training dependencies
pip install -r requirements.txt

# Validate configuration
python train.py --config configs/training/default.yaml --validate-only

# Quick test run (small dataset, 1 epoch)
python train.py --config configs/training/quick-test.yaml

# Full production training
python train.py --config configs/training/default.yaml
```

## Configuration

### Configuration Files

- `configs/training/default.yaml` - Production training configuration
- `configs/training/quick-test.yaml` - Fast testing configuration

### Configuration Structure

```yaml
# Experiment metadata
config_version: "1.0.0"
experiment_name: "my-experiment"

# Model configuration
model:
  base_model: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  model_max_length: 4096

# LoRA configuration (parameter-efficient fine-tuning)
lora:
  enabled: true
  r: 16                    # Rank (higher = more parameters)
  alpha: 32                # Scaling factor (typically 2x rank)
  dropout: 0.05
  use_4bit: true          # QLoRA 4-bit quantization
  bnb_4bit_quant_type: "nf4"

# Training hyperparameters
training:
  num_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch = 4 * 4 = 16
  learning_rate: 0.0002
  optimizer: "paged_adamw_32bit"
  lr_scheduler: "cosine"
  precision: "bf16"
  gradient_checkpointing: true

# Data configuration
data:
  train_file: "data/training/train.jsonl"
  val_file: "data/training/val.jsonl"
  quality_threshold: 0.7

# Checkpointing
checkpoint:
  output_dir: "./checkpoints/my-experiment"
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 3
  load_best_model_at_end: true

# Logging
logging:
  logging_dir: "./logs/my-experiment"
  logging_steps: 10
  eval_steps: 500
  report_to: ["tensorboard"]

# Reproducibility
reproducibility:
  seed: 42
  deterministic: true
  save_environment: true
  save_rng_state: true
```

### Configuration Overrides

Override any configuration value from the command line:

```bash
python train.py --config configs/training/default.yaml \
    --override training.num_epochs=5 \
    --override training.learning_rate=0.0001 \
    --override checkpoint.save_steps=250
```

## Data Format

Training data must be in JSONL format with chat messages:

```jsonl
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is sentiment analysis?"},
  {"role": "assistant", "content": "Sentiment analysis is..."}
]}
```

### Data Preparation

1. **Format**: Each line is a JSON object with a `messages` array
2. **Quality**: Set `quality_threshold` to filter low-quality examples
3. **Splits**: Provide separate train/val files or use automatic splitting
4. **Size**: Minimum 100 examples recommended, 1000+ for production

## Training Features

### Reproducibility

- **Global Seeding**: All random number generators seeded
- **Deterministic Algorithms**: PyTorch deterministic mode enabled
- **Environment Capture**: Saves pip freeze, git commit, system info
- **RNG State Checkpointing**: Exact training resumption

### Safety & Stability

- **NaN/Inf Detection**: Automatic numerical instability monitoring
- **Gradient Clipping**: Prevents exploding gradients
- **Safe Checkpointing**: Atomic writes prevent corruption
- **Validation**: Configuration validation before training

### Efficiency

- **QLoRA**: 4-bit quantization reduces memory by 75%
- **Gradient Checkpointing**: Trade compute for memory
- **Paged Optimizers**: Efficient memory management
- **Mixed Precision**: BF16/FP16 training

## Monitoring

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/

# View at http://localhost:6006
```

### Metrics Tracked

- Training loss
- Validation loss
- Learning rate
- Gradient norm
- Training speed (samples/sec)

## Checkpoints

### Checkpoint Structure

```
checkpoints/my-experiment/
├── checkpoint-epoch1-step500/
│   ├── adapter_model.bin      # LoRA weights
│   ├── adapter_config.json
│   ├── training_state.pt      # Optimizer, scheduler, RNG states
│   └── metadata.json          # Epoch, step, metrics
├── final_model/               # Best model
└── config.yaml                # Training configuration
```

### Resume Training

```bash
python train.py --config configs/training/default.yaml \
    --override checkpoint.resume_from_checkpoint=./checkpoints/my-experiment/checkpoint-epoch1-step500
```

## Advanced Usage

### Multi-GPU Training

```yaml
distributed:
  enabled: true
  backend: "nccl"
  world_size: 4  # Number of GPUs
```

```bash
torchrun --nproc_per_node=4 train.py --config configs/training/default.yaml
```

### Custom Model

```yaml
model:
  base_model: "mistralai/Mistral-7B-Instruct-v0.2"
  model_max_length: 8192
```

### Hyperparameter Tuning

Key parameters to tune:

1. **Learning Rate** (0.0001 - 0.0003)
2. **LoRA Rank** (8, 16, 32, 64)
3. **Batch Size** (effective batch 16-64)
4. **Warmup Ratio** (0.03 - 0.1)

## Troubleshooting

### Out of Memory

1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Enable `gradient_checkpointing`
4. Use 4-bit quantization
5. Reduce `model_max_length`

### Poor Performance

1. Increase training data size
2. Increase `num_epochs`
3. Tune learning rate
4. Increase LoRA rank
5. Check data quality

### Training Instability

1. Reduce learning rate
2. Increase warmup ratio
3. Enable gradient clipping
4. Check for NaN/Inf in logs
5. Validate data format

## Best Practices

1. **Start Small**: Use quick-test config first
2. **Monitor Closely**: Watch TensorBoard during training
3. **Validate Often**: Use validation set to prevent overfitting
4. **Save Everything**: Enable environment and RNG state saving
5. **Document**: Keep notes on experiments and results
6. **Version Control**: Track config changes in git
7. **Test Thoroughly**: Evaluate on held-out test set

## Production Checklist

- [ ] Configuration validated
- [ ] Training data quality checked
- [ ] Validation set prepared
- [ ] Monitoring setup (TensorBoard)
- [ ] Checkpointing configured
- [ ] Reproducibility enabled
- [ ] Resource requirements verified
- [ ] Backup strategy in place

