# Multi-GPU Training Guide

This guide explains how to train using multiple GPUs with the DINO LoRA/DoRA framework.

## Overview

The framework supports multi-GPU training using PyTorch's `DataParallel` wrapper. This allows you to:
- Distribute model across multiple GPUs
- Parallelize computations for faster training
- Automatically handle gradient synchronization

## Quick Start

### Basic Multi-GPU Training

To use all available GPUs:

```bash
python scripts/train.py \
  --config configs/default_config.yaml \
  --data-dir data/train \
  --val-data-dir data/val \
  --multi-gpu
```

### Specify Which GPUs to Use

To use only specific GPUs (e.g., GPUs 0 and 1):

```bash
python scripts/train.py \
  --config configs/default_config.yaml \
  --data-dir data/train \
  --val-data-dir data/val \
  --multi-gpu \
  --gpu-ids 0 1
```

To use GPUs 1, 2, and 3:

```bash
python scripts/train.py \
  --config configs/default_config.yaml \
  --data-dir data/train \
  --val-data-dir data/val \
  --multi-gpu \
  --gpu-ids 1 2 3
```

## How It Works

### DataParallel Architecture

The framework uses `torch.nn.DataParallel` which:

1. **Splits batches**: Each batch is divided across GPUs
2. **Replicates model**: The model is replicated on each GPU
3. **Computes in parallel**: Forward/backward passes run on each GPU
4. **Synchronizes gradients**: All GPU gradients are averaged before the optimizer step

Example with 2 GPUs:
- Batch size 128 → Each GPU processes 64 samples
- Model on GPU 0 and GPU 1
- Gradients synchronized across GPUs
- Single optimizer step on GPU 0

### Device Management

The trainer automatically:
- Detects available GPUs
- Configures CUDA environment
- Logs GPU information
- Handles model checkpointing correctly (saves unwrapped state dict)

## Usage Examples

### Example 1: Basic Training with 2 GPUs

```bash
python scripts/train.py \
  --config configs/default_config.yaml \
  --data-dir data/train \
  --val-data-dir data/val \
  --multi-gpu \
  --batch-size 256 \
  --num-epochs 20
```

**Important**: When using DataParallel, you can increase batch size since it's split across GPUs.
- 2 GPUs with batch_size 256 → 128 samples per GPU
- Better GPU utilization and faster training

### Example 2: Specific GPUs with LoRA

```bash
python scripts/train.py \
  --config configs/default_config.yaml \
  --method lora \
  --data-dir data/train \
  --val-data-dir data/val \
  --multi-gpu \
  --gpu-ids 0 2 \
  --batch-size 256 \
  --learning-rate 1e-3
```

### Example 3: Specific GPUs with DoRA

```bash
python scripts/train.py \
  --config configs/default_config.yaml \
  --method dora \
  --data-dir data/train \
  --val-data-dir data/val \
  --multi-gpu \
  --gpu-ids 0 1 \
  --batch-size 256
```

## Checking GPU Availability

Before training, verify your GPUs:

```bash
# Check GPU availability
nvidia-smi

# Find available GPUs in Python
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"

# List all GPU names
python -c "
import torch
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

## Performance Considerations

### Batch Size

When using multiple GPUs, increase batch size for better utilization:

```bash
# Single GPU
--batch-size 32

# 2 GPUs (total batch size 64, 32 per GPU)
--batch-size 64 --multi-gpu

# 4 GPUs (total batch size 256, 64 per GPU)
--batch-size 256 --multi-gpu
```

### Number of Workers

Adjust `num_workers` in config based on your CPU:

```yaml
training:
  num_workers: 8  # For 4 GPUs
  # Each worker should have ~2 CPUs available
```

### Memory Management

Each GPU gets a portion of the batch:
- GPU memory usage = model size + (batch_size / num_gpus) * sample features

Monitor GPU memory:

```bash
# Watch GPU memory during training
watch -n 1 nvidia-smi
```

## Distributed Data Parallel (DDP) - Advanced

For even better performance on multiple nodes, consider using Distributed Data Parallel (DDP):

```python
# This is for future development
# DDP is more efficient for large-scale training
# Contact developers for implementation
```

## Troubleshooting

### GPUs Not Being Used

Check if CUDA is available:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory (OOM)

Reduce batch size or number of GPUs:

```bash
# Reduce batch size
--batch-size 64 --multi-gpu

# Or use fewer GPUs
--multi-gpu --gpu-ids 0 1
```

### Uneven GPU Utilization

DataParallel exhibits uneven GPU utilization due to kernel launch overhead on non-primary GPUs. This is expected behavior. If it's a bottleneck, consider DDP implementation.

### Training Speed Issues

If training is slow:
1. Check if GPUs are actually in use: `nvidia-smi`
2. Verify batch size is sufficient for GPU RAM
3. Check `num_workers` setting (should be > 0)
4. Ensure data preprocessing isn't bottlenecking

### Checkpoint Compatibility

Checkpoints saved with multi-GPU training load fine in single-GPU mode:

```bash
# Train with multi-GPU
python scripts/train.py --multi-gpu ...

# Evaluate with single GPU (automatic state dict handling)
python scripts/evaluate.py --checkpoint best_model.pt ...
```

## Monitoring Training

The framework logs:
- Number of available GPUs
- GPU names and models
- Multi-GPU status
- Per-GPU information

View logs during training:

```bash
# Real-time logs
tail -f logs/training.log

# Check GPU usage
nvidia-smi
```

## Tips for Best Results

1. **Start with batch size 2x**: Increase batch size when using multi-GPU
2. **Monitor first few batches**: Ensure GPUs are being utilized
3. **Use pin_memory=True**: Already enabled in default config
4. **Gradient synchronization**: Automatic, no manual intervention needed
5. **Checkpoint saving**: Automatically handles DataParallel unwrapping

## Comparison: Single vs Multi-GPU

| Aspect | Single GPU | Multi-GPU |
|--------|-----------|-----------|
| Scalability | Limited by GPU RAM | Can use multiple GPUs |
| Batch Size | Limited | Can increase 2-4x |
| Training Speed | Baseline | 1.5x-3.5x faster* |
| Model Code | No changes needed | No changes needed |
| Checkpoint Compatibility | N/A | Automatic handling |

*Speed improvement depends on GPU model, interconnect, and data loading efficiency.

## Configuration File Settings

No configuration changes needed! Multi-GPU is enabled via command-line arguments:

```yaml
# configs/default_config.yaml
# No GPU-specific settings required
# Just use --multi-gpu flag when running
```

## Next Steps

- Monitor training with TensorBoard: `tensorboard --logdir logs/`
- Track experiments with Weights & Biases: See [WANDB_GUIDE.md](WANDB_GUIDE.md)
- Evaluate model performance: See [README.md](../README.md)
