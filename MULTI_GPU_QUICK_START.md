# Multi-GPU Training - Quick Reference

## Common Commands

### Check GPU Availability
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
```

### Use All Available GPUs
```bash
python scripts/train.py \
  --config configs/default_config.yaml \
  --data-dir data/train \
  --val-data-dir data/val \
  --multi-gpu
```

### Use Specific GPUs (e.g., 0 and 1)
```bash
python scripts/train.py \
  --config configs/default_config.yaml \
  --data-dir data/train \
  --val-data-dir data/val \
  --multi-gpu \
  --gpu-ids 0 1
```

### Use Specific GPUs with Larger Batch Size
```bash
python scripts/train.py \
  --config configs/default_config.yaml \
  --data-dir data/train \
  --val-data-dir data/val \
  --multi-gpu \
  --gpu-ids 0 1 2 3 \
  --batch-size 512
```

### Train with DoRA on Multiple GPUs
```bash
python scripts/train.py \
  --config configs/default_config.yaml \
  --method dora \
  --data-dir data/train \
  --val-data-dir data/val \
  --multi-gpu
```

### Monitor Training in Another Terminal
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# View logs
tail -f logs/training.log

# TensorBoard visualization
tensorboard --logdir logs/
```

## Recommended Settings by # of GPUs

### 2 GPUs
```bash
python scripts/train.py \
  --config configs/default_config.yaml \
  --data-dir data/train \
  --val-data-dir data/val \
  --multi-gpu \
  --batch-size 128
```

### 4 GPUs
```bash
python scripts/train.py \
  --config configs/default_config.yaml \
  --data-dir data/train \
  --val-data-dir data/val \
  --multi-gpu \
  --batch-size 256
```

### 8 GPUs
```bash
python scripts/train.py \
  --config configs/default_config.yaml \
  --data-dir data/train \
  --val-data-dir data/val \
  --multi-gpu \
  --batch-size 512
```

## Environment Variables

```bash
# Show which GPUs to use (alternative to --gpu-ids)
export CUDA_VISIBLE_DEVICES=0,1

# Control memory allocation
export CUDA_LAUNCH_BLOCKING=1  # For debugging (slower)

# Combined example
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py \
  --config configs/default_config.yaml \
  --data-dir data/train \
  --multi-gpu
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `--batch-size` |
| GPUs not utilized | Check `nvidia-smi`, verify batch size is set |
| Slow training | Increase `--batch-size`, check `num_workers` in config |
| CUDA not found | Ensure CUDA toolkit is installed |
| GPU out of memory but small batch | Try fewer GPUs with `--gpu-ids` |

## File References

- **Full Guide**: [MULTI_GPU_GUIDE.md](MULTI_GPU_GUIDE.md)
- **Trainer Code**: [src/training/trainer.py](../src/training/trainer.py)
- **Training Script**: [scripts/train.py](../scripts/train.py)
- **Main README**: [README.md](../README.md)
