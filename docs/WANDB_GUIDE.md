# Weights & Biases (W&B) Integration Guide

This guide explains how to use Weights & Biases for experiment tracking and logging with the DINO LoRA framework.

## Installation

W&B is included in the requirements. Install it with:

```bash
pip install wandb
```

## Quick Start

### 1. Create a W&B Account

Visit [wandb.ai](https://wandb.ai) and create a free account.

### 2. Get Your API Key

After logging in:
1. Go to Settings → API Keys
2. Copy your API key

### 3. Authenticate

Run the following command and paste your API key when prompted:

```bash
wandb login
```

### 4. Configure Your Project

Edit `configs/default_config.yaml`:

```yaml
logging:
  log_dir: "logs"
  tensorboard: true
  wandb:
    enabled: true
    project: "dino-lora"          # Your project name
    entity: "your-username"       # Your W&B username
    name: null                    # Auto-generated if null
    tags: ["dino", "lora", "experiment-1"]
    notes: "Training DINO with LoRA on custom dataset"
```

### 5. Train and Monitor

```bash
python scripts/train.py --config configs/default_config.yaml
```

The training will automatically log to W&B. Monitor your experiment at:
https://wandb.ai/your-username/dino-lora

## Configuration Options

### `enabled` (boolean)
- Enable/disable W&B logging
- Default: `true`

### `project` (string)
- W&B project name where runs will be logged
- Default: `"dino-lora"`

### `entity` (string or null)
- Your W&B username or team name
- If `null`, uses default from `wandb login`
- Example: `"your-username"` or `"your-team"`

### `name` (string or null)
- Custom name for this run
- If `null`, W&B generates a name automatically
- Example: `"experiment-001"`, `"lora-r8-lr1e3"`

### `tags` (list)
- Tags to organize and filter experiments
- Example: `["dino", "lora", "baseline"]`

### `notes` (string)
- Additional notes about the experiment
- Example: `"Testing different LoRA ranks"`

## Logged Metrics

The framework automatically logs:

### Training Metrics
- `training/loss` - Batch-level loss
- `training/epoch_loss` - Epoch-level training loss
- `training/epoch_accuracy` - Epoch-level training accuracy
- `training/batch` - Current batch number
- `training/global_step` - Total optimization steps

### Validation Metrics
- `validation/loss` - Validation loss
- `validation/accuracy` - Validation accuracy

### Hyperparameters
- `learning_rate`
- `weight_decay`
- `num_epochs`
- `batch_size`

## Example Configurations

### Baseline Experiment
```yaml
wandb:
  enabled: true
  project: "dino-lora"
  entity: "myusername"
  name: "baseline-vit-b16"
  tags: ["baseline", "vitb16"]
  notes: "Baseline DINO-LoRA with default hyperparameters"
```

### Hyperparameter Search
```yaml
wandb:
  enabled: true
  project: "dino-lora"
  entity: "myusername"
  name: "lora-rank-search"
  tags: ["hyperparameter-search", "lora-rank"]
  notes: "Testing LoRA rank: 4, 8, 16, 32"
```

### Quick Testing
```yaml
wandb:
  enabled: false  # Disable for quick tests
```

## Viewing Results

### On wandb.ai
1. Go to your project page
2. Compare runs side-by-side
3. View training curves, metrics, and logs
4. Create custom charts and reports

### Features
- **Parallel Coordinates Plot**: Compare multiple hyperparameters
- **Scatter Plot**: Visualize relationship between metrics
- **System Metrics**: CPU, GPU, memory usage
- **Model Checkpoints**: Log and compare model versions

## Offline Mode

If you don't have internet connectivity, run in offline mode:

```bash
WANDB_MODE=offline python scripts/train.py --config configs/default_config.yaml
```

Runs will be synced when you reconnect to the internet:

```bash
wandb sync ./wandb/offline-run-id
```

## Disabling W&B

To disable W&B logging:

```yaml
wandb:
  enabled: false
```

Or via command line (if supported):

```bash
DISABLED_WANDB=true python scripts/train.py
```

## Troubleshooting

### "wandb not installed"
```bash
pip install wandb
```

### "Failed to initialize W&B"
- Check internet connection
- Verify API key with `wandb login`
- Check W&B status at [status.wandb.com](https://status.wandb.com)

### No runs appearing on wandb.ai
- Verify `entity` and `project` are correct
- Check W&B login with `wandb login`
- Check console output for W&B initialization messages

## Advanced Usage

### Log Model Checkpoints

To automatically save your best model checkpoint to W&B, modify the trainer:

```python
if is_best:
    wandb.save(str(best_path))
```

### Log Predictions

Log sample predictions during validation:

```python
if self.wandb_enabled:
    for images, labels in val_dataloader:
        predictions = model(images)
        wandb.log({"predictions": wandb.Table(...)})
        break
```

### Custom Metrics

Log custom metrics during training:

```python
if self.wandb_enabled:
    wandb.log({
        "custom_metric": value,
        "epoch": epoch,
    })
```

## References

- [W&B Documentation](https://docs.wandb.ai/)
- [W&B Quick Start](https://docs.wandb.ai/quickstart)
- [Logging Best Practices](https://docs.wandb.ai/guides/track)
- [W&B API Reference](https://docs.wandb.ai/ref/python)
