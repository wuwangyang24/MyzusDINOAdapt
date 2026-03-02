# Triple-Check Loss for DINO LoRA

This guide explains how to use the Triple-Check Loss with the DINO LoRA framework for learning consistent treatment effects across multiple bioassays.

## Overview

The Triple-Check Loss is a novel approach to ensure that the learned representations capture consistent treatment effects across different bioassays. It works by:

1. **Extract Delta 1**: Computing the feature difference between treated and untreated aphids from Bioassay 1
   $$\Delta_1 = z_{T1} - z_{U1}$$

2. **Extract Delta 2**: Computing the feature difference between treated and untreated aphids from Bioassay 2
   $$\Delta_2 = z_{T2} - z_{U2}$$

3. **Minimize Loss**: Ensuring the two deltas are similar
   $$\mathcal{L} = d(\Delta_1, \Delta_2)$$

This approach forces the model to learn representations where the treatment effect is consistent across different bioassays, improving generalization and robustness.

## Data Structure

Triple-Check training requires paired bioassay samples with the following directory structure:

```
paired_bioassay_data/
├── metadata.json
├── bioassay_1/
│   ├── treated/
│   │   ├── sample_1.jpg
│   │   ├── sample_2.jpg
│   │   └── ...
│   └── untreated/
│       ├── sample_1.jpg
│       ├── sample_2.jpg
│       └── ...
└── bioassay_2/
    ├── treated/
    │   ├── sample_1.jpg
    │   ├── sample_2.jpg
    │   └── ...
    └── untreated/
        ├── sample_1.jpg
        ├── sample_2.jpg
        └── ...
```

### Creating metadata.json

The framework can automatically create `metadata.json` by pairing samples with matching filenames:

**Option 1: Automatic Generation**
```bash
python scripts/check.py \
  --data-dir paired_bioassay_data \
  --create-metadata \
  --num-epochs 20
```

**Option 2: Manual Creation**

Use Python to create the metadata:
```python
from src.data import create_paired_metadata

create_paired_metadata(
    root_dir="paired_bioassay_data",
    bioassay_1_dir="bioassay_1",
    bioassay_2_dir="bioassay_2",
    treated_dir="treated",
    untreated_dir="untreated",
    output_file="metadata.json"
)
```

This creates a `metadata.json` containing pairs like:

```json
{
  "pairs": [
    {
      "id": 0,
      "bioassay_1_treated": "bioassay_1/treated/sample_1.jpg",
      "bioassay_1_untreated": "bioassay_1/untreated/sample_1.jpg",
      "bioassay_2_treated": "bioassay_2/treated/sample_1.jpg",
      "bioassay_2_untreated": "bioassay_2/untreated/sample_1.jpg"
    },
    ...
  ]
}
```

## Quick Start

### 1. Prepare Your Data

Organize your bioassay images following the structure above. Ensure treated and untreated samples from different bioassays have matching filenames.

### 2. Train Model

```bash
python scripts/check.py \
  --data-dir paired_bioassay_data \
  --val-data-dir paired_bioassay_data_val \
  --num-epochs 50 \
  --batch-size 32 \
  --distance-metric l2
```

### 3. Monitor Training

Training metrics are logged to TensorBoard and W&B (if enabled):

```bash
tensorboard --logdir logs
```

## Configuration

### Command-line Arguments

```
--config                Path to YAML config file
--data-dir             Path to paired bioassay training data
--val-data-dir         Path to paired bioassay validation data (optional)
--create-metadata      Automatically create metadata.json
--num-epochs           Number of training epochs
--batch-size           Batch size
--learning-rate        Learning rate
--distance-metric      Distance metric: l2, cosine, or kl
--num-untreated-samples  Number of untreated samples to average per pair (default: 1)
--device               Device: cuda or cpu
```

### Distance Metrics

The framework supports three distance metrics for the Triple-Check loss:

#### L2 Distance (Default)
```
Loss = ||Δ₁ - Δ₂||₂
```
- **Pros**: Intuitive, measures Euclidean distance
- **Cons**: Scale-sensitive

**Use when**: You want direct feature space alignment

#### Cosine Distance
```
Loss = 1 - cosine_similarity(Δ₁, Δ₂)
```
- **Pros**: Scale-invariant, focuses on direction
- **Cons**: Ignores magnitude

**Use when**: You care about the direction of treatment effect, not magnitude

#### KL Divergence
```
Loss = Σ p(i) log(p(i)/q(i))
```
Where p and q are softmax-normalized deltas.

- **Pros**: Probabilistic interpretation
- **Cons**: Requires careful temperature tuning

**Use when**: You want probabilistic features

## Loss Functions

### TripleCheckLoss (Basic)

Minimizes the distance between Δ₁ and Δ₂:

```python
from src.losses import TripleCheckLoss

loss_fn = TripleCheckLoss(
    distance_metric="l2",  # l2, cosine, or kl
    temperature=1.0,
    reduction="mean"  # mean, sum, or none
)
```

### TripleCheckWithContrastiveLoss (Advanced)

Combines triple-check consistency with contrastive learning to ensure treated and untreated samples are well-separated:

```python
from src.losses import TripleCheckWithContrastiveLoss

loss_fn = TripleCheckWithContrastiveLoss(
    distance_metric="l2",
    temperature=1.0,
    contrastive_weight=0.5,  # Weight for contrastive term
    margin=1.0,  # Margin for contrastive loss
    reduction="mean"
)
```

This is useful when you want to enforce both:
1. **Consistency**: Treat Effect is similar across bioassays
2. **Separation**: Treated and untreated samples are distinguishable

## Training Examples

### Basic Training with L2 Distance

```bash
python scripts/check.py \
  --data-dir data/paired_bioassay \
  --distance-metric l2 \
  --num-epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001
```

### With Validation Set and W&B Logging

```bash
python scripts/check.py \
  --config configs/default_config.yaml \
  --data-dir data/paired_bioassay_train \
  --val-data-dir data/paired_bioassay_val \
  --distance-metric cosine \
  --num-epochs 100 \
  --batch-size 64
```

### With Custom Metadata

```python
from src.models import DINOWithLoRA, LoRAConfig
from src.losses import TripleCheckWithContrastiveLoss
from src.data import PairedBioassayDataset, get_default_transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.training import TripleCheckModule
from torch.utils.data import DataLoader

# Setup model and loss
model = DINOWithLoRA(
    backbone_name="dino_vitb16",
    lora_config=LoRAConfig(r=8),
    num_classes=None
)

loss_fn = TripleCheckWithContrastiveLoss(
    distance_metric="l2",
    contrastive_weight=0.3,
    margin=1.5
)

# Create datasets
transform = get_default_transforms(image_size=224, is_train=True)
dataset = PairedBioassayDataset(
    root_dir="data/paired_bioassay",
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create Lightning module
module = TripleCheckModule(
    model=model,
    loss_fn=loss_fn,
    learning_rate=1e-3,
    weight_decay=1e-4,
)

# Train
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="gpu",
    devices=1,
    callbacks=[ModelCheckpoint(dirpath="checkpoints", monitor="val/loss", save_top_k=1)],
)
trainer.fit(module, train_dataloaders=dataloader)
```

## Feature Analysis

After training, you can analyze the learned features:

```python
import torch
from src.models import DINOWithLoRA

# Load model
model = DINOWithLoRA(...)
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Extract features for analysis
with torch.no_grad():
    feat_treated = model.backbone(treated_images)
    feat_untreated = model.backbone(untreated_images)
    
    delta = feat_treated - feat_untreated
    
    # Compute statistics
    delta_mean = delta.mean(dim=0)
    delta_std = delta.std(dim=0)
    delta_norm = torch.norm(delta, dim=1)
    
    print(f"Mean delta magnitude: {delta_norm.mean().item():.4f}")
    print(f"Delta covariance: {torch.cov(delta.T)}")
```

## Interpretation and Evaluation

### Metrics to Monitor

1. **Training Loss**: Should decrease over time
2. **Validation Loss**: Should decrease and plateau
3. **Delta Magnitude**: The norm of the difference vector
4. **Feature Variance**: How consistent the treatment effect is

### Understanding the Results

- **Low Loss**: Δ₁ ≈ Δ₂ (treatment effect consistent across bioassays)
- **High Loss**: Δ₁ ≠ Δ₂ (treatment effect differs across bioassays)

### Biological Interpretation

If the model achieves low triple-check loss, it means:
- The learned representations capture a consistent treatment effect
- The chemical compound has a stable, predictable effect
- The model should generalize well to new bioassays

## Tips and Best Practices

### Data Preparation

1. **Pairing Strategy**: 
   - Ensure treated and untreated samples are from the same aphid population
   - Use matching filenames for paired samples
   - Verify image quality is consistent

2. **Data Balance**:
   - Keep equal numbers of treated/untreated per bioassay
   - Balance sample sizes across bioassays

### Hyperparameter Tuning

1. **Learning Rate**: Start with 1e-3, adjust if loss doesn't decrease
2. **Batch Size**: Larger batches (32-128) often work better
3. **Distance Metric**: 
   - L2 for general purpose
   - Cosine for direction-focused tasks
   - KL for probabilistic features
4. **Contrastive Weight**: 0.1-0.5 usually works well

### Training Tips

1. **Warm Up**: Consider learning rate warmup for first few epochs
2. **Checkpointing**: Save best model based on validation loss
3. **Monitoring**: Use TensorBoard to monitor loss curves
4. **Early Stopping**: Stop if validation loss plateaus

## Troubleshooting

### Loss Not Decreasing

- Check data loading (verify images are being read correctly)
- Try higher learning rate (1e-2 to 1e-4)
- Ensure paired samples are actually paired correctly
- Try different distance metric

### High Variance in Loss

- Increase batch size
- Use gradient accumulation
- Try cosine distance instead of L2

### Memory Issues

- Reduce batch size
- Use smaller image size (but not below 224)
- Use smaller backbone (dino_vits14)

### Model Not Learning Consistent Effects

- More training epochs
- More paired bioassay data
- Try contrastive loss variant
- Adjust distance metric

## References

- [DINO Paper](https://arxiv.org/abs/2104.14294)
- [LoRA Paper](https://arxiv.org/abs/2106.09714)
- [Metric Learning Review](https://arxiv.org/abs/2103.14025)
