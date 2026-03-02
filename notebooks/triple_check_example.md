# Triple-Check Loss Example Notebook

This notebook demonstrates how to use the Triple-Check Loss with DINO LoRA.

## Setup

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import DINOWithLoRA, LoRAConfig
from src.losses import TripleCheckLoss, TripleCheckWithContrastiveLoss
from src.data.paired_dataset import PairedBioassayDataset, create_paired_metadata
from src.data import get_default_transforms
from src.training.triple_check_trainer import TripleCheckTrainer
from src.utils import setup_logger
```

## 1. Prepare Data

### Option A: Auto-generate metadata.json with Multiple Untreated Samples

```python
from src.data import create_paired_metadata

# Create metadata from directory structure
# num_untreated_samples=3 means each treated sample will be paired with 3 untreated samples
# These will be averaged after passing through the model
create_paired_metadata(
    root_dir="path/to/paired_bioassay_data",
    bioassay_1_dir="bioassay_1",
    bioassay_2_dir="bioassay_2",
    treated_dir="treated",
    untreated_dir="untreated",
    num_untreated_samples=3  # Default: 1
)
```

### Option B: Load existing metadata.json

```python
# Verify directory structure
import os
data_dir = "path/to/paired_bioassay_data"
print(os.listdir(data_dir))
# Should contain: metadata.json, bioassay_1/, bioassay_2/
```

## 2. Create Model

```python
# Configure LoRA
lora_config = LoRAConfig(
    r=8,
    lora_alpha=16.0,
    lora_dropout=0.1,
)

# Create DINO with LoRA (no classification head for triple-check)
model = DINOWithLoRA(
    backbone_name="dino_vitb16",
    lora_config=lora_config,
    num_classes=None  # Important: no classification head
)

print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
```

## 3. Load Paired Bioassay Data

```python
# Create datasets
# num_untreated_samples must match the value used in create_paired_metadata
transform = get_default_transforms(image_size=224, is_train=True)

train_dataset = PairedBioassayDataset(
    root_dir="path/to/paired_bioassay_data/train",
    transform=transform,
    num_untreated_samples=3  # Match the value from metadata creation
)

val_dataset = PairedBioassayDataset(
    root_dir="path/to/paired_bioassay_data/val",
    transform=get_default_transforms(image_size=224, is_train=False),
    num_untreated_samples=3  # Match the value from metadata creation
)

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True,
    pin_memory=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=32,
    num_workers=4,
    shuffle=False,
    pin_memory=True
)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Note: Untreated samples will be {3}x3 = 9 images averaged before each forward pass")
```

## 4. Choose Loss Function

### Option A: Basic Triple-Check Loss (L2)

```python
loss_fn = TripleCheckLoss(
    distance_metric="l2",
    temperature=1.0,
    reduction="mean"
)
```

### Option B: Triple-Check with Cosine Distance

```python
loss_fn = TripleCheckLoss(
    distance_metric="cosine",
    temperature=1.0,
    reduction="mean"
)
```

### Option C: Triple-Check with Contrastive Learning

```python
loss_fn = TripleCheckWithContrastiveLoss(
    distance_metric="l2",
    temperature=1.0,
    contrastive_weight=0.5,  # Balance consistency vs separation
    margin=1.0,  # Minimum separation between treated/untreated
    reduction="mean"
)
```

## 5. Create Trainer

```python
trainer = TripleCheckTrainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    loss_fn=loss_fn,
    learning_rate=1e-3,
    weight_decay=1e-4,
    num_epochs=50,
    device="cuda",
    checkpoint_dir="checkpoints",
    log_dir="logs",
    save_interval=5,
    wandb_config={
        "enabled": True,
        "project": "dino-lora-triple-check",
        "entity": "your-username",
        "tags": ["triple-check", "bioassay"]
    }
)
```

## 6. Train Model

```python
history = trainer.train()
```

## 7. Analyze Results

### Load Best Model

```python
model = DINOWithLoRA(...)
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

### Extract and Analyze Features

```python
import torch

# Get a batch of data
img_t1, img_u1, img_t2, img_u2 = next(iter(val_dataloader))
img_t1, img_u1, img_t2, img_u2 = (
    img_t1.to("cuda"),
    img_u1.to("cuda"),
    img_t2.to("cuda"),
    img_u2.to("cuda")
)

# Extract features
with torch.no_grad():
    feat_t1 = model.backbone(img_t1)
    feat_u1 = model.backbone(img_u1)
    feat_t2 = model.backbone(img_t2)
    feat_u2 = model.backbone(img_u2)

# Compute deltas
delta_1 = feat_t1 - feat_u1
delta_2 = feat_t2 - feat_u2

# Analyze consistency
delta_diff = torch.norm(delta_1 - delta_2, dim=1)
print(f"Delta difference mean: {delta_diff.mean().item():.6f}")
print(f"Delta difference std:  {delta_diff.std().item():.6f}")

# Analyze magnitude
delta_1_norm = torch.norm(delta_1, dim=1)
delta_2_norm = torch.norm(delta_2, dim=1)
print(f"Δ₁ norm mean: {delta_1_norm.mean().item():.6f}")
print(f"Δ₂ norm mean: {delta_2_norm.mean().item():.6f}")
```

### Visualize Features

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce features to 2D for visualization
pca = PCA(n_components=2)
feat_2d = pca.fit_transform(feat_t1.cpu().numpy())

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(feat_2d[:, 0], feat_2d[:, 1], alpha=0.6)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
plt.title("Treated Aphid Features (2D PCA)")
plt.colorbar(label="Class")
plt.show()
```

## 8. Evaluate Consistency

```python
def evaluate_consistency(model, dataloader, device="cuda"):
    """Evaluate how consistent the treatment effect is across bioassays."""
    model.eval()
    
    all_losses = []
    
    with torch.no_grad():
        for img_t1, img_u1, img_t2, img_u2 in dataloader:
            img_t1, img_u1, img_t2, img_u2 = (
                img_t1.to(device),
                img_u1.to(device),
                img_t2.to(device),
                img_u2.to(device)
            )
            
            # Get features
            feat_t1 = model.backbone(img_t1)
            feat_u1 = model.backbone(img_u1)
            feat_t2 = model.backbone(img_t2)
            feat_u2 = model.backbone(img_u2)
            
            # Compute consistency
            delta_1 = feat_t1 - feat_u1
            delta_2 = feat_t2 - feat_u2
            loss = torch.norm(delta_1 - delta_2, dim=1)
            
            all_losses.append(loss.cpu().numpy())
    
    import numpy as np
    all_losses = np.concatenate(all_losses)
    
    print(f"Consistency Metrics:")
    print(f"  Mean Loss:    {all_losses.mean():.6f}")
    print(f"  Std Loss:     {all_losses.std():.6f}")
    print(f"  Min Loss:     {all_losses.min():.6f}")
    print(f"  Max Loss:     {all_losses.max():.6f}")
    
    return all_losses

consistency_losses = evaluate_consistency(model, val_dataloader)
```

## 9. Export and Use

```python
# Save model with metadata
torch.save({
    'model_state_dict': model.state_dict(),
    'backbone_name': 'dino_vitb16',
    'lora_config': {
        'r': 8,
        'lora_alpha': 16.0,
        'lora_dropout': 0.1,
    }
}, 'dino_triple_check_final.pt')

# Later, reload
model = DINOWithLoRA(backbone_name='dino_vitb16', ...)
checkpoint = torch.load('dino_triple_check_final.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 10. Batch Inference

```python
# Predict on new bioassay images
def predict_treatment_effect(model, treated_img, untreated_img, device="cuda"):
    """
    Predict the treatment effect (feature difference) for a pair of images.
    """
    model.eval()
    
    with torch.no_grad():
        feat_t = model.backbone(treated_img.unsqueeze(0).to(device))
        feat_u = model.backbone(untreated_img.unsqueeze(0).to(device))
        
        delta = feat_t - feat_u
        
    return delta.squeeze(0).cpu().numpy()

# Usage
from torchvision import transforms
from PIL import Image

transform = get_default_transforms(image_size=224, is_train=False)
treated_img = transform(Image.open("treated.jpg"))
untreated_img = transform(Image.open("untreated.jpg"))

effect = predict_treatment_effect(model, treated_img, untreated_img)
print(f"Treatment effect vector shape: {effect.shape}")
print(f"Treatment effect magnitude: {np.linalg.norm(effect):.6f}")
```
