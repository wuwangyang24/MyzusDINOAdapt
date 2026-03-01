# DINO LoRA Adaptation - Example Notebook

This notebook demonstrates how to use the DINO LoRA adaptation framework.

## Setup

```python
import torch
from src.models import DINOWithLoRA, LoRAConfig
from src.data import create_dataloader
from src.training import Trainer
from src.evaluation import Evaluator
from src.utils import load_config
```

## 1. Create Model with LoRA

```python
# Configure LoRA
lora_config = LoRAConfig(
    r=8,
    lora_alpha=16.0,
    lora_dropout=0.1,
)

# Create DINO with LoRA
model = DINOWithLoRA(
    backbone_name="dino_vitb16",
    pretrained=True,
    lora_config=lora_config,
    num_classes=10  # Your number of classes
)

print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
```

## 2. Load Data

```python
# Create data loaders
train_dataloader = create_dataloader(
    "data/train",
    batch_size=32,
    num_workers=4,
    is_train=True,
    image_size=224
)

val_dataloader = create_dataloader(
    "data/val",
    batch_size=32,
    num_workers=4,
    is_train=False,
    image_size=224
)
```

## 3. Train Model

```python
# Create trainer
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    learning_rate=1e-3,
    weight_decay=1e-4,
    num_epochs=10,
    device="cuda",
    checkpoint_dir="checkpoints",
    log_dir="logs"
)

# Train
history = trainer.train()
```

## 4. Evaluate Model

```python
# Create evaluator
evaluator = Evaluator(model, device="cuda")

# Evaluate on validation set
metrics = evaluator.evaluate_and_log(val_dataloader, "Validation Set")

# Extract features
features, labels = evaluator.get_features(val_dataloader)
print(f"Extracted features shape: {features.shape}")
```

## 5. Load Pre-trained Checkpoint

```python
# Load checkpoint
checkpoint = torch.load("checkpoints/best_model.pt", map_location="cuda")
model.load_state_dict(checkpoint["model_state_dict"])

# Evaluate loaded model
metrics = evaluator.evaluate_and_log(val_dataloader, "Test Set")
```
