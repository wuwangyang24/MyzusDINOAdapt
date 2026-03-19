# DINO LoRA/DoRA Adaptation Framework

A comprehensive framework for adapting DINO (Dense Interaction Neural Net Objects) models using Low-Rank Adaptation (LoRA) and Dimensional and Rank Adaptation (DoRA) for efficient fine-tuning on custom datasets.

## Features

- **LoRA & DoRA Adaptation**: Efficient parameter-efficient fine-tuning using low-rank decomposition and dimensional scaling
- **DINO Integration**: Built-in support for DINO vision transformers (ViT-B/16, ViT-S/14, ViT-L/14)
- **Complete Pipeline**: End-to-end training, evaluation, and feature extraction
- **Flexible Configuration**: YAML-based configuration for easy experimentation
- **Logging & Monitoring**: TensorBoard integration for training visualization
- **Modular Design**: Clean separation of concerns with models, data, training, and evaluation modules

## Project Structure

```
.
├── src/
│   ├── models/              # Model definitions and adaptation implementations
│   │   ├── lora.py         # Core LoRA modules
│   │   ├── dino_lora.py    # DINO with LoRA wrapper
│   │   ├── dora.py         # Core DoRA modules
│   │   └── dino_dora.py    # DINO with DoRA wrapper
│   ├── data/               # Data loading and preprocessing
│   │   ├── dataset.py      # Dataset class
│   │   └── dataloader.py   # DataLoader utilities
│   ├── training/           # Training utilities
│   │   └── trainer.py      # Trainer class
│   ├── evaluation/         # Evaluation metrics
│   │   └── evaluator.py    # Evaluator class
│   └── utils/              # Utility functions
│       ├── config_utils.py # Configuration loading
│       └── logger_utils.py # Logging setup
├── configs/                # Configuration files
│   └── default_config.yaml # Default training configuration
├── scripts/                # Training and evaluation scripts
│   ├── train.py           # Training script (supports both LoRA and DoRA)
│   └── evaluate.py        # Evaluation script
├── notebooks/             # Jupyter notebooks for experiments
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd MyzusDINOAdapt
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Organize your dataset in the following structure:
```
data/
├── train/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class_2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── class_1/
    ├── class_2/
    └── ...
```

### 2. Configure Training

Edit `configs/default_config.yaml` to set:
- Model backbone (dino_vitb16, dino_vits14, etc.)
- Number of classes
- LoRA hyperparameters
- Training parameters

Example:
```yaml
model:
  backbone: "dino_vitb16"
  num_classes: 10

lora:
  r: 8
  lora_alpha: 16.0
  lora_dropout: 0.1

training:
  num_epochs: 10
  batch_size: 32
  learning_rate: 1.0e-3
```

### 3. Train the Model

**Using LoRA (default):**
```bash
python MyzusDINOAdapt/scripts/train.py --config MyzusDINOAdapt/configs/default_config.yaml --train-dir Data/Myzus --num-epochs 20 --batch-size 1 --warmup_epochs 2
```

**Using DoRA:**
```bash
python scripts/train.py \
  --config configs/default_config.yaml \
  --method dora \
  --train-dir data/train \
  --val-dir data/val \
  --num-epochs 20 \
  --batch-size 32
```

Or modify the config file to set `adaptation.method: dora`.

### 4. Evaluate the Model

```bash
# For LoRA model
python scripts/evaluate.py \
  --config configs/default_config.yaml \
  --checkpoint checkpoints/best_model.pt \
  --data-dir data/val \
  --batch-size 32

# For DoRA model
python scripts/evaluate.py \
  --config configs/default_config.yaml \
  --method dora \
  --checkpoint checkpoints/best_model.pt \
  --data-dir data/val \
  --batch-size 32
```

## Key Components

### LoRA Implementation

The framework includes a complete LoRA implementation in `src/models/lora.py`:

- **LoRALinear**: Low-rank adapted linear layer that replaces standard linear operations
- **LoRAConfig**: Configuration class for LoRA hyperparameters

### DoRA Implementation

The framework also includes DoRA (Dimensional and Rank Adaptation) in `src/models/dora.py`:

- **DoRALinear**: Dimensional and rank adapted linear layer combining low-rank decomposition with dimensional scaling
- **DoRAConfig**: Configuration class for DoRA hyperparameters

DoRA extends LoRA by adding learnable magnitude scaling vectors for each output dimension, which can improve adaptation performance.

### DINO with LoRA/DoRA

- `src/models/dino_lora.py` provides `DINOWithLoRA` class
- `src/models/dino_dora.py` provides `DINOWithDoRA` class

Both classes:
- Load pretrained DINO backbones
- Apply the respective adaptation method to attention and MLP layers
- Support adding classification heads for downstream tasks
- Provide trainable parameter utilities

### Training

The `Trainer` class in `src/training/trainer.py` handles:
- Forward/backward passes
- Gradient updates with AdamW optimizer
- Epoch-based training loops
- Validation and checkpointing
- TensorBoard logging

### Evaluation

The `Evaluator` class in `src/evaluation/evaluator.py` provides:
- Accuracy, precision, recall, F1 metrics
- Feature extraction from backbone
- Comprehensive evaluation logging

## Configuration

### Model Configuration
- `backbone`: DINO model type (dino_vitb16, dino_vits14, dino_vitl14)
- `num_classes`: Number of output classes
- `hub_source`: Source for loading DINO model ("github" for remote, "local" for local directory)
- `hub_source_dir`: Path to local DINO torch.hub directory (required when `hub_source` is "local")
- `weights_path`: Path to pretrained weights file for custom model initialization (optional)

### LoRA Configuration
- `r`: Rank of low-rank decomposition (typically 4-64)
- `lora_alpha`: Scaling factor for LoRA outputs
- `lora_dropout`: Dropout rate in LoRA layers

### Training Configuration
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate for optimizer
- `weight_decay`: L2 regularization coefficient
- `num_workers`: Parallel data loading workers

## Advanced Usage

### Custom Dataset

```python
from src.data import create_dataloader

# Use custom dataset
dataloader = create_dataloader(
    data_dir="path/to/data",
    batch_size=32,
    is_train=True,
    image_size=224
)
```

### Feature Extraction

```python
from src.evaluation import Evaluator
from src.models import DINOWithLoRA

model = DINOWithLoRA(backbone_name="dino_vitb16")
evaluator = Evaluator(model, device="cuda")

# Extract features
features, labels = evaluator.get_features(dataloader)
```

### Loading Models from Local Torch.Hub

For offline environments or to use a local copy of DINO, you can load models from a local directory:

**Using direct instantiation:**

```python
from src.models import DINOWithLoRA, LoRAConfig

model = DINOWithLoRA(
    backbone_name="dino_vitb16",
    lora_config=LoRAConfig(),
    hub_source="local",
    hub_source_dir="/path/to/local/dino/hub",  # Directory containing torch.hub hubconf.py
    num_classes=10
)
```

**Using configuration file:**

```yaml
model:
  backbone: "dino_vitb16"
  num_classes: 10
  hub_source: "local"
  hub_source_dir: "/path/to/local/dino/hub"
```

Then train with:
```bash
python scripts/train.py --config configs/default_config.yaml
```

The local hub directory should have the same structure as the `facebookresearch/dino` repository with a `hubconf.py` file containing model definitions.

### Loading Custom Pretrained Weights

When using a local torch.hub source, you can specify a path to custom pretrained weights:

**Using direct instantiation:**

```python
from src.models import DINOWithLoRA, LoRAConfig

model = DINOWithLoRA(
    backbone_name="dino_vitb16",
    lora_config=LoRAConfig(),
    hub_source="local",
    hub_source_dir="/path/to/local/dino/hub",
    weights_path="/path/to/custom/weights.pth",  # Path to pretrained weights file
    num_classes=10
)
```

**Using configuration file:**

```yaml
model:
  backbone: "dino_vitb16"
  num_classes: 10
  hub_source: "local"
  hub_source_dir: "/path/to/local/dino/hub"
  weights_path: "/path/to/custom/weights.pth"
```

This enables fine-tuning from custom pretrained models in offline or air-gapped environments.

## Understanding LoRA

LoRA (Low-Rank Adaptation) is an efficient fine-tuning technique that:

1. **Freezes** the original model weights
2. **Injects** trainable low-rank matrices: W' = W₀ + ΔW = W₀ + BA
3. **Trains** only A and B matrices, drastically reducing parameters

**Advantages**:
- Reduces trainable parameters by 100-1000x
- Maintains model performance
- Enables efficient multi-task adaptation
- Reduces memory requirements

**Visualization**:
```
Original: W₀ (d_in × d_out)
LoRA:     ΔW = B × A where B: (d_out × r), A: (r × d_in)
Total:    Forward pass = W₀ × x + α/r × B × A × x
```

## Tips for Best Results

1. **Hyperparameter Tuning**:
   - LoRA rank (r): Start with 8, try 4, 16, 32
   - Learning rate: Often lower than full fine-tuning (1e-4 to 1e-3)
   - Batch size: Larger batches usually help (32-128)

2. **Data Preparation**:
   - Resize images consistently
   - Normalize using ImageNet statistics
   - Use data augmentation for better generalization

3. **Training**:
   - Monitor validation loss for early stopping
   - Use warmup for initial epochs
   - Save best model based on validation metrics

4. **Model Selection**:
   - ViT-S/14: Smaller, faster, for limited resources
   - ViT-B/16: Good balance of speed and accuracy
   - ViT-L/14: Larger, better performance, more memory

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Use gradient accumulation
- Try LoRA with smaller rank

### Poor Performance
- Increase learning rate slightly
- Use more training data
- Try larger LoRA rank
- Freeze fewer backbone layers

### Slow Training
- Increase batch size
- Use DistributedDataParallel for multi-GPU training
- Reduce image size (be careful with this)

## Citation

If you use this framework, please cite:

```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phil and Allen-Zhu, Zeyuan and Li, Yuanzhi},
  journal={arXiv preprint arXiv:2106.09714},
  year={2021}
}

@article{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Vedaldi, Albedo and Wolf, Christoph},
  journal={arXiv preprint arXiv:2104.14294},
  year={2021}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09714)
- [DINO Paper](https://arxiv.org/abs/2104.14294)
- [DINO GitHub](https://github.com/facebookresearch/dino)
- [PyTorch Documentation](https://pytorch.org/docs/)
