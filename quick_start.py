#!/usr/bin/env python
"""Quick start guide - Simple end-to-end example."""

import torch
from pathlib import Path
from src.models import DINOWithLoRA, LoRAConfig
from src.data import create_dataloader
from src.training import Trainer
from src.evaluation import Evaluator
from src.utils import setup_logger

# Setup
logger = setup_logger("quick_start")

# 1. Configure LoRA
logger.info("Setting up LoRA configuration...")
lora_config = LoRAConfig(
    r=8,
    lora_alpha=16.0,
    lora_dropout=0.1,
)

# 2. Create model
logger.info("Creating DINO model with LoRA...")
model = DINOWithLoRA(
    backbone_name="dino_vitb16",
    pretrained=True,
    lora_config=lora_config,
    num_classes=10,  # Change to your number of classes
)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({100*trainable_params/total_params:.2f}%)")

# 3. Create data loaders (adjust paths to your data)
# Assuming you have data in: data/train and data/val
data_exists = Path("data/train").exists() and Path("data/val").exists()

if data_exists:
    logger.info("Loading data...")
    train_dataloader = create_dataloader(
        "data/train",
        batch_size=32,
        num_workers=0,  # Set to 0 for debugging, increase for actual training
        is_train=True,
        image_size=224
    )
    
    val_dataloader = create_dataloader(
        "data/val",
        batch_size=32,
        num_workers=0,
        is_train=False,
        image_size=224
    )
    
    # 4. Train
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=5,  # Use 5 for quick test, increase for actual training
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir="checkpoints",
        log_dir="logs",
        save_interval=1,
    )
    
    history = trainer.train()
    
    # 5. Evaluate
    logger.info("Evaluating model...")
    evaluator = Evaluator(model, device="cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluator.evaluate_and_log(val_dataloader, "Validation Set")
    
else:
    logger.warning(
        "Data directories not found. Please create data/train and data/val directories "
        "and organize your images by class:\n"
        "  data/train/class_1/image1.jpg\n"
        "  data/train/class_1/image2.jpg\n"
        "  data/train/class_2/image1.jpg\n"
        "  ...\n"
        "\nThen run this script again."
    )
    logger.info("Model created and ready for training once data is provided!")
