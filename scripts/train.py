"""Training script for DINO with LoRA/DoRA adaptation."""

import argparse
import torch
import yaml
from pathlib import Path

from src.models import DINOWithLoRA, LoRAConfig, DINOWithDoRA, DoRAConfig
from src.data import create_dataloader
from src.training import Trainer
from src.utils import setup_logger, load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DINO with LoRA/DoRA adaptation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["lora", "dora"],
        help="Adaptation method (overrides config)"
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        help="Path to training data directory (overrides config)"
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        help="Path to validation data directory (overrides config)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to train on (overrides config)"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.method:
        config["adaptation"]["method"] = args.method
    if args.train_dir:
        config["data"]["train_dir"] = args.train_dir
    if args.val_dir:
        config["data"]["val_dir"] = args.val_dir
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.device:
        config["device"] = args.device
    
    # Setup logger
    logger = setup_logger(
        "train",
        log_file=f"{config['logging']['log_dir']}/training.log"
    )
    
    adaptation_method = config["adaptation"]["method"]
    
    logger.info("=" * 50)
    logger.info(f"DINO {adaptation_method.upper()} Training")
    logger.info("=" * 50)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Adaptation Method: {adaptation_method}")
    
    # Set random seed
    torch.manual_seed(config.get("seed", 42))
    
    # Create model based on adaptation method
    logger.info(f"Creating model: {config['model']['backbone']} with {adaptation_method.upper()}")
    
    if adaptation_method == "lora":
        lora_config = LoRAConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            lora_dropout=config["lora"]["lora_dropout"],
        )
        
        model = DINOWithLoRA(
            backbone_name=config["model"]["backbone"],
            lora_config=lora_config,
            num_classes=config["model"]["num_classes"],
            hub_source=config["model"].get("hub_source", "github"),
            hub_source_dir=config["model"].get("hub_source_dir"),
            weights_path=config["model"].get("weights_path"),
        )
    elif adaptation_method == "dora":
        dora_config = DoRAConfig(
            r=config["dora"]["r"],
            dora_alpha=config["dora"]["dora_alpha"],
            dora_dropout=config["dora"]["dora_dropout"],
        )
        
        model = DINOWithDoRA(
            backbone_name=config["model"]["backbone"],
            dora_config=dora_config,
            num_classes=config["model"]["num_classes"],
            hub_source=config["model"].get("hub_source", "github"),
            hub_source_dir=config["model"].get("hub_source_dir"),
            weights_path=config["model"].get("weights_path"),
        )
    else:
        raise ValueError(f"Unknown adaptation method: {adaptation_method}")
    
    logger.info("Model created successfully")
    
    # Create data loaders
    logger.info(f"Loading training data from: {config['data']['train_dir']}")
    train_dataloader = create_dataloader(
        config["data"]["train_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        is_train=True,
        image_size=config["data"]["image_size"],
    )
    
    val_dataloader = None
    if config["data"].get("val_dir"):
        logger.info(f"Loading validation data from: {config['data']['val_dir']}")
        val_dataloader = create_dataloader(
            config["data"]["val_dir"],
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
            is_train=False,
            image_size=config["data"]["image_size"],
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        num_epochs=config["training"]["num_epochs"],
        device=config["device"],
        checkpoint_dir=config["checkpoint"]["save_dir"],
        log_dir=config["logging"]["log_dir"],
        save_interval=config["checkpoint"]["save_interval"],
        wandb_config=config["logging"].get("wandb"),
    )
    
    # Train
    logger.info("Starting training...")
    history = trainer.train()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
