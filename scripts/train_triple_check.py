"""Training script for DINO with Triple-Check Loss and LoRA/DoRA adaptation."""

import argparse
import torch
from pathlib import Path

from src.models import DINOWithLoRA, LoRAConfig, DINOWithDoRA, DoRAConfig
from src.losses import TripleCheckLoss
from src.data.paired_dataset import PairedBioassayDataset, create_paired_metadata
from src.data import get_default_transforms
from src.training import TripleCheckTrainer
from src.utils import setup_logger, load_config
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DINO with LoRA/DoRA using Triple-Check Loss"
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
        "--data-dir",
        type=str,
        required=True,
        help="Path to paired bioassay data directory"
    )
    parser.add_argument(
        "--val-data-dir",
        type=str,
        help="Path to validation data directory (optional)"
    )
    parser.add_argument(
        "--create-metadata",
        action="store_true",
        help="Create metadata.json from directory structure"
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
        "--distance-metric",
        type=str,
        choices=["l2", "cosine", "kl"],
        default="l2",
        help="Distance metric for triple-check loss"
    )
    parser.add_argument(
        "--num-untreated-samples",
        type=int,
        help="Number of untreated samples to average per pair (default: 1)"
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
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.num_untreated_samples:
        config["data"]["num_untreated_samples"] = args.num_untreated_samples
    if args.device:
        config["device"] = args.device
    
    # Set default num_untreated_samples if not specified
    if "num_untreated_samples" not in config.get("data", {}):
        config.setdefault("data", {})[
            "num_untreated_samples"
        ] = 1
    
    # Setup logger
    logger = setup_logger(
        "train_triple_check",
        log_file=f"{config['logging']['log_dir']}/triple_check_training.log"
    )
    
    adaptation_method = config["adaptation"]["method"]
    
    logger.info("=" * 50)
    logger.info(f"DINO {adaptation_method.upper()} Triple-Check Training")
    logger.info("=" * 50)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Adaptation Method: {adaptation_method}")
    
    # Set random seed
    torch.manual_seed(config.get("seed", 42))
    
    # Create metadata if needed
    if args.create_metadata:
        logger.info("Creating metadata from directory structure...")
        create_paired_metadata(args.data_dir)
    
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
            num_classes=None,  # No classification head for triple-check
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
            num_classes=None,  # No classification head for triple-check
            hub_source=config["model"].get("hub_source", "github"),
            hub_source_dir=config["model"].get("hub_source_dir"),
            weights_path=config["model"].get("weights_path"),
        )
    else:
        raise ValueError(f"Unknown adaptation method: {adaptation_method}")
    
    logger.info("Model created successfully")
    
    # Create datasets
    logger.info(f"Loading paired bioassay data from: {args.data_dir}")
    transform = get_default_transforms(
        image_size=config["data"]["image_size"],
        is_train=True
    )
    
    train_dataset = PairedBioassayDataset(
        root_dir=args.data_dir,
        transform=transform,
        num_untreated_samples=config["data"].get("num_untreated_samples", 1)
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_dataloader = None
    if args.val_data_dir:
        logger.info(f"Loading validation data from: {args.val_data_dir}")
        val_transform = get_default_transforms(
            image_size=config["data"]["image_size"],
            is_train=False
        )
        val_dataset = PairedBioassayDataset(
            root_dir=args.val_data_dir,
            transform=val_transform,
            num_untreated_samples=config["data"].get("num_untreated_samples", 1)
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
    
    # Create loss function
    loss_fn = TripleCheckLoss(
        distance_metric=args.distance_metric,
        temperature=1.0,
        reduction="mean"
    )
    
    # Create trainer
    trainer = TripleCheckTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
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
