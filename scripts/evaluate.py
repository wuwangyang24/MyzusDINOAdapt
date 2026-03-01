"""Evaluation script for DINO LoRA models."""

import argparse
import torch
from pathlib import Path

from src.models import DINOWithLoRA, LoRAConfig
from src.data import create_dataloader
from src.evaluation import Evaluator
from src.utils import setup_logger, load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate DINO LoRA model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to evaluation data directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to evaluate on"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logger
    logger = setup_logger("evaluate")
    
    logger.info("=" * 50)
    logger.info("DINO LoRA Evaluation")
    logger.info("=" * 50)
    
    # Create model
    logger.info(f"Creating model: {config['model']['backbone']}")
    lora_config = LoRAConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
    )
    
    model = DINOWithLoRA(
        backbone_name=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        lora_config=lora_config,
        num_classes=config["model"]["num_classes"],
    )
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create data loader
    logger.info(f"Loading evaluation data from: {args.data_dir}")
    dataloader = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        is_train=False,
        image_size=config["data"]["image_size"],
        shuffle=False,
    )
    
    # Evaluate
    evaluator = Evaluator(model, device=args.device)
    metrics = evaluator.evaluate_and_log(dataloader, "Test Set")
    
    logger.info("\nEvaluation completed!")


if __name__ == "__main__":
    main()
