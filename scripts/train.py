"""Training script for DINO with Triple-Check Loss and LoRA/DoRA adaptation."""

import argparse
import json
import sys
import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Ensure the project root (parent of scripts/) is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import DINOWithLoRA, LoRAConfig, DINOWithDoRA, DoRAConfig
from src.losses import TripleCheckLoss
from src.data import CompoundPlateDataset, auto_create_compound_plate_metadata, get_default_transforms, compound_collate_fn
from src.training import TripleCheckModule
from src.utils import setup_logger, load_config
from torch.utils.data import DataLoader, RandomSampler

try:
    from pytorch_lightning.loggers import WandbLogger
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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
        "--backbone",
        type=str,
        choices=[
            "dino_vits8", "dino_vits16", "dino_vitb8", "dino_vitb16",
            "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
            "dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg",
        ],
        help="DINO/DINOv2 backbone variant (overrides config)"
    )
    parser.add_argument(
        "--data-root-dir",
        type=str,
        default="",
        help="Root directory prepended to --data-dir"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to paired bioassay data directory (relative to --data-root-dir if set)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="metadata.json",
        help="Path to metadata JSON file (relative to data-dir, or absolute)"
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
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to train on (overrides config)"
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Enable multi-GPU training using DDP"
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="GPU IDs to use (e.g., --gpu-ids 0 1 2). If not specified, uses all available GPUs"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision: 32 (full), 16-mixed (AMP float16), bf16-mixed (AMP bfloat16)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        help="Number of warmup steps (overrides config)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=4,
        help="Max images per plate per type per step (default: 4)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Fraction of metadata entries to use as validation set (0.0 = no split)"
    )
    parser.add_argument(
        "--val-every-steps",
        type=int,
        default=None,
        help="Run validation every N training steps (default: once per epoch)"
    )
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (default: 1.0, 0 to disable)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers (overrides config; 0 = main process only)"
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Batches pre-loaded per DataLoader worker (default: 2; lower = less RAM)"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        help="LoRA/DoRA rank (overrides config)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        help="LoRA/DoRA alpha scaling factor (overrides config)"
    )
    parser.add_argument(
        "--train-layernorm",
        action="store_true",
        help="Unfreeze LayerNorm parameters during LoRA/DoRA training (overrides config)"
    )
    parser.add_argument(
        "--random_sampler",
        action="store_true",
        help="Sample compounds with replacement (allows overlap across batches)"
    )

    # ── Efficacy classifier validation ──
    parser.add_argument(
        "--efficacy-val",
        action="store_true",
        help="Run efficacy classification as a validation metric"
    )
    parser.add_argument(
        "--efficacy-train-metadata",
        type=str,
        default="Data/Metadata/metadata_compound_all20ppm.json",
        help="Path to training metadata JSON for efficacy classifier"
    )
    parser.add_argument(
        "--efficacy-train-labels",
        type=str,
        default="Data/Embeddings/efficacy.pt",
        help="Path to training efficacy .pt file"
    )
    parser.add_argument(
        "--efficacy-inference-metadata",
        type=str,
        default="Data/Metadata/metadata_compound_all100ppm.json",
        help="Path to inference metadata JSON for efficacy classifier"
    )
    parser.add_argument(
        "--efficacy-inference-labels",
        type=str,
        default="Data/Metadata/compounds500ppm.csv",
        help="Path to inference efficacy CSV (columns: 'Compound No', 'Active')"
    )
    parser.add_argument(
        "--efficacy-image-root",
        type=str,
        default="",
        help="Root directory for efficacy classifier image paths (defaults to --data-root-dir)"
    )
    parser.add_argument(
        "--efficacy-threshold",
        type=float,
        default=70.0,
        help="Efficacy binarisation threshold (default: 70.0)"
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
    if args.backbone:
        config["model"]["backbone"] = args.backbone
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.device:
        config["device"] = args.device
    if args.warmup_steps:
        config["training"]["warmup_steps"] = args.warmup_steps
    if args.lora_r is not None:
        config["lora"]["r"] = args.lora_r
        config["dora"]["r"] = args.lora_r
    if args.lora_alpha is not None:
        config["lora"]["lora_alpha"] = args.lora_alpha
        config["dora"]["dora_alpha"] = args.lora_alpha
    if args.train_layernorm:
        config["lora"]["train_layernorm"] = True
        config["dora"]["train_layernorm"] = True

    # Setup logger
    logger = setup_logger(
        "check",
        log_file=f"{config['logging']['log_dir']}/triple_check_training.log"
    )

    pl.seed_everything(config.get("seed", 42))

    adaptation_method = config["adaptation"]["method"]

    logger.info("=" * 50)
    logger.info(f"DINO {adaptation_method.upper()} Triple-Check Training")
    logger.info("=" * 50)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Adaptation Method: {adaptation_method}")

    # Create metadata if needed
    if args.create_metadata:
        logger.info("Creating metadata from directory structure...")
        auto_create_compound_plate_metadata(args.data_dir)

    # Create model based on adaptation method
    logger.info(f"Creating model: {config['model']['backbone']} with {adaptation_method.upper()}")

    if adaptation_method == "lora":
        lora_config = LoRAConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            lora_dropout=config["lora"]["lora_dropout"],
            train_layernorm=config["lora"].get("train_layernorm", False),
        )
        model = DINOWithLoRA(
            backbone_name=config["model"]["backbone"],
            lora_config=lora_config,
            num_classes=None,
            hub_source=config["model"].get("hub_source", "github"),
            hub_source_dir=config["model"].get("hub_source_dir"),
            weights_path=config["model"].get("weights_path"),
        )
    elif adaptation_method == "dora":
        dora_config = DoRAConfig(
            r=config["dora"]["r"],
            dora_alpha=config["dora"]["dora_alpha"],
            dora_dropout=config["dora"]["dora_dropout"],
            train_layernorm=config["dora"].get("train_layernorm", False),
        )
        model = DINOWithDoRA(
            backbone_name=config["model"]["backbone"],
            dora_config=dora_config,
            num_classes=None,
            hub_source=config["model"].get("hub_source", "github"),
            hub_source_dir=config["model"].get("hub_source_dir"),
            weights_path=config["model"].get("weights_path"),
        )
    else:
        raise ValueError(f"Unknown adaptation method: {adaptation_method}")

    logger.info("Model created successfully")

    # Resolve paths relative to data-root-dir
    if args.data_root_dir:
        args.data_dir = str(Path(args.data_root_dir) / args.data_dir)
        args.metadata = str(Path(args.data_root_dir) / args.metadata)
    
    # root_dir for image loading is data-root-dir (not data-dir)
    # since metadata paths already include the subdirectory
    image_root_dir = args.data_root_dir if args.data_root_dir else args.data_dir

    # Create datasets
    logger.info(f"Loading paired bioassay data from: {args.data_dir}")
    transform = get_default_transforms(
        image_size=config["data"]["image_size"],
        is_train=True
    )

    # Load and pre-filter compounds with >= 2 valid plates
    metadata_path = Path(args.metadata)
    with open(metadata_path, 'r') as f:
        raw_metadata = json.load(f)
    all_compounds = raw_metadata if isinstance(raw_metadata, list) else raw_metadata.get("compounds", [])
    before = len(all_compounds)
    all_compounds = [
        c for c in all_compounds
        if CompoundPlateDataset._count_valid_plates(c) >= 2
    ]
    if len(all_compounds) < before:
        logger.info(f"Filtered {before - len(all_compounds)} compounds with <2 valid plates ({before} -> {len(all_compounds)})")

    # Split metadata into train/val if --val-ratio is set
    train_compounds = all_compounds
    val_compounds = None
    if args.val_ratio > 0.0:
        if args.val_ratio >= 1.0:
            raise ValueError(f"--val-ratio must be < 1.0, got {args.val_ratio}")
        n_val = max(1, int(len(all_compounds) * args.val_ratio))
        # Deterministic shuffle for reproducibility
        import random
        rng = random.Random(config.get("seed", 42))
        indices = list(range(len(all_compounds)))
        rng.shuffle(indices)
        val_indices = sorted(indices[:n_val])
        train_indices = sorted(indices[n_val:])
        train_compounds = [all_compounds[i] for i in train_indices]
        val_compounds = [all_compounds[i] for i in val_indices]
        logger.info(f"Split metadata: {len(train_compounds)} train, {len(val_compounds)} val (ratio={args.val_ratio})")

    train_dataset = CompoundPlateDataset(
        root_dir=image_root_dir,
        metadata_file=args.metadata,
        transform=transform,
        compounds_list=train_compounds,
        num_plates=2,
        max_samples=args.max_samples,
    )
    num_workers = args.num_workers if args.num_workers is not None else config["training"]["num_workers"]
    prefetch = args.prefetch_factor if args.prefetch_factor is not None else (2 if num_workers > 0 else None)
    if args.random_sampler:
        train_sampler = RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=len(train_dataset),
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=num_workers,
            sampler=train_sampler,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch,
            collate_fn=compound_collate_fn,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=num_workers,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch,
            collate_fn=compound_collate_fn,
        )

    val_dataloader = None
    if val_compounds is not None:
        logger.info(f"Creating validation set from metadata split ({len(val_compounds)} compounds)")
        val_transform = get_default_transforms(
            image_size=config["data"]["image_size"],
            is_train=False
        )
        val_dataset = CompoundPlateDataset(
            root_dir=image_root_dir,
            metadata_file=args.metadata,
            transform=val_transform,
            compounds_list=val_compounds,
            num_plates=2,
            max_samples=args.max_samples,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch,
            collate_fn=compound_collate_fn,
        )
    elif args.val_data_dir:
        logger.info(f"Loading validation data from: {args.val_data_dir}")
        val_transform = get_default_transforms(
            image_size=config["data"]["image_size"],
            is_train=False
        )
        val_dataset = CompoundPlateDataset(
            root_dir=args.val_data_dir,
            transform=val_transform,
            num_plates=2,
            max_samples=args.max_samples,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch,
            collate_fn=compound_collate_fn,
        )

    # Create loss function and Lightning module
    loss_fn = TripleCheckLoss(
        distance_metric=args.distance_metric,
        temperature=1.0,
        reduction="mean",
    )
    # Compute step counts for LR scheduler
    steps_per_epoch = len(train_dataloader)
    num_epochs = config["training"]["num_epochs"]
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = config["training"].get("warmup_steps", 100)

    module = TripleCheckModule(
        model=model,
        loss_fn=loss_fn,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        max_samples=args.max_samples,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # --- Callbacks ---
    backbone = config["model"]["backbone"]
    r = config[adaptation_method]["r"]
    alpha = config[adaptation_method].get(f"{adaptation_method}_alpha", r)
    alpha_str = int(alpha) if alpha == int(alpha) else alpha
    ckpt_tag = f"{backbone}_{adaptation_method}_r{r}a{alpha_str}"
    if config[adaptation_method].get("train_layernorm", False):
        ckpt_tag += "_LN"
    ckpt_subdir = Path(config["checkpoint"]["save_dir"]) / ckpt_tag
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_subdir),
        every_n_epochs=config["checkpoint"]["save_interval"],
        monitor="val/loss" if val_dataloader is not None else None,
        save_top_k=1,
        filename="best",
        save_last=False,
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_subdir),
        every_n_epochs=config["training"]["num_epochs"],
        save_top_k=1,
        monitor=None,
        filename="last",
        save_on_train_epoch_end=True,
    )
    callbacks = [checkpoint_callback, last_checkpoint_callback, LearningRateMonitor(logging_interval="step")]

    # Efficacy classifier callback (optional)
    if args.efficacy_val:
        from src.training.efficacy_callback import EfficacyClassifierCallback
        efficacy_image_root = args.efficacy_image_root or args.data_root_dir or args.data_dir
        efficacy_callback = EfficacyClassifierCallback(
            train_metadata_path=args.efficacy_train_metadata,
            train_efficacy_path=args.efficacy_train_labels,
            inference_metadata_path=args.efficacy_inference_metadata,
            inference_efficacy_csv=args.efficacy_inference_labels,
            image_root_dir=efficacy_image_root,
            threshold=args.efficacy_threshold,
            every_n_steps=args.val_every_steps or 0,
        )
        callbacks.append(efficacy_callback)
        logger.info("Efficacy classifier validation enabled")

    # --- Loggers ---
    tb_logger = TensorBoardLogger(
        save_dir=config["logging"]["log_dir"],
        name="lightning_logs",
    )
    pl_loggers = [tb_logger]

    wandb_cfg = config["logging"].get("wandb", {})
    if WANDB_AVAILABLE and wandb_cfg.get("enabled", False):
        # Log all config + CLI args + computed values
        wandb_config = dict(config)
        wandb_config["cli"] = vars(args)
        wandb_config.update({
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
        })
        wandb_logger = WandbLogger(
            project=wandb_cfg.get("project", "dino-lora-triple-check"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name"),
            tags=wandb_cfg.get("tags", []),
            notes=wandb_cfg.get("notes", ""),
            config=wandb_config,
        )
        # Use trainer/global_step as x-axis for all metrics so the
        # step counter is monotonically increasing in W&B charts.
        wandb_logger.experiment.define_metric("trainer/global_step")
        wandb_logger.experiment.define_metric("*", step_metric="trainer/global_step")
        pl_loggers.append(wandb_logger)

    # --- Accelerator / devices / strategy ---
    use_cuda = config.get("device", "cuda") == "cuda" and torch.cuda.is_available()
    accelerator = "gpu" if use_cuda else "cpu"
    if args.multi_gpu:
        devices = args.gpu_ids if args.gpu_ids else "auto"
        strategy = "ddp"
    else:
        devices = [args.gpu_ids[0]] if args.gpu_ids else 1
        strategy = "auto"

    # --- Build Lightning Trainer ---
    trainer_kwargs = dict(
        max_epochs=config["training"]["num_epochs"],
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=args.precision,
        callbacks=callbacks,
        logger=pl_loggers,
        accumulate_grad_batches=config["training"].get("gradient_accumulation_steps", 1),
        log_every_n_steps=10,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0 else None,
    )
    if args.val_every_steps is not None and val_dataloader is not None:
        trainer_kwargs["val_check_interval"] = args.val_every_steps
    trainer = pl.Trainer(**trainer_kwargs)

    logger.info("Starting training...")
    trainer.fit(module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
