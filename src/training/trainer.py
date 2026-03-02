"""Trainer for DINO LoRA with Triple-Check loss."""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils import setup_logger
from src.models import DINOWithLoRA
from src.losses import TripleCheckLoss

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TripleCheckTrainer:
    """Trainer for DINO with LoRA adaptation using Triple-Check loss."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 10,
        device: Union[str, List[int]] = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        save_interval: int = 1,
        wandb_config: Optional[Dict[str, Any]] = None,
        multi_gpu: bool = False,
        gpu_ids: Optional[List[int]] = None,
    ):
        """
        Initialize Triple-Check trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader (yields 4 image tensors per batch)
            val_dataloader: Validation data loader
            loss_fn: Loss function (default: TripleCheckLoss with L2 distance)
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            num_epochs: Number of training epochs
            device: Device to train on (cuda, cpu, or list of GPU IDs)
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for logs
            save_interval: Epoch interval for saving checkpoints
            wandb_config: W&B configuration dictionary
            multi_gpu: Whether to use DataParallel for multiple GPUs
            gpu_ids: List of GPU IDs to use (if None, uses all available)
        """
        # Handle device and multi-GPU setup
        self.device = self._setup_device(device, multi_gpu, gpu_ids)
        self.multi_gpu = multi_gpu and torch.cuda.device_count() > 1
        self.model_device = self.device  # Device for loss and other components
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Apply DataParallel if multi-GPU is enabled
        if self.multi_gpu:
            if gpu_ids is None:
                gpu_ids = list(range(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.save_interval = save_interval
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = setup_logger(
            "TripleCheckTrainer",
            log_file=str(self.log_dir / "training.log")
        )
        
        # Log device information
        self._log_device_info()
        
        # Setup loss function
        if loss_fn is None:
            loss_fn = TripleCheckLoss(distance_metric="l2", reduction="mean")
        self.loss_fn = loss_fn.to(self.model_device)
        
        # Setup W&B
        self.wandb_enabled = False
        if WANDB_AVAILABLE and wandb_config and wandb_config.get("enabled", False):
            try:
                wandb.init(
                    project=wandb_config.get("project", "dino-lora-triple-check"),
                    entity=wandb_config.get("entity"),
                    name=wandb_config.get("name"),
                    tags=wandb_config.get("tags", []),
                    notes=wandb_config.get("notes", ""),
                    config={
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "num_epochs": num_epochs,
                        "batch_size": train_dataloader.batch_size,
                        "loss_function": self.loss_fn.__class__.__name__,
                        "multi_gpu": self.multi_gpu,
                        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    }
                )
                self.wandb_enabled = True
                self.logger.info("W&B logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize W&B: {e}")
        
        # Setup optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup TensorBoard
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Training stats
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    @staticmethod
    def _setup_device(
        device: Union[str, List[int]],
        multi_gpu: bool,
        gpu_ids: Optional[List[int]] = None
    ) -> str:
        """
        Setup device for training.
        
        Args:
            device: Device specification (cuda, cpu, or list of GPU IDs)
            multi_gpu: Whether to use multi-GPU
            gpu_ids: List of GPU IDs to use
            
        Returns:
            Device string (cuda or cpu)
        """
        if isinstance(device, list):
            # Convert list of GPU IDs to 'cuda'
            if len(device) > 0 and torch.cuda.is_available():
                torch.cuda.set_device(device[0])
                return "cuda"
            else:
                return "cpu"
        
        # Handle string device
        if device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            else:
                print("Warning: CUDA requested but not available. Using CPU.")
                return "cpu"
        
        return device
    
    def _log_device_info(self) -> None:
        """Log device and GPU information."""
        if torch.cuda.is_available():
            self.logger.info(f"CUDA is available")
            self.logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                self.logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            if self.multi_gpu:
                self.logger.info("Using DataParallel for multi-GPU training")
            else:
                self.logger.info(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("Using CPU for training")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with epoch statistics
        """
        self.model.train()  # Enable training mode for LoRA layers and dropout
        
        total_loss = 0.0
        
        for batch_idx, (img_t1, img_u1, img_t2, img_u2) in enumerate(self.train_dataloader):
            img_t1 = img_t1.to(self.device)
            img_u1 = img_u1.to(self.device)
            img_t2 = img_t2.to(self.device)
            img_u2 = img_u2.to(self.device)
            
            # Forward pass - get features (without classification head)
            self.optimizer.zero_grad()
            
            # Extract features from DINO backbone
            with torch.no_grad():
                feat_t1 = self.model.backbone(img_t1)
                feat_u1 = self._process_features(img_u1, self.model.backbone)
                feat_t2 = self.model.backbone(img_t2)
                feat_u2 = self._process_features(img_u2, self.model.backbone)
            
            # Compute loss
            loss = self.loss_fn(feat_t1, feat_u1, feat_t2, feat_u2)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            self.global_step += 1
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                self.writer.add_scalar(
                    "training/loss",
                    loss.item(),
                    self.global_step
                )
                
                # Log to W&B
                if self.wandb_enabled:
                    wandb.log({
                        "training/loss": loss.item(),
                        "training/batch": batch_idx,
                        "training/global_step": self.global_step,
                    })
            
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"Batch [{batch_idx}/{len(self.train_dataloader)}] "
                    f"Loss: {loss.item():.6f}"
                )
        
        avg_loss = total_loss / len(self.train_dataloader)
        
        return {
            "loss": avg_loss,
        }
    
    def _process_features(self, img_tensor: torch.Tensor, backbone) -> torch.Tensor:
        """
        Process image tensor through backbone, handling multiple untreated samples.
        
        Args:
            img_tensor: Image tensor with shape (B, C, H, W) or (N, B, C, H, W)
            backbone: Model backbone to extract features
            
        Returns:
            Features with shape (B, D) - averaged if multiple untreated samples
        """
        # Check if we have multiple untreated samples (5D tensor)
        if img_tensor.dim() == 5:
            # Shape: (N, B, C, H, W) where N is number of untreated samples
            n_samples, batch_size = img_tensor.shape[:2]
            
            # Reshape to (N*B, C, H, W) for efficient batch processing
            img_reshaped = img_tensor.view(-1, img_tensor.shape[2], img_tensor.shape[3], img_tensor.shape[4])
            
            # Forward pass
            features = backbone(img_reshaped)  # Shape: (N*B, D)
            
            # Reshape back to (N, B, D)
            feature_dim = features.shape[-1]
            features_reshaped = features.view(n_samples, batch_size, feature_dim)
            
            # Average across samples: (B, D)
            features_avg = features_reshaped.mean(dim=0)
            
            return features_avg
        else:
            # Regular 4D tensor (B, C, H, W)
            return backbone(img_tensor)
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            Dictionary with validation statistics
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        
        for img_t1, img_u1, img_t2, img_u2 in self.val_dataloader:
            img_t1 = img_t1.to(self.device)
            img_u1 = img_u1.to(self.device)
            img_t2 = img_t2.to(self.device)
            img_u2 = img_u2.to(self.device)
            
            # Extract features
            feat_t1 = self.model.backbone(img_t1)
            feat_u1 = self._process_features(img_u1, self.model.backbone)
            feat_t2 = self.model.backbone(img_t2)
            feat_u2 = self._process_features(img_u2, self.model.backbone)
            
            # Compute loss
            loss = self.loss_fn(feat_t1, feat_u1, feat_t2, feat_u2)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_dataloader)
        
        return {
            "loss": avg_loss,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best checkpoint
        """
        # Handle DataParallel wrapper
        model_state = self.model.module.state_dict() if self.multi_gpu else self.model.state_dict()
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best checkpoint saved to {best_path}")
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model for specified number of epochs.
        
        Returns:
            Dictionary with training history
        """
        self.logger.info("Starting Triple-Check training...")
        self.logger.info(f"Total epochs: {self.num_epochs}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Loss function: {self.loss_fn.__class__.__name__}")
        
        history = {
            "train_loss": [],
            "val_loss": [],
        }
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_stats = self.train_epoch()
            self.logger.info(f"Train Loss: {train_stats['loss']:.6f}")
            
            history["train_loss"].append(train_stats["loss"])
            
            # Log to W&B
            if self.wandb_enabled:
                wandb.log({
                    "epoch": epoch,
                    "training/epoch_loss": train_stats["loss"],
                })
            
            # Validate
            if self.val_dataloader is not None:
                val_stats = self.validate()
                self.logger.info(f"Val Loss: {val_stats['loss']:.6f}")
                
                self.writer.add_scalar("validation/loss", val_stats["loss"], epoch)
                
                # Log to W&B
                if self.wandb_enabled:
                    wandb.log({
                        "validation/loss": val_stats["loss"],
                        "epoch": epoch,
                    })
                
                history["val_loss"].append(val_stats["loss"])
                
                # Save best model
                if val_stats["loss"] < self.best_val_loss:
                    self.best_val_loss = val_stats["loss"]
                    self.save_checkpoint(epoch, is_best=True)
            
            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch)
            
            self.writer.add_scalar("training/loss", train_stats["loss"], epoch)
        
        self.logger.info("Training completed!")
        self.writer.close()
        
        # Finish W&B run
        if self.wandb_enabled:
            wandb.finish()
        
        return history
