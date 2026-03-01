"""Training logic for DINO LoRA adaptation."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils import setup_logger
from src.models import DINOWithLoRA

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    """Trainer for DINO with LoRA adaptation."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        save_interval: int = 1,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            num_epochs: Number of training epochs
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for logs
            save_interval: Epoch interval for saving checkpoints
            wandb_config: W&B configuration dictionary
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.save_interval = save_interval
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = setup_logger(
            "Trainer",
            log_file=str(self.log_dir / "training.log")
        )
        
        # Setup W&B
        self.wandb_enabled = False
        if WANDB_AVAILABLE and wandb_config and wandb_config.get("enabled", False):
            try:
                wandb.init(
                    project=wandb_config.get("project", "dino-lora"),
                    entity=wandb_config.get("entity"),
                    name=wandb_config.get("name"),
                    tags=wandb_config.get("tags", []),
                    notes=wandb_config.get("notes", ""),
                    config={
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "num_epochs": num_epochs,
                        "batch_size": train_dataloader.batch_size,
                    }
                )
                self.wandb_enabled = True
                self.logger.info("W&B logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize W&B: {e}")
        elif wandb_config and wandb_config.get("enabled", False) and not WANDB_AVAILABLE:
            self.logger.warning("W&B enabled in config but wandb not installed. "
                              "Install with: pip install wandb")
        
        # Setup optimizer and loss
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup TensorBoard
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Training stats
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with epoch statistics
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
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
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_loss = total_loss / len(self.train_dataloader)
        accuracy = 100.0 * correct / total
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }
    
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
        correct = 0
        total = 0
        
        for images, labels in self.val_dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = 100.0 * correct / total
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
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
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.num_epochs}")
        self.logger.info(f"Device: {self.device}")
        
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_stats = self.train_epoch()
            self.logger.info(
                f"Train Loss: {train_stats['loss']:.4f}, "
                f"Train Accuracy: {train_stats['accuracy']:.2f}%"
            )
            
            history["train_loss"].append(train_stats["loss"])
            history["train_accuracy"].append(train_stats["accuracy"])
            
            # Log to W&B
            if self.wandb_enabled:
                wandb.log({
                    "epoch": epoch,
                    "training/epoch_loss": train_stats["loss"],
                    "training/epoch_accuracy": train_stats["accuracy"],
                })
            
            # Validate
            if self.val_dataloader is not None:
                val_stats = self.validate()
                self.logger.info(
                    f"Val Loss: {val_stats['loss']:.4f}, "
                    f"Val Accuracy: {val_stats['accuracy']:.2f}%"
                )
                
                self.writer.add_scalar("validation/loss", val_stats["loss"], epoch)
                self.writer.add_scalar("validation/accuracy", val_stats["accuracy"], epoch)
                
                # Log to W&B
                if self.wandb_enabled:
                    wandb.log({
                        "validation/loss": val_stats["loss"],
                        "validation/accuracy": val_stats["accuracy"],
                        "epoch": epoch,
                    })
                
                history["val_loss"].append(val_stats["loss"])
                history["val_accuracy"].append(val_stats["accuracy"])
                
                # Save best model
                if val_stats["loss"] < self.best_val_loss:
                    self.best_val_loss = val_stats["loss"]
                    self.save_checkpoint(epoch, is_best=True)
            
            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch)
            
            self.writer.add_scalar("training/loss", train_stats["loss"], epoch)
            self.writer.add_scalar("training/accuracy", train_stats["accuracy"], epoch)
        
        self.logger.info("Training completed!")
        self.writer.close()
        
        # Finish W&B run
        if self.wandb_enabled:
            wandb.finish()
        
        return history
