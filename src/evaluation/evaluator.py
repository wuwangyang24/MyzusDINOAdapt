"""Evaluation utilities for DINO LoRA models."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from src.utils import setup_logger


class Evaluator:
    """Evaluator for DINO LoRA models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to evaluate on
        """
        self.model = model.to(device)
        self.device = device
        self.logger = setup_logger("Evaluator")
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: Data loader for evaluation
            return_predictions: Whether to return predictions and labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = {
            "loss": total_loss / len(dataloader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
            "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        }
        
        if return_predictions:
            return metrics, all_preds, all_labels
        
        return metrics
    
    def evaluate_and_log(
        self,
        dataloader: DataLoader,
        dataset_name: str = "Dataset",
    ) -> Dict[str, float]:
        """
        Evaluate and log metrics.
        
        Args:
            dataloader: Data loader for evaluation
            dataset_name: Name of dataset for logging
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = self.evaluate(dataloader)
        
        self.logger.info(f"\n{dataset_name} Evaluation Results:")
        self.logger.info(f"  Loss:      {metrics['loss']:.4f}")
        self.logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
        
        return metrics
    
    @torch.no_grad()
    def get_features(
        self,
        dataloader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from model.
        
        Args:
            dataloader: Data loader
            
        Returns:
            Tuple of (features, labels)
        """
        self.model.eval()
        
        all_features = []
        all_labels = []
        
        for images, labels in dataloader:
            images = images.to(self.device)
            
            # Get features (remove classification head if present)
            if hasattr(self.model, 'backbone'):
                features = self.model.backbone(images)
            else:
                # Forward until last layer
                features = self.model(images)
            
            all_features.extend(features.cpu().numpy())
            all_labels.extend(labels.numpy())
        
        return np.array(all_features), np.array(all_labels)
