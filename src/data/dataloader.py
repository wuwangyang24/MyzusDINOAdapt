"""Data loading utilities."""

from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from .dataset import ImageClassificationDataset, get_default_transforms


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    is_train: bool = True,
    image_size: int = 224,
    shuffle: bool = True,
    dataset: Optional[Dataset] = None,
) -> DataLoader:
    """
    Create a dataloader for image classification.
    
    Args:
        data_dir: Directory containing data (if dataset is None)
        batch_size: Batch size
        num_workers: Number of data loading workers
        is_train: Whether to use training transforms
        image_size: Target image size
        shuffle: Whether to shuffle data
        dataset: Custom dataset (if None, ImageClassificationDataset is used)
        
    Returns:
        DataLoader instance
    """
    if dataset is None:
        transform = get_default_transforms(image_size, is_train)
        dataset = ImageClassificationDataset(data_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle and is_train,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train,
    )
    
    return dataloader
