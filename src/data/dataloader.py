"""Data loading utilities."""

from typing import Optional
import torch
from torch.utils.data import DataLoader, Dataset


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
    Create a dataloader.
    
    Args:
        data_dir: Deprecated. Use dataset parameter
        batch_size: Batch size
        num_workers: Number of data loading workers
        is_train: Deprecated. Configure transforms in your dataset
        image_size: Deprecated. Configure transforms in your dataset
        shuffle: Whether to shuffle data
        dataset: Dataset instance (required - use CompoundPlateDataset)
        
    Returns:
        DataLoader instance
    """
    if dataset is None:
        raise ValueError(
            "dataset parameter is required. "
            "Use CompoundPlateDataset for compound plate data."
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle and is_train,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train,
    )
    
    return dataloader
