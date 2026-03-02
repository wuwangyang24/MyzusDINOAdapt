"""Dataset implementations for DINO training."""

from typing import Optional, Callable, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class ImageClassificationDataset(Dataset):
    """
    Image classification dataset.
    
    Expects directory structure:
        root_dir/
            class_1/
                image1.jpg
                image2.jpg
            class_2/
                image3.jpg
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.png', '.jpeg'),
    ):
        """
        Initialize image classification dataset.
        
        Args:
            root_dir: Root directory containing class folders
            transform: Image transformations to apply
            extensions: Supported image file extensions
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions
        
        self.samples = []
        self.class_to_idx = {}
        
        # Build class to index mapping and sample list
        self._build_dataset()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root_dir}")
    
    def _build_dataset(self) -> None:
        """Build dataset by scanning directory structure."""
        class_idx = 0
        
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            self.class_to_idx[class_name] = class_idx
            
            # Find all images in class directory
            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in self.extensions:
                    self.samples.append((image_path, class_idx))
            
            class_idx += 1
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image tensor, class label)
        """
        image_path, label = self.samples[idx]
        
        # Load and convert image to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return len(self.class_to_idx)


def get_default_transforms(
    image_size: int = 224,
    is_train: bool = True
) -> transforms.Compose:
    """
    Get default image transforms for DINO.
    
    Args:
        image_size: Target image size
        is_train: Whether to use training transforms (with augmentation)
        
    Returns:
        Composed transform
    """
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    return transform
