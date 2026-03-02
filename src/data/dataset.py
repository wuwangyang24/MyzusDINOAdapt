"""Dataset for paired bioassay samples."""

from typing import Optional, Callable, Tuple, Dict, List
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os


class PairedBioassayDataset(Dataset):
    """
    Paired bioassay dataset for triple-check loss training.
    
    Supports averaging multiple untreated samples per pair.
    
    Expects directory structure:
        root_dir/
            metadata.json  (contains pairing information)
            bioassay_1/
                treated/
                    sample_1.jpg
                    sample_2.jpg
                untreated/
                    sample_1.jpg
                    sample_2.jpg
            bioassay_2/
                treated/
                    sample_1.jpg
                    sample_2.jpg
                untreated/
                    sample_1.jpg
                    sample_2.jpg
    
    metadata.json format (with num_untreated_samples=1):
    {
        "pairs": [
            {
                "id": 1,
                "bioassay_1_treated": "bioassay_1/treated/sample_1.jpg",
                "bioassay_1_untreated": ["bioassay_1/untreated/sample_1.jpg"],
                "bioassay_2_treated": "bioassay_2/treated/sample_1.jpg",
                "bioassay_2_untreated": ["bioassay_2/untreated/sample_1.jpg"]
            },
            ...
        ]
    }
    
    metadata.json format (with num_untreated_samples > 1):
    {
        "pairs": [
            {
                "id": 1,
                "bioassay_1_treated": "bioassay_1/treated/sample_1.jpg",
                "bioassay_1_untreated": ["bioassay_1/untreated/sample_1.jpg", 
                                         "bioassay_1/untreated/sample_2.jpg"],
                "bioassay_2_treated": "bioassay_2/treated/sample_1.jpg",
                "bioassay_2_untreated": ["bioassay_2/untreated/sample_1.jpg",
                                         "bioassay_2/untreated/sample_2.jpg"]
            },
            ...
        ]
    }
    """
    
    def __init__(
        self,
        root_dir: str,
        metadata_file: str = "metadata.json",
        transform: Optional[Callable] = None,
        num_untreated_samples: int = 1,
    ):
        """
        Initialize paired bioassay dataset.
        
        Args:
            root_dir: Root directory containing bioassay samples
            metadata_file: Name of metadata JSON file
            transform: Image transformations to apply
            num_untreated_samples: Number of untreated samples to average for each pair.
                                   If > 1, untreated samples in metadata are expected to be lists.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.num_untreated_samples = num_untreated_samples
        self.metadata_path = self.root_dir / metadata_file
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.pairs = self.metadata.get("pairs", [])
        
        if len(self.pairs) == 0:
            raise RuntimeError(f"No pairs found in metadata: {self.metadata_path}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int):
        """
        Get paired samples by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (treated_bioassay1, untreated_bioassay1, treated_bioassay2, untreated_bioassay2)
            
            If num_untreated_samples == 1:
                - untreated samples are shape (C, H, W)
                
            If num_untreated_samples > 1:
                - untreated samples are shape (N, C, H, W) where N = num_untreated_samples
                - This allows averaging after model forward pass
        """
        pair = self.pairs[idx]
        
        # Load treated images
        img_t1 = self._load_image(pair["bioassay_1_treated"])
        img_t2 = self._load_image(pair["bioassay_2_treated"])
        
        # Load untreated images (may return single image or list of images)
        untreated_u1 = self._load_untreated_images(pair["bioassay_1_untreated"])
        untreated_u2 = self._load_untreated_images(pair["bioassay_2_untreated"])
        
        # Apply transforms - handle both single image and list of images
        if self.transform:
            img_t1 = self.transform(img_t1)
            img_t2 = self.transform(img_t2)
            
            # For untreated images
            if isinstance(untreated_u1, list):
                untreated_u1 = [self.transform(img) for img in untreated_u1]
            else:
                untreated_u1 = self.transform(untreated_u1)
                
            if isinstance(untreated_u2, list):
                untreated_u2 = [self.transform(img) for img in untreated_u2]
            else:
                untreated_u2 = self.transform(untreated_u2)
        
        # Convert to tensors - stack if multiple samples
        if isinstance(untreated_u1, list):
            img_u1 = torch.stack(untreated_u1)  # Shape: (N, C, H, W)
        else:
            img_u1 = untreated_u1
            
        if isinstance(untreated_u2, list):
            img_u2 = torch.stack(untreated_u2)  # Shape: (N, C, H, W)
        else:
            img_u2 = untreated_u2
        
        return img_t1, img_u1, img_t2, img_u2
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and convert image to RGB."""
        full_path = self.root_dir / image_path
        image = Image.open(full_path).convert('RGB')
        return image
    
    def _load_untreated_images(self, untreated_paths):
        """
        Load untreated sample(s) without averaging.
        
        Args:
            untreated_paths: Either a string (single path) or list of strings (multiple paths)
                            
        Returns:
            PIL Image if single path, or list of PIL Images if multiple paths.
            The caller will handle stacking them into a tensor.
        """
        # Handle both old format (single string) and new format (list of strings)
        if isinstance(untreated_paths, str):
            return self._load_image(untreated_paths)
        
        # Load multiple untreated samples as a list
        if isinstance(untreated_paths, list):
            images = [self._load_image(path) for path in untreated_paths]
            return images
        
        raise ValueError(f"Unexpected untreated_paths type: {type(untreated_paths)}")


def create_paired_metadata(
    root_dir: str,
    bioassay_1_dir: str = "bioassay_1",
    bioassay_2_dir: str = "bioassay_2",
    treated_dir: str = "treated",
    untreated_dir: str = "untreated",
    output_file: str = "metadata.json",
    num_untreated_samples: int = 1,
) -> None:
    """
    Create metadata.json file by pairing samples from two bioassays.
    
    Assumes treated and untreated samples have matching filenames.
    
    Args:
        root_dir: Root directory
        bioassay_1_dir: Directory name for bioassay 1
        bioassay_2_dir: Directory name for bioassay 2
        treated_dir: Directory name for treated samples
        untreated_dir: Directory name for untreated samples
        output_file: Output metadata filename
        num_untreated_samples: Number of untreated samples to group per treated sample.
                              If > 1, each pair will have multiple untreated samples that will be averaged.
    """
    root = Path(root_dir)
    
    # Get treated/untreated samples from both bioassays
    treated_1_path = root / bioassay_1_dir / treated_dir
    untreated_1_path = root / bioassay_1_dir / untreated_dir
    treated_2_path = root / bioassay_2_dir / treated_dir
    untreated_2_path = root / bioassay_2_dir / untreated_dir
    
    # Check all directories exist
    for path in [treated_1_path, untreated_1_path, treated_2_path, untreated_2_path]:
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
    
    # Get filenames
    treated_1_files = sorted([f.name for f in treated_1_path.glob("*") if f.is_file()])
    untreated_1_files = sorted([f.name for f in untreated_1_path.glob("*") if f.is_file()])
    treated_2_files = sorted([f.name for f in treated_2_path.glob("*") if f.is_file()])
    untreated_2_files = sorted([f.name for f in untreated_2_path.glob("*") if f.is_file()])
    
    # Verify matching files
    if not (len(treated_1_files) == len(treated_2_files)):
        raise RuntimeError(
            f"Mismatched treated file counts: "
            f"T1={len(treated_1_files)}, T2={len(treated_2_files)}"
        )
    
    # Check that untreated files can be divided evenly
    if len(untreated_1_files) % num_untreated_samples != 0:
        raise RuntimeError(
            f"Bioassay 1: {len(untreated_1_files)} untreated samples cannot be evenly divided "
            f"into groups of {num_untreated_samples}"
        )
    
    if len(untreated_2_files) % num_untreated_samples != 0:
        raise RuntimeError(
            f"Bioassay 2: {len(untreated_2_files)} untreated samples cannot be evenly divided "
            f"into groups of {num_untreated_samples}"
        )
    
    num_treated = len(treated_1_files)
    expected_untreated_per_bioassay = num_treated * num_untreated_samples
    
    if len(untreated_1_files) != expected_untreated_per_bioassay:
        raise RuntimeError(
            f"Bioassay 1: Expected {expected_untreated_per_bioassay} untreated samples "
            f"({num_treated} treated × {num_untreated_samples}), got {len(untreated_1_files)}"
        )
    
    if len(untreated_2_files) != expected_untreated_per_bioassay:
        raise RuntimeError(
            f"Bioassay 2: Expected {expected_untreated_per_bioassay} untreated samples "
            f"({num_treated} treated × {num_untreated_samples}), got {len(untreated_2_files)}"
        )
    
    # Create pairs
    pairs = []
    for pair_idx in range(num_treated):
        # Get the N untreated samples for this treated sample
        start_idx = pair_idx * num_untreated_samples
        end_idx = start_idx + num_untreated_samples
        
        u1_files = untreated_1_files[start_idx:end_idx]
        u2_files = untreated_2_files[start_idx:end_idx]
        
        pair = {
            "id": pair_idx,
            "bioassay_1_treated": str(Path(bioassay_1_dir) / treated_dir / treated_1_files[pair_idx]),
            "bioassay_1_untreated": [str(Path(bioassay_1_dir) / untreated_dir / f) for f in u1_files],
            "bioassay_2_treated": str(Path(bioassay_2_dir) / treated_dir / treated_2_files[pair_idx]),
            "bioassay_2_untreated": [str(Path(bioassay_2_dir) / untreated_dir / f) for f in u2_files],
        }
        pairs.append(pair)
    
    # Save metadata
    metadata = {"pairs": pairs, "num_untreated_samples": num_untreated_samples}
    output_path = root / output_file
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created metadata with {len(pairs)} pairs "
          f"({num_untreated_samples} untreated samples per treated) at {output_path}")


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
