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
    
    metadata.json format:
    {
        "pairs": [
            {
                "id": 1,
                "bioassay_1_treated": "bioassay_1/treated/sample_1.jpg",
                "bioassay_1_untreated": "bioassay_1/untreated/sample_1.jpg",
                "bioassay_2_treated": "bioassay_2/treated/sample_1.jpg",
                "bioassay_2_untreated": "bioassay_2/untreated/sample_1.jpg"
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
    ):
        """
        Initialize paired bioassay dataset.
        
        Args:
            root_dir: Root directory containing bioassay samples
            metadata_file: Name of metadata JSON file
            transform: Image transformations to apply
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get paired samples by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (treated_bioassay1, untreated_bioassay1, treated_bioassay2, untreated_bioassay2)
        """
        pair = self.pairs[idx]
        
        # Load images
        img_t1 = self._load_image(pair["bioassay_1_treated"])
        img_u1 = self._load_image(pair["bioassay_1_untreated"])
        img_t2 = self._load_image(pair["bioassay_2_treated"])
        img_u2 = self._load_image(pair["bioassay_2_untreated"])
        
        # Apply transforms
        if self.transform:
            img_t1 = self.transform(img_t1)
            img_u1 = self.transform(img_u1)
            img_t2 = self.transform(img_t2)
            img_u2 = self.transform(img_u2)
        
        return img_t1, img_u1, img_t2, img_u2
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and convert image to RGB."""
        full_path = self.root_dir / image_path
        image = Image.open(full_path).convert('RGB')
        return image


def create_paired_metadata(
    root_dir: str,
    bioassay_1_dir: str = "bioassay_1",
    bioassay_2_dir: str = "bioassay_2",
    treated_dir: str = "treated",
    untreated_dir: str = "untreated",
    output_file: str = "metadata.json",
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
    if not (len(treated_1_files) == len(untreated_1_files) == len(treated_2_files) == len(untreated_2_files)):
        raise RuntimeError(
            f"Mismatched file counts: "
            f"T1={len(treated_1_files)}, U1={len(untreated_1_files)}, "
            f"T2={len(treated_2_files)}, U2={len(untreated_2_files)}"
        )
    
    # Create pairs
    pairs = []
    for i, (t1, u1, t2, u2) in enumerate(zip(treated_1_files, untreated_1_files, treated_2_files, untreated_2_files)):
        pair = {
            "id": i,
            "bioassay_1_treated": str(Path(bioassay_1_dir) / treated_dir / t1),
            "bioassay_1_untreated": str(Path(bioassay_1_dir) / untreated_dir / u1),
            "bioassay_2_treated": str(Path(bioassay_2_dir) / treated_dir / t2),
            "bioassay_2_untreated": str(Path(bioassay_2_dir) / untreated_dir / u2),
        }
        pairs.append(pair)
    
    # Save metadata
    metadata = {"pairs": pairs}
    output_path = root / output_file
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created metadata with {len(pairs)} pairs at {output_path}")
