"""Dataset for paired bioassay samples and compound-plate-well structures."""

from typing import Optional, Callable, Tuple, Dict, List
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os
from collections import defaultdict


class CompoundPlateDataset(Dataset):
    """
    Dataset for compound screening across multiple plates and wells.
    
    Directory structure:
        root_dir/
            metadata.json
            94000/
                well_2_1/
                    treated/
                        sample_1.png
                        sample_2.png
                well_1_3/
                    control/
                        sample_1.png
                        sample_2.png
                well_3_6/
                    treated/
                        sample_1.png
            131000/
                well_1_2/
                    control/
                        sample_1.png
                        sample_2.png
    
    Note: treated and control samples can be in different wells on the same plate.
    
    metadata.json format:
    {
        "compounds": [
            {
                "id": 1,
                "94000": {
                    "treated": ["94000/well_2_1/treated/sample_1.png", ...],
                    "control": ["94000/well_1_3/control/sample_1.png", ...]
                },
                "131000": {
                    "treated": ["131000/well_3_6/treated/sample_1.png", ...],
                    "control": ["131000/well_1_2/control/sample_1.png", ...]
                }
            },
            {
                "id": 2,
                ...
            }
        ]
    }
    
    Data flow:
    1. Get compound data (all treated/control samples from all plates)
    2. For each plate: average control latents
    3. For each treated sample on a plate: subtract that plate's control average
    4. For each plate: average the adjusted treated latents
    5. Loss: minimize distance between plate-average treated latents for same compound
    """
    
    def __init__(
        self,
        root_dir: str,
        metadata_file: str = "metadata.json",
        transform: Optional[Callable] = None,
    ):
        """
        Initialize compound-plate dataset.
        
        Args:
            root_dir: Root directory containing plate samples
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
        
        self.compounds = self.metadata.get("compounds", [])
        
        if len(self.compounds) == 0:
            raise RuntimeError(f"No compounds found in metadata: {self.metadata_path}")
    
    def __len__(self) -> int:
        """Return dataset size (number of compounds)."""
        return len(self.compounds)
    
    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get compound data organized by plate and type.
        
        Args:
            idx: Compound index
            
        Returns:
            Dict with structure:
            {
                "id": compound_id,
                "plates": {
                    "plate_1": {
                        "treated": torch.Tensor of shape (N, C, H, W),
                        "control": torch.Tensor of shape (M, C, H, W)
                    },
                    "plate_2": {
                        "treated": torch.Tensor of shape (K, C, H, W),
                        "control": torch.Tensor of shape (L, C, H, W)
                    }
                }
            }
        """
        compound = self.compounds[idx]
        compound_id = compound["id"]
        
        # Group by plate
        plates_data = {}
        
        # Extract all plate data
        for key, plate_data in compound.items():
            if key == "id":
                continue
            
            plate_name = key  # e.g., "plate_1"
            
            if not isinstance(plate_data, dict):
                continue
            
            plates_data[plate_name] = {}
            
            # Load treated and control images for this plate
            for sample_type in ["treated", "control"]:
                if sample_type not in plate_data:
                    continue
                
                paths = plate_data[sample_type]
                
                # Load images
                images = []
                if isinstance(paths, list):
                    for path in paths:
                        images.append(self._load_image(path))
                else:
                    images.append(self._load_image(paths))
                
                # Apply transforms
                if self.transform:
                    images = [self.transform(img) for img in images]
                
                # Stack into tensor
                if images:
                    images_tensor = torch.stack(images)  # Shape: (N, C, H, W)
                    plates_data[plate_name][sample_type] = images_tensor
        
        # Ensure all plates have both treated and control
        for plate_name in plates_data:
            if "treated" not in plates_data[plate_name]:
                raise ValueError(f"Compound {compound_id}: {plate_name} missing treated samples")
            if "control" not in plates_data[plate_name]:
                raise ValueError(f"Compound {compound_id}: {plate_name} missing control samples")
        
        return {
            "id": compound_id,
            "plates": plates_data
        }
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and convert image to RGB."""
        full_path = self.root_dir / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")
        image = Image.open(full_path).convert('RGB')
        return image


def create_compound_plate_metadata(
    root_dir: str,
    compound_mapping_file: str,
    output_file: str = "metadata.json",
) -> None:
    """
    Create metadata.json for compound-plate-well structure.
    
    This function reads a compound mapping file that specifies which plates/wells
    belong to which compound, then generates the metadata.json.
    
    Args:
        root_dir: Root directory containing plate data
        compound_mapping_file: JSON file mapping compounds to plates/wells.
                              Format:
                              {
                                  "1": {
                                      "plate_1": {
                                          "treated": ["well_A1", "well_A2"],
                                          "control": ["well_B1", "well_B2"]
                                      },
                                      "plate_2": {
                                          "treated": ["well_C1"],
                                          "control": ["well_D1"]
                                      }
                                  },
                                  "2": { ... }
                              }
        output_file: Output metadata filename
    """
    root = Path(root_dir)
    
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    
    # Load compound mapping
    if not Path(compound_mapping_file).exists():
        raise FileNotFoundError(f"Compound mapping file not found: {compound_mapping_file}")
    
    with open(compound_mapping_file, 'r') as f:
        compound_mapping = json.load(f)
    
    compounds = []
    
    for compound_id, plates_dict in compound_mapping.items():
        compound_entry = {"id": str(compound_id)}
        
        for plate_name, types_dict in plates_dict.items():
            # plate_name should be like "plate_1"
            plate_path = root / plate_name
            
            if not plate_path.exists():
                raise FileNotFoundError(f"Plate directory not found: {plate_path}")
            
            plate_entry = {}
            
            # Process treated and control separately
            for sample_type in ["treated", "control"]:
                if sample_type not in types_dict:
                    continue
                
                wells = types_dict[sample_type]
                samples = []
                
                for well_name in wells:
                    well_path = plate_path / well_name / sample_type
                    
                    if not well_path.exists():
                        raise FileNotFoundError(f"Well type directory not found: {well_path}")
                    
                    # Get samples
                    files = sorted([f for f in well_path.glob("*.png")])
                    for f in files:
                        rel_path = f.relative_to(root)
                        samples.append(str(rel_path))
                
                if samples:
                    plate_entry[sample_type] = samples
            
            # Verify plate has both treated and control
            if "treated" in plate_entry and "control" in plate_entry:
                compound_entry[plate_name] = plate_entry
        
        # Verify compound has at least one plate with both treated and control
        has_complete_plate = False
        for key, plate_data in compound_entry.items():
            if key != "id" and isinstance(plate_data, dict):
                if "treated" in plate_data and "control" in plate_data:
                    has_complete_plate = True
                    break
        
        if not has_complete_plate:
            raise RuntimeError(
                f"Compound {compound_id} does not have any plate with both treated and control"
            )
        
        compounds.append(compound_entry)
    
    # Save metadata
    metadata = {"compounds": compounds}
    output_path = root / output_file
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created metadata with {len(compounds)} compounds at {output_path}")


def auto_create_compound_plate_metadata(
    root_dir: str,
    output_file: str = "metadata.json",
) -> None:
    """
    Automatically create metadata by scanning directory structure.
    
    Assumes directory structure: plate_X/well_Y/{treated|control}/*.png
    
    Creates compounds where:
    - Each compound gets all treated samples from all wells on a plate
    - Each compound gets all control samples from all wells on a plate
    - This ensures each compound has treated and control data per plate
    
    Args:
        root_dir: Root directory containing plate data
        output_file: Output metadata filename
    """
    root = Path(root_dir)
    
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    
    # Scan directory structure
    plate_dirs = sorted([d for d in root.iterdir() 
                        if d.is_dir() and d.name.startswith("plate_")])
    
    if not plate_dirs:
        raise RuntimeError(f"No plate directories found in {root}")
    
    # For each plate, collect all treated and control samples
    plates_data = {}
    
    for plate_dir in plate_dirs:
        plate_name = plate_dir.name
        treated_samples = []
        control_samples = []
        
        well_dirs = sorted([d for d in plate_dir.iterdir() if d.is_dir()])
        
        for well_dir in well_dirs:
            well_name = well_dir.name
            
            # Check for treated subdirectory
            treated_dir = well_dir / "treated"
            if treated_dir.exists():
                treated_files = sorted([f for f in treated_dir.glob("*.png")])
                for f in treated_files:
                    rel_path = f.relative_to(root)
                    treated_samples.append(str(rel_path))
            
            # Check for control subdirectory
            control_dir = well_dir / "control"
            if control_dir.exists():
                control_files = sorted([f for f in control_dir.glob("*.png")])
                for f in control_files:
                    rel_path = f.relative_to(root)
                    control_samples.append(str(rel_path))
        
        if treated_samples and control_samples:
            plates_data[plate_name] = {
                "treated": treated_samples,
                "control": control_samples
            }
    
    if not plates_data:
        raise RuntimeError("No plates with both treated and control samples found")
    
    # Create a single compound with all plates
    compounds = [{
        "id": 1,
        **plates_data
    }]
    
    # Save metadata
    metadata = {"compounds": compounds}
    output_path = root / output_file
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Auto-created metadata with {len(compounds)} compound(s) "
          f"from {len(plates_data)} plates at {output_path}")


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
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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


