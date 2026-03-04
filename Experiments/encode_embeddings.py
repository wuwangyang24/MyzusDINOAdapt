"""
encode_embeddings.py

Encode all images in a metadata file using a pretrained DINO backbone.

For each compound and each plate:
  - treated images are encoded individually and stored as a (N, D) tensor.
  - control images are encoded and averaged across all samples on that plate,
    stored as a single (D,) vector.

Metadata format (list of dicts, one per compound):
    [
        {
            "Compound": 1,
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
            "Compound": 2,
            ...
        }
    ]

Output .pt file structure (dict):
    {
        <compound_id (int)>: {
            <plate_id (str)>: {
                "treated": torch.Tensor,   # shape (N, D) — one row per image
                "control": torch.Tensor,   # shape (D,)   — averaged across all controls
            }
        }
    }

Usage:
    # Plain pretrained DINO (no adaptation)
    python encode_embeddings.py \\
        --metadata   /path/to/metadata.json \\
        --root_dir   /path/to/images \\
        --output     /path/to/embeddings.pt \\
        --backbone   dino_vitb16 \\
        --model_type dino \\
        --batch_size 64 \\
        --device     cuda

    # DINO + LoRA  (load fine-tuned weights from a checkpoint)
    python encode_embeddings.py \\
        --metadata     /path/to/metadata.json \\
        --root_dir     /path/to/images \\
        --output       /path/to/embeddings.pt \\
        --backbone     dino_vitb16 \\
        --model_type   dino_lora \\
        --weights_path /path/to/lora_checkpoint.pt \\
        --lora_r       8 --lora_alpha 16.0

    # DINO + DoRA  (load fine-tuned weights from a checkpoint)
    python encode_embeddings.py \\
        --metadata     /path/to/metadata.json \\
        --root_dir     /path/to/images \\
        --output       /path/to/embeddings.pt \\
        --backbone     dino_vitb16 \\
        --model_type   dino_dora \\
        --weights_path /path/to/dora_checkpoint.pt \\
        --dora_r       8 --dora_alpha 16.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Make sure the project's src/ package is importable regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.dino_lora import DINOWithLoRA
from src.models.dino_dora import DINOWithDoRA
from src.models.lora import LoRAConfig
from src.models.dora import DoRAConfig


# ---------------------------------------------------------------------------
# DINO standard preprocessing (matches the pretrained model's training setup)
# ---------------------------------------------------------------------------
DINO_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# Mapping backbone name → CLS-token feature dimension (for reference / assertions)
BACKBONE_DIM = {
    "dino_vits8":  384,
    "dino_vits16": 384,
    "dino_vitb8":  768,
    "dino_vitb16": 768,
    "dino_vitl14": 1024,
    "dino_vitg14": 1536,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _freeze(model: nn.Module) -> None:
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


def _load_checkpoint(model: nn.Module, weights_path: str) -> None:
    """
    Load a state-dict checkpoint into *model* in-place.
    Handles both raw state-dicts and checkpoint dicts with a 'state_dict'
    or 'model_state_dict' key.
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        # Use explicit `is not None` to avoid falsy empty-dict pitfall
        if ckpt.get("state_dict") is not None:
            state_dict = ckpt["state_dict"]
        elif ckpt.get("model_state_dict") is not None:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt   # raw state-dict
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys in checkpoint ({len(missing)}): "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected keys in checkpoint ({len(unexpected)}): "
              f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"  ✓ Loaded weights from: {path}")


def load_model(
    model_type: str,
    backbone_name: str,
    device: torch.device,
    weights_path: Optional[str] = None,
    hub_source: str = "github",
    hub_source_dir: Optional[str] = None,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.1,
    dora_r: int = 8,
    dora_alpha: float = 16.0,
    dora_dropout: float = 0.1,
) -> nn.Module:
    """
    Build and return a model ready for inference.

    Args:
        model_type:      One of 'dino', 'dino_lora', 'dino_dora'.
        backbone_name:   DINO backbone variant (e.g. 'dino_vitb16').
        device:          Torch device.
        weights_path:    Optional path to a fine-tuned checkpoint (.pt/.pth).
                         Required for 'dino_lora' / 'dino_dora' if you want to
                         use adapted weights; skipped for plain 'dino'.
        hub_source:      'github' or 'local'.
        hub_source_dir:  Local DINO hub directory (only for hub_source='local').
        lora_r/alpha/dropout:  LoRA hyper-parameters.
        dora_r/alpha/dropout:  DoRA hyper-parameters.

    Returns:
        nn.Module in eval mode, frozen, on *device*.
    """
    print(f"Building model  : {model_type}  (backbone: {backbone_name})")

    common_kwargs = dict(
        backbone_name=backbone_name,
        num_classes=None,          # features only
        hub_source=hub_source,
        hub_source_dir=hub_source_dir,
    )

    if model_type == "dino":
        model = torch.hub.load(
            "facebookresearch/dino:main"
            if hub_source == "github"
            else hub_source_dir,
            backbone_name,
            **(dict(source="local") if hub_source == "local" else {}),
        )

    elif model_type == "dino_lora":
        lora_cfg = LoRAConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        model = DINOWithLoRA(
            lora_config=lora_cfg,
            **common_kwargs,
        )
        if weights_path:
            _load_checkpoint(model, weights_path)

    elif model_type == "dino_dora":
        dora_cfg = DoRAConfig(
            r=dora_r,
            dora_alpha=dora_alpha,
            dora_dropout=dora_dropout,
        )
        model = DINOWithDoRA(
            dora_config=dora_cfg,
            **common_kwargs,
        )
        if weights_path:
            _load_checkpoint(model, weights_path)

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose from: dino, dino_lora, dino_dora."
        )

    model.to(device)
    model.eval()
    _freeze(model)
    print(f"  ✓ Model ready  "
          f"(feature dim ≈ {BACKBONE_DIM.get(backbone_name, '?')})")
    return model


# ---------------------------------------------------------------------------
# Per-batch encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_paths(
    image_paths: List[str],
    root_dir: Path,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    transform: transforms.Compose = DINO_TRANSFORM,
) -> torch.Tensor:
    """
    Encode a list of image paths and return a (N, D) float32 CPU tensor.

    Args:
        image_paths: Relative paths from root_dir.
        root_dir:    Base directory prepended to every path.
        model:       DINO backbone (eval mode, no grad).
        device:      Torch device.
        batch_size:  Number of images per forward pass.
        transform:   Torchvision transform applied to each PIL image.

    Returns:
        Tensor of shape (N, D) on CPU.
    """
    all_features: List[torch.Tensor] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start: start + batch_size]
        batch_tensors: List[torch.Tensor] = []

        for rel_path in batch_paths:
            full_path = root_dir / rel_path
            try:
                img = Image.open(full_path).convert("RGB")
                batch_tensors.append(transform(img))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load image '{full_path}': {exc}"
                ) from exc

        batch = torch.stack(batch_tensors, dim=0).to(device)   # (B, 3, 224, 224)
        features = model(batch)                                  # (B, D)
        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)  # (N, D)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def encode_metadata(
    metadata: List[Dict],
    root_dir: Path,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
) -> Dict:
    """
    Iterate over compounds and plates and build the embedding dictionary.

    Returns:
        Nested dict: compound_id → plate_id → {"treated": Tensor, "control": Tensor}
    """
    COMPOUND_KEY = "Compound"
    result: Dict = {}

    for compound_entry in tqdm(metadata, desc="Compounds", unit="compound"):
        compound_id: int = compound_entry[COMPOUND_KEY]
        result[compound_id] = {}

        # All keys that are NOT "Compound" are plate identifiers
        plate_ids = [k for k in compound_entry.keys() if k != COMPOUND_KEY]

        for plate_id in tqdm(plate_ids, desc=f"  Compound {compound_id} plates",
                             leave=False, unit="plate"):
            plate_data = compound_entry[plate_id]

            treated_paths: List[str] = plate_data.get("treated", [])
            control_paths: List[str] = plate_data.get("control", [])

            if not treated_paths and not control_paths:
                print(f"  [WARN] Compound {compound_id}, plate {plate_id}: "
                      f"no images found — skipping.")
                continue

            plate_result: Dict[str, torch.Tensor] = {}

            # ---- Treated: encode each image individually ----
            if treated_paths:
                treated_feats = encode_paths(
                    treated_paths, root_dir, model, device, batch_size
                )  # (N_treated, D)
                plate_result["treated"] = treated_feats
            else:
                print(f"  [WARN] Compound {compound_id}, plate {plate_id}: "
                      f"no treated images.")

            # ---- Control: encode then average ----
            if control_paths:
                control_feats = encode_paths(
                    control_paths, root_dir, model, device, batch_size
                )  # (N_control, D)
                control_avg = control_feats.mean(dim=0)  # (D,)
                plate_result["control"] = control_avg
            else:
                print(f"  [WARN] Compound {compound_id}, plate {plate_id}: "
                      f"no control images.")

            result[compound_id][plate_id] = plate_result

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode images with a DINO (or DINO+LoRA/DoRA) model and save embeddings."
    )

    # ---- Required ----
    parser.add_argument(
        "--metadata", type=str, required=True,
        help="Path to the metadata JSON file (list of compound dicts).",
    )
    parser.add_argument(
        "--root_dir", type=str, required=True,
        help="Root directory that the relative image paths in metadata are relative to.",
    )

    # ---- Model selection ----
    parser.add_argument(
        "--model_type", type=str, default="dino",
        choices=["dino", "dino_lora", "dino_dora"],
        help="Which model to use for encoding. Default: dino",
    )
    parser.add_argument(
        "--backbone", type=str, default="dino_vitb16",
        choices=list(BACKBONE_DIM.keys()),
        help="DINO backbone variant. Default: dino_vitb16",
    )
    parser.add_argument(
        "--weights_path", type=str, default=None,
        help="Path to a fine-tuned checkpoint (.pt/.pth). "
             "Used with dino_lora / dino_dora to load adapted weights.",
    )

    # ---- Hub source ----
    parser.add_argument(
        "--hub_source", type=str, default="github",
        choices=["github", "local"],
        help="Where to download the DINO backbone from. Default: github",
    )
    parser.add_argument(
        "--hub_source_dir", type=str, default=None,
        help="Local DINO hub directory (required when --hub_source local).",
    )

    # ---- LoRA hyper-parameters ----
    lora_grp = parser.add_argument_group("LoRA (used when --model_type dino_lora)")
    lora_grp.add_argument("--lora_r",       type=int,   default=8,    help="LoRA rank. Default: 8")
    lora_grp.add_argument("--lora_alpha",   type=float, default=16.0, help="LoRA alpha. Default: 16.0")
    lora_grp.add_argument("--lora_dropout", type=float, default=0.1,  help="LoRA dropout. Default: 0.1")

    # ---- DoRA hyper-parameters ----
    dora_grp = parser.add_argument_group("DoRA (used when --model_type dino_dora)")
    dora_grp.add_argument("--dora_r",       type=int,   default=8,    help="DoRA rank. Default: 8")
    dora_grp.add_argument("--dora_alpha",   type=float, default=16.0, help="DoRA alpha. Default: 16.0")
    dora_grp.add_argument("--dora_dropout", type=float, default=0.1,  help="DoRA dropout. Default: 0.1")

    # ---- Misc ----
    parser.add_argument(
        "--output", type=str, default="embeddings.pt",
        help="Output file path (.pt). Default: embeddings.pt",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Number of images per forward pass. Default: 64",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device (e.g. 'cuda', 'cuda:1', 'cpu'). Auto-detected if not specified.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Accept either a bare list or a dict with a "compounds" key
    if isinstance(metadata, dict):
        if "compounds" in metadata:
            metadata = metadata["compounds"]
        else:
            raise ValueError(
                "Metadata JSON must be a list of compound dicts, "
                "or a dict with a 'compounds' key."
            )
    if not isinstance(metadata, list):
        raise TypeError(f"Expected list of compound dicts, got {type(metadata)}.")

    print(f"Loaded metadata: {len(metadata)} compounds.")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    model = load_model(
        model_type=args.model_type,
        backbone_name=args.backbone,
        device=device,
        weights_path=args.weights_path,
        hub_source=args.hub_source,
        hub_source_dir=args.hub_source_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dora_r=args.dora_r,
        dora_alpha=args.dora_alpha,
        dora_dropout=args.dora_dropout,
    )

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    embeddings = encode_metadata(
        metadata=metadata,
        root_dir=root_dir,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"\nEmbeddings saved to: {output_path}")

    # Print a brief summary
    total_treated = total_control = 0
    for cid, plates in embeddings.items():
        for pid, data in plates.items():
            if "treated" in data:
                total_treated += data["treated"].shape[0]
            if "control" in data:
                total_control += 1   # one averaged vector per plate
    print(f"Summary: {total_treated} treated embeddings, "
          f"{total_control} averaged control embeddings "
          f"(across {len(embeddings)} compounds).")


if __name__ == "__main__":
    main()
