"""
encode_embeddings.py

Encode all images in a metadata file using a pretrained DINO or DINOv2 backbone.

For each compound and each plate:
  - treated images are encoded individually and stored as a (N, D) tensor.
  - control images are encoded and averaged across all samples on that plate,
    stored as a single (D,) vector.

Metadata format (list of dicts, one per compound):
    [
        {
            "Compound": "1",
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
            "Compound": "2",
            ...
        }
    ]

Output .pt file structure (dict):
    {
        <compound_id (str)>: {
            <plate_id (str)>: {
                "treated": torch.Tensor,   # shape (N, D) — one row per image
                "control": torch.Tensor,   # shape (D,)   — averaged across all controls
                # When --return_reg_tokens is used, features come from the mean
                # of the register tokens instead of the CLS token.
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

    # Plain pretrained DINOv2
    python encode_embeddings.py \\
        --metadata   /path/to/metadata.json \\
        --root_dir   /path/to/images \\
        --output     /path/to/embeddings.pt \\
        --backbone   dinov2_vitb14 \\
        --model_type dino

    # DINOv2 with register tokens
    python encode_embeddings.py \\
        --metadata   /path/to/metadata.json \\
        --root_dir   /path/to/images \\
        --output     /path/to/embeddings.pt \\
        --backbone   dinov2_vitl14_reg \\
        --model_type dino

    # DINO + LoRA  (load fine-tuned weights from a checkpoint)
    python encode_embeddings.py \\
        --metadata     /path/to/metadata.json \\
        --root_dir     /path/to/images \\
        --output       /path/to/embeddings.pt \\
        --backbone     dino_vitb16 \\
        --model_type   dino_lora \\
        --weights_path /path/to/lora_checkpoint.pt \\
        --lora_r       8 --lora_alpha 16.0

    # DINOv2 + LoRA
    python encode_embeddings.py \\
        --metadata     /path/to/metadata.json \\
        --root_dir     /path/to/images \\
        --output       /path/to/embeddings.pt \\
        --backbone     dinov2_vitb14 \\
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

    # DINOv2 + DoRA
    python encode_embeddings.py \\
        --metadata     /path/to/metadata.json \\
        --root_dir     /path/to/images \\
        --output       /path/to/embeddings.pt \\
        --backbone     dinov2_vitb14 \\
        --model_type   dino_dora \\
        --weights_path /path/to/dora_checkpoint.pt \\
        --dora_r       8 --dora_alpha 16.0

    # Custom VAE encoder
    python encode_embeddings.py \\
        --metadata       /path/to/metadata.json \\
        --root_dir       /path/to/images \\
        --output         /path/to/embeddings.pt \\
        --model_type     custom_vae \\
        --vae_checkpoint /path/to/tiltedvae2.ckpt \\
        --vae_latent_dim 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.io import decode_image, ImageReadMode
from tqdm import tqdm

# Make sure the project's src/ package is importable regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.dino_lora import DINOWithLoRA
from src.models.dino_dora import DINOWithDoRA
from src.models.lora import LoRAConfig
from src.models.dora import DoRAConfig
from VAE.model2 import Encoder


# ---------------------------------------------------------------------------
# DINO standard preprocessing (matches the pretrained model's training setup)
# ---------------------------------------------------------------------------
DINO_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# VAE preprocessing (96×64 input, [0,1] range, no ImageNet normalisation)
VAE_TRANSFORM = transforms.Compose([
    transforms.Resize((96, 64), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ConvertImageDtype(torch.float32),
])

# Mapping backbone name → CLS-token feature dimension (for reference / assertions)
BACKBONE_DIM = {
    # DINOv1
    "dino_vits8":  384,
    "dino_vits16": 384,
    "dino_vitb8":  768,
    "dino_vitb16": 768,
    "dino_vitl14": 1024,
    "dino_vitg14": 1536,
    # DINOv2
    "dinov2_vits14":     384,
    "dinov2_vitb14":     768,
    "dinov2_vitl14":     1024,
    "dinov2_vitg14":     1536,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14_reg": 1536,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_backbone(model: nn.Module) -> nn.Module:
    """Return the raw backbone, unwrapping LoRA/DoRA wrappers if present."""
    if hasattr(model, "backbone"):
        return model.backbone
    return model


class _VAEEncoderWrapper(nn.Module):
    """Wraps a VAE Encoder so that forward() returns only mu (B, D)."""
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _z, mu, _log_var = self.encoder(x)
        return mu


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
    vae_checkpoint: Optional[str] = None,
    vae_latent_dim: int = 100,
) -> nn.Module:
    """
    Build and return a model ready for inference.

    Args:
        model_type:      One of 'dino', 'dino_lora', 'dino_dora'.
        backbone_name:   DINO/DINOv2 backbone variant (e.g. 'dino_vitb16',
                         'dinov2_vitb14', 'dinov2_vitl14_reg').
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

    hub_repo = (
        "facebookresearch/dinov2:main"
        if backbone_name.startswith("dinov2_")
        else "facebookresearch/dino:main"
    )

    if model_type == "dino":
        model = torch.hub.load(
            hub_repo
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

    elif model_type == "custom_vae":
        if not vae_checkpoint:
            raise ValueError("--vae_checkpoint is required for model_type 'custom_vae'.")
        ckpt_path = Path(vae_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"VAE checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        encoder_weights = {
            k.split('.', 1)[1]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("encoder.")
        }
        encoder = Encoder(vae_latent_dim)
        encoder.load_state_dict(encoder_weights)
        model = _VAEEncoderWrapper(encoder)
        print(f"  ✓ Loaded VAE encoder from: {ckpt_path}")

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose from: dino, dino_lora, dino_dora, custom_vae."
        )

    model.to(device)
    model.eval()
    _freeze(model)
    feat_dim = vae_latent_dim if model_type == "custom_vae" else BACKBONE_DIM.get(backbone_name, '?')
    print(f"  ✓ Model ready  (feature dim ≈ {feat_dim})")
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
    return_reg_tokens: bool = False,
    use_amp: bool = True,
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
        return_reg_tokens: If True, also return register tokens via
                           DINOv2's ``forward_features`` method.

    Returns:
        If return_reg_tokens is False:
            Tensor of shape (N, D) on CPU  (CLS token features).
        If return_reg_tokens is True:
            Tensor of shape (N, D) on CPU  (register tokens averaged
            over the N_reg dimension per image).
    """
    all_features: List[torch.Tensor] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start: start + batch_size]
        batch_tensors: List[torch.Tensor] = []

        for rel_path in batch_paths:
            full_path = root_dir / rel_path
            try:
                img = decode_image(str(full_path), mode=ImageReadMode.RGB)
                batch_tensors.append(transform(img))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load image '{full_path}': {exc}"
                ) from exc

        batch = torch.stack(batch_tensors, dim=0).to(device)   # (B, 3, 224, 224)

        amp_enabled = use_amp and device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            if return_reg_tokens:
                backbone = _get_backbone(model)
                out = backbone.forward_features(batch)
                reg_tok = out["x_norm_regtokens"]             # (B, N_reg, D)
                reg_tok_mean = reg_tok.mean(dim=1)             # (B, D)
                all_features.append(reg_tok_mean.float().cpu())
            else:
                features = model(batch)                        # (B, D)
                all_features.append(features.float().cpu())

    return torch.cat(all_features, dim=0)                  # (N, D)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def encode_metadata(
    metadata: List[Dict],
    root_dir: Path,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    return_reg_tokens: bool = False,
    use_amp: bool = True,
    transform: transforms.Compose = DINO_TRANSFORM,
) -> Dict:
    """
    Iterate over compounds and plates and build the embedding dictionary.

    Returns:
        Nested dict: compound_id → plate_id → {"treated": Tensor, "control": Tensor}
        When *return_reg_tokens* is True the dict also contains
        "treated_reg_tokens" and "control_reg_tokens".
    """
    COMPOUND_KEY = "Compound"
    result: Dict = {}

    for compound_entry in tqdm(metadata, desc="Compounds", unit="compound"):
        compound_id: str = str(compound_entry[COMPOUND_KEY])
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
                    treated_paths, root_dir, model, device, batch_size,
                    transform=transform,
                    return_reg_tokens=return_reg_tokens, use_amp=use_amp,
                )  # (N_treated, D)
                plate_result["treated"] = treated_feats
            else:
                print(f"  [WARN] Compound {compound_id}, plate {plate_id}: "
                      f"no treated images.")

            # ---- Control: encode then average ----
            if control_paths:
                control_feats = encode_paths(
                    control_paths, root_dir, model, device, batch_size,
                    transform=transform,
                    return_reg_tokens=return_reg_tokens, use_amp=use_amp,
                )  # (N_control, D)
                control_avg = control_feats.mean(dim=0)   # (D,)
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
        choices=["dino", "dino_lora", "dino_dora", "custom_vae"],
        help="Which model to use for encoding. Default: dino",
    )
    parser.add_argument(
        "--backbone", type=str, default="dino_vitb16",
        choices=list(BACKBONE_DIM.keys()),
        help="DINO/DINOv2 backbone variant. Default: dino_vitb16",
    )
    parser.add_argument(
        "--weights_path", type=str, default=None,
        help="Path to a fine-tuned checkpoint (.pt/.pth). "
             "Used with dino_lora / dino_dora to load adapted weights.",
    )
    parser.add_argument(
        "--return_reg_tokens", action="store_true", default=False,
        help="Use mean register tokens instead of CLS token as features "
             "(DINOv2 _reg backbones only).",
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

    # ---- Custom VAE hyper-parameters ----
    vae_grp = parser.add_argument_group("VAE (used when --model_type custom_vae)")
    vae_grp.add_argument(
        "--vae_checkpoint", type=str, default=None,
        help="Path to VAE checkpoint (.ckpt). Required for custom_vae.",
    )
    vae_grp.add_argument(
        "--vae_latent_dim", type=int, default=100,
        help="Latent dimension of the VAE encoder. Default: 100",
    )

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
        "--no_amp", action="store_true", default=False,
        help="Disable float16 automatic mixed precision (enabled by default on GPU).",
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
        vae_checkpoint=args.vae_checkpoint,
        vae_latent_dim=args.vae_latent_dim,
    )

    # ------------------------------------------------------------------
    # Validate --return_reg_tokens usage
    # ------------------------------------------------------------------
    if args.return_reg_tokens:
        if args.model_type == "custom_vae":
            raise ValueError("--return_reg_tokens is not supported with custom_vae.")
        if not args.backbone.endswith("_reg"):
            raise ValueError(
                f"--return_reg_tokens requires a DINOv2 _reg backbone "
                f"(e.g. dinov2_vitb14_reg), but got '{args.backbone}'."
            )

    # ------------------------------------------------------------------
    # Select transform
    # ------------------------------------------------------------------
    transform = VAE_TRANSFORM if args.model_type == "custom_vae" else DINO_TRANSFORM

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    embeddings = encode_metadata(
        metadata=metadata,
        root_dir=root_dir,
        model=model,
        device=device,
        batch_size=args.batch_size,
        return_reg_tokens=args.return_reg_tokens,
        use_amp=not args.no_amp,
        transform=transform,
    )

    # ------------------------------------------------------------------
    # Save — inject model name into the output filename
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    suffix = output_path.suffix or ".pt"
    stem = output_path.stem
    if args.model_type == "custom_vae":
        model_tag = f"custom_vae_dim{args.vae_latent_dim}"
    else:
        model_tag = f"{args.backbone}_{args.model_type}"
        if args.return_reg_tokens:
            model_tag += "_reg"
    output_path = output_path.with_name(f"{stem}_{model_tag}{suffix}")
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
