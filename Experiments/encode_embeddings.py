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
import gc
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


class _ImageDataset(torch.utils.data.Dataset):
    """Lightweight map-style dataset that decodes images from disk."""

    def __init__(self, paths: List[str], root_dir: Path,
                 transform: transforms.Compose) -> None:
        self.paths = paths
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        full_path = self.root_dir / self.paths[idx]
        try:
            img = decode_image(str(full_path), mode=ImageReadMode.RGB)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image '{full_path}': {exc}"
            ) from exc
        return self.transform(img)


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

    # Lightning wraps the model under a 'model.' prefix — strip it
    state_dict = {
        k.replace("model.", "", 1) if k.startswith("model.") else k: v
        for k, v in state_dict.items()
    }

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
    lora_train_layernorm: bool = False,
    dora_r: int = 8,
    dora_alpha: float = 16.0,
    dora_dropout: float = 0.1,
    dora_train_layernorm: bool = False,
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
            train_layernorm=lora_train_layernorm,
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
            train_layernorm=dora_train_layernorm,
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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model = torch.compile(model)
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
    num_workers: int = 4,
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
        num_workers: DataLoader workers for parallel image loading.

    Returns:
        If return_reg_tokens is False:
            Tensor of shape (N, D) on CPU  (CLS token features).
        If return_reg_tokens is True:
            Tensor of shape (N, D) on CPU  (register tokens averaged
            over the N_reg dimension per image).
    """
    dataset = _ImageDataset(image_paths, root_dir, transform)
    pin = device.type == "cuda"
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=num_workers > 0,
    )

    all_features: List[torch.Tensor] = []
    amp_enabled = use_amp and device.type == "cuda"

    for batch in loader:
        batch = batch.to(device, non_blocking=pin)
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

    result = torch.cat(all_features, dim=0)                # (N, D)
    del all_features
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _encode_segment(
    image_paths: List[str],
    root_dir: Path,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    transform: transforms.Compose,
    return_reg_tokens: bool,
    use_amp: bool,
    num_workers: int,
) -> torch.Tensor:
    """Encode a single segment of images and return (N, D) CPU tensor."""
    dataset = _ImageDataset(image_paths, root_dir, transform)
    pin = device.type == "cuda"
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=False,
    )
    amp_enabled = use_amp and device.type == "cuda"
    parts: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_tensor in loader:
            batch_tensor = batch_tensor.to(device, non_blocking=pin)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                if return_reg_tokens:
                    backbone = _get_backbone(model)
                    out = backbone.forward_features(batch_tensor)
                    reg_tok = out["x_norm_regtokens"]
                    parts.append(reg_tok.mean(dim=1).float().cpu())
                else:
                    features = model(batch_tensor)
                    parts.append(features.float().cpu())

    result = torch.cat(parts, dim=0)
    del parts
    return result


def encode_metadata(
    metadata: List[Dict],
    root_dir: Path,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    return_reg_tokens: bool = False,
    use_amp: bool = True,
    transform: transforms.Compose = DINO_TRANSFORM,
    num_workers: int = 4,
) -> Dict:
    """
    Iterate over compounds and plates and build the embedding dictionary.

    Processes each segment (compound / plate / role) independently so that
    only one segment's worth of images lives in RAM at a time.

    Returns:
        Nested dict: compound_id -> plate_id -> {"treated": Tensor, "control": Tensor}
    """
    COMPOUND_KEY = "Compound"

    # ------------------------------------------------------------------
    # Collect segments (paths grouped by compound / plate / role)
    # ------------------------------------------------------------------
    segments: List[tuple] = []  # (compound_id, plate_id, role, paths)

    for compound_entry in metadata:
        compound_id = str(compound_entry[COMPOUND_KEY])
        plate_ids = [k for k in compound_entry.keys() if k != COMPOUND_KEY]

        for plate_id in plate_ids:
            plate_data = compound_entry[plate_id]
            treated_paths: List[str] = plate_data.get("treated", [])
            control_paths: List[str] = plate_data.get("control", [])

            if not treated_paths and not control_paths:
                print(f"  [WARN] Compound {compound_id}, plate {plate_id}: "
                      f"no images found — skipping.")
                continue

            if treated_paths:
                segments.append((compound_id, plate_id, "treated", treated_paths))
            else:
                print(f"  [WARN] Compound {compound_id}, plate {plate_id}: "
                      f"no treated images.")

            if control_paths:
                segments.append((compound_id, plate_id, "control", control_paths))
            else:
                print(f"  [WARN] Compound {compound_id}, plate {plate_id}: "
                      f"no control images.")

    total_images = sum(len(paths) for _, _, _, paths in segments)
    if total_images == 0:
        return {}
    print(f"Encoding {total_images} images across {len(segments)} segments...")

    # ------------------------------------------------------------------
    # Encode one segment at a time to keep RAM usage flat
    # ------------------------------------------------------------------
    result: Dict = {}

    for compound_id, plate_id, role, paths in tqdm(segments, desc="Segments"):
        feats = _encode_segment(
            paths, root_dir, model, device, batch_size,
            transform, return_reg_tokens, use_amp, num_workers,
        )

        if compound_id not in result:
            result[compound_id] = {}
        if plate_id not in result[compound_id]:
            result[compound_id][plate_id] = {}

        if role == "treated":
            result[compound_id][plate_id]["treated"] = feats       # (N, D)
        else:
            result[compound_id][plate_id]["control"] = feats.mean(dim=0)  # (D,)
            del feats
        gc.collect()

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
    lora_grp.add_argument("--lora_train_layernorm", action="store_true", help="Unfreeze LayerNorm params in LoRA model")

    # ---- DoRA hyper-parameters ----
    dora_grp = parser.add_argument_group("DoRA (used when --model_type dino_dora)")
    dora_grp.add_argument("--dora_r",       type=int,   default=8,    help="DoRA rank. Default: 8")
    dora_grp.add_argument("--dora_alpha",   type=float, default=16.0, help="DoRA alpha. Default: 16.0")
    dora_grp.add_argument("--dora_dropout", type=float, default=0.1,  help="DoRA dropout. Default: 0.1")
    dora_grp.add_argument("--dora_train_layernorm", action="store_true", help="Unfreeze LayerNorm params in DoRA model")

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
        "--num_workers", type=int, default=4,
        help="DataLoader workers for parallel image loading. Default: 4",
    )
    parser.add_argument(
        "--no_amp", action="store_true", default=False,
        help="Disable float16 automatic mixed precision (enabled by default on GPU).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device (e.g. 'cuda', 'cuda:1', 'cpu'). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--name_suffix", type=str, default=None,
        help="Custom suffix appended to the output filename (e.g. '_exp1').",
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
        lora_train_layernorm=getattr(args, 'lora_train_layernorm', False),
        dora_r=args.dora_r,
        dora_alpha=args.dora_alpha,
        dora_dropout=args.dora_dropout,
        dora_train_layernorm=getattr(args, 'dora_train_layernorm', False),
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
        num_workers=args.num_workers,
    )

    # Free model memory before saving
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
        if args.model_type == "dino_lora":
            model_tag += f"_r{args.lora_r}a{args.lora_alpha}"
            if getattr(args, 'lora_train_layernorm', False):
                model_tag += "_LN"
        elif args.model_type == "dino_dora":
            model_tag += f"_r{args.dora_r}a{args.dora_alpha}"
            if getattr(args, 'dora_train_layernorm', False):
                model_tag += "_LN"
        if args.return_reg_tokens:
            model_tag += "_reg"
    if args.name_suffix:
        model_tag += f"_{args.name_suffix}"
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
