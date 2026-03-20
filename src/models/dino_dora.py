"""DINO model with DoRA adaptation."""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torchvision.models as models
from .dora import DoRAConfig, DoRALinear


class DINOWithDoRA(nn.Module):
    """
    DINO model with DoRA adaptation.
    
    Supports adapting DINO and DINOv2 backbones (dino_vitb16, dinov2_vitb14, etc.)
    with dimensional and rank adaptation for efficient fine-tuning.
    """
    
    def __init__(
        self,
        backbone_name: str = "dino_vitb16",
        dora_config: Optional[DoRAConfig] = None,
        num_classes: Optional[int] = None,
        hub_source: str = "github",
        hub_source_dir: Optional[str] = None,
        weights_path: Optional[str] = None,
    ):
        """
        Initialize DINO with DoRA adaptation.
        
        Args:
            backbone_name: Name of DINO/DINOv2 backbone (dino_vitb16, dinov2_vitb14, etc.)
            dora_config: DoRA configuration
            num_classes: Number of output classes (optional, for classification head)
            hub_source: Source for torch.hub ("github" or "local"), defaults to "github"
            hub_source_dir: Local directory path when hub_source is "local"
            weights_path: Path to pretrained weights file for local source loading
        """
        super().__init__()
        
        # Load DINO / DINOv2 backbone
        hub_repo = (
            'facebookresearch/dinov2:main'
            if backbone_name.startswith('dinov2_')
            else 'facebookresearch/dino:main'
        )
        try:
            if hub_source.lower() == "local":
                if hub_source_dir is None:
                    raise ValueError(
                        "hub_source_dir must be provided when hub_source='local'"
                    )
                self.backbone = torch.hub.load(
                    hub_source_dir,
                    backbone_name,
                    source="local",
                    weights=weights_path,
                )
            else:
                self.backbone = torch.hub.load(
                    hub_repo,
                    backbone_name,
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {backbone_name} from {hub_source}. "
                f"Make sure DINO is available. Error: {e}"
            )
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        # Apply DoRA if config provided
        if dora_config is None:
            dora_config = DoRAConfig()
        
        self.dora_config = dora_config
        self._apply_dora()
        
        # Add classification head if needed
        if num_classes is not None:
            self._add_classification_head(num_classes)
    
    def _apply_dora(self) -> None:
        """Apply DoRA adaptation to the model."""
        # Freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Replace linear layers with DoRA layers in attention blocks
        self._replace_attention_layers()
        self._replace_mlp_layers()
    
    def _replace_attention_layers(self) -> None:
        """Replace attention projection layers with DoRA."""
        for name, module in self.backbone.named_modules():
            if hasattr(module, 'qkv'):  # Vision Transformer attention
                # Replace qkv projection
                if isinstance(module.qkv, nn.Linear):
                    original_qkv = module.qkv
                    module.qkv = DoRALinear(
                        original_qkv.in_features,
                        original_qkv.out_features,
                        r=self.dora_config.r,
                        dora_alpha=self.dora_config.dora_alpha,
                        dora_dropout=self.dora_config.dora_dropout,
                        bias=original_qkv.bias is not None,
                    )
                    # Copy original weights
                    module.qkv.linear.weight.data = original_qkv.weight.data.clone()
                    if original_qkv.bias is not None:
                        module.qkv.linear.bias.data = original_qkv.bias.data.clone()
            
            if hasattr(module, 'proj') and isinstance(module.proj, nn.Linear):
                # Replace output projection
                original_proj = module.proj
                module.proj = DoRALinear(
                    original_proj.in_features,
                    original_proj.out_features,
                    r=self.dora_config.r,
                    dora_alpha=self.dora_config.dora_alpha,
                    dora_dropout=self.dora_config.dora_dropout,
                    bias=original_proj.bias is not None,
                )
                module.proj.linear.weight.data = original_proj.weight.data.clone()
                if original_proj.bias is not None:
                    module.proj.linear.bias.data = original_proj.bias.data.clone()
    
    def _replace_mlp_layers(self) -> None:
        """Replace MLP layers with DoRA."""
        for name, module in self.backbone.named_modules():
            if hasattr(module, 'fc1') and hasattr(module, 'fc2'):
                # Replace fc1 (first linear layer in MLP)
                if isinstance(module.fc1, nn.Linear):
                    original_fc1 = module.fc1
                    module.fc1 = DoRALinear(
                        original_fc1.in_features,
                        original_fc1.out_features,
                        r=self.dora_config.r,
                        dora_alpha=self.dora_config.dora_alpha,
                        dora_dropout=self.dora_config.dora_dropout,
                        bias=original_fc1.bias is not None,
                    )
                    module.fc1.linear.weight.data = original_fc1.weight.data.clone()
                    if original_fc1.bias is not None:
                        module.fc1.linear.bias.data = original_fc1.bias.data.clone()
                # Replace fc2 (second linear layer in MLP)
                if isinstance(module.fc2, nn.Linear):
                    original_fc2 = module.fc2
                    module.fc2 = DoRALinear(
                        original_fc2.in_features,
                        original_fc2.out_features,
                        r=self.dora_config.r,
                        dora_alpha=self.dora_config.dora_alpha,
                        dora_dropout=self.dora_config.dora_dropout,
                        bias=original_fc2.bias is not None,
                    )
                    module.fc2.linear.weight.data = original_fc2.weight.data.clone()
                    if original_fc2.bias is not None:
                        module.fc2.linear.bias.data = original_fc2.bias.data.clone()
    
    def _add_classification_head(self, num_classes: int) -> None:
        """Add classification head on top of DINO backbone."""
        # Get the output dimension of DINO backbone
        if "vits14" in self.backbone_name:
            feat_dim = 384
        elif "vitb16" in self.backbone_name:
            feat_dim = 768
        elif "vitl14" in self.backbone_name:
            feat_dim = 1024
        else:
            feat_dim = 768  # Default
        
        self.classification_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Output tensor
        """
        # Get backbone features
        features = self.backbone(x)
        
        # Apply classification head if available
        if self.num_classes is not None:
            output = self.classification_head(features)
            return output
        
        return features
    
    def get_trainable_params(self):
        """Get list of trainable parameters (DoRA layers only)."""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
        return trainable_params
    
    def get_dora_params(self):
        """Get DoRA-specific parameters."""
        dora_params = []
        for name, param in self.named_parameters():
            if 'dora_' in name or 'magnitude' in name or 'classification_head' in name:
                if param.requires_grad:
                    dora_params.append((name, param))
        return dora_params
