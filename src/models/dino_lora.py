"""DINO model with LoRA adaptation."""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torchvision.models as models
from .lora import LoRAConfig, LoRALinear


class DINOWithLoRA(nn.Module):
    """
    DINO model with LoRA adaptation.
    
    Supports adapting DINO backbones (dino_vitb16, dino_vits14, etc.)
    with low-rank adaptation for efficient fine-tuning.
    """
    
    def __init__(
        self,
        backbone_name: str = "dino_vitb16",
        pretrained: bool = True,
        lora_config: Optional[LoRAConfig] = None,
        num_classes: Optional[int] = None,
    ):
        """
        Initialize DINO with LoRA adaptation.
        
        Args:
            backbone_name: Name of DINO backbone (dino_vitb16, dino_vits14, etc.)
            pretrained: Whether to load pretrained weights
            lora_config: LoRA configuration
            num_classes: Number of output classes (optional, for classification head)
        """
        super().__init__()
        
        # Load DINO backbone
        try:
            self.backbone = torch.hub.load(
                'facebookresearch/dino:main',
                backbone_name
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {backbone_name}. "
                f"Make sure DINO is available via torch.hub. Error: {e}"
            )
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        # Apply LoRA if config provided
        if lora_config is None:
            lora_config = LoRAConfig()
        
        self.lora_config = lora_config
        self._apply_lora()
        
        # Add classification head if needed
        if num_classes is not None:
            self._add_classification_head(num_classes)
    
    def _apply_lora(self) -> None:
        """Apply LoRA adaptation to the model."""
        # Freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Replace linear layers with LoRA layers in attention blocks
        self._replace_attention_layers()
        self._replace_mlp_layers()
        
        # Unfreeze normalization layers if in target modules
        if any('norm' in m for m in self.lora_config.target_modules):
            self._unfreeze_normalization_layers()
    
    def _replace_attention_layers(self) -> None:
        """Replace attention projection layers with LoRA."""
        for name, module in self.backbone.named_modules():
            if hasattr(module, 'qkv'):  # Vision Transformer attention
                # Replace qkv projection
                if isinstance(module.qkv, nn.Linear):
                    original_qkv = module.qkv
                    module.qkv = LoRALinear(
                        original_qkv.in_features,
                        original_qkv.out_features,
                        r=self.lora_config.r,
                        lora_alpha=self.lora_config.lora_alpha,
                        lora_dropout=self.lora_config.lora_dropout,
                        bias=original_qkv.bias is not None,
                    )
                    # Copy original weights
                    module.qkv.linear.weight.data = original_qkv.weight.data.clone()
                    if original_qkv.bias is not None:
                        module.qkv.linear.bias.data = original_qkv.bias.data.clone()
            
            if hasattr(module, 'proj') and isinstance(module, type(module)):
                # Replace output projection
                if isinstance(module.proj, nn.Linear):
                    original_proj = module.proj
                    module.proj = LoRALinear(
                        original_proj.in_features,
                        original_proj.out_features,
                        r=self.lora_config.r,
                        lora_alpha=self.lora_config.lora_alpha,
                        lora_dropout=self.lora_config.lora_dropout,
                        bias=original_proj.bias is not None,
                    )
                    module.proj.linear.weight.data = original_proj.weight.data.clone()
                    if original_proj.bias is not None:
                        module.proj.linear.bias.data = original_proj.bias.data.clone()
    
    def _replace_mlp_layers(self) -> None:
        """Replace MLP layers with LoRA."""
        for name, module in self.backbone.named_modules():
            if hasattr(module, 'fc1') and hasattr(module, 'fc2'):
                # Replace fc1 (first linear layer in MLP)
                if isinstance(module.fc1, nn.Linear):
                    original_fc1 = module.fc1
                    module.fc1 = LoRALinear(
                        original_fc1.in_features,
                        original_fc1.out_features,
                        r=self.lora_config.r,
                        lora_alpha=self.lora_config.lora_alpha,
                        lora_dropout=self.lora_config.lora_dropout,
                        bias=original_fc1.bias is not None,
                    )
                    module.fc1.linear.weight.data = original_fc1.weight.data.clone()
                    if original_fc1.bias is not None:
                        module.fc1.linear.bias.data = original_fc1.bias.data.clone()
    
    def _unfreeze_normalization_layers(self) -> None:
        """Unfreeze and make trainable all normalization layers."""
        for name, param in self.backbone.named_parameters():
            if 'norm' in name:
                param.requires_grad = True
    
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
        """Get list of trainable parameters (LoRA layers only)."""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
        return trainable_params
    
    def get_lora_params(self):
        """Get LoRA-specific parameters."""
        lora_params = []
        for name, param in self.named_parameters():
            if 'lora_' in name or 'classification_head' in name:
                if param.requires_grad:
                    lora_params.append((name, param))
        return lora_params
