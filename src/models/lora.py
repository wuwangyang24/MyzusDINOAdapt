"""LoRA (Low-Rank Adaptation) implementation."""

from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    r: int = 8  # Rank of low-rank decomposition
    lora_alpha: float = 16.0  # Scaling factor
    target_modules: list = None  # Target modules to apply LoRA
    lora_dropout: float = 0.1  # Dropout rate for LoRA layers
    bias: str = "none"  # Bias handling: "none", "all", "lora_only"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class LoRALinear(nn.Module):
    """LoRA-adapted Linear layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Initialize LoRA-adapted linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            r: Rank of low-rank matrices
            lora_alpha: Scaling factor for LoRA outputs
            lora_dropout: Dropout rate for LoRA
            bias: Whether to use bias
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)
        
        # Original linear layer (frozen during training)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA weight matrices
        self.lora_a = nn.Linear(in_features, r, bias=False)
        self.lora_b = nn.Linear(r, out_features, bias=False)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)
        
        # Scaling factor
        self.scaling = lora_alpha / r
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with LoRA adaptation
        """
        # Original linear transformation
        out = self.linear(x)
        
        # Add LoRA adaptation
        lora_out = self.lora_b(self.lora_dropout(self.lora_a(x)))
        out = out + self.scaling * lora_out
        
        return out


def get_peft_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    Convert a model to use LoRA adaptation.
    
    Args:
        model: Original model
        config: LoRA configuration
        
    Returns:
        Model with LoRA layers
    """
    for name, module in model.named_modules():
        for target in config.target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Replace linear layer with LoRA linear layer
                parent_name, child_name = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]
                
                lora_linear = LoRALinear(
                    module.in_features,
                    module.out_features,
                    r=config.r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    bias=module.bias is not None,
                )
                
                # Copy original weights
                lora_linear.linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_linear.linear.bias.data = module.bias.data.clone()
                
                setattr(parent, child_name, lora_linear)
    
    return model
