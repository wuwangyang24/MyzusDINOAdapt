"""DoRA (Dimensional and Rank Adaptation) implementation."""

from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DoRAConfig:
    """Configuration for DoRA adaptation."""
    r: int = 8  # Rank of low-rank decomposition
    dora_alpha: float = 16.0  # Scaling factor
    target_modules: list = None  # Target modules to apply DoRA
    dora_dropout: float = 0.1  # Dropout rate for DoRA layers
    bias: str = "none"  # Bias handling: "none", "all", "dora_only"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class DoRALinear(nn.Module):
    """DoRA-adapted Linear layer (Dimensional and Rank Adaptation)."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        dora_alpha: float = 16.0,
        dora_dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Initialize DoRA-adapted linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            r: Rank of low-rank matrices
            dora_alpha: Scaling factor for DoRA outputs
            dora_dropout: Dropout rate for DoRA
            bias: Whether to use bias
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.dora_alpha = dora_alpha
        self.dora_dropout = nn.Dropout(dora_dropout)
        
        # Original linear layer (frozen during training)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # DoRA weight matrices
        self.dora_a = nn.Linear(in_features, r, bias=False)
        self.dora_b = nn.Linear(r, out_features, bias=False)
        
        # Dimensional vector for DoRA (column-wise scaling)
        self.magnitude = nn.Parameter(torch.ones(out_features))
        
        # Ensure DoRA parameters are trainable
        self.dora_a.weight.requires_grad = True
        self.dora_b.weight.requires_grad = True
        self.magnitude.requires_grad = True
        
        # Initialize DoRA weights
        nn.init.kaiming_uniform_(self.dora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.dora_b.weight)
        
        # Scaling factor
        self.scaling = dora_alpha / r
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with DoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with DoRA adaptation
        """
        # Original linear transformation
        out = self.linear(x)
        
        # Add DoRA adaptation with dimensional scaling
        dora_out = self.dora_b(self.dora_dropout(self.dora_a(x)))
        dora_scaled = dora_out * self.magnitude * self.scaling
        out = out + dora_scaled
        
        return out


def get_peft_model_dora(model: nn.Module, config: DoRAConfig) -> nn.Module:
    """
    Convert a model to use DoRA adaptation.
    
    Args:
        model: Original model
        config: DoRA configuration
        
    Returns:
        Model with DoRA layers
    """
    for name, module in model.named_modules():
        for target in config.target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Replace linear layer with DoRA linear layer
                parent_name, child_name = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]
                
                dora_linear = DoRALinear(
                    module.in_features,
                    module.out_features,
                    r=config.r,
                    dora_alpha=config.dora_alpha,
                    dora_dropout=config.dora_dropout,
                    bias=module.bias is not None,
                )
                
                # Copy original weights
                dora_linear.linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    dora_linear.linear.bias.data = module.bias.data.clone()
                
                setattr(parent, child_name, dora_linear)
    
    return model
