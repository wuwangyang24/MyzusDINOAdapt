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
    lora_dropout: float = 0.1  # Dropout rate for LoRA layers
    bias: str = "none"  # Bias handling: "none", "all", "lora_only"
    train_layernorm: bool = False  # Unfreeze LayerNorm parameters during training


class LoRALinear(nn.Module):
    """LoRA-adapted Linear layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.,
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
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False
        
        # LoRA weight matrices
        self.lora_a = nn.Linear(in_features, r, bias=False)
        self.lora_b = nn.Linear(r, out_features, bias=False)
        
        # Ensure LoRA parameters are trainable
        self.lora_a.weight.requires_grad = True
        self.lora_b.weight.requires_grad = True
        
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
