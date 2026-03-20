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
    dora_dropout: float = 0.1  # Dropout rate for DoRA layers
    bias: str = "none"  # Bias handling: "none", "all", "dora_only"


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
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False
        
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
