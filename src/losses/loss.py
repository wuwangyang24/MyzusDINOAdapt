"""Triple-Check Loss for bioassay consistency."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripleCheckLoss(nn.Module):
    """
    Triple-Check Loss for learning consistent treatment effects across bioassays.
    
    The logic:
    1. Extract Delta 1: Δ₁ = z_T1 - z_U1 (difference between treated and untreated from Bioassay 1)
    2. Extract Delta 2: Δ₂ = z_T2 - z_U2 (difference between treated and untreated from Bioassay 2)
    3. Loss: Minimize the distance between Δ₁ and Δ₂
    
    This ensures the model learns consistent treatment effects across different bioassays.
    """
    
    def __init__(
        self,
        distance_metric: str = "l2",
        temperature: float = 1.0,
        reduction: str = "mean",
        normalize_embeddings: bool = False,
    ):
        """
        Initialize Triple-Check Loss.
        
        Args:
            distance_metric: Distance metric to use ("l2", "cosine", "kl")
            temperature: Temperature for softening the loss
            reduction: Reduction method ("mean", "sum", "none")
            normalize_embeddings: If True, L2-normalize all embeddings before
                computing deltas. Recommended when control embeddings are
                frozen and treated embeddings come from an adapted backbone.
        """
        super().__init__()
        self.distance_metric = distance_metric
        self.temperature = temperature
        self.reduction = reduction
        self.normalize_embeddings = normalize_embeddings
        
        # Validate parameters
        if distance_metric not in ["l2", "cosine", "kl"]:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    def forward(
        self,
        features_t1: torch.Tensor,
        features_u1: torch.Tensor,
        features_t2: torch.Tensor,
        features_u2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Triple-Check Loss.
        
        Args:
            features_t1: Treated aphid features from Bioassay 1, shape (B, D)
            features_u1: Untreated aphid features from Bioassay 1, shape (B, D)
            features_t2: Treated aphid features from Bioassay 2, shape (B, D)
            features_u2: Untreated aphid features from Bioassay 2, shape (B, D)
            
        Returns:
            Loss value (scalar if reduction != "none")
        """
        # Control features: either pre-averaged (1, D) from file, or (N, D)
        # from backbone. mean(dim=0) handles both cases correctly.
        u1 = features_u1.mean(dim=0)  # (D,)
        u2 = features_u2.mean(dim=0)  # (D,)

        # Optionally L2-normalize all embeddings onto the unit hypersphere
        if self.normalize_embeddings:
            features_t1 = F.normalize(features_t1, dim=-1)
            features_t2 = F.normalize(features_t2, dim=-1)
            u1 = F.normalize(u1, dim=-1)
            u2 = F.normalize(u2, dim=-1)

        delta_1 = (features_t1 - u1).mean(dim=0, keepdim=True)  # Δ₁ = mean(z_T1 - z_U1)
        delta_2 = (features_t2 - u2).mean(dim=0, keepdim=True)  # Δ₂ = mean(z_T2 - z_U2)
        
        # Compute distance between deltas
        if self.distance_metric == "l2":
            loss = self._l2_distance(delta_1, delta_2)
        elif self.distance_metric == "cosine":
            loss = self._cosine_distance(delta_1, delta_2)
        elif self.distance_metric == "kl":
            loss = self._kl_divergence(delta_1, delta_2)
        
        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss
    
    def _l2_distance(self, delta_1: torch.Tensor, delta_2: torch.Tensor) -> torch.Tensor:
        """Compute L2 distance between deltas in float32 to prevent fp16 overflow."""
        diff = (delta_1 - delta_2).float()
        distance = torch.norm(diff, p=2, dim=1)
        return distance
    
    def _cosine_distance(self, delta_1: torch.Tensor, delta_2: torch.Tensor) -> torch.Tensor:
        """Compute cosine distance between deltas in float32 to prevent fp16 overflow."""
        delta_1 = delta_1.float()
        delta_2 = delta_2.float()
        eps = 1e-6
        # Clamp norms to avoid division by near-zero when deltas vanish
        norm_1 = delta_1.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
        norm_2 = delta_2.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
        delta_1_norm = delta_1 / norm_1
        delta_2_norm = delta_2 / norm_2

        # Cosine distance: 1 - cosine_similarity
        cosine_sim = torch.sum(delta_1_norm * delta_2_norm, dim=1)
        cosine_sim = cosine_sim.clamp(-1.0, 1.0)
        distance = 1.0 - cosine_sim
        return distance
    
    def _kl_divergence(self, delta_1: torch.Tensor, delta_2: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between softmax-normalized deltas."""
        delta_1 = delta_1.float()
        delta_2 = delta_2.float()
        # Convert to probabilities using softmax (along feature dimension)
        p = F.softmax(delta_1 / self.temperature, dim=1)
        q = F.softmax(delta_2 / self.temperature, dim=1)
        
        # KL divergence: sum(p * log(p/q))
        kl_div = torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10)), dim=1)
        return kl_div


class TripleCheckWithContrastiveLoss(nn.Module):
    """
    Triple-Check Loss combined with contrastive learning.
    
    Combines triple-check consistency loss with a contrastive term that
    encourages treated and untreated samples to be distinguishable.
    """
    
    def __init__(
        self,
        distance_metric: str = "l2",
        temperature: float = 1.0,
        contrastive_weight: float = 0.5,
        margin: float = 1.0,
        reduction: str = "mean",
    ):
        """
        Initialize Triple-Check + Contrastive Loss.
        
        Args:
            distance_metric: Distance metric for triple-check ("l2", "cosine", "kl")
            temperature: Temperature for loss
            contrastive_weight: Weight for contrastive loss term
            margin: Margin for contrastive loss
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.triple_check = TripleCheckLoss(
            distance_metric=distance_metric,
            temperature=temperature,
            reduction="none",
        )
        self.contrastive_weight = contrastive_weight
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        features_t1: torch.Tensor,
        features_u1: torch.Tensor,
        features_t2: torch.Tensor,
        features_u2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            features_t1: Treated aphid features from Bioassay 1, shape (B, D)
            features_u1: Untreated aphid features from Bioassay 1, shape (B, D)
            features_t2: Treated aphid features from Bioassay 2, shape (B, D)
            features_u2: Untreated aphid features from Bioassay 2, shape (B, D)
            
        Returns:
            Combined loss value
        """
        # Triple-check consistency loss
        tc_loss = self.triple_check(features_t1, features_u1, features_t2, features_u2)
        
        # Contrastive loss: encourage separation between treated and untreated
        contrastive_loss = self._contrastive_loss(
            features_t1, features_u1, features_t2, features_u2
        )
        
        # Combine losses
        total_loss = tc_loss + self.contrastive_weight * contrastive_loss
        
        if self.reduction == "mean":
            return total_loss.mean()
        elif self.reduction == "sum":
            return total_loss.sum()
        else:  # "none"
            return total_loss
    
    def _contrastive_loss(
        self,
        features_t1: torch.Tensor,
        features_u1: torch.Tensor,
        features_t2: torch.Tensor,
        features_u2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        Encourages treated-untreated pairs to be far apart.
        """
        # L2 distances
        dist_1 = torch.norm(features_t1 - features_u1, p=2, dim=1)
        dist_2 = torch.norm(features_t2 - features_u2, p=2, dim=1)
        
        # Hinge loss: max(0, margin - distance)
        # We want the distance to be at least margin
        loss_1 = F.relu(self.margin - dist_1)
        loss_2 = F.relu(self.margin - dist_2)
        
        return loss_1 + loss_2
