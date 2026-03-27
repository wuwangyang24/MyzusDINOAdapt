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


class TripleCheckBatchLoss(nn.Module):
    """
    Batch-level Triple-Check Loss with contrastive repulsion.

    For each compound *i* in the batch, two plate deltas are computed:
        Δ_i1 = mean(z_T1 - z_U1),  Δ_i2 = mean(z_T2 - z_U2)

    The loss has two terms:

    1. **Alignment** (per-compound): encourage Δ_i1 ≈ Δ_i2 for the same
       compound (same as the original TripleCheckLoss).
    2. **Repulsion** (cross-compound): push the average delta of compound *i*
       away from the average delta of compound *j* (j ≠ i) within the batch,
       preventing the trivial solution where all deltas collapse to the same
       direction.

    The final loss is:
        L = L_align + repulsion_weight * L_repel
    """

    def __init__(
        self,
        distance_metric: str = "l2",
        temperature: float = 0.1,
        repulsion_weight: float = 1.0,
        normalize_embeddings: bool = False,
        reduction: str = "mean",
        add_align_loss: bool = True,
    ):
        """
        Args:
            distance_metric: "l2" or "cosine" for the alignment term.
            temperature: Temperature for InfoNCE-style softmax.
            repulsion_weight: Weight λ for the repulsion term.
            normalize_embeddings: L2-normalize embeddings before computing deltas.
            reduction: "mean" or "sum".
            add_align_loss: If True, add explicit alignment loss on top of InfoNCE.
                If False, use pure InfoNCE (alignment is implicit in the positive pair).
        """
        super().__init__()
        self.distance_metric = distance_metric
        self.temperature = temperature
        self.repulsion_weight = repulsion_weight
        self.normalize_embeddings = normalize_embeddings
        self.reduction = reduction
        self.add_align_loss = add_align_loss

    def forward(
        self,
        deltas_plate1: torch.Tensor,
        deltas_plate2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute batch-level alignment + repulsion loss.

        Args:
            deltas_plate1: (K, D) — per-compound delta from plate 1.
            deltas_plate2: (K, D) — per-compound delta from plate 2.
                K = number of valid compounds in the batch.

        Returns:
            Tuple of (total_loss, align_loss, repel_loss) scalars.
        """
        K = deltas_plate1.shape[0]

        # ---- Alignment: same-compound plate deltas should match ----
        align_loss = torch.tensor(0.0, device=deltas_plate1.device)
        if self.add_align_loss:
            if self.distance_metric == "cosine":
                align = 1.0 - F.cosine_similarity(deltas_plate1, deltas_plate2, dim=-1)  # (K,)
            else:  # l2
                align = (deltas_plate1 - deltas_plate2).float().norm(p=2, dim=-1)  # (K,)

            if self.reduction == "mean":
                align_loss = align.mean()
            else:
                align_loss = align.sum()

        # ---- Repulsion: different-compound deltas should differ ----
        if K < 2 or self.repulsion_weight == 0.0:
            repel_loss = torch.tensor(0.0, device=deltas_plate1.device)
            return align_loss + self.repulsion_weight * repel_loss, align_loss, repel_loss

        # Use both plate deltas as anchors and targets (symmetric).
        # Anchors: [plate1; plate2] of each compound  →  (2K, D)
        # Positive for anchor i is the other plate of the same compound.
        # Negatives are all deltas from other compounds.
        all_deltas = torch.cat([deltas_plate1, deltas_plate2], dim=0)  # (2K, D)
        all_deltas = F.normalize(all_deltas.float(), dim=-1)

        # Similarity matrix: (2K, 2K)
        sim = torch.mm(all_deltas, all_deltas.T) / self.temperature

        # Mask out self-similarity on the diagonal — a vector dotted with
        # itself is always the largest entry and would dominate the softmax.
        sim.fill_diagonal_(float('-inf'))

        # Labels: anchor i (plate1 of compound j) pairs with i+K (plate2 of compound j)
        # and vice versa.
        labels = torch.cat([
            torch.arange(K, 2 * K, device=sim.device),  # plate1 → plate2
            torch.arange(0, K, device=sim.device),       # plate2 → plate1
        ])  # (2K,)

        repel_loss = F.cross_entropy(sim, labels)

        return align_loss + self.repulsion_weight * repel_loss, align_loss, repel_loss

    def compute_deltas(
        self,
        features_t: torch.Tensor,
        features_u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mean delta for a single compound-plate pair.

        Args:
            features_t: (N, D) treated features.
            features_u: (1, D) or (N, D) control features.

        Returns:
            (D,) delta vector.
        """
        u = features_u.mean(dim=0)
        if self.normalize_embeddings:
            features_t = F.normalize(features_t, dim=-1)
            u = F.normalize(u, dim=-1)
        return (features_t - u).mean(dim=0)


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
