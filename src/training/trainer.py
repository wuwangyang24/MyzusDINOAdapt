"""PyTorch Lightning module for DINO LoRA with Triple-Check loss."""

import random
import warnings
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*")

from src.losses import TripleCheckLoss, TripleCheckBatchLoss, DCL


class TripleCheckModule(pl.LightningModule):
    """
    PyTorch Lightning module for DINO with LoRA/DoRA adaptation using Triple-Check loss.

    Device placement, logging, checkpointing, mixed precision, and multi-GPU (DDP)
    are all delegated to the ``pytorch_lightning.Trainer`` — this class only contains
    the forward logic, loss computation, and optimizer configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_samples: int = 4,
        warmup_steps: int = 0,
        total_steps: int = 0,
    ):
        """
        Args:
            model: DINOWithLoRA or DINOWithDoRA instance.
            loss_fn: Loss function (default: TripleCheckLoss with L2 distance).
            learning_rate: AdamW learning rate.
            weight_decay: AdamW weight decay.
            max_samples: Max images per plate per type per step.
            warmup_steps: Number of linear warmup steps.
            total_steps: Total training steps (for cosine decay).
        """
        super().__init__()
        self.model = model
        self.loss_fn = (
            loss_fn
            if loss_fn is not None
            else TripleCheckLoss(distance_metric="l2", reduction="mean")
        )
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.max_samples = max_samples
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        # Save lr / weight_decay to hparams; skip non-serialisable objects
        self.save_hyperparameters(ignore=["model", "loss_fn"])

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features_batched(self, all_images: list, chunk_size: int = 32) -> list:
        """Extract features for multiple image groups via the backbone.

        Concatenates groups then processes in GPU-friendly chunks to avoid
        OOM when the total image count is large (e.g. 120 with max_samples=30).

        Args:
            all_images: List of ``(N_i, C, H, W)`` tensors, one per group.
            chunk_size: Max images per backbone forward pass.

        Returns:
            List of ``(N_i, D)`` tensors — per-image features for each group.
        """
        sizes = [t.shape[0] for t in all_images]
        big_batch = torch.cat(all_images, dim=0)        # (sum(N_i), C, H, W)
        total = big_batch.shape[0]

        # Forward in chunks
        if total <= chunk_size:
            all_feats = self.model.backbone(big_batch)
        else:
            feat_parts = []
            for start in range(0, total, chunk_size):
                feat_parts.append(self.model.backbone(big_batch[start:start + chunk_size]))
            all_feats = torch.cat(feat_parts, dim=0)    # (total, D)

        results = []
        offset = 0
        for n in sizes:
            results.append(all_feats[offset : offset + n])  # (N_i, D)
            offset += n
        return results

    # ------------------------------------------------------------------
    # Shared forward / loss step
    # ------------------------------------------------------------------

    def _process_single_compound(self, compound):
        """Extract image groups for one compound, returning 4 tensors or None.

        Returns:
            None if the compound has fewer than 2 plates, otherwise a tuple
            ``(groups, is_precomputed)`` where *groups* is
            ``[treated_p1, control_p1, treated_p2, control_p2]`` and
            *is_precomputed* is a parallel boolean list indicating whether
            each element is a pre-computed feature vector (``True``) or a
            batch of images that still needs a backbone forward pass
            (``False``).
        """
        plates = compound["plates"]
        plate_names = list(plates.keys())

        if len(plate_names) < 2:
            return None

        if len(plate_names) > 2:
            p1, p2 = random.sample(plate_names, 2)
        else:
            p1, p2 = plate_names[0], plate_names[1]

        groups = []
        is_precomputed = []
        for pname in [p1, p2]:
            for stype in ["treated", "control"]:
                data = plates[pname][stype]
                if data.dim() <= 2:
                    # Pre-computed feature embedding: (D,) or (N, D)
                    is_precomputed.append(True)
                    groups.append(data)
                else:
                    if data.dim() == 5:
                        data = data.squeeze(0)
                    is_precomputed.append(False)
                    groups.append(data)
        return groups, is_precomputed

    def _shared_step(self, batch, batch_idx=None):
        """Process a batch of compounds from CompoundPlateDataset.

        Accepts either a single compound dict (batch_size=1 with default
        collate) or a list of compound dicts (batch_size>1 with
        compound_collate_fn). All images across compounds are batched into
        a single backbone forward pass for maximum GPU utilisation.
        """
        # Normalise to list of compounds
        if isinstance(batch, dict):
            compounds = [batch]
        else:
            compounds = batch

        # Collect image groups from all valid compounds
        all_image_groups = []   # only groups that need backbone encoding
        group_layout = []       # per compound: list of ('image', idx) or ('precomputed', tensor)
        compound_indices = []   # which compound each group-of-4 belongs to
        for i, compound in enumerate(compounds):
            result = self._process_single_compound(compound)
            if result is None:
                continue
            groups, is_precomputed = result
            layout = []
            for g, is_pre in zip(groups, is_precomputed):
                if is_pre:
                    layout.append(('precomputed', g))
                else:
                    layout.append(('image', len(all_image_groups)))
                    all_image_groups.append(g)
            group_layout.append(layout)
            compound_indices.append(i)

        if not compound_indices:
            return None

        # Single backbone forward pass for image groups only
        encoded_feats = self._extract_features_batched(all_image_groups) if all_image_groups else []

        # Assemble all features in original order
        all_feats = []
        for layout in group_layout:
            for kind, data in layout:
                if kind == 'image':
                    all_feats.append(encoded_feats[data])
                else:
                    feat = data.detach().to(self.device)
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(0)  # (D,) -> (1, D)
                    all_feats.append(feat)

        # Compute per-compound deltas
        deltas_p1 = []   # plate-1 delta per compound
        deltas_p2 = []   # plate-2 delta per compound
        treated_stds = []
        for j in range(len(compound_indices)):
            base = j * 4
            feat_t1 = all_feats[base]      # (N, D)
            feat_u1 = all_feats[base + 1]  # (1, D)
            feat_t2 = all_feats[base + 2]  # (N, D)
            feat_u2 = all_feats[base + 3]  # (1, D)

            if isinstance(self.loss_fn, (TripleCheckBatchLoss, DCL)):
                deltas_p1.append(self.loss_fn.compute_deltas(feat_t1, feat_u1))
                deltas_p2.append(self.loss_fn.compute_deltas(feat_t2, feat_u2))
            else:
                deltas_p1.append((feat_t1.float() - feat_u1.float().mean(dim=0)).mean(dim=0))
                deltas_p2.append((feat_t2.float() - feat_u2.float().mean(dim=0)).mean(dim=0))

            # Per-compound std of treated embeddings (across both plates)
            all_treated = torch.cat([feat_t1.float(), feat_t2.float()], dim=0)
            treated_stds.append(all_treated.std(dim=0).mean().item())

        deltas_p1_stack = torch.stack(deltas_p1, dim=0)  # (K, D)
        deltas_p2_stack = torch.stack(deltas_p2, dim=0)  # (K, D)

        # Compute loss
        if isinstance(self.loss_fn, DCL):
            # DCL expects L2-normalized embeddings of shape (K, D)
            z1 = F.normalize(deltas_p1_stack.float(), dim=-1)
            z2 = F.normalize(deltas_p2_stack.float(), dim=-1)
            loss_12 = self.loss_fn(z1, z2)
            loss_21 = self.loss_fn(z2, z1)
            loss = (loss_12 + loss_21) / 2
            if self.training:
                self.log("train/dcl_loss_12", loss_12.detach(),
                         on_step=True, on_epoch=True, rank_zero_only=True,
                         batch_size=len(compound_indices))
                self.log("train/dcl_loss_21", loss_21.detach(),
                         on_step=True, on_epoch=True, rank_zero_only=True,
                         batch_size=len(compound_indices))
        elif isinstance(self.loss_fn, TripleCheckBatchLoss):
            loss, align_loss, repel_loss = self.loss_fn(deltas_p1_stack, deltas_p2_stack)
            if self.training:
                self.log("train/align_loss", align_loss.detach(),
                         on_step=True, on_epoch=True, rank_zero_only=True,
                         batch_size=len(compound_indices))
                self.log("train/repel_loss", repel_loss.detach(),
                         on_step=True, on_epoch=True, rank_zero_only=True,
                         batch_size=len(compound_indices))
        else:
            # Legacy per-compound loss
            losses = []
            for j in range(len(compound_indices)):
                base = j * 4
                losses.append(self.loss_fn(
                    all_feats[base], all_feats[base + 1],
                    all_feats[base + 2], all_feats[base + 3],
                ))
            loss = torch.stack(losses).mean()

        # Log diagnostics on the same schedule as PL's log_every_n_steps
        if self.training:
            with torch.no_grad():
                all_deltas = torch.cat([deltas_p1_stack.detach(), deltas_p2_stack.detach()], dim=0)  # (2K, D)
                delta_norms = all_deltas.float().norm(p=2, dim=1)
                self.log("diag/delta_norm_mean", delta_norms.mean().item(),
                         on_step=True, on_epoch=False, rank_zero_only=True,
                         batch_size=len(compound_indices))
                self.log("diag/delta_norm_std", delta_norms.std().item(),
                         on_step=True, on_epoch=False, rank_zero_only=True,
                         batch_size=len(compound_indices))

                # Mean norm of treated embeddings across all compounds
                treated_norms = []
                for j in range(len(compound_indices)):
                    base = j * 4
                    treated_norms.append(all_feats[base].float().norm(p=2, dim=1).mean().item())
                    treated_norms.append(all_feats[base + 2].float().norm(p=2, dim=1).mean().item())
                self.log("diag/treated_norm_mean", sum(treated_norms) / len(treated_norms),
                         on_step=True, on_epoch=False, rank_zero_only=True,
                         batch_size=len(compound_indices))

                # Per-compound: cosine sim between mean embeddings of the two plates
                intra_cos_sims = []
                plate_means = []  # (K, D) — per-compound average of both plate means
                for j in range(len(compound_indices)):
                    base = j * 4
                    mean_p1 = all_feats[base].float().mean(dim=0)      # (D,)
                    mean_p2 = all_feats[base + 2].float().mean(dim=0)  # (D,)
                    intra_cos_sims.append(
                        F.cosine_similarity(mean_p1.unsqueeze(0), mean_p2.unsqueeze(0)).item()
                    )
                    plate_means.append((mean_p1 + mean_p2) / 2.0)

                # Per-compound: mean cosine sim of compound avg vs all other compounds' avg
                inter_cos_sims = []
                if len(plate_means) > 1:
                    plate_means_stack = torch.stack(plate_means, dim=0)  # (K, D)
                    plate_means_norm = F.normalize(plate_means_stack, dim=-1)
                    cos_matrix = torch.mm(plate_means_norm, plate_means_norm.T)  # (K, K)
                    for j in range(len(compound_indices)):
                        mask = torch.ones(len(compound_indices), dtype=torch.bool, device=cos_matrix.device)
                        mask[j] = False
                        inter_cos_sims.append(cos_matrix[j][mask].mean().item())

                # ── Delta-based cosine similarities (more informative than raw embeddings) ──
                # Intra: cos sim between Δ_i1 and Δ_i2 for the same compound
                intra_delta_cos_sims = F.cosine_similarity(
                    deltas_p1_stack.float(), deltas_p2_stack.float(), dim=-1
                ).tolist()  # (K,)

                # Inter: cos sim between mean deltas of different compounds
                inter_delta_cos_sims = []
                K = deltas_p1_stack.shape[0]
                if K > 1:
                    mean_deltas = ((deltas_p1_stack + deltas_p2_stack) / 2.0).float()  # (K, D)
                    mean_deltas_norm = F.normalize(mean_deltas, dim=-1)
                    delta_cos_matrix = torch.mm(mean_deltas_norm, mean_deltas_norm.T)  # (K, K)
                    for j in range(K):
                        mask = torch.ones(K, dtype=torch.bool, device=delta_cos_matrix.device)
                        mask[j] = False
                        inter_delta_cos_sims.append(delta_cos_matrix[j][mask].mean().item())

                # Log delta-based metrics via PL (scalars)
                if intra_delta_cos_sims:
                    self.log("diag/intra_delta_cos_sim_mean",
                             sum(intra_delta_cos_sims) / len(intra_delta_cos_sims),
                             on_step=True, on_epoch=False, rank_zero_only=True,
                             batch_size=len(compound_indices))
                if inter_delta_cos_sims:
                    self.log("diag/inter_delta_cos_sim_mean",
                             sum(inter_delta_cos_sims) / len(inter_delta_cos_sims),
                             on_step=True, on_epoch=False, rank_zero_only=True,
                             batch_size=len(compound_indices))

                # Log per-compound treated std as a W&B histogram (vertical distribution)
                try:
                    import wandb
                    if wandb.run is not None:
                        log_dict = {
                            "diag/treated_std_distribution": wandb.Histogram(treated_stds),
                            "diag/treated_std_mean": sum(treated_stds) / len(treated_stds),
                            "diag/intra_compound_cos_sim": wandb.Histogram(intra_cos_sims),
                            "diag/intra_compound_cos_sim_mean": sum(intra_cos_sims) / len(intra_cos_sims),
                        }
                        if inter_cos_sims:
                            log_dict["diag/inter_compound_cos_sim"] = wandb.Histogram(inter_cos_sims)
                            log_dict["diag/inter_compound_cos_sim_mean"] = sum(inter_cos_sims) / len(inter_cos_sims)
                        # Delta-based histograms
                        log_dict["diag/intra_delta_cos_sim"] = wandb.Histogram(intra_delta_cos_sims)
                        if inter_delta_cos_sims:
                            log_dict["diag/inter_delta_cos_sim"] = wandb.Histogram(inter_delta_cos_sims)
                        wandb.log(log_dict, commit=False)
                except ImportError:
                    pass

        return loss

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        if loss is None:
            return None
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        bs = len(batch) if isinstance(batch, list) else 1
        self.log(
            "train/loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
            batch_size=bs,
        )
        return loss

    def on_after_backward(self):
        """Log total gradient norm (frequency controlled by Trainer's log_every_n_steps)."""
        grad_norm_total = 0.0
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm_total += param.grad.norm(2).item() ** 2
        self.log("grad_norm/total", grad_norm_total ** 0.5, on_step=True, on_epoch=False, prog_bar=True, rank_zero_only=True, batch_size=1)

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        if loss is None:
            return None
        bs = len(batch) if isinstance(batch, list) else 1
        self.log(
            "val/loss", loss,
            on_epoch=True, prog_bar=True, sync_dist=True,
            batch_size=bs,
        )

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError(
                "No trainable parameters found. "
                "Make sure LoRA/DoRA layers have requires_grad=True."
            )
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.total_steps <= 0:
            return optimizer

        from torch.optim.lr_scheduler import LambdaLR
        import math

        warmup = self.warmup_steps
        total = self.total_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup:
                # Linear warmup: 0 → 1
                return max(current_step / max(1, warmup), 1e-7)
            # Cosine decay: 1 → 0
            progress = (current_step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "strict": False,
            },
        }
