"""PyTorch Lightning module for DINO LoRA with Triple-Check loss."""

import random
import warnings
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*")

from src.losses import TripleCheckLoss


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
            List of ``(D,)`` tensors — mean feature per group.
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
            results.append(all_feats[offset : offset + n].mean(dim=0))  # (D,)
            offset += n
        return results

    # ------------------------------------------------------------------
    # Shared forward / loss step
    # ------------------------------------------------------------------

    def _process_single_compound(self, compound):
        """Extract image groups for one compound, returning 4 tensors or None."""
        plates = compound["plates"]
        plate_names = list(plates.keys())

        if len(plate_names) < 2:
            return None

        if len(plate_names) > 2:
            p1, p2 = random.sample(plate_names, 2)
        else:
            p1, p2 = plate_names[0], plate_names[1]

        groups = []
        for pname in [p1, p2]:
            for stype in ["treated", "control"]:
                imgs = plates[pname][stype]
                if imgs.dim() == 5:
                    imgs = imgs.squeeze(0)
                groups.append(imgs)
        return groups  # [treated_p1, control_p1, treated_p2, control_p2]

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
        all_groups = []       # flat list of image tensors
        compound_indices = [] # which compound each group-of-4 belongs to
        for i, compound in enumerate(compounds):
            groups = self._process_single_compound(compound)
            if groups is not None:
                all_groups.extend(groups)
                compound_indices.append(i)

        if not compound_indices:
            return None

        # Single backbone forward pass for ALL images across ALL compounds
        all_feats = self._extract_features_batched(all_groups)

        # Compute per-compound losses
        losses = []
        delta_norms = []
        feat_norms = []
        for j in range(len(compound_indices)):
            base = j * 4
            feat_t1 = all_feats[base].unsqueeze(0)
            feat_u1 = all_feats[base + 1].unsqueeze(0)
            feat_t2 = all_feats[base + 2].unsqueeze(0)
            feat_u2 = all_feats[base + 3].unsqueeze(0)
            losses.append(self.loss_fn(feat_t1, feat_u1, feat_t2, feat_u2))

            # Diagnostics: delta and feature norms
            delta1 = (feat_t1 - feat_u1).float()
            delta2 = (feat_t2 - feat_u2).float()
            delta_norms.append(delta1.norm().item())
            delta_norms.append(delta2.norm().item())
            feat_norms.append(feat_t1.float().norm().item())
            feat_norms.append(feat_u1.float().norm().item())
            feat_norms.append(feat_t2.float().norm().item())
            feat_norms.append(feat_u2.float().norm().item())

        loss = torch.stack(losses).mean()

        # Log diagnostics on the same schedule as PL's log_every_n_steps
        if self.training:
            all_feat_tensor = torch.stack([f.float() for f in all_feats])
            self.log("diag/feat_norm_mean", sum(feat_norms) / len(feat_norms),
                     on_step=True, on_epoch=False, rank_zero_only=True)
            self.log("diag/delta_norm_mean", sum(delta_norms) / len(delta_norms),
                     on_step=True, on_epoch=False, rank_zero_only=True)
            self.log("diag/feat_std", all_feat_tensor.std(dim=0).mean().item(),
                     on_step=True, on_epoch=False, rank_zero_only=True)
            normed = torch.nn.functional.normalize(all_feat_tensor, dim=-1)
            cos_sim = (normed @ normed.T).fill_diagonal_(0)
            n = cos_sim.shape[0]
            self.log("diag/cos_sim_mean", cos_sim.sum().item() / (n * (n - 1)),
                     on_step=True, on_epoch=False, rank_zero_only=True)

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
        """Log gradient norms (frequency controlled by Trainer's log_every_n_steps)."""
        grad_norm_total = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                grad_norm_total += grad_norm ** 2
                self.log(f"grad_norm/{name}", grad_norm, on_step=True, on_epoch=False, rank_zero_only=True)
        self.log("grad_norm/total", grad_norm_total ** 0.5, on_step=True, on_epoch=False, prog_bar=True, rank_zero_only=True)

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
