"""PyTorch Lightning module for DINO LoRA with Triple-Check loss."""

import random
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

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
    ):
        """
        Args:
            model: DINOWithLoRA or DINOWithDoRA instance.
            loss_fn: Loss function (default: TripleCheckLoss with L2 distance).
            learning_rate: AdamW learning rate.
            weight_decay: AdamW weight decay.
            max_samples: Max images per plate per type per step.
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

    def _shared_step(self, batch):
        """Process batch from CompoundPlateDataset.
        
        Batches all images into a single backbone forward pass for
        maximum GPU utilisation.
        """
        plates = batch["plates"]
        plate_names = list(plates.keys())
        
        if len(plate_names) < 2:
            return None
        
        # Pick 2 plates
        if len(plate_names) > 2:
            p1, p2 = random.sample(plate_names, 2)
        else:
            p1, p2 = plate_names[0], plate_names[1]
        
        # Gather all image tensors, removing collate batch dim
        groups = []  # order: treated_p1, control_p1, treated_p2, control_p2
        for pname in [p1, p2]:
            for stype in ["treated", "control"]:
                imgs = plates[pname][stype]
                if imgs.dim() == 5:
                    imgs = imgs.squeeze(0)
                groups.append(imgs)
        
        # Single backbone forward pass
        feats = self._extract_features_batched(groups)
        feat_t1, feat_u1, feat_t2, feat_u2 = [f.unsqueeze(0) for f in feats]
        
        return self.loss_fn(feat_t1, feat_u1, feat_t2, feat_u2)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        if loss is None:
            return None
        self.log(
            "train/loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        return loss

    def on_after_backward(self):
        """Log gradient norms every 200 steps."""
        if self.global_step % 200 != 0:
            return
        grad_norm_total = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                grad_norm_total += grad_norm ** 2
                self.log(f"grad_norm/{name}", grad_norm, on_step=True, on_epoch=False, rank_zero_only=True)
        self.log("grad_norm/total", grad_norm_total ** 0.5, on_step=True, on_epoch=False, prog_bar=True, rank_zero_only=True)

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        if loss is None:
            return None
        self.log(
            "val/loss", loss,
            on_epoch=True, prog_bar=True, sync_dist=True,
        )

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError(
                "No trainable parameters found. "
                "Make sure LoRA/DoRA layers have requires_grad=True."
            )
        return optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
