"""PyTorch Lightning module for DINO LoRA with Triple-Check loss."""

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
        run_validation: bool = False,
    ):
        """
        Args:
            model: DINOWithLoRA or DINOWithDoRA instance.
            loss_fn: Loss function (default: TripleCheckLoss with L2 distance).
            learning_rate: AdamW learning rate.
            weight_decay: AdamW weight decay.
            run_validation: Whether to run validation steps.
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
        self.run_validation = run_validation
        # Save lr / weight_decay to hparams; skip non-serialisable objects
        self.save_hyperparameters(ignore=["model", "loss_fn"])

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Extract DINO backbone features, averaging over N untreated samples when needed.

        Args:
            img_tensor: ``(B, C, H, W)`` for a single sample, or
                        ``(N, B, C, H, W)`` for N untreated samples.

        Returns:
            Feature tensor of shape ``(B, D)``.
        """
        if img_tensor.dim() == 5:
            n_samples, batch_size = img_tensor.shape[:2]
            # Collapse N and B into a single batch dimension for one forward pass
            img_flat = img_tensor.view(-1, *img_tensor.shape[2:])   # (N*B, C, H, W)
            feats = self.model.backbone(img_flat)                    # (N*B, D)
            feats = feats.view(n_samples, batch_size, -1).mean(0)   # (B, D)
            return feats
        return self.model.backbone(img_tensor)

    # ------------------------------------------------------------------
    # Shared forward / loss step
    # ------------------------------------------------------------------

    def _shared_step(self, batch):
        img_t1, img_u1, img_t2, img_u2 = batch
        feat_t1 = self._extract_features(img_t1)
        feat_u1 = self._extract_features(img_u1)
        feat_t2 = self._extract_features(img_t2)
        feat_u2 = self._extract_features(img_u2)
        return self.loss_fn(feat_t1, feat_u1, feat_t2, feat_u2)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log(
            "train/loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if not self.run_validation:
            return
        loss = self._shared_step(batch)
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
