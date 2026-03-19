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

    def _extract_features(self, img_tensor: torch.Tensor, chunk_size: int = 4) -> torch.Tensor:
        """Extract DINO backbone features, averaging over N samples.

        Processes images in chunks to avoid OOM on large sample sets.

        Args:
            img_tensor: ``(N, C, H, W)`` for N samples from one plate/type.
            chunk_size: Number of images to process per forward pass.

        Returns:
            Feature tensor of shape ``(D,)`` — mean over N samples.
        """
        n = img_tensor.shape[0]
        if n <= chunk_size:
            feats = self.model.backbone(img_tensor)  # (N, D)
            return feats.mean(dim=0)  # (D,)
        
        # Process in chunks and accumulate weighted sum
        feat_sum = None
        for start in range(0, n, chunk_size):
            chunk = img_tensor[start : start + chunk_size]
            chunk_feats = self.model.backbone(chunk)  # (chunk_len, D)
            chunk_mean = chunk_feats.sum(dim=0)       # (D,)
            if feat_sum is None:
                feat_sum = chunk_mean
            else:
                feat_sum = feat_sum + chunk_mean
        return feat_sum / n  # (D,)

    # ------------------------------------------------------------------
    # Shared forward / loss step
    # ------------------------------------------------------------------

    def _shared_step(self, batch):
        """Process batch from CompoundPlateDataset.
        
        Randomly samples 2 plates and a small number of images per plate
        to keep memory bounded.
        """
        plates = batch["plates"]
        plate_names = list(plates.keys())
        
        if len(plate_names) < 2:
            return None
        
        # Randomly sample 2 plates
        p1, p2 = random.sample(plate_names, 2)
        
        max_samples = self.max_samples
        
        selected = {}
        for pname in [p1, p2]:
            treated = plates[pname]["treated"]
            control = plates[pname]["control"]
            # Remove collate batch dim: (1, N, C, H, W) -> (N, C, H, W)
            if treated.dim() == 5:
                treated = treated.squeeze(0)
            if control.dim() == 5:
                control = control.squeeze(0)
            # Subsample images to limit memory
            if treated.shape[0] > max_samples:
                idx = torch.randperm(treated.shape[0])[:max_samples]
                treated = treated[idx]
            if control.shape[0] > max_samples:
                idx = torch.randperm(control.shape[0])[:max_samples]
                control = control[idx]
            # Extract features
            selected[pname] = {
                "treated": self._extract_features(treated),
                "control": self._extract_features(control),
            }
        
        feat_t1 = selected[p1]["treated"].unsqueeze(0)   # (1, D)
        feat_u1 = selected[p1]["control"].unsqueeze(0)    # (1, D)
        feat_t2 = selected[p2]["treated"].unsqueeze(0)    # (1, D)
        feat_u2 = selected[p2]["control"].unsqueeze(0)    # (1, D)
        
        return self.loss_fn(feat_t1, feat_u1, feat_t2, feat_u2)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            # Return a zero loss that still has grad to avoid corrupting the model
            zero = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.log(
                "train/loss", 0.0,
                on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
            )
            return zero
        self.log(
            "train/loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        return loss

    def on_after_backward(self):
        """Log gradient norms every 1000 steps."""
        if self.global_step % 200 != 0:
            return
        grad_norm_total = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                grad_norm_total += grad_norm ** 2
                self.log(f"grad_norm/{name}", grad_norm, on_step=True, on_epoch=False)
        self.log("grad_norm/total", grad_norm_total ** 0.5, on_step=True, on_epoch=False, prog_bar=True)

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
