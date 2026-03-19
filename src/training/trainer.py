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
        
        batch is a dict: {"id": [...], "plates": {"plate_name": {"treated": Tensor, "control": Tensor}, ...}}
        With batch_size=1, each tensor has shape (1, N, C, H, W) where the leading dim is the batch dim
        added by the DataLoader collate.
        
        For each compound, picks 2 plates and computes TripleCheckLoss between their deltas.
        If more than 2 plates, averages loss over all plate pairs.
        """
        plates = batch["plates"]
        plate_names = list(plates.keys())
        
        if len(plate_names) < 2:
            raise ValueError(
                f"Need at least 2 plates per compound for Triple-Check loss, got {len(plate_names)}: {plate_names}"
            )
        
        # Extract features for each plate (squeeze batch dim from collate)
        plate_feats = {}
        for pname in plate_names:
            treated = plates[pname]["treated"]
            control = plates[pname]["control"]
            # Remove collate batch dim: (1, N, C, H, W) -> (N, C, H, W)
            if treated.dim() == 5:
                treated = treated.squeeze(0)
            if control.dim() == 5:
                control = control.squeeze(0)
            plate_feats[pname] = {
                "treated": self._extract_features(treated),   # (D,)
                "control": self._extract_features(control),   # (D,)
            }
        
        # Compute loss over all plate pairs
        total_loss = torch.tensor(0.0, device=self.device)
        num_pairs = 0
        for i in range(len(plate_names)):
            for j in range(i + 1, len(plate_names)):
                p1, p2 = plate_names[i], plate_names[j]
                feat_t1 = plate_feats[p1]["treated"].unsqueeze(0)   # (1, D)
                feat_u1 = plate_feats[p1]["control"].unsqueeze(0)   # (1, D)
                feat_t2 = plate_feats[p2]["treated"].unsqueeze(0)   # (1, D)
                feat_u2 = plate_feats[p2]["control"].unsqueeze(0)   # (1, D)
                total_loss = total_loss + self.loss_fn(feat_t1, feat_u1, feat_t2, feat_u2)
                num_pairs += 1
        
        return total_loss / num_pairs

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
