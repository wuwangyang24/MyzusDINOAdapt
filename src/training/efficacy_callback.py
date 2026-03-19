"""Lightning Callback: run efficacy classification as a validation metric.

After each validation epoch the callback:
  1. Encodes training & inference embeddings using the current backbone.
  2. Trains a lightweight XGBoost classifier on the training embeddings.
  3. Evaluates on the inference embeddings.
  4. Logs AUROC, balanced accuracy, and weighted F1 to the Lightning loggers.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.io import decode_image, ImageReadMode
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False


# ── Shared encoding helpers ──────────────────────────────────────────────────

DINO_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])


@torch.no_grad()
def _encode_paths(
    image_paths: List[str],
    root_dir: Path,
    backbone: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode images → (N, D) CPU tensor using the given backbone."""
    all_feats: List[torch.Tensor] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        tensors = []
        for rp in batch_paths:
            img = decode_image(str(root_dir / rp), mode=ImageReadMode.RGB)
            tensors.append(DINO_TRANSFORM(img))
        batch = torch.stack(tensors).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            feats = backbone(batch)
        all_feats.append(feats.float().cpu())
    return torch.cat(all_feats, dim=0)


def _encode_metadata(
    metadata: List[Dict],
    root_dir: Path,
    backbone: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> Dict:
    """Encode a metadata list → nested dict (same format as encode_embeddings)."""
    result: Dict = {}
    for entry in metadata:
        cid = str(entry["Compound"])
        result[cid] = {}
        for key, plate_data in entry.items():
            if key == "Compound" or not isinstance(plate_data, dict):
                continue
            plate_result: Dict[str, torch.Tensor] = {}
            treated_paths = plate_data.get("treated", [])
            control_paths = plate_data.get("control", [])
            if treated_paths:
                plate_result["treated"] = _encode_paths(
                    treated_paths, root_dir, backbone, device, batch_size)
            if control_paths:
                ctrl = _encode_paths(
                    control_paths, root_dir, backbone, device, batch_size)
                plate_result["control"] = ctrl.mean(dim=0)
            if plate_result:
                result[cid][key] = plate_result
    return result


def _build_mean_features(
    embeddings: Dict,
    cid2label: Dict[str, int],
):
    """Build (N, D) mean-pooled features + labels from embedding dict."""
    X, y, cids = [], [], []
    for cid, plates in embeddings.items():
        if cid not in cid2label:
            continue
        latents = []
        for pdata in plates.values():
            t = pdata.get("treated")
            if t is not None and t.numel() > 0:
                latents.append(t.float())
        if not latents:
            continue
        mean = torch.cat(latents, dim=0).mean(dim=0).numpy()
        X.append(mean)
        y.append(cid2label[cid])
        cids.append(cid)
    return np.stack(X), np.array(y, dtype=int), cids


# ── Callback ─────────────────────────────────────────────────────────────────

class EfficacyClassifierCallback(pl.Callback):
    """Run efficacy classification after each validation epoch and log metrics.

    Args:
        train_metadata_path: Path to training metadata JSON.
        train_efficacy_path: Path to training efficacy .pt file.
        inference_metadata_path: Path to inference metadata JSON.
        inference_efficacy_csv: Path to inference efficacy CSV.
        image_root_dir: Root directory for image paths.
        threshold: Binarisation threshold for efficacy.
        batch_size: Batch size for encoding images.
    """

    def __init__(
        self,
        train_metadata_path: str,
        train_efficacy_path: str,
        inference_metadata_path: str,
        inference_efficacy_csv: str,
        image_root_dir: str,
        threshold: float = 70.0,
        batch_size: int = 64,
    ):
        super().__init__()
        if not _HAS_XGBOOST:
            raise ImportError(
                "xgboost is required for EfficacyClassifierCallback. "
                "Install with: pip install xgboost"
            )

        self.image_root_dir = Path(image_root_dir)
        self.threshold = threshold
        self.batch_size = batch_size

        # Load training metadata
        with open(train_metadata_path, "r") as f:
            raw = json.load(f)
        self.train_metadata = raw if isinstance(raw, list) else raw.get("compounds", [])

        # Load training efficacy and binarise
        efficacy_data = torch.load(train_efficacy_path, map_location="cpu", weights_only=False)
        efficacy_map = {str(e["Compound"]): float(e["Efficacy"]) for e in efficacy_data}
        self.train_labels = {cid: int(v >= threshold) for cid, v in efficacy_map.items()}

        # Load inference metadata
        with open(inference_metadata_path, "r") as f:
            raw_inf = json.load(f)
        self.inference_metadata = raw_inf if isinstance(raw_inf, list) else raw_inf.get("compounds", [])

        # Load inference labels
        import pandas as pd
        df = pd.read_csv(inference_efficacy_csv)
        self.inference_labels = {str(row["Compound No"]): int(row["Active"]) for _, row in df.iterrows()}

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Encode, train classifier, evaluate, log."""
        backbone = pl_module.model.backbone
        backbone.eval()
        device = pl_module.device

        # 1. Encode training embeddings
        train_emb = _encode_metadata(
            self.train_metadata, self.image_root_dir, backbone, device, self.batch_size)
        X_train, y_train, _ = _build_mean_features(train_emb, self.train_labels)

        if X_train.shape[0] == 0:
            return

        # 2. Encode inference embeddings
        inf_emb = _encode_metadata(
            self.inference_metadata, self.image_root_dir, backbone, device, self.batch_size)
        X_inf, y_inf, _ = _build_mean_features(inf_emb, self.inference_labels)

        if X_inf.shape[0] == 0:
            return

        # 3. Train XGBoost
        clf = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            objective="binary:logistic",
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train, verbose=False)

        # 4. Evaluate
        preds = clf.predict(X_inf)
        proba = clf.predict_proba(X_inf)[:, 1]

        auroc = roc_auc_score(y_inf, proba)
        bal_acc = balanced_accuracy_score(y_inf, preds)
        f1 = f1_score(y_inf, preds, average="weighted", zero_division=0)

        # 5. Log
        pl_module.log("val/efficacy_auroc", auroc, prog_bar=True, sync_dist=True)
        pl_module.log("val/efficacy_bal_acc", bal_acc, sync_dist=True)
        pl_module.log("val/efficacy_f1", f1, sync_dist=True)

        step = trainer.global_step
        print(f"  [EfficacyClassifier] step={step}  AUROC={auroc:.4f}  "
              f"BalAcc={bal_acc:.4f}  F1={f1:.4f}")

        # Put backbone back to training mode
        backbone.train()
