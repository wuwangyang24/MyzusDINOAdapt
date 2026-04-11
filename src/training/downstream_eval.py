"""
PyTorch Lightning callback for periodic downstream classification evaluation.

Every *eval_every_n_steps* training steps, this callback:
  1. Encodes training and inference embeddings using the current model state.
  2. Trains a quick XGBoost classifier on the training embeddings.
  3. Evaluates on inference embeddings and logs AUROC (+ balanced accuracy, F1).
"""

import gc
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from Experiments.encode_embeddings import encode_metadata, DINO_TRANSFORM
from Experiments.Efficacy500_classifier.classifier_utils import (
    load_efficacy,
    binarize_efficacy,
    load_inference_labels,
    build_mean_latent_features,
)

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score


class DownstreamEvalCallback(pl.Callback):
    """Periodically evaluate the current model on a downstream efficacy classification task.

    Parameters
    ----------
    eval_every_n_steps : int
        Run evaluation every N global training steps.
    train_metadata_path : str
        Path to the JSON metadata for the classifier training images (e.g. 20 ppm).
    train_root_dir : str
        Root directory for the classifier training images.
    train_efficacy_path : str
        Path to efficacy.pt with training compound efficacy values.
    inference_metadata_path : str
        Path to the JSON metadata for the inference images (e.g. 100 ppm).
    inference_root_dir : str
        Root directory for the inference images.
    inference_efficacy_path : str
        Path to CSV with inference ground-truth labels ('Compound No', 'Active').
    efficacy_threshold : float
        Threshold for binarising efficacy (default: 70.0).
    subtract_control : bool
        Subtract per-plate averaged control embedding from treated embeddings.
    normalize_before_subtract : bool
        L2-normalize embeddings before control subtraction.
    encode_batch_size : int
        Batch size for encoding images (default: 64).
    encode_num_workers : int
        DataLoader workers for image encoding (default: 4).
    train_control_embeddings_path : str or None
        Path to pre-computed control embeddings .pt for training data (optional).
    inf_control_embeddings_path : str or None
        Path to pre-computed control embeddings .pt for inference data (optional).
    scale_pos_weight : bool
        Use scale_pos_weight=n_neg/n_pos in XGBoost to handle class imbalance.
    ckpt_dir : str or None
        Directory to save the best-AUROC checkpoint. If None, no checkpoint is saved.
    """

    def __init__(
        self,
        eval_every_n_steps: int,
        train_metadata_path: str,
        train_root_dir: str,
        train_efficacy_path: str,
        inference_metadata_path: str,
        inference_root_dir: str,
        inference_efficacy_path: str,
        efficacy_threshold: float = 70.0,
        subtract_control: bool = False,
        normalize_before_subtract: bool = False,
        encode_batch_size: int = 64,
        encode_num_workers: int = 4,
        train_control_embeddings_path: Optional[str] = None,
        inf_control_embeddings_path: Optional[str] = None,
        scale_pos_weight: bool = False,
        ckpt_dir: Optional[str] = None,
        num_samples_control: Optional[int] = None,
    ):
        super().__init__()
        if not _HAS_XGBOOST:
            raise ImportError(
                "xgboost is required for DownstreamEvalCallback. "
                "Install it with:  pip install xgboost"
            )

        self.eval_every_n_steps = eval_every_n_steps
        self.train_metadata_path = train_metadata_path
        self.train_root_dir = Path(train_root_dir)
        self.train_efficacy_path = train_efficacy_path
        self.inference_metadata_path = inference_metadata_path
        self.inference_root_dir = Path(inference_root_dir)
        self.inference_efficacy_path = inference_efficacy_path
        self.efficacy_threshold = efficacy_threshold
        self.subtract_control = subtract_control
        self.normalize_before_subtract = normalize_before_subtract
        self.encode_batch_size = encode_batch_size
        self.encode_num_workers = encode_num_workers
        self.scale_pos_weight = scale_pos_weight
        self.num_samples_control = num_samples_control
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else None
        self.best_auroc = -1.0

        # Lazy-loaded on first evaluation
        self._train_metadata: Optional[List[Dict]] = None
        self._inference_metadata: Optional[List[Dict]] = None
        self._cid2label: Optional[Dict[str, int]] = None
        self._inf_cid2label: Optional[Dict[str, int]] = None
        self._train_control_embeddings: Optional[Dict] = None
        self._inf_control_embeddings: Optional[Dict] = None
        self._train_control_embeddings_path = train_control_embeddings_path
        self._inf_control_embeddings_path = inf_control_embeddings_path
        self._compiled_model: Optional[torch.nn.Module] = None

    # ------------------------------------------------------------------
    # Lazy loaders (run once)
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load metadata, efficacy labels, and control embeddings once."""
        if self._train_metadata is not None:
            return

        # Validate that all required files exist
        required_files = {
            "train_metadata": self.train_metadata_path,
            "train_root_dir": self.train_root_dir,
            "train_efficacy": self.train_efficacy_path,
            "inference_metadata": self.inference_metadata_path,
            "inference_root_dir": self.inference_root_dir,
            "inference_efficacy": self.inference_efficacy_path,
        }
        if self._train_control_embeddings_path is not None:
            required_files["train_control_embeddings"] = self._train_control_embeddings_path
        if self._inf_control_embeddings_path is not None:
            required_files["inf_control_embeddings"] = self._inf_control_embeddings_path

        missing = [name for name, path in required_files.items() if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(
                f"[DownstreamEval] Missing files/directories: "
                + ", ".join(f"{name}={required_files[name]}" for name in missing)
            )

        # Training metadata
        with open(self.train_metadata_path, "r") as f:
            raw = json.load(f)
        self._train_metadata = raw if isinstance(raw, list) else raw.get("compounds", raw)

        # Inference metadata
        with open(self.inference_metadata_path, "r") as f:
            raw = json.load(f)
        self._inference_metadata = raw if isinstance(raw, list) else raw.get("compounds", raw)

        # Training efficacy labels
        efficacy = load_efficacy(self.train_efficacy_path)
        self._cid2label = binarize_efficacy(efficacy, threshold=self.efficacy_threshold)

        # Inference efficacy labels
        self._inf_cid2label = load_inference_labels(self.inference_efficacy_path)

        # Pre-computed control embeddings
        if self._train_control_embeddings_path is not None:
            self._train_control_embeddings = torch.load(
                self._train_control_embeddings_path, map_location="cpu", weights_only=False,
            )
            n_cids = len(self._train_control_embeddings)
            print(f"  [DownstreamEval] Loaded train control embeddings: "
                  f"{n_cids} compounds from {self._train_control_embeddings_path}")
        if self._inf_control_embeddings_path is not None:
            self._inf_control_embeddings = torch.load(
                self._inf_control_embeddings_path, map_location="cpu", weights_only=False,
            )
            n_cids = len(self._inf_control_embeddings)
            print(f"  [DownstreamEval] Loaded inf control embeddings: "
                  f"{n_cids} compounds from {self._inf_control_embeddings_path}")

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step % self.eval_every_n_steps != 0:
            return

        # Only run on rank 0 in multi-GPU
        if trainer.global_rank != 0:
            return

        self._ensure_loaded()

        try:
            auroc, bal_acc, f1, pred_df = self._evaluate(pl_module)
        except Exception as e:
            warnings.warn(f"[DownstreamEval] Evaluation failed at step {step}: {e}")
            return

        # Save predictions CSV
        if self.ckpt_dir is not None and pred_df is not None:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self.ckpt_dir / f"downstream_preds_step{step}.csv"
            pred_df.to_csv(csv_path, index=False)

        # Log to all PL loggers (TensorBoard, W&B, etc.)
        pl_module.log("downstream/auroc", auroc, on_step=True, on_epoch=False,
                      rank_zero_only=True, batch_size=1)
        pl_module.log("downstream/balanced_accuracy", bal_acc, on_step=True, on_epoch=False,
                      rank_zero_only=True, batch_size=1)
        pl_module.log("downstream/f1", f1, on_step=True, on_epoch=False,
                      rank_zero_only=True, batch_size=1)

        # Also log to W&B directly for immediate visibility
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "downstream/auroc": auroc,
                    "downstream/balanced_accuracy": bal_acc,
                    "downstream/f1": f1,
                }, commit=False)
        except ImportError:
            pass

        # Save checkpoint if this is the best AUROC so far
        is_best = auroc > self.best_auroc
        if is_best:
            self.best_auroc = auroc
            if self.ckpt_dir is not None:
                self.ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = self.ckpt_dir / "best_auroc.ckpt"
                trainer.save_checkpoint(str(ckpt_path))
                print(f"  [DownstreamEval] New best AUROC={auroc:.4f} — saved {ckpt_path}")

        best_tag = " (best)" if is_best else ""
        print(f"  [DownstreamEval] step={step}  AUROC={auroc:.4f}{best_tag}  "
              f"BalAcc={bal_acc:.4f}  F1={f1:.4f}")

    # ------------------------------------------------------------------
    # Core evaluation logic
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, pl_module) -> tuple:
        """Encode embeddings, train XGBoost, evaluate, return (auroc, bal_acc, f1)."""
        model = pl_module.model
        was_training = model.training
        model.eval()
        device = pl_module.device

        # Compile once for faster encoding; reuses current weights each call
        if self._compiled_model is None and device.type == "cuda":
            self._compiled_model = torch.compile(model)
        eval_model = self._compiled_model if self._compiled_model is not None else model

        try:
            # Encode training embeddings
            train_embeddings = encode_metadata(
                metadata=self._train_metadata,
                root_dir=self.train_root_dir,
                model=eval_model,
                device=device,
                batch_size=self.encode_batch_size,
                return_reg_tokens=False,
                use_amp=device.type == "cuda",
                transform=DINO_TRANSFORM,
                num_workers=self.encode_num_workers,
                control_embeddings=self._train_control_embeddings,
                num_samples_control=self.num_samples_control,
            )

            # Encode inference embeddings
            inf_embeddings = encode_metadata(
                metadata=self._inference_metadata,
                root_dir=self.inference_root_dir,
                model=eval_model,
                device=device,
                batch_size=self.encode_batch_size,
                return_reg_tokens=False,
                use_amp=device.type == "cuda",
                transform=DINO_TRANSFORM,
                num_workers=self.encode_num_workers,
                control_embeddings=self._inf_control_embeddings,
                num_samples_control=self.num_samples_control,
            )

            # Build mean-pooled features
            X_train, y_train, _ = build_mean_latent_features(
                train_embeddings, self._cid2label,
                self.subtract_control, self.normalize_before_subtract,
            )
            X_inf, y_inf, inf_cids = build_mean_latent_features(
                inf_embeddings, self._inf_cid2label,
                self.subtract_control, self.normalize_before_subtract,
            )

            if X_train.shape[0] == 0 or X_inf.shape[0] == 0:
                raise RuntimeError("No compounds matched between embeddings and labels.")

            # Train a quick XGBoost classifier
            xgb_params = dict(
                n_estimators=1000,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                objective="binary:logistic",
                eval_metric="auc",
                use_label_encoder=False,
                random_state=42,
                early_stopping_rounds=20,
                verbosity=0,
            )
            if self.scale_pos_weight:
                n_pos = int(y_train.sum())
                n_neg = len(y_train) - n_pos
                if n_pos > 0:
                    xgb_params["scale_pos_weight"] = n_neg / n_pos
            clf = xgb.XGBClassifier(**xgb_params)
            clf.fit(X_train, y_train, eval_set=[(X_inf, y_inf)], verbose=False)

            # Evaluate
            inf_preds = clf.predict(X_inf)
            inf_proba = clf.predict_proba(X_inf)[:, 1]
            auroc = roc_auc_score(y_inf, inf_proba)
            bal_acc = balanced_accuracy_score(y_inf, inf_preds)
            f1 = f1_score(y_inf, inf_preds, average="weighted", zero_division=0)

            # Build predictions dataframe
            pred_df = pd.DataFrame({
                "compound_id": inf_cids,
                "true_label": y_inf,
                "pred_label": inf_preds,
                "pred_proba": inf_proba,
            })

        finally:
            # Restore model training state
            if was_training:
                model.train()
            # Clean up
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return auroc, bal_acc, f1, pred_df
