"""
classifier_utils.py

Shared utilities for synthesis-program classifiers:
  - Data loading & feature builders
  - GatedABMIL model + train / infer helpers
  - Label encoding
  - Result saving (report, confusion matrix, predictions CSV)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, classification_report, confusion_matrix,
    top_k_accuracy_score,
)
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_efficacy_data(path: str) -> Dict[str, float]:
    """
    Load efficacy.pt → {compound_id: efficacy_value}.

    Expected format: [{'Compound': '...', 'Efficacy': float}, ...]
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {str(entry["Compound"]): float(entry["Efficacy"]) for entry in data}


def build_efficacy_features(
    efficacy: Dict[str, float],
    compound_col: pd.Series,
    label_col: pd.Series,
    label2idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build an (N, 1) feature matrix from efficacy values."""
    comp2label: Dict[str, int] = {}
    for comp, prog in zip(compound_col, label_col):
        comp2label[str(comp)] = label2idx[str(prog)]

    X_rows, y_rows, cids = [], [], []
    for cid, eff_val in efficacy.items():
        if cid not in comp2label:
            continue
        X_rows.append([eff_val])
        y_rows.append(comp2label[cid])
        cids.append(cid)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows), cids


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize along *dim*."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def build_mil_bags(
    embeddings: Dict,
    compound_col: pd.Series,
    label_col: pd.Series,
    label2idx: Dict[str, int],
    subtract_control: bool = False,
    normalize_before_subtract: bool = False,
) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """Build variable-length bags of instance embeddings per compound."""
    comp2label: Dict[str, int] = {}
    for comp, prog in zip(compound_col, label_col):
        comp2label[str(comp)] = label2idx[str(prog)]

    bags, labels, cids = [], [], []
    for compound_id, plates in embeddings.items():
        cid = str(compound_id)
        if cid not in comp2label:
            continue
        plate_latents: List[torch.Tensor] = []
        for plate_data in plates.values():
            treated = plate_data.get("treated")
            if treated is None or treated.numel() == 0:
                continue
            if subtract_control and "control" in plate_data:
                control = plate_data["control"]
                if normalize_before_subtract:
                    treated = _l2_normalize(treated)
                    control = _l2_normalize(control)
                treated = treated - control.unsqueeze(0)
            plate_latents.append(treated.float())
        if not plate_latents:
            continue
        bags.append(torch.cat(plate_latents, dim=0))
        labels.append(comp2label[cid])
        cids.append(cid)
    return bags, labels, cids


def build_mean_latent_features(
    embeddings: Dict,
    compound_col: pd.Series,
    label_col: pd.Series,
    label2idx: Dict[str, int],
    subtract_control: bool = False,
    normalize_before_subtract: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build a (num_compounds, D) feature matrix where each row is the mean
    of all treated latents for a compound (optionally control-subtracted).

    Returns
    -------
    X         : (N, D) float32 array
    y         : (N,) int array
    cids      : list of compound ID strings
    """
    comp2label: Dict[str, int] = {}
    for comp, prog in zip(compound_col, label_col):
        comp2label[str(comp)] = label2idx[str(prog)]

    X_rows, y_rows, cids = [], [], []

    for compound_id, plates in embeddings.items():
        cid = str(compound_id)
        if cid not in comp2label:
            continue

        plate_latents: List[torch.Tensor] = []
        for plate_data in plates.values():
            treated = plate_data.get("treated")
            if treated is None or treated.numel() == 0:
                continue
            if subtract_control and "control" in plate_data:
                control = plate_data["control"]
                if normalize_before_subtract:
                    treated = _l2_normalize(treated)
                    control = _l2_normalize(control)
                treated = treated - control.unsqueeze(0)
            plate_latents.append(treated.float())

        if not plate_latents:
            continue

        all_latents = torch.cat(plate_latents, dim=0)       # (M, D)
        mean_latent = all_latents.mean(dim=0).numpy()       # (D,)
        X_rows.append(mean_latent)
        y_rows.append(comp2label[cid])
        cids.append(cid)

    return np.stack(X_rows), np.array(y_rows), cids


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Model — Gated Attention-Based MIL (ABMIL)
# ═══════════════════════════════════════════════════════════════════════════════


class GatedABMIL(nn.Module):
    """Gated Attention-Based Multiple Instance Learning classifier (multi-class)."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        bag : (N_instances, D)

        Returns
        -------
        logits : (num_classes,) raw logits
        att    : (N_instances,) attention weights
        """
        V = self.attention_V(bag)             # (N, H)
        U = self.attention_U(bag)             # (N, H)
        att_logits = self.attention_w(V * U)  # (N, 1)
        att = torch.softmax(att_logits, dim=0)  # (N, 1)
        bag_repr = (att * bag).sum(dim=0, keepdim=True)  # (1, D)
        logits = self.classifier(bag_repr).squeeze(0)     # (num_classes,)
        return logits, att.squeeze()


def train_abmil(
    bags: List[torch.Tensor],
    labels: List[int],
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> GatedABMIL:
    """Train Gated ABMIL on all data, return trained model."""
    input_dim = bags[0].shape[1]

    torch.manual_seed(args.seed)
    model = GatedABMIL(input_dim, num_classes, args.abmil_hidden, args.abmil_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.abmil_lr, weight_decay=args.abmil_wd)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    print(f"\nTraining ABMIL on {len(bags)} compounds ({num_classes} classes) ...")
    for epoch in tqdm(range(args.abmil_epochs), desc="ABMIL Training"):
        model.train()
        indices = np.random.permutation(len(bags))
        epoch_loss = 0.0
        for i in indices:
            bag = bags[i].to(device)
            label = torch.tensor(labels[i], dtype=torch.long, device=device)
            logits, _ = model(bag)
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    print("Training done.\n")
    return model


def infer_abmil(
    model: GatedABMIL,
    bags: List[torch.Tensor],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference, return (predictions, class-probabilities)."""
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for bag in bags:
            logits, _ = model(bag.to(device))
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(int(probs.argmax()))
    return np.array(all_preds), np.stack(all_probs)


# ═══════════════════════════════════════════════════════════════════════════════
# 2a-½. Class-weight computation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_class_weights(
    labels: List[int],
    num_classes: int,
    mode: str = "balanced",
) -> torch.Tensor:
    """Compute per-class weights for CrossEntropyLoss.

    Parameters
    ----------
    labels     : list of integer class indices
    num_classes: total number of classes
    mode       : ``"balanced"`` — inverse frequency (N / (K * n_k))
                 ``"sqrt_balanced"`` — sqrt of inverse frequency
                 ``"none"`` — uniform weights (all ones)

    Returns
    -------
    weights : (num_classes,) float tensor
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)          # avoid division by zero
    n_samples = float(len(labels))

    if mode == "balanced":
        w = n_samples / (num_classes * counts)
    elif mode == "sqrt_balanced":
        w = np.sqrt(n_samples / (num_classes * counts))
    else:  # "none"
        w = np.ones(num_classes, dtype=np.float64)

    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) with optional per-class weights.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma > 0 reduces the loss for well-classified examples, forcing the
    model to focus on hard / minority samples.
    """

    def __init__(
        self,
        weight: torch.Tensor = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)  # p_t for the true class
        return ((1 - pt) ** self.gamma * ce).mean()


def _balanced_epoch_indices(
    labels: List[int],
    num_classes: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Return shuffled indices that oversample minority classes.

    Each class contributes *max_count* samples (with replacement for
    minority classes), so every class is seen equally often per epoch.
    """
    per_class: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for idx, lab in enumerate(labels):
        per_class[lab].append(idx)

    max_count = max(len(v) for v in per_class.values())
    epoch_indices = []
    for c in range(num_classes):
        members = per_class[c]
        if len(members) == 0:
            continue
        if len(members) >= max_count:
            epoch_indices.extend(rng.choice(members, size=max_count, replace=False))
        else:
            epoch_indices.extend(rng.choice(members, size=max_count, replace=True))
    rng.shuffle(epoch_indices)
    return np.array(epoch_indices)


# ═══════════════════════════════════════════════════════════════════════════════
# 2b. Model — LogSumExp MIL
# ═══════════════════════════════════════════════════════════════════════════════


class LogSumExpMIL(nn.Module):
    """LogSumExp pooling MIL classifier.

    Pooling:  z = (1/r) * log( (1/M) * sum_i exp(r * h_i) )

    The learnable parameter *r* smoothly interpolates between mean pooling
    (r → 0) and max pooling (r → ∞).  A small FC head is applied before
    and after pooling.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        r_init: float = 1.0,
    ):
        super().__init__()
        self.instance_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # learnable temperature
        self.log_r = nn.Parameter(torch.tensor(float(np.log(max(r_init, 1e-4)))))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        bag : (M, D)  — instance embeddings for one compound

        Returns
        -------
        logits : (num_classes,)
        """
        h = self.instance_transform(bag)          # (M, hidden)
        r = self.log_r.exp().clamp(min=1e-4)      # positive scalar
        # LogSumExp pooling (numerically stable via torch.logsumexp)
        z = torch.logsumexp(r * h, dim=0) / r - np.log(h.shape[0]) / r  # (hidden,)
        logits = self.classifier(z)               # (num_classes,)
        return logits


def train_logsumexp(
    bags: List[torch.Tensor],
    labels: List[int],
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    class_weights: str = "none",
    oversample: bool = False,
    focal_gamma: float = 0.0,
) -> LogSumExpMIL:
    """Train LogSumExpMIL on all data, return trained model."""
    input_dim = bags[0].shape[1]

    torch.manual_seed(args.seed)
    model = LogSumExpMIL(
        input_dim, num_classes,
        hidden_dim=args.lse_hidden,
        dropout=args.lse_dropout,
        r_init=args.lse_r_init,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lse_lr, weight_decay=args.lse_wd)

    # ── Loss function ───────────────────────────────────────────────────────
    cw = compute_class_weights(labels, num_classes, mode=class_weights)
    if class_weights != "none":
        print(f"  Class weights ({class_weights}): "
              f"{[f'{v:.3f}' for v in cw.tolist()]}")

    if focal_gamma > 0:
        criterion = FocalLoss(
            weight=cw.to(device),
            gamma=focal_gamma,
            label_smoothing=args.label_smoothing,
        )
        print(f"  Using Focal Loss (gamma={focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=cw.to(device),
            label_smoothing=args.label_smoothing,
        )

    # ── Oversampling setup ──────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    if oversample:
        print("  Class-balanced oversampling enabled")

    print(f"\nTraining LogSumExp MIL on {len(bags)} compounds ({num_classes} classes) ...")
    for epoch in tqdm(range(args.lse_epochs), desc="LogSumExp Training"):
        model.train()
        if oversample:
            indices = _balanced_epoch_indices(labels, num_classes, rng)
        else:
            indices = rng.permutation(len(bags))
        for i in indices:
            bag = bags[i].to(device)
            label = torch.tensor(labels[i], dtype=torch.long, device=device)
            logits = model(bag)
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("Training done.\n")
    return model


def infer_logsumexp(
    model: LogSumExpMIL,
    bags: List[torch.Tensor],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference, return (predictions, class-probabilities)."""
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for bag in bags:
            logits = model(bag.to(device))
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(int(probs.argmax()))
    return np.array(all_preds), np.stack(all_probs)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Label encoding
# ═══════════════════════════════════════════════════════════════════════════════

def build_label_encoder(series: pd.Series) -> Tuple[Dict[str, int], List[str]]:
    classes = sorted(series.astype(str).unique().tolist())
    str2idx = {c: i for i, c in enumerate(classes)}
    return str2idx, classes


def save_label_encoder(classes: List[str], str2idx: Dict[str, int], path: Path) -> None:
    with open(path, "w") as f:
        json.dump({"classes": classes, "str2idx": str2idx}, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Result saving
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    num_classes: int,
    title: str,
    save_path: Path,
) -> None:
    """Plot and save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.5), max(7, num_classes * 0.45)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=range(num_classes),
        yticks=range(num_classes),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _topk_predictions(val_probs: np.ndarray, k: int) -> np.ndarray:
    """Return the top-k predicted class indices for each sample, shape (N, k)."""
    return np.argsort(val_probs, axis=1)[:, -k:][:, ::-1]


def _topk_confusion_matrix(
    val_true: np.ndarray,
    val_probs: np.ndarray,
    k: int,
    num_classes: int,
) -> np.ndarray:
    """Build a confusion matrix where a prediction counts as correct if the
    true label is within the top-k predictions.  Off-diagonal entries show
    which class was predicted as #1 when the true label was NOT in top-k."""
    topk_idx = _topk_predictions(val_probs, k)  # (N, k)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i, (true, top_classes) in enumerate(zip(val_true, topk_idx)):
        if true in top_classes:
            cm[true, true] += 1       # correct under top-k
        else:
            cm[true, top_classes[0]] += 1  # wrong: attribute to top-1 pred
    return cm


def save_results(
    val_true: np.ndarray,
    val_preds: np.ndarray,
    val_probs: np.ndarray,
    val_cids: List[str],
    classes: List[str],
    num_classes: int,
    output_dir: Path,
    cm_title: str,
    file_suffix: str,
    report_header: str,
    save_predictions: bool,
    topk: Tuple[int, ...] = (1, 3, 5),
) -> None:
    """Save classification report, confusion matrix, top-k accuracies, and (optionally) predictions CSV."""
    val_acc = balanced_accuracy_score(val_true, val_preds)
    val_f1 = f1_score(val_true, val_preds, average="weighted", zero_division=0)

    # ── Top-k accuracies (always include top-1) ─────────────────────────────
    all_k = sorted(set((1,) + tuple(topk)))
    topk_results = {}
    for k in all_k:
        if k > num_classes:
            continue
        if k == 1:
            topk_results[k] = float((val_preds == val_true).mean())
        else:
            topk_results[k] = float(top_k_accuracy_score(
                val_true, val_probs, k=k, labels=list(range(num_classes)),
            ))

    # ══════════════════════════════════════════════════════════════════════════
    # Top-1 report & confusion matrix
    # ══════════════════════════════════════════════════════════════════════════
    report_str = classification_report(
        val_true, val_preds,
        labels=list(range(num_classes)),
        target_names=classes,
        zero_division=0,
    )
    print("\n── Top-1 Classification Report ──")
    print(report_str)
    print(f"Balanced accuracy : {val_acc:.4f}")
    print(f"Weighted F1       : {val_f1:.4f}")
    print(f"Top-1 accuracy    : {topk_results[1]:.4f}")

    report_path = output_dir / f"classification_report_top1{file_suffix}.txt"
    with open(report_path, "w") as f:
        f.write(report_header)
        f.write("── Top-1 Classification Report ──\n\n")
        f.write(report_str)
        f.write(f"\nBalanced accuracy : {val_acc:.4f}\n")
        f.write(f"Weighted F1       : {val_f1:.4f}\n")
        f.write(f"Top-1 accuracy    : {topk_results[1]:.4f}\n")
    print(f"Report saved to    : {report_path}")

    cm = confusion_matrix(val_true, val_preds, labels=list(range(num_classes)))
    cm_path = output_dir / f"confusion_matrix_top1{file_suffix}.png"
    _plot_confusion_matrix(cm, classes, num_classes, f"{cm_title} (Top-1)", cm_path)
    print(f"Confusion matrix   : {cm_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Top-k reports & confusion matrices (k > 1)
    # ══════════════════════════════════════════════════════════════════════════
    for k, k_acc in sorted(topk_results.items()):
        if k == 1:
            continue

        # Build top-k adjusted predictions: if true label is in top-k,
        # count as correct (pred = true); otherwise use top-1 prediction.
        topk_idx = _topk_predictions(val_probs, k)
        topk_preds = np.array([
            true if true in row else row[0]
            for true, row in zip(val_true, topk_idx)
        ])

        topk_acc = balanced_accuracy_score(val_true, topk_preds)
        topk_f1 = f1_score(val_true, topk_preds, average="weighted", zero_division=0)

        report_k_str = classification_report(
            val_true, topk_preds,
            labels=list(range(num_classes)),
            target_names=classes,
            zero_division=0,
        )

        print(f"\n── Top-{k} Classification Report ──")
        print(report_k_str)
        print(f"Balanced accuracy : {topk_acc:.4f}")
        print(f"Weighted F1       : {topk_f1:.4f}")
        print(f"Top-{k} accuracy    : {k_acc:.4f}")

        report_k_path = output_dir / f"classification_report_top{k}{file_suffix}.txt"
        with open(report_k_path, "w") as f:
            f.write(report_header)
            f.write(f"── Top-{k} Classification Report ──\n\n")
            f.write(report_k_str)
            f.write(f"\nBalanced accuracy : {topk_acc:.4f}\n")
            f.write(f"Weighted F1       : {topk_f1:.4f}\n")
            f.write(f"Top-{k} accuracy    : {k_acc:.4f}\n")
        print(f"Report saved to    : {report_k_path}")

        # Top-k confusion matrix
        cm_k = _topk_confusion_matrix(val_true, val_probs, k, num_classes)
        cm_k_path = output_dir / f"confusion_matrix_top{k}{file_suffix}.png"
        _plot_confusion_matrix(cm_k, classes, num_classes, f"{cm_title} (Top-{k})", cm_k_path)
        print(f"Confusion matrix   : {cm_k_path}")

    # ── Summary of all top-k accuracies ───────────────────────────────────────
    summary_path = output_dir / f"topk_summary{file_suffix}.txt"
    with open(summary_path, "w") as f:
        f.write(report_header)
        f.write("── Top-k Accuracy Summary ──\n\n")
        for k, acc in sorted(topk_results.items()):
            f.write(f"Top-{k} accuracy : {acc:.4f}\n")
    print(f"\nTop-k summary      : {summary_path}")

    # ── Predictions CSV ──────────────────────────────────────────────────────
    if save_predictions:
        pred_rows = {
            "compound_id":     val_cids,
            "true_label":      [classes[i] for i in val_true],
            "predicted_label": [classes[i] for i in val_preds],
            "correct":         [t == p for t, p in zip(val_true, val_preds)],
        }
        for cls_idx, cls_name in enumerate(classes):
            pred_rows[f"prob_{cls_name}"] = val_probs[:, cls_idx].tolist()
        # Add top-k correctness columns
        for k in sorted(topk_results):
            if k == 1:
                continue
            topk_idx = _topk_predictions(val_probs, k)
            pred_rows[f"correct_top{k}"] = [t in row for t, row in zip(val_true, topk_idx)]

        pred_df = pd.DataFrame(pred_rows)
        pred_path = output_dir / f"predictions{file_suffix}.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"Predictions saved to: {pred_path}")

    print(f"Outputs saved to   : {output_dir}")
