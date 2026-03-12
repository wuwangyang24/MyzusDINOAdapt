"""
train_synthesis_classifier.py

Train a classifier to predict the synthesis program of a compound from its
DINO embeddings.  Two classifiers are available:

  1. **Gated ABMIL** (default) — Gated Attention-Based MIL over variable-length
     bags of instance embeddings (no mean-pooling)
  2. **XGBoost** — gradient-boosted trees on the mean latent per compound

DATA FLOW  (Gated ABMIL)
-------------------------
For each compound:
  1. Collect all treated latent vectors across every plate  →  (M, D)
     (optionally subtract the per-plate averaged control embedding first)
  2. Compute gated attention weights over instances          →  (M,)
  3. Weighted-sum bag representation                         →  (D,) → FC → num_classes

DATA FLOW  (XGBoost)
---------------------
For each compound:
  1. Collect all treated latent vectors across every plate   →  (M, D)
     (optionally subtract the per-plate averaged control embedding first)
  2. Compute the element-wise mean across M images           →  (D,)
  3. Feed the (N, D) feature matrix into XGBoost             →  num_classes

Inputs
------
  --embeddings   Experiments/embeddings.pt   (optional if --efficacy is given)
                 Output of encode_embeddings.py:
                    { compound_id: { plate_id: {"treated": (N,D), "control": (D,)} } }

  --efficacy     Experiments/efficacy.pt     (optional, alternative to --embeddings)
                 List of dicts: [{"Compound": str, "Efficacy": float}, ...]
                 Uses the efficacy value as a single scalar feature per compound.

  --metadata     CSV / Excel file with at least two columns:
                    "compound"           (str)  — must match compound_id keys in .pt file
                    "synthesis_program"  (str)  — class label

Usage examples
--------------
  # Gated ABMIL (default)
  python Experiments/train_synthesis_classifier.py \\
      --embeddings Experiments/embeddings.pt \\
      --metadata   data/compound_metadata.csv \\
      --output_dir Experiments/runs/classifier

  # Gated ABMIL with control subtraction, custom hyper-parameters
  python Experiments/train_synthesis_classifier.py \\
      --embeddings       Experiments/embeddings.pt \\
      --metadata         data/compound_metadata.csv \\
      --subtract_control \\
      --abmil_hidden 128 --abmil_dropout 0.25 \\
      --abmil_epochs 200 --abmil_lr 1e-3

  # XGBoost on mean latents with control subtraction
  python Experiments/train_synthesis_classifier.py \\
      --embeddings       Experiments/embeddings.pt \\
      --metadata         data/compound_metadata.csv \\
      --classifier       xgboost \\
      --subtract_control \\
      --xgb_n_estimators 500 --xgb_max_depth 8

  # XGBoost with efficacy values instead of embeddings
  python Experiments/train_synthesis_classifier.py \\
      --efficacy   Experiments/efficacy.pt \\
      --metadata   data/compound_metadata.csv \\
      --classifier xgboost

Output
------
  <output_dir>/
      best_model.pt | xgboost_model.json  — saved model
      label_encoder.json                   — { "classes": [...], "str2idx": {...} }
      training_log.csv                     — per-epoch train/val metrics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, classification_report, confusion_matrix
)
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

# ── project imports ──────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_efficacy_data(path: str) -> Dict[str, float]:
    """
    Load efficacy.pt → {compound_id: efficacy_value}.

    Expected format: [{'Compound': '...', 'Efficacy': float}, ...]
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {str(entry['Compound']): float(entry['Efficacy']) for entry in data}


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


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  ABMIL Training & Inference helpers
# ═══════════════════════════════════════════════════════════════════════════════


def build_mil_bags(
    embeddings: Dict,
    compound_col: pd.Series,
    label_col: pd.Series,
    label2idx: Dict[str, int],
    subtract_control: bool = False,
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
                treated = treated - control.unsqueeze(0)
            plate_latents.append(treated.float())
        if not plate_latents:
            continue
        bags.append(torch.cat(plate_latents, dim=0))
        labels.append(comp2label[cid])
        cids.append(cid)
    return bags, labels, cids


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
# 4.  Label encoding helpers
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 3b. Mean-latent feature builder (for XGBoost)
# ═══════════════════════════════════════════════════════════════════════════════

def build_mean_latent_features(
    embeddings: Dict,
    compound_col: pd.Series,
    label_col: pd.Series,
    label2idx: Dict[str, int],
    subtract_control: bool = False,
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


def build_label_encoder(series: pd.Series) -> Tuple[Dict[str, int], List[str]]:
    classes = sorted(series.astype(str).unique().tolist())
    str2idx = {c: i for i, c in enumerate(classes)}
    return str2idx, classes


def save_label_encoder(classes: List[str], str2idx: Dict[str, int], path: Path) -> None:
    with open(path, "w") as f:
        json.dump({"classes": classes, "str2idx": str2idx}, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a classifier for compound synthesis programs (Gated ABMIL or XGBoost)."
    )

    # ---- Data ----
    p.add_argument("--embeddings", default=None,
                   help="Path to the .pt embeddings file from encode_embeddings.py")
    p.add_argument("--efficacy", default=None,
                   help="Path to efficacy.pt ([{'Compound': str, 'Efficacy': float}, ...]). "
                        "Alternative to --embeddings; uses efficacy as single-scalar feature.")
    p.add_argument("--metadata", required=True,
                   help="CSV or Excel file with 'compound' and 'synthesis_program' columns")
    p.add_argument("--compound_col", default="compound",
                   help="Name of the compound ID column in metadata. Default: compound")
    p.add_argument("--label_col", default="synthesis_program",
                   help="Name of the synthesis program column in metadata. Default: synthesis_program")
    p.add_argument("--subtract_control", action="store_true",
                   help="Subtract per-plate averaged control embedding from treated embeddings")
    p.add_argument("--val_split", type=float, default=0.2,
                   help="Fraction of compounds used for validation. Default: 0.2")

    # ---- ABMIL hyper-parameters ----
    p.add_argument("--abmil_hidden", type=int, default=128,
                   help="ABMIL attention hidden dim. Default: 128")
    p.add_argument("--abmil_dropout", type=float, default=0.25,
                   help="ABMIL dropout. Default: 0.25")
    p.add_argument("--abmil_lr", type=float, default=1e-3,
                   help="ABMIL learning rate. Default: 1e-3")
    p.add_argument("--abmil_wd", type=float, default=1e-4,
                   help="ABMIL weight decay. Default: 1e-4")
    p.add_argument("--abmil_epochs", type=int, default=200,
                   help="ABMIL training epochs. Default: 200")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="Label smoothing for CrossEntropyLoss. Default: 0.1")

    # ---- Misc ----
    p.add_argument("--output_dir",  default="Experiments/runs/classifier",
                   help="Directory for checkpoints and logs. Default: Experiments/runs/classifier")
    p.add_argument("--device",      default=None,
                   help="Torch device. Auto-detected if not specified.")
    p.add_argument("--seed",        type=int, default=42, help="Random seed. Default: 42")
    p.add_argument("--save_predictions", action="store_true",
                   help="Save validation predictions + ground truth to predictions.csv")

    p.add_argument("--min_compounds_per_class", type=int, default=2,
                   help="Drop synthesis programs with fewer compounds than this. Default: 2")

    # ---- Classifier selection ----
    p.add_argument("--classifier", choices=["abmil", "xgboost"],
                   default="abmil",
                   help="Which classifier to use. Default: abmil")

    # ---- XGBoost hyper-parameters ----
    p.add_argument("--xgb_n_estimators", type=int, default=300,
                   help="[XGBoost] Number of boosting rounds. Default: 300")
    p.add_argument("--xgb_max_depth", type=int, default=6,
                   help="[XGBoost] Max tree depth. Default: 6")
    p.add_argument("--xgb_learning_rate", type=float, default=0.1,
                   help="[XGBoost] Boosting learning rate. Default: 0.1")
    p.add_argument("--xgb_subsample", type=float, default=0.8,
                   help="[XGBoost] Row subsampling ratio. Default: 0.8")
    p.add_argument("--xgb_colsample_bytree", type=float, default=0.8,
                   help="[XGBoost] Column subsampling ratio. Default: 0.8")
    p.add_argument("--xgb_early_stopping", type=int, default=20,
                   help="[XGBoost] Early stopping rounds. Default: 20")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  XGBoost training pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _run_xgboost(
    args: argparse.Namespace,
    embeddings: Optional[Dict],
    df: pd.DataFrame,
    str2idx: Dict[str, int],
    classes: List[str],
    num_classes: int,
    output_dir: Path,
    efficacy: Optional[Dict[str, float]] = None,
) -> None:
    """Train an XGBoost classifier on per-compound features (mean latent or efficacy)."""
    if not _HAS_XGBOOST:
        raise ImportError(
            "xgboost is required for --classifier xgboost. "
            "Install it with:  pip install xgboost"
        )

    from sklearn.model_selection import train_test_split

    # ── Build feature matrix ─────────────────────────────────────────────────
    if efficacy is not None:
        X, y, cids = build_efficacy_features(
            efficacy=efficacy,
            compound_col=df[args.compound_col],
            label_col=df[args.label_col],
            label2idx=str2idx,
        )
    else:
        X, y, cids = build_mean_latent_features(
            embeddings=embeddings,
            compound_col=df[args.compound_col],
            label_col=df[args.label_col],
            label2idx=str2idx,
            subtract_control=args.subtract_control,
        )
    print(f"  {X.shape[0]} compounds with valid features.")
    print(f"  Feature dim (D) : {X.shape[1]}")

    if X.shape[0] == 0:
        raise RuntimeError(
            "Dataset is empty. Check that compound IDs in the input file "
            "match the compound IDs in the metadata."
        )

    # ── Remove classes with fewer than min_compounds_per_class compounds ──
    min_cpc = max(args.min_compounds_per_class, 2)  # need ≥ 2 for train/val split
    class_counts = np.bincount(y)
    valid_classes = set(np.where(class_counts >= min_cpc)[0])
    keep_mask = np.array([yi in valid_classes for yi in y])
    n_removed = len(y) - keep_mask.sum()
    if n_removed > 0:
        removed_names = sorted({classes[yi] for yi in y if yi not in valid_classes})
        print(f"  Dropped {n_removed} compound(s) from {len(removed_names)} "
              f"class(es) with <{min_cpc} compounds: {removed_names}")
        X, y, cids = X[keep_mask], y[keep_mask], [c for c, k in zip(cids, keep_mask) if k]

    # Remap labels to contiguous 0..K-1 (required by XGBoost)
    remaining = sorted(set(y.tolist()))
    old2new = {old: new for new, old in enumerate(remaining)}
    y = np.array([old2new[yi] for yi in y])
    classes = [classes[old] for old in remaining]
    num_classes = len(classes)
    print(f"  {num_classes} classes after filtering, {len(y)} compounds remaining.")

    # ── Train / val split ────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val, cids_train, cids_val = train_test_split(
        X, y, cids,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    print(f"  Train: {len(y_train)}  |  Val: {len(y_val)}")

    # ── XGBoost model ────────────────────────────────────────────────────────
    objective = "multi:softprob" if num_classes > 2 else "binary:logistic"
    xgb_params = dict(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        objective=objective,
        eval_metric="mlogloss" if num_classes > 2 else "logloss",
        use_label_encoder=False,
        random_state=args.seed,
        n_jobs=-1,
        early_stopping_rounds=args.xgb_early_stopping,
    )
    if num_classes > 2:
        xgb_params["num_class"] = num_classes

    clf = xgb.XGBClassifier(**xgb_params)

    print(f"\nTraining XGBoost ({args.xgb_n_estimators} rounds, "
          f"max_depth={args.xgb_max_depth}, lr={args.xgb_learning_rate}) ...")
    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True,
    )

    # ── Evaluation ───────────────────────────────────────────────────────────
    val_preds = clf.predict(X_val)
    val_acc = balanced_accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)

    print("\nClassification Report (validation set):")
    report_str = classification_report(
        y_val, val_preds,
        labels=list(range(num_classes)),
        target_names=classes,
        zero_division=0,
    )
    print(report_str)
    print(f"Val accuracy : {val_acc:.4f}")
    print(f"Val F1       : {val_f1:.4f}")

    # ── Save classification report ───────────────────────────────────────────
    _input_path = args.efficacy if args.efficacy else args.embeddings
    emb_stem = Path(_input_path).stem              # e.g. "embeddings_tiltedvae_dim100"
    report_path = output_dir / f"classification_report_{emb_stem}.txt"
    with open(report_path, "w") as f:
        f.write(f"Input file       : {_input_path}\n")
        f.write(f"Subtract control : {args.subtract_control}\n\n")
        f.write(report_str)
        f.write(f"\nVal accuracy : {val_acc:.4f}\n")
        f.write(f"Val F1       : {val_f1:.4f}\n")
    print(f"Report saved to    : {report_path}")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_val, val_preds, labels=list(range(num_classes)))
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
        title=f"Confusion Matrix — {emb_stem}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)
    fig.tight_layout()
    cm_path = output_dir / f"confusion_matrix_{emb_stem}.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix   : {cm_path}")

    # ── Save model ───────────────────────────────────────────────────────────
    model_path = output_dir / f"xgboost_model_{emb_stem}.json"
    clf.save_model(str(model_path))
    print(f"Model saved to     : {model_path}")

    # ── Save predictions ─────────────────────────────────────────────────────
    if args.save_predictions:
        val_probs = clf.predict_proba(X_val)  # (N, num_classes)
        pred_rows = {
            "compound_id":     cids_val,
            "true_label":      [classes[i] for i in y_val],
            "predicted_label": [classes[i] for i in val_preds],
            "correct":         [t == p for t, p in zip(y_val, val_preds)],
        }
        for cls_idx, cls_name in enumerate(classes):
            pred_rows[f"prob_{cls_name}"] = val_probs[:, cls_idx].tolist()

        pred_df = pd.DataFrame(pred_rows)
        pred_path = output_dir / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"Predictions saved to: {pred_path}")

    # ── Training log (evals_result) ──────────────────────────────────────────
    evals = clf.evals_result()
    if evals:
        metric_key = "mlogloss" if num_classes > 2 else "logloss"
        log_df = pd.DataFrame({
            "epoch": list(range(1, len(evals["validation_0"][metric_key]) + 1)),
            f"train_{metric_key}": evals["validation_0"][metric_key],
            f"val_{metric_key}": evals["validation_1"][metric_key],
        })
        log_df.to_csv(output_dir / f"training_log_{emb_stem}.csv", index=False)

    print(f"Outputs saved to   : {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}")

    # ── Output directory ─────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate inputs ──────────────────────────────────────────────────────
    if not args.embeddings and not args.efficacy:
        raise ValueError("Provide at least one of --embeddings or --efficacy.")
    if args.embeddings and args.efficacy:
        raise ValueError("Provide only one of --embeddings or --efficacy, not both.")

    # ── Load input data ──────────────────────────────────────────────────────
    efficacy: Optional[Dict[str, float]] = None
    embeddings: Optional[Dict] = None
    if args.efficacy:
        print(f"Loading efficacy   : {args.efficacy}")
        efficacy = load_efficacy_data(args.efficacy)
        print(f"  {len(efficacy)} compounds found in efficacy file.")
    else:
        print(f"Loading embeddings : {args.embeddings}")
        embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=False)
        print(f"  {len(embeddings)} compounds found in embeddings file.")

    # ── Load metadata ─────────────────────────────────────────────────────────
    print(f"Loading metadata   : {args.metadata}")
    meta_path = Path(args.metadata)
    if meta_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(meta_path)
    else:
        df = pd.read_csv(meta_path)

    required_cols = {args.compound_col, args.label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Metadata is missing column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[args.compound_col, args.label_col]].dropna()
    print(f"  {len(df)} compound rows after dropping NaN.")

    # ── Label encoding ────────────────────────────────────────────────────────
    str2idx, classes = build_label_encoder(df[args.label_col])
    num_classes = len(classes)
    print(f"  {num_classes} synthesis programs: {classes}")
    save_label_encoder(classes, str2idx, output_dir / "label_encoder.json")

    # ── Route to XGBoost or ABMIL ─────────────────────────────────────────
    if args.classifier == "xgboost":
        _run_xgboost(
            args=args,
            embeddings=embeddings,
            df=df,
            str2idx=str2idx,
            classes=classes,
            num_classes=num_classes,
            output_dir=output_dir,
            efficacy=efficacy,
        )
        return

    # ══════════════════════════════════════════════════════════════════════════
    # Gated ABMIL path
    # ══════════════════════════════════════════════════════════════════════════
    if embeddings is None:
        raise ValueError("Gated ABMIL requires --embeddings (not --efficacy).")

    bags, labels, cids = build_mil_bags(
        embeddings,
        compound_col=df[args.compound_col],
        label_col=df[args.label_col],
        label2idx=str2idx,
        subtract_control=args.subtract_control,
    )
    print(f"  {len(bags)} compounds with valid bags, feature dim {bags[0].shape[1]}.")

    if len(bags) == 0:
        raise RuntimeError(
            "Dataset is empty. Check that compound IDs in the embeddings file "
            "match the compound IDs in the metadata."
        )

    # ── Train / val split ─────────────────────────────────────────────────────
    n_total = len(bags)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    indices = np.random.permutation(n_total)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_bags   = [bags[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_bags     = [bags[i] for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]
    val_cids     = [cids[i] for i in val_idx]
    print(f"  Train: {n_train}  |  Val: {n_val}")

    # ── Train model ───────────────────────────────────────────────────────────
    model = train_abmil(train_bags, train_labels, num_classes, args, device)

    # ── Save model ────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), output_dir / "best_model.pt")

    # ── Evaluate on validation set ────────────────────────────────────────────
    val_preds, val_probs = infer_abmil(model, val_bags, device)
    val_true = np.array(val_labels)

    val_acc = balanced_accuracy_score(val_true, val_preds)
    val_f1 = f1_score(val_true, val_preds, average="weighted", zero_division=0)

    report_str = classification_report(
        val_true, val_preds,
        labels=list(range(num_classes)),
        target_names=classes,
        zero_division=0,
    )
    print("\nClassification Report (validation set):")
    print(report_str)
    print(f"Val accuracy : {val_acc:.4f}")
    print(f"Val F1       : {val_f1:.4f}")

    # ── Save classification report ───────────────────────────────────────────
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Classifier       : abmil\n")
        f.write(f"Embeddings       : {args.embeddings}\n")
        f.write(f"Subtract control : {args.subtract_control}\n\n")
        f.write(report_str)
        f.write(f"\nVal accuracy : {val_acc:.4f}\n")
        f.write(f"Val F1       : {val_f1:.4f}\n")
    print(f"Report saved to    : {report_path}")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(val_true, val_preds, labels=list(range(num_classes)))
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
        title="Confusion Matrix — Gated ABMIL",
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
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Confusion matrix   : {output_dir / 'confusion_matrix.png'}")

    # ── Save predictions ─────────────────────────────────────────────────────
    if args.save_predictions:
        pred_rows = {
            "compound_id":     val_cids,
            "true_label":      [classes[i] for i in val_true],
            "predicted_label": [classes[i] for i in val_preds],
            "correct":         [t == p for t, p in zip(val_true, val_preds)],
        }
        for cls_idx, cls_name in enumerate(classes):
            pred_rows[f"prob_{cls_name}"] = val_probs[:, cls_idx].tolist()

        pred_df = pd.DataFrame(pred_rows)
        pred_path = output_dir / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"Predictions saved to: {pred_path}")

    print(f"Outputs saved to  : {output_dir}")


if __name__ == "__main__":
    main()
