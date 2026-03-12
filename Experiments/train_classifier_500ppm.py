"""
train_classifier_500ppm.py

Train a binary classifier (XGBoost or Gated ABMIL) on embeddings_20ppm +
efficacy.pt, then run inference on embeddings_100ppm and evaluate against
efficacy_500ppm.

Classifiers
-----------
  - xgboost : mean-pools instance embeddings per compound, then XGBoost.
  - abmil   : Gated Attention-Based MIL — learns instance-level attention
               weights over variable-length bags (no mean-pooling).

Workflow
--------
  1. TRAIN  — fit classifier on embeddings_20ppm / efficacy.pt
  2. INFER  — predict on embeddings_100ppm, evaluate vs efficacy_500ppm
             Logs: classification report, confusion matrix, AUROC curve,
             predictions CSV.

Usage
-----
  # XGBoost (default)
  python Experiments/train_classifier_500ppm.py \\
      --classifier xgboost \\
      --embeddings           Experiments/embeddings_20ppm.pt \\
      --efficacy             Experiments/efficacy.pt \\
      --inference_embeddings Experiments/embeddings_100ppm.pt \\
      --inference_efficacy   Experiments/efficacy_500ppm.csv

  # Gated ABMIL
  python Experiments/train_classifier_500ppm.py \\
      --classifier abmil \\
      --embeddings           Experiments/embeddings_20ppm.pt \\
      --efficacy             Experiments/efficacy.pt \\
      --inference_embeddings Experiments/embeddings_100ppm.pt \\
      --inference_efficacy   Experiments/efficacy_500ppm.csv

Output
------
  <output_dir>/
      classification_report.txt
      confusion_matrix.png
      auroc_curve.png
      predictions.csv
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_val_score
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
# 1.  Efficacy loading & binarisation
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = ["inactive", "active"]  # 0 = < threshold, 1 = >= threshold


def load_efficacy(path: str) -> Dict[str, float]:
    """
    Load efficacy.pt → {compound_id: efficacy_value}.

    Expected format: [{'Compound': '...', 'Efficacy': float}, ...]
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {str(entry["Compound"]): float(entry["Efficacy"]) for entry in data}


def binarize_efficacy(
    efficacy: Dict[str, float], threshold: float = 70.0,
) -> Dict[str, int]:
    """Return {compound_id: 0 or 1} where 1 means efficacy >= threshold."""
    return {cid: int(val >= threshold) for cid, val in efficacy.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 2a.  Gated Attention-Based MIL (ABMIL)
# ═══════════════════════════════════════════════════════════════════════════════


class GatedABMIL(nn.Module):
    """Gated Attention-Based Multiple Instance Learning classifier."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.25):
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
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        bag : (N_instances, D)

        Returns
        -------
        logit  : (1,) raw logit
        att    : (N_instances,) attention weights
        """
        V = self.attention_V(bag)  # (N, H)
        U = self.attention_U(bag)  # (N, H)
        att_logits = self.attention_w(V * U)  # (N, 1)
        att = torch.softmax(att_logits, dim=0)  # (N, 1)
        bag_repr = (att * bag).sum(dim=0, keepdim=True)  # (1, D)
        logit = self.classifier(bag_repr).squeeze()  # scalar
        return logit, att.squeeze()


class MILBagDataset:
    """Dataset that yields one bag (variable-length tensor) per compound."""

    def __init__(self, bags: List[torch.Tensor], labels: List[int]):
        self.bags = bags
        self.labels = labels

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int):
        return self.bags[idx], self.labels[idx]


def build_mil_bags(
    embeddings: Dict,
    cid2label: Dict[str, int],
    subtract_control: bool = False,
) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """Build variable-length bags of instance embeddings per compound."""
    bags, labels, cids = [], [], []
    for compound_id, plates in embeddings.items():
        cid = str(compound_id)
        if cid not in cid2label:
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
        labels.append(cid2label[cid])
        cids.append(cid)
    return bags, labels, cids


def train_abmil(
    bags: List[torch.Tensor],
    labels: List[int],
    args: argparse.Namespace,
    device: torch.device,
) -> GatedABMIL:
    """Train Gated ABMIL with 5-fold CV reporting, return model trained on all data."""
    input_dim = bags[0].shape[1]
    all_labels = np.array(labels)

    # ── 5-Fold Cross Validation ──────────────────────────────────────────
    print("\n5-Fold Cross Validation (ABMIL) on training data ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_accs, fold_f1s, fold_aurocs = [], [], []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(bags, all_labels), 1):
        torch.manual_seed(args.seed + fold_idx)
        fold_model = GatedABMIL(input_dim, args.abmil_hidden, args.abmil_dropout).to(device)
        optimizer = torch.optim.Adam(fold_model.parameters(), lr=args.abmil_lr, weight_decay=args.abmil_wd)
        criterion = nn.BCEWithLogitsLoss()

        tr_bags = [bags[i] for i in tr_idx]
        tr_labels = [labels[i] for i in tr_idx]
        va_bags = [bags[i] for i in va_idx]
        va_labels = [labels[i] for i in va_idx]

        for epoch in range(args.abmil_epochs):
            fold_model.train()
            indices = np.random.permutation(len(tr_bags))
            for i in indices:
                bag = tr_bags[i].to(device)
                label = torch.tensor(float(tr_labels[i]), device=device)
                logit, _ = fold_model(bag)
                loss = criterion(logit, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        fold_model.eval()
        va_proba, va_true = [], []
        with torch.no_grad():
            for i in range(len(va_bags)):
                logit, _ = fold_model(va_bags[i].to(device))
                va_proba.append(torch.sigmoid(logit).cpu().item())
                va_true.append(va_labels[i])
        va_preds = [int(p >= 0.5) for p in va_proba]
        va_true_arr = np.array(va_true)
        va_preds_arr = np.array(va_preds)
        va_proba_arr = np.array(va_proba)
        fold_accs.append(balanced_accuracy_score(va_true_arr, va_preds_arr))
        fold_f1s.append(f1_score(va_true_arr, va_preds_arr, average="weighted", zero_division=0))
        fold_aurocs.append(roc_auc_score(va_true_arr, va_proba_arr))
        print(f"  Fold {fold_idx}: Acc={fold_accs[-1]:.4f}  F1={fold_f1s[-1]:.4f}  AUROC={fold_aurocs[-1]:.4f}")

    print(f"  Mean : Acc={np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}  "
          f"F1={np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}  "
          f"AUROC={np.mean(fold_aurocs):.4f} +/- {np.std(fold_aurocs):.4f}")

    # ── Train final model on all data ────────────────────────────────────
    print(f"\nTraining final ABMIL on all {len(bags)} training compounds ...")
    torch.manual_seed(args.seed)
    model = GatedABMIL(input_dim, args.abmil_hidden, args.abmil_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.abmil_lr, weight_decay=args.abmil_wd)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.abmil_epochs):
        model.train()
        indices = np.random.permutation(len(bags))
        epoch_loss = 0.0
        for i in indices:
            bag = bags[i].to(device)
            label = torch.tensor(float(labels[i]), device=device)
            logit, _ = model(bag)
            loss = criterion(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.abmil_epochs}  loss={epoch_loss/len(bags):.4f}")

    print("Training done.\n")
    return model


def infer_abmil(
    model: GatedABMIL,
    bags: List[torch.Tensor],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference, return (predictions, probabilities)."""
    model.eval()
    probas, preds = [], []
    with torch.no_grad():
        for bag in bags:
            logit, _ = model(bag.to(device))
            p = torch.sigmoid(logit).cpu().item()
            probas.append(p)
            preds.append(int(p >= 0.5))
    return np.array(preds), np.array(probas)


# ═══════════════════════════════════════════════════════════════════════════════
# 2b.  Mean-latent feature builder (for XGBoost)
# ═══════════════════════════════════════════════════════════════════════════════


def build_mean_latent_features(
    embeddings: Dict,
    cid2label: Dict[str, int],
    subtract_control: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build (N, D) feature matrix from per-compound mean latents.

    Returns
    -------
    X    : (N, D)
    y    : (N,) int labels (0 or 1)
    cids : list of compound IDs
    """
    X_rows, y_rows, cids = [], [], []

    for compound_id, plates in embeddings.items():
        cid = str(compound_id)
        if cid not in cid2label:
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

        all_latents = torch.cat(plate_latents, dim=0)
        mean_latent = all_latents.mean(dim=0).numpy()
        X_rows.append(mean_latent)
        y_rows.append(cid2label[cid])
        cids.append(cid)

    return np.stack(X_rows), np.array(y_rows, dtype=int), cids


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Binary XGBoost classifier: predict efficacy >= threshold "
        "(active) vs < threshold (inactive) from DINO embeddings.",
    )

    # ── Classifier choice ──
    p.add_argument(
        "--classifier",
        choices=["xgboost", "abmil"],
        default="xgboost",
        help="Classifier to use: 'xgboost' (mean-pooled) or 'abmil' (gated attention MIL) (default: xgboost)",
    )

    # ── Training data ──
    p.add_argument(
        "--embeddings",
        default="Experiments/embeddings_20ppm.pt",
        help="Training embeddings (default: Experiments/embeddings_20ppm.pt)",
    )
    p.add_argument(
        "--efficacy",
        default="Experiments/efficacy.pt",
        help="Training efficacy labels (default: Experiments/efficacy.pt)",
    )
    p.add_argument(
        "--subtract_control",
        action="store_true",
        help="Subtract per-plate averaged control embedding from treated embeddings",
    )
    p.add_argument(
        "--balance",
        action="store_true",
        help="Undersample majority class to balance the training set",
    )
    p.add_argument(
        "--tune",
        action="store_true",
        help="Run randomized hyperparameter search before training",
    )
    p.add_argument("--tune_iter", type=int, default=100, help="Number of random search iterations (default: 50)")

    # ── Inference data ──
    p.add_argument(
        "--inference_embeddings",
        default="Experiments/embeddings_100ppm.pt",
        help="Inference embeddings (default: Experiments/embeddings_100ppm.pt)",
    )
    p.add_argument(
        "--inference_efficacy",
        default="Experiments/efficacy_500ppm.csv",
        help="Ground-truth efficacy for inference evaluation CSV with 'Compound No' and 'Active' columns (default: Experiments/efficacy_500ppm.csv)",
    )

    # ── Threshold ──
    p.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Efficacy threshold for binary classification: >= threshold → active (default: 70)",
    )

    # ── XGBoost hyper-parameters ──
    p.add_argument("--xgb_n_estimators", type=int, default=1000, help="XGBoost rounds (default: 300)")
    p.add_argument("--xgb_max_depth", type=int, default=6, help="XGBoost max depth (default: 6)")
    p.add_argument("--xgb_learning_rate", type=float, default=0.1, help="XGBoost lr (default: 0.1)")
    p.add_argument("--xgb_subsample", type=float, default=0.8, help="XGBoost row subsample (default: 0.8)")
    p.add_argument("--xgb_colsample_bytree", type=float, default=0.8, help="XGBoost col subsample (default: 0.8)")
    p.add_argument("--xgb_early_stopping", type=int, default=20, help="XGBoost early stopping (default: 20)")

    # ── ABMIL hyper-parameters ──
    p.add_argument("--abmil_hidden", type=int, default=128, help="ABMIL attention hidden dim (default: 128)")
    p.add_argument("--abmil_dropout", type=float, default=0.25, help="ABMIL dropout (default: 0.25)")
    p.add_argument("--abmil_lr", type=float, default=1e-3, help="ABMIL learning rate (default: 1e-3)")
    p.add_argument("--abmil_wd", type=float, default=1e-4, help="ABMIL weight decay (default: 1e-4)")
    p.add_argument("--abmil_epochs", type=int, default=200, help="ABMIL training epochs (default: 200)")

    # ── Misc ──
    p.add_argument(
        "--model_name",
        default="dinov2_vits14",
        help="Name of the model that produced the embeddings (included in output path and report)",
    )
    p.add_argument(
        "--output_dir",
        default="Experiments/runs/efficacy_classifier",
        help="Output directory (default: Experiments/runs/efficacy_classifier)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()

    if args.classifier == "xgboost" and not _HAS_XGBOOST:
        raise ImportError("xgboost is required. Install it with:  pip install xgboost")

    # ── Reproducibility ──────────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading embeddings : {args.embeddings}")
    embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    print(f"  {len(embeddings)} compounds in embeddings.")

    print(f"Loading efficacy   : {args.efficacy}")
    efficacy = load_efficacy(args.efficacy)
    print(f"  {len(efficacy)} compounds in efficacy file.")

    # Binarize efficacy
    cid2label = binarize_efficacy(efficacy, threshold=args.threshold)
    n_active = sum(v == 1 for v in cid2label.values())
    n_inactive = sum(v == 0 for v in cid2label.values())
    print(f"  Threshold: {args.threshold}  →  {n_active} active, {n_inactive} inactive")

    # ══════════════════════════════════════════════════════════════════════════
    # Load inference data (shared by both classifiers)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"Loading inference embeddings : {args.inference_embeddings}")
    inf_embeddings = torch.load(args.inference_embeddings, map_location="cpu", weights_only=False)
    print(f"  {len(inf_embeddings)} compounds in inference embeddings.")

    print(f"Loading inference efficacy   : {args.inference_efficacy}")
    inf_efficacy_df = pd.read_csv(args.inference_efficacy)
    inf_cid2label = {
        str(row["Compound No"]): int(row["Active"])
        for _, row in inf_efficacy_df.iterrows()
    }
    print(f"  {len(inf_cid2label)} compounds in inference efficacy file.")

    # ══════════════════════════════════════════════════════════════════════════
    # ABMIL path
    # ══════════════════════════════════════════════════════════════════════════
    if args.classifier == "abmil":
        train_bags, train_labels, _ = build_mil_bags(
            embeddings, cid2label, args.subtract_control,
        )
        print(f"  {len(train_bags)} training compounds (bags), feature dim {train_bags[0].shape[1]}.")
        if len(train_bags) == 0:
            raise RuntimeError("No compounds matched between embeddings and efficacy.")

        model = train_abmil(train_bags, train_labels, args, device)

        inf_bags, inf_labels, cids_inf = build_mil_bags(
            inf_embeddings, inf_cid2label, args.subtract_control,
        )
        y_inf = np.array(inf_labels)
        print(f"  {len(inf_bags)} inference compounds (bags).")
        if len(inf_bags) == 0:
            raise RuntimeError("No compounds matched between inference embeddings and efficacy.")

        inf_preds, inf_proba = infer_abmil(model, inf_bags, device)
        classifier_label = "ABMIL"

    # ══════════════════════════════════════════════════════════════════════════
    # XGBoost path
    # ══════════════════════════════════════════════════════════════════════════
    else:
        # ── Build training features ──────────────────────────────────────
        X_train, y_train, _ = build_mean_latent_features(
            embeddings, cid2label, args.subtract_control,
        )
        print(f"  {X_train.shape[0]} training compounds, feature dim {X_train.shape[1]}.")

        if X_train.shape[0] == 0:
            raise RuntimeError("No compounds matched between embeddings and efficacy.")

        # ── Optionally balance training set (undersample majority) ──────
        if args.balance:
            active_idx = np.where(y_train == 1)[0]
            inactive_idx = np.where(y_train == 0)[0]
            n_minority = min(len(active_idx), len(inactive_idx))
            rng = np.random.RandomState(args.seed)
            active_sampled = rng.choice(active_idx, size=n_minority, replace=False)
            inactive_sampled = rng.choice(inactive_idx, size=n_minority, replace=False)
            balanced_idx = np.sort(np.concatenate([active_sampled, inactive_sampled]))
            X_train = X_train[balanced_idx]
            y_train = y_train[balanced_idx]
            print(f"  Balanced training set: {n_minority} active + {n_minority} inactive = {len(y_train)} compounds.")

        # ── XGBoost parameters (defaults or from CLI) ────────────────────
        xgb_params = dict(
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
        )

        # ── Optional hyperparameter tuning ───────────────────────────────
        if args.tune:
            print(f"\nHyperparameter tuning ({args.tune_iter} iterations, 5-fold CV) ...")
            param_distributions = {
                "n_estimators": [500, 1000, 2000],
                "max_depth": [5, 10, 20],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "min_child_weight": [1, 3, 5, 7],
                "gamma": [0, 0.3, 0.6, 1.0],
                "reg_alpha": [0, 0.1, 0.5, 1.0],
                "reg_lambda": [0.5, 1.0, 5.0],
            }
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
            param_list = list(ParameterSampler(param_distributions, n_iter=args.tune_iter, random_state=args.seed))
            best_score, best_params = -1, None
            for params in tqdm(param_list, desc="Tuning XGBoost"):
                tmp_clf = xgb.XGBClassifier(
                    **params,
                    objective="binary:logistic",
                    eval_metric="auc",
                    use_label_encoder=False,
                    random_state=args.seed,
                    n_jobs=-1,
                )
                scores = cross_val_score(tmp_clf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
                mean_score = scores.mean()
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
            xgb_params = dict(best_params)
            print(f"  Best AUROC: {best_score:.4f}")
            print(f"  Best params: {xgb_params}")

        # ── 5-Fold Cross Validation ──────────────────────────────────────
        print("\n5-Fold Cross Validation on training data ...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        fold_accs, fold_f1s, fold_aurocs = [], [], []

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            fold_clf = xgb.XGBClassifier(
                **xgb_params,
                objective="binary:logistic",
                eval_metric="auc",
                use_label_encoder=False,
                random_state=args.seed,
                n_jobs=-1,
                early_stopping_rounds=args.xgb_early_stopping,
            )
            fold_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            va_preds = fold_clf.predict(X_va)
            va_proba = fold_clf.predict_proba(X_va)[:, 1]
            fold_accs.append(balanced_accuracy_score(y_va, va_preds))
            fold_f1s.append(f1_score(y_va, va_preds, average="weighted", zero_division=0))
            fold_aurocs.append(roc_auc_score(y_va, va_proba))
            print(f"  Fold {fold_idx}: Acc={fold_accs[-1]:.4f}  F1={fold_f1s[-1]:.4f}  AUROC={fold_aurocs[-1]:.4f}")

        print(f"  Mean : Acc={np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}  "
              f"F1={np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}  "
              f"AUROC={np.mean(fold_aurocs):.4f} +/- {np.std(fold_aurocs):.4f}")

        # ── Train final model on all training data ───────────────────────
        clf = xgb.XGBClassifier(
            **xgb_params,
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=args.seed,
            n_jobs=-1,
            early_stopping_rounds=args.xgb_early_stopping,
        )
        print(f"\nTraining final XGBoost on all {X_train.shape[0]} training compounds ...")
        clf.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=500)
        print("Training done.\n")

        # ── Inference features ───────────────────────────────────────────
        X_inf, y_inf, cids_inf = build_mean_latent_features(
            inf_embeddings, inf_cid2label, args.subtract_control,
        )
        print(f"  {X_inf.shape[0]} inference compounds, feature dim {X_inf.shape[1]}.")
        if X_inf.shape[0] == 0:
            raise RuntimeError("No compounds matched between inference embeddings and efficacy.")

        inf_preds = clf.predict(X_inf)
        inf_proba = clf.predict_proba(X_inf)[:, 1]
        classifier_label = "XGBoost"

    # ══════════════════════════════════════════════════════════════════════════
    # Shared evaluation & reporting
    # ══════════════════════════════════════════════════════════════════════════
    inf_acc = balanced_accuracy_score(y_inf, inf_preds)
    inf_f1 = f1_score(y_inf, inf_preds, average="weighted", zero_division=0)
    inf_auroc = roc_auc_score(y_inf, inf_proba)

    report_str = classification_report(
        y_inf, inf_preds, labels=[0, 1],
        target_names=CLASS_NAMES, zero_division=0,
    )
    print("Classification Report (inference):")
    print(report_str)
    print(f"Inference accuracy : {inf_acc:.4f}")
    print(f"Inference F1       : {inf_f1:.4f}")
    print(f"Inference AUROC    : {inf_auroc:.4f}")

    # Save report
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Classifier           : {args.classifier}\n")
        f.write(f"Model                : {args.model_name}\n")
        f.write(f"Train embeddings     : {args.embeddings}\n")
        f.write(f"Train efficacy       : {args.efficacy}\n")
        f.write(f"Inference embeddings : {args.inference_embeddings}\n")
        f.write(f"Inference efficacy   : {args.inference_efficacy}\n")
        f.write(f"Threshold            : {args.threshold}\n")
        f.write(f"Classes              : {CLASS_NAMES}\n\n")
        f.write(report_str)
        f.write(f"\nInference accuracy : {inf_acc:.4f}")
        f.write(f"\nInference F1       : {inf_f1:.4f}")
        f.write(f"\nInference AUROC    : {inf_auroc:.4f}\n")
    print(f"Report saved       : {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_inf, inf_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ylabel="True", xlabel="Predicted",
        title=f"Inference: Efficacy Binary (threshold={args.threshold})",
    )
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # AUROC curve
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_inf, inf_proba, name=classifier_label, ax=ax_roc,
    )
    ax_roc.set_title(f"ROC Curve (AUROC = {inf_auroc:.4f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    fig_roc.tight_layout()
    fig_roc.savefig(output_dir / "auroc_curve.png", dpi=150)
    plt.close(fig_roc)

    # Predictions CSV
    pred_df = pd.DataFrame({
        "compound_id": cids_inf,
        "true_label": [CLASS_NAMES[i] for i in y_inf],
        "predicted_label": [CLASS_NAMES[i] for i in inf_preds],
        "probability_active": inf_proba,
        "correct": [int(t == p) for t, p in zip(y_inf, inf_preds)],
    })
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    print(f"Outputs saved to   : {output_dir}")


if __name__ == "__main__":
    main()
