"""
train_classifier_500ppm.py

Binary XGBoost classifier: predict whether a compound's efficacy is **>= 70**
(active) or **< 70** (inactive) from its DINO embeddings.

DATA FLOW
---------
For each compound:
  1. Collect all treated latent vectors across every plate   →  (M, D)
     (optionally subtract the per-plate averaged control embedding first)
  2. Compute the element-wise mean across M images           →  (D,)
  3. Feed the (N, D) feature matrix into XGBoost             →  binary label

Inputs
------
  --embeddings   Experiments/embeddings_20ppm.pt
                 Output of encode_embeddings.py:
                    { compound_id: { plate_id: {"treated": (N,D), "control": (D,)} } }

  --efficacy     Experiments/efficacy.pt
                 List of dicts: [{"Compound": str, "Efficacy": float}, ...]

Usage examples
--------------
  # default (threshold=70)
  python Experiments/train_classifier_500ppm.py \\
      --embeddings Experiments/embeddings_20ppm.pt \\
      --efficacy   Experiments/efficacy.pt

  # with control subtraction
  python Experiments/train_classifier_500ppm.py \\
      --embeddings       Experiments/embeddings_20ppm.pt \\
      --efficacy         Experiments/efficacy.pt \\
      --subtract_control

  # custom threshold
  python Experiments/train_classifier_500ppm.py \\
      --embeddings Experiments/embeddings_20ppm.pt \\
      --efficacy   Experiments/efficacy.pt \\
      --threshold  80

Output
------
  <output_dir>/
      xgboost_model.json           — saved model
      label_encoder.json            — class names
      training_log.csv              — per-round train/val metrics
      confusion_matrix.png
      classification_report.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
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
# 2.  Mean-latent feature builder
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

    # ── Data ──
    p.add_argument(
        "--embeddings",
        default="Experiments/embeddings_20ppm.pt",
        help="Path to embeddings .pt file (default: Experiments/embeddings_20ppm.pt)",
    )
    p.add_argument(
        "--efficacy",
        default="Experiments/efficacy.pt",
        help="Path to efficacy.pt (default: Experiments/efficacy.pt)",
    )
    p.add_argument(
        "--subtract_control",
        action="store_true",
        help="Subtract per-plate averaged control embedding from treated embeddings",
    )
    p.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of compounds for validation (default: 0.2)",
    )

    # ── Threshold ──
    p.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Efficacy threshold for binary classification: >= threshold → active (default: 70)",
    )

    # ── XGBoost hyper-parameters ──
    p.add_argument("--xgb_n_estimators", type=int, default=300, help="XGBoost rounds (default: 300)")
    p.add_argument("--xgb_max_depth", type=int, default=6, help="XGBoost max depth (default: 6)")
    p.add_argument("--xgb_learning_rate", type=float, default=0.1, help="XGBoost lr (default: 0.1)")
    p.add_argument("--xgb_subsample", type=float, default=0.8, help="XGBoost row subsample (default: 0.8)")
    p.add_argument("--xgb_colsample_bytree", type=float, default=0.8, help="XGBoost col subsample (default: 0.8)")
    p.add_argument("--xgb_early_stopping", type=int, default=20, help="XGBoost early stopping (default: 20)")

    # ── Misc ──
    p.add_argument(
        "--output_dir",
        default="Experiments/runs/efficacy_classifier",
        help="Output directory (default: Experiments/runs/efficacy_classifier)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--save_predictions", action="store_true", help="Save predictions CSV")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()

    if not _HAS_XGBOOST:
        raise ImportError("xgboost is required. Install it with:  pip install xgboost")

    # ── Reproducibility ──────────────────────────────────────────────────────
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
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

    # Save label encoder
    with open(output_dir / "label_encoder.json", "w") as f:
        json.dump({"classes": CLASS_NAMES, "threshold": args.threshold}, f, indent=2)

    # ── Build features ───────────────────────────────────────────────────────
    X, y, cids = build_mean_latent_features(
        embeddings, cid2label, args.subtract_control,
    )
    print(f"  {X.shape[0]} compounds, feature dim {X.shape[1]}, 2 classes.")

    if X.shape[0] == 0:
        raise RuntimeError("No compounds matched between embeddings and efficacy.")

    X_train, X_val, y_train, y_val, _, cids_val = train_test_split(
        X, y, cids,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    print(f"  Train: {len(y_train)}  |  Val: {len(y_val)}")

    # ── Train XGBoost ────────────────────────────────────────────────────────
    metric = "logloss"
    clf = xgb.XGBClassifier(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        objective="binary:logistic",
        eval_metric=metric,
        use_label_encoder=False,
        random_state=args.seed,
        n_jobs=-1,
        early_stopping_rounds=args.xgb_early_stopping,
    )
    print(f"\nTraining XGBoost classifier ({args.xgb_n_estimators} rounds) ...")
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True)

    # ── Evaluation ───────────────────────────────────────────────────────────
    val_preds = clf.predict(X_val)
    val_acc = balanced_accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)

    report_str = classification_report(
        y_val, val_preds, labels=[0, 1],
        target_names=CLASS_NAMES, zero_division=0,
    )
    print("\nClassification Report (validation):")
    print(report_str)
    print(f"Val accuracy : {val_acc:.4f}")
    print(f"Val F1       : {val_f1:.4f}")

    # Save report
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Embeddings : {args.embeddings}\n")
        f.write(f"Efficacy   : {args.efficacy}\n")
        f.write(f"Threshold  : {args.threshold}\n")
        f.write(f"Classes    : {CLASS_NAMES}\n\n")
        f.write(report_str)
        f.write(f"\nVal accuracy : {val_acc:.4f}\nVal F1 : {val_f1:.4f}\n")
    print(f"Report saved       : {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_val, val_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion Matrix:")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ylabel="True", xlabel="Predicted",
        title=f"Efficacy Binary Classification (threshold={args.threshold})",
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
    val_proba = clf.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, val_proba)
    print(f"Val AUROC    : {auroc:.4f}")
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_val, val_proba, name="XGBoost", ax=ax_roc,
    )
    ax_roc.set_title(f"ROC Curve (AUROC = {auroc:.4f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    fig_roc.tight_layout()
    fig_roc.savefig(output_dir / "auroc_curve.png", dpi=150)
    plt.close(fig_roc)

    # Save predictions
    if args.save_predictions:
        pred_df = pd.DataFrame({
            "compound_id": cids_val,
            "true_label": [CLASS_NAMES[i] for i in y_val],
            "predicted_label": [CLASS_NAMES[i] for i in val_preds],
            "probability_active": val_proba,
            "correct": [int(t == p) for t, p in zip(y_val, val_preds)],
        })
        pred_df.to_csv(output_dir / "predictions.csv", index=False)

    print(f"Outputs saved to   : {output_dir}")


if __name__ == "__main__":
    main()
