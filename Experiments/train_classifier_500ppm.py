"""
train_classifier_500ppm.py

Train a binary XGBoost classifier on embeddings_20ppm + efficacy.pt, then run
inference on embeddings_100ppm and evaluate against efficacy_500ppm.pt.

Workflow
--------
  1. TRAIN  — fit XGBoost on embeddings_20ppm / efficacy.pt  (no logging)
  2. INFER  — predict on embeddings_100ppm, evaluate vs efficacy_500ppm.pt
             Logs: classification report, confusion matrix, AUROC curve,
             predictions CSV.

Usage
-----
  python Experiments/train_classifier_500ppm.py \\
      --embeddings           Experiments/embeddings_20ppm.pt \\
      --efficacy             Experiments/efficacy.pt \\
      --inference_embeddings Experiments/embeddings_100ppm.pt \\
      --inference_efficacy   Experiments/efficacy_500ppm.pt

Output
------
  <output_dir>/
      label_encoder.json
      classification_report.txt
      confusion_matrix.png
      auroc_curve.png
      predictions.csv
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

    # ── Inference data ──
    p.add_argument(
        "--inference_embeddings",
        default="Experiments/embeddings_100ppm.pt",
        help="Inference embeddings (default: Experiments/embeddings_100ppm.pt)",
    )
    p.add_argument(
        "--inference_efficacy",
        default="Experiments/efficacy_500ppm.pt",
        help="Ground-truth efficacy for inference evaluation (default: Experiments/efficacy_500ppm.pt)",
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

    # ── Build training features ──────────────────────────────────────────────
    X_train, y_train, _ = build_mean_latent_features(
        embeddings, cid2label, args.subtract_control,
    )
    print(f"  {X_train.shape[0]} training compounds, feature dim {X_train.shape[1]}.")

    if X_train.shape[0] == 0:
        raise RuntimeError("No compounds matched between embeddings and efficacy.")

    # ── Train XGBoost (no logging) ───────────────────────────────────────────
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
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=True)
    print("Training done.\n")

    # ══════════════════════════════════════════════════════════════════════════
    # INFERENCE on embeddings_100ppm  →  evaluate vs efficacy_500ppm
    # ══════════════════════════════════════════════════════════════════════════
    print(f"Loading inference embeddings : {args.inference_embeddings}")
    inf_embeddings = torch.load(args.inference_embeddings, map_location="cpu", weights_only=False)
    print(f"  {len(inf_embeddings)} compounds in inference embeddings.")

    print(f"Loading inference efficacy   : {args.inference_efficacy}")
    inf_efficacy = load_efficacy(args.inference_efficacy)
    print(f"  {len(inf_efficacy)} compounds in inference efficacy file.")

    inf_cid2label = binarize_efficacy(inf_efficacy, threshold=args.threshold)

    X_inf, y_inf, cids_inf = build_mean_latent_features(
        inf_embeddings, inf_cid2label, args.subtract_control,
    )
    print(f"  {X_inf.shape[0]} inference compounds, feature dim {X_inf.shape[1]}.")

    if X_inf.shape[0] == 0:
        raise RuntimeError("No compounds matched between inference embeddings and efficacy.")

    # ── Inference predictions ────────────────────────────────────────────────
    inf_preds = clf.predict(X_inf)
    inf_proba = clf.predict_proba(X_inf)[:, 1]
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
        y_inf, inf_proba, name="XGBoost", ax=ax_roc,
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
