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

  # CatBoost with balanced class weights (default)
  python Experiments/train_synthesis_classifier.py \\
      --embeddings       Experiments/embeddings.pt \\
      --metadata         data/compound_metadata.csv \\
      --classifier       catboost

  # CatBoost with softer balancing and control subtraction
  python Experiments/train_synthesis_classifier.py \\
      --embeddings       Experiments/embeddings.pt \\
      --metadata         data/compound_metadata.csv \\
      --classifier       catboost \\
      --subtract_control \\
      --cb_auto_class_weights SqrtBalanced \\
      --cb_iterations 1000 --cb_depth 8

  # XGBoost with hyperparameter tuning
  python Experiments/train_synthesis_classifier.py \\
      --embeddings       Experiments/embeddings.pt \\
      --metadata         data/compound_metadata.csv \\
      --classifier       xgboost \\
      --tune --tune_iter 50

  # CatBoost with hyperparameter tuning
  python Experiments/train_synthesis_classifier.py \\
      --embeddings       Experiments/embeddings.pt \\
      --metadata         data/compound_metadata.csv \\
      --classifier       catboost \\
      --tune --tune_iter 50

Output
------
  <output_dir>/
      best_model.pt | xgboost_model.json  — saved model
      label_encoder.json                   — { "classes": [...], "str2idx": {...} }
      training_log.csv                     — per-epoch train/val metrics
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

# ── project imports ──────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from classifier_utils import (
    load_efficacy_data,
    build_efficacy_features,
    build_mil_bags,
    build_mean_latent_features,
    GatedABMIL,
    train_abmil,
    infer_abmil,
    build_label_encoder,
    save_label_encoder,
    save_results,
)
from classifier_tuning import _tune_xgboost, _tune_catboost





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
    p.add_argument("--normalize_before_subtract", action="store_true",
                   help="L2-normalize treated and control embeddings before subtraction (requires --subtract_control)")
    p.add_argument("--val_split", type=float, default=0.2,
                   help="Fraction of compounds used for validation (early stopping / tuning). Default: 0.2")
    p.add_argument("--test_split", type=float, default=0.2,
                   help="Fraction of compounds held out for final evaluation (not seen during training). Default: 0.15")

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

    # ---- Tuning ----
    p.add_argument("--tune", action="store_true",
                   help="Run randomized hyperparameter search before final training")
    p.add_argument("--tune_iter", type=int, default=50,
                   help="Number of random search iterations. Default: 50")

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
    p.add_argument("--classifier", choices=["abmil", "xgboost", "catboost"],
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

    # ---- CatBoost hyper-parameters ----
    p.add_argument("--cb_iterations", type=int, default=500,
                   help="[CatBoost] Number of boosting iterations. Default: 500")
    p.add_argument("--cb_depth", type=int, default=5,
                   help="[CatBoost] Tree depth. Default: 5")
    p.add_argument("--cb_learning_rate", type=float, default=0.1,
                   help="[CatBoost] Learning rate. Default: 0.1")
    p.add_argument("--cb_l2_leaf_reg", type=float, default=1.0,
                   help="[CatBoost] L2 regularization. Default: 1.0")
    p.add_argument("--cb_auto_class_weights", choices=["None", "Balanced", "SqrtBalanced"],
                   default="Balanced",
                   help="[CatBoost] Auto class weighting. Default: Balanced")
    p.add_argument("--cb_early_stopping", type=int, default=50,
                   help="[CatBoost] Early stopping rounds. Default: 50")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 5b. Gated ABMIL training pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _run_abmil(
    args: argparse.Namespace,
    embeddings: Dict,
    df: pd.DataFrame,
    str2idx: Dict[str, int],
    classes: List[str],
    num_classes: int,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Train a Gated ABMIL classifier on per-compound bags of embeddings."""
    bags, labels, cids = build_mil_bags(
        embeddings,
        compound_col=df[args.compound_col],
        label_col=df[args.label_col],
        label2idx=str2idx,
        subtract_control=args.subtract_control,
        normalize_before_subtract=args.normalize_before_subtract,
    )
    print(f"  {len(bags)} compounds with valid bags, feature dim {bags[0].shape[1]}.")

    if len(bags) == 0:
        raise RuntimeError(
            "Dataset is empty. Check that compound IDs in the embeddings file "
            "match the compound IDs in the metadata."
        )

    # ── Remove classes with fewer than min_compounds_per_class compounds ──
    min_cpc = max(args.min_compounds_per_class, 2)
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr)
    valid_classes = set(np.where(class_counts >= min_cpc)[0])
    keep_mask = np.array([li in valid_classes for li in labels_arr])
    n_removed = len(labels) - keep_mask.sum()
    if n_removed > 0:
        removed_names = sorted({classes[li] for li in labels_arr if li not in valid_classes})
        print(f"  Dropped {n_removed} compound(s) from {len(removed_names)} "
              f"class(es) with <{min_cpc} compounds: {removed_names}")
        bags   = [b for b, k in zip(bags, keep_mask) if k]
        labels = [l for l, k in zip(labels, keep_mask) if k]
        cids   = [c for c, k in zip(cids, keep_mask) if k]

    # Remap labels to contiguous 0..K-1
    remaining = sorted(set(labels))
    old2new = {old: new for new, old in enumerate(remaining)}
    labels = [old2new[li] for li in labels]
    classes = [classes[old] for old in remaining]
    num_classes = len(classes)
    print(f"  {num_classes} classes after filtering, {len(labels)} compounds remaining.")

    # ── Train / val / test split ─────────────────────────────────────────────
    n_total = len(bags)
    n_test = max(1, int(n_total * args.test_split))
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val - n_test
    if n_train < 1:
        raise RuntimeError(f"Not enough data for 3-way split: {n_total} total, need at least 3.")
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_bags   = [bags[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_bags     = [bags[i] for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]
    test_bags    = [bags[i] for i in test_idx]
    test_labels  = [labels[i] for i in test_idx]
    test_cids    = [cids[i] for i in test_idx]
    print(f"  Train: {n_train}  |  Val: {n_val}  |  Test: {n_test}")

    # ── Train on train+val (val only used for final eval split) ──────────────
    trainval_bags   = train_bags + val_bags
    trainval_labels = train_labels + val_labels
    model = train_abmil(trainval_bags, trainval_labels, num_classes, args, device)

    # ── Save model ──────────────────────────────────────────────────────────
    torch.save(model.state_dict(), output_dir / "best_model.pt")

    # ── Evaluate on held-out test set ────────────────────────────────────────
    test_preds, test_probs = infer_abmil(model, test_bags, device)
    test_true = np.array(test_labels)

    save_results(
        val_true=test_true,
        val_preds=test_preds,
        val_probs=test_probs,
        val_cids=test_cids,
        classes=classes,
        num_classes=num_classes,
        output_dir=output_dir,
        cm_title="Confusion Matrix — Gated ABMIL",
        file_suffix="",
        report_header=(
            f"Classifier       : abmil\n"
            f"Embeddings       : {args.embeddings}\n"
            f"Subtract control : {args.subtract_control}\n"
            f"Normalize before subtract : {args.normalize_before_subtract}\n\n"
        ),
        save_predictions=args.save_predictions,
    )


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
            normalize_before_subtract=args.normalize_before_subtract,
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

    # ── Train / val / test split ─────────────────────────────────────────────
    strat = y if len(np.unique(y)) > 1 else None
    X_trainval, X_test, y_trainval, y_test, cids_trainval, cids_test = train_test_split(
        X, y, cids,
        test_size=args.test_split,
        random_state=args.seed,
        stratify=strat,
    )
    strat_tv = y_trainval if len(np.unique(y_trainval)) > 1 else None
    relative_val = args.val_split / (1.0 - args.test_split)
    X_train, X_val, y_train, y_val, cids_train, cids_val = train_test_split(
        X_trainval, y_trainval, cids_trainval,
        test_size=relative_val,
        random_state=args.seed,
        stratify=strat_tv,
    )
    print(f"  Train: {len(y_train)}  |  Val: {len(y_val)}  |  Test: {len(y_test)}")

    # ── Optional hyperparameter tuning ────────────────────────────────────────
    if args.tune:
        best_params = _tune_xgboost(
            X_train, y_train, X_val, y_val, num_classes, args,
        )
        args.xgb_n_estimators = best_params["n_estimators"]
        args.xgb_max_depth = best_params["max_depth"]
        args.xgb_learning_rate = best_params["learning_rate"]
        args.xgb_subsample = best_params["subsample"]
        args.xgb_colsample_bytree = best_params["colsample_bytree"]
        print(f"\n  Final XGBoost config: n_estimators={args.xgb_n_estimators}  "
              f"max_depth={args.xgb_max_depth}  lr={args.xgb_learning_rate}  "
              f"subsample={args.xgb_subsample}  colsample={args.xgb_colsample_bytree}")

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

    # ── Record best iteration, retrain on train+val ──────────────────────────
    best_n = clf.best_iteration + 1 if hasattr(clf, 'best_iteration') and clf.best_iteration is not None else args.xgb_n_estimators
    print(f"\nRetraining XGBoost on train+val ({len(y_trainval)} compounds, {best_n} rounds) ...")
    xgb_final_params = {k: v for k, v in xgb_params.items() if k != 'early_stopping_rounds'}
    xgb_final_params['n_estimators'] = best_n
    clf = xgb.XGBClassifier(**xgb_final_params)
    clf.fit(X_trainval, y_trainval, verbose=500)

    # ── Evaluation on held-out test set ───────────────────────────────────────
    test_preds = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)  # (N, num_classes)

    _input_path = args.efficacy if args.efficacy else args.embeddings
    emb_stem = Path(_input_path).stem              # e.g. "embeddings_tiltedvae_dim100"

    save_results(
        val_true=y_test,
        val_preds=test_preds,
        val_probs=test_probs,
        val_cids=cids_test,
        classes=classes,
        num_classes=num_classes,
        output_dir=output_dir,
        cm_title=f"Confusion Matrix — {emb_stem}",
        file_suffix=f"_{emb_stem}",
        report_header=(
            f"Input file       : {_input_path}\n"
            f"Subtract control : {args.subtract_control}\n"
            f"Normalize before subtract : {args.normalize_before_subtract}\n\n"
        ),
        save_predictions=args.save_predictions,
    )

    # ── Save model ───────────────────────────────────────────────────────────
    model_path = output_dir / f"xgboost_model_{emb_stem}.json"
    clf.save_model(str(model_path))
    print(f"Model saved to     : {model_path}")

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


# ═══════════════════════════════════════════════════════════════════════════════
# 6b. CatBoost training pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _run_catboost(
    args: argparse.Namespace,
    embeddings: Optional[Dict],
    df: pd.DataFrame,
    str2idx: Dict[str, int],
    classes: List[str],
    num_classes: int,
    output_dir: Path,
    efficacy: Optional[Dict[str, float]] = None,
) -> None:
    """Train a CatBoost classifier on per-compound features."""
    if not _HAS_CATBOOST:
        raise ImportError(
            "catboost is required for --classifier catboost. "
            "Install it with:  pip install catboost"
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
            normalize_before_subtract=args.normalize_before_subtract,
        )
    print(f"  {X.shape[0]} compounds with valid features.")
    print(f"  Feature dim (D) : {X.shape[1]}")

    if X.shape[0] == 0:
        raise RuntimeError(
            "Dataset is empty. Check that compound IDs in the input file "
            "match the compound IDs in the metadata."
        )

    # ── Remove classes with fewer than min_compounds_per_class compounds ──
    min_cpc = max(args.min_compounds_per_class, 2)
    class_counts = np.bincount(y)
    valid_classes = set(np.where(class_counts >= min_cpc)[0])
    keep_mask = np.array([yi in valid_classes for yi in y])
    n_removed = len(y) - keep_mask.sum()
    if n_removed > 0:
        removed_names = sorted({classes[yi] for yi in y if yi not in valid_classes})
        print(f"  Dropped {n_removed} compound(s) from {len(removed_names)} "
              f"class(es) with <{min_cpc} compounds: {removed_names}")
        X, y, cids = X[keep_mask], y[keep_mask], [c for c, k in zip(cids, keep_mask) if k]

    # Remap labels to contiguous 0..K-1
    remaining = sorted(set(y.tolist()))
    old2new = {old: new for new, old in enumerate(remaining)}
    y = np.array([old2new[yi] for yi in y])
    classes = [classes[old] for old in remaining]
    num_classes = len(classes)
    print(f"  {num_classes} classes after filtering, {len(y)} compounds remaining.")

    # ── Class distribution ───────────────────────────────────────────────────
    for ci, cname in enumerate(classes):
        print(f"    {cname}: {(y == ci).sum()} compounds")

    # ── Train / val / test split ─────────────────────────────────────────────
    strat = y if len(np.unique(y)) > 1 else None
    X_trainval, X_test, y_trainval, y_test, cids_trainval, cids_test = train_test_split(
        X, y, cids,
        test_size=args.test_split,
        random_state=args.seed,
        stratify=strat,
    )
    strat_tv = y_trainval if len(np.unique(y_trainval)) > 1 else None
    relative_val = args.val_split / (1.0 - args.test_split)
    X_train, X_val, y_train, y_val, cids_train, cids_val = train_test_split(
        X_trainval, y_trainval, cids_trainval,
        test_size=relative_val,
        random_state=args.seed,
        stratify=strat_tv,
    )
    print(f"  Train: {len(y_train)}  |  Val: {len(y_val)}  |  Test: {len(y_test)}")

    # ── Optional hyperparameter tuning ────────────────────────────────────────
    if args.tune:
        best_params = _tune_catboost(
            X_train, y_train, X_val, y_val, num_classes, args,
        )
        args.cb_iterations = best_params["iterations"]
        args.cb_depth = best_params["depth"]
        args.cb_learning_rate = best_params["learning_rate"]
        args.cb_l2_leaf_reg = best_params["l2_leaf_reg"]
        args.cb_auto_class_weights = best_params["auto_class_weights"]
        print(f"\n  Final CatBoost config: iterations={args.cb_iterations}  "
              f"depth={args.cb_depth}  lr={args.cb_learning_rate}  "
              f"l2_leaf_reg={args.cb_l2_leaf_reg}  "
              f"class_weights={args.cb_auto_class_weights}")

    # ── CatBoost model ───────────────────────────────────────────────────────
    auto_cw = None if args.cb_auto_class_weights == "None" else args.cb_auto_class_weights
    cb_params = dict(
        iterations=args.cb_iterations,
        depth=args.cb_depth,
        learning_rate=args.cb_learning_rate,
        l2_leaf_reg=args.cb_l2_leaf_reg,
        auto_class_weights=auto_cw,
        loss_function="MultiClass" if num_classes > 2 else "Logloss",
        eval_metric="TotalF1:average=Macro" if num_classes > 2 else "F1",
        random_seed=args.seed,
        verbose=50,
        early_stopping_rounds=args.cb_early_stopping,
    )

    clf = CatBoostClassifier(**cb_params)

    print(f"\nTraining CatBoost ({args.cb_iterations} iters, depth={args.cb_depth}, "
          f"lr={args.cb_learning_rate}, class_weights={auto_cw}) ...")
    clf.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
    )

    # ── Record best iteration, retrain on train+val ──────────────────────────
    best_n = clf.get_best_iteration() + 1 if clf.get_best_iteration() is not None else args.cb_iterations
    print(f"\nRetraining CatBoost on train+val ({len(y_trainval)} compounds, {best_n} iterations) ...")
    cb_final_params = {k: v for k, v in cb_params.items() if k != 'early_stopping_rounds'}
    cb_final_params['iterations'] = best_n
    clf = CatBoostClassifier(**cb_final_params)
    clf.fit(X_trainval, y_trainval)

    # ── Evaluation on held-out test set ───────────────────────────────────────
    test_preds = clf.predict(X_test).astype(int).ravel()
    test_probs = clf.predict_proba(X_test)

    _input_path = args.efficacy if args.efficacy else args.embeddings
    emb_stem = Path(_input_path).stem

    save_results(
        val_true=y_test,
        val_preds=test_preds,
        val_probs=test_probs,
        val_cids=cids_test,
        classes=classes,
        num_classes=num_classes,
        output_dir=output_dir,
        cm_title=f"Confusion Matrix — CatBoost — {emb_stem}",
        file_suffix=f"_{emb_stem}",
        report_header=(
            f"Classifier       : catboost\n"
            f"Input file       : {_input_path}\n"
            f"Subtract control : {args.subtract_control}\n"
            f"Normalize before subtract : {args.normalize_before_subtract}\n"
            f"Auto class weights : {auto_cw}\n\n"
        ),
        save_predictions=args.save_predictions,
    )

    # ── Save model ───────────────────────────────────────────────────────────
    model_path = output_dir / f"catboost_model_{emb_stem}.cbm"
    clf.save_model(str(model_path))
    print(f"Model saved to     : {model_path}")

    # ── Training log ─────────────────────────────────────────────────────────
    evals = clf.get_evals_result()
    if evals and "validation" in evals:
        metrics = evals["validation"]
        first_key = next(iter(metrics))
        log_df = pd.DataFrame({
            "epoch": list(range(1, len(metrics[first_key]) + 1)),
            **{k: v for k, v in metrics.items()},
        })
        log_df.to_csv(output_dir / f"training_log_{emb_stem}.csv", index=False)


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
    output_dir = Path(args.output_dir) / args.classifier
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

    # ── Route to classifier ───────────────────────────────────────────────
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
    elif args.classifier == "catboost":
        _run_catboost(
            args=args,
            embeddings=embeddings,
            df=df,
            str2idx=str2idx,
            classes=classes,
            num_classes=num_classes,
            output_dir=output_dir,
            efficacy=efficacy,
        )
    else:
        if embeddings is None:
            raise ValueError("Gated ABMIL requires --embeddings (not --efficacy).")
        _run_abmil(
            args=args,
            embeddings=embeddings,
            df=df,
            str2idx=str2idx,
            classes=classes,
            num_classes=num_classes,
            output_dir=output_dir,
            device=device,
        )


if __name__ == "__main__":
    main()
