"""
classifier_tuning.py

Randomized hyperparameter search for XGBoost and ABMIL classifiers
used in efficacy-500ppm binary classification.
"""

import argparse
import copy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_val_score
from tqdm import tqdm

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

try:
    from .classifier_utils import GatedABMIL, train_abmil, infer_abmil, LogSumExpMIL, train_logsumexp, infer_logsumexp
except ImportError:
    from classifier_utils import GatedABMIL, train_abmil, infer_abmil, LogSumExpMIL, train_logsumexp, infer_logsumexp


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ABMIL tuning
# ═══════════════════════════════════════════════════════════════════════════════


def tune_abmil(
    bags: List[torch.Tensor],
    labels: List[int],
    eval_bags: List[torch.Tensor],
    eval_labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict:
    """Random search over ABMIL hyperparameters, return best config."""
    param_space = {
        "hidden": [64, 128, 256],
        "dropout": [0.1, 0.25, 0.4],
        "lr": [5e-5, 1e-4, 2e-4, 5e-4],
        "wd": [1e-5, 1e-4, 1e-3],
        "instance_dropout": [0.0, 0.1, 0.2],
    }

    rng = np.random.RandomState(args.seed)
    n_trials = args.abmil_tune_iter
    print(f"\nABMIL hyperparameter tuning ({n_trials} trials, {args.abmil_tune_epochs} epochs each) ...")

    best_auroc = -1.0
    best_params = {}
    results = []

    for trial in range(n_trials):
        config = {k: rng.choice(v) for k, v in param_space.items()}

        trial_args = copy.deepcopy(args)
        trial_args.abmil_hidden = int(config["hidden"])
        trial_args.abmil_dropout = float(config["dropout"])
        trial_args.abmil_lr = float(config["lr"])
        trial_args.abmil_wd = float(config["wd"])
        trial_args.abmil_instance_dropout = float(config["instance_dropout"])
        trial_args.abmil_epochs = args.abmil_tune_epochs
        trial_args.abmil_patience = 5

        print(f"  Trial {trial+1}/{n_trials}  hidden={config['hidden']}  dropout={config['dropout']:.2f}  "
              f"lr={config['lr']:.1e}  wd={config['wd']:.1e}  inst_drop={config['instance_dropout']:.1f}")

        torch.manual_seed(args.seed + trial)
        model = train_abmil(
            bags, labels, trial_args, device,
            eval_bags=eval_bags, eval_labels=eval_labels,
            verbose=True,
        )

        preds, probas = infer_abmil(model, eval_bags, device)
        auroc = roc_auc_score(eval_labels, probas)
        trial_f1 = f1_score(eval_labels, preds, average="weighted", zero_division=0)
        results.append({**config, "auroc": auroc, "f1": trial_f1})
        print(f"  →  AUROC={auroc:.4f}  F1={trial_f1:.4f}{'  ★ BEST' if auroc > best_auroc else ''}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_params = dict(config)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n  Best trial AUROC: {best_auroc:.4f}")
    print(f"  Best params: {best_params}")

    results_df = pd.DataFrame(results).sort_values("auroc", ascending=False)
    print(f"\n  Top 5 configs:")
    print(results_df.head().to_string(index=False))

    return best_params


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  XGBoost tuning
# ═══════════════════════════════════════════════════════════════════════════════


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> Dict:
    """Random search over XGBoost hyperparameters, return best config."""
    if not _HAS_XGBOOST:
        raise ImportError("xgboost is required for tuning. Install with: pip install xgboost")

    param_distributions = {
        "n_estimators": [1000, 3000, 5000],
        "max_depth": [2, 4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.3, 0.6, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 5.0],
    }

    print(f"\nHyperparameter tuning ({args.tune_iter} iterations, 5-fold CV) ...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    param_list = list(ParameterSampler(
        param_distributions, n_iter=args.tune_iter, random_state=args.seed,
    ))
    best_score, best_params = -1, None

    for i, params in enumerate(tqdm(param_list, desc="Tuning XGBoost"), 1):
        tmp_clf = xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=args.seed,
            n_jobs=-1,
        )
        scores = cross_val_score(
            tmp_clf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        mean_score = scores.mean()
        is_best = mean_score > best_score
        if is_best:
            best_score = mean_score
            best_params = params
        tqdm.write(f"  [{i}/{len(param_list)}] AUROC={mean_score:.4f}  "
                   f"depth={params.get('max_depth')}  lr={params.get('learning_rate')}  "
                   f"n_est={params.get('n_estimators')}  sub={params.get('subsample')}  "
                   f"col={params.get('colsample_bytree')}  mcw={params.get('min_child_weight')}  "
                   f"gamma={params.get('gamma')}  alpha={params.get('reg_alpha')}  "
                   f"lambda={params.get('reg_lambda')}"
                   f"{'  ★ BEST' if is_best else ''}")

    print(f"  Best AUROC: {best_score:.4f}")
    print(f"  Best params: {best_params}")
    return dict(best_params)


# ═══════════════════════════════════════════════════════════════════════════════
# 2b. LogSumExp MIL tuning
# ═══════════════════════════════════════════════════════════════════════════════


def tune_logsumexp(
    bags: List[torch.Tensor],
    labels: List[int],
    eval_bags: List[torch.Tensor],
    eval_labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict:
    """Random search over LogSumExp MIL hyperparameters, return best config."""
    param_space = {
        "hidden": [64, 128, 256],
        "dropout": [0.1, 0.25, 0.4],
        "lr": [5e-5, 1e-4, 2e-4, 5e-4],
        "wd": [1e-5, 1e-4, 1e-3],
        "init_r": [0.1, 0.5, 1.0, 2.0, 5.0],
        "instance_dropout": [0.0, 0.1, 0.2],
    }

    rng = np.random.RandomState(args.seed)
    n_trials = args.lse_tune_iter
    print(f"\nLogSumExp MIL hyperparameter tuning ({n_trials} trials, {args.lse_tune_epochs} epochs each) ...")

    best_auroc = -1.0
    best_params = {}
    results = []

    for trial in range(n_trials):
        config = {k: rng.choice(v) for k, v in param_space.items()}

        trial_args = copy.deepcopy(args)
        trial_args.lse_hidden = int(config["hidden"])
        trial_args.lse_dropout = float(config["dropout"])
        trial_args.lse_lr = float(config["lr"])
        trial_args.lse_wd = float(config["wd"])
        trial_args.lse_init_r = float(config["init_r"])
        trial_args.lse_instance_dropout = float(config["instance_dropout"])
        trial_args.lse_epochs = args.lse_tune_epochs
        trial_args.lse_patience = 5

        print(f"  Trial {trial+1}/{n_trials}  hidden={config['hidden']}  dropout={config['dropout']:.2f}  "
              f"lr={config['lr']:.1e}  wd={config['wd']:.1e}  init_r={config['init_r']:.1f}  "
              f"inst_drop={config['instance_dropout']:.1f}")

        torch.manual_seed(args.seed + trial)
        model = train_logsumexp(
            bags, labels, trial_args, device,
            eval_bags=eval_bags, eval_labels=eval_labels,
            verbose=True,
        )

        preds, probas = infer_logsumexp(model, eval_bags, device)
        auroc = roc_auc_score(eval_labels, probas)
        trial_f1 = f1_score(eval_labels, preds, average="weighted", zero_division=0)
        results.append({**config, "auroc": auroc, "f1": trial_f1})
        print(f"  ->  AUROC={auroc:.4f}  F1={trial_f1:.4f}{'  * BEST' if auroc > best_auroc else ''}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_params = dict(config)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n  Best trial AUROC: {best_auroc:.4f}")
    print(f"  Best params: {best_params}")

    results_df = pd.DataFrame(results).sort_values("auroc", ascending=False)
    print(f"\n  Top 5 configs:")
    print(results_df.head().to_string(index=False))

    return best_params


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CatBoost tuning
# ═══════════════════════════════════════════════════════════════════════════════


def tune_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> Dict:
    """Random search over CatBoost hyperparameters, return best config."""
    if not _HAS_CATBOOST:
        raise ImportError("catboost is required for tuning. Install with: pip install catboost")

    param_distributions = {
        "iterations": [500, 1000, 2000],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "l2_leaf_reg": [1.0, 3.0, 5.0, 7.0],
        "subsample": [0.7, 0.8, 1.0],
        "rsm": [0.7, 0.8, 1.0],
    }

    print(f"\nCatBoost hyperparameter tuning ({args.tune_iter} iterations, 5-fold CV) ...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    param_list = list(ParameterSampler(
        param_distributions, n_iter=args.tune_iter, random_state=args.seed,
    ))
    best_score, best_params = -1, None

    for i, params in enumerate(tqdm(param_list, desc="Tuning CatBoost"), 1):
        scores = []
        for tr_idx, va_idx in cv.split(X_train, y_train):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            tmp_clf = CatBoostClassifier(
                **params,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=args.seed,
                verbose=0,
                early_stopping_rounds=50,
            )
            tmp_clf.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
            va_proba = tmp_clf.predict_proba(X_va)[:, 1]
            scores.append(roc_auc_score(y_va, va_proba))
        mean_score = np.mean(scores)
        is_best = mean_score > best_score
        if is_best:
            best_score = mean_score
            best_params = params
        tqdm.write(f"  [{i}/{len(param_list)}] AUROC={mean_score:.4f}  "
                   f"depth={params.get('depth')}  lr={params.get('learning_rate')}  "
                   f"iter={params.get('iterations')}  sub={params.get('subsample')}  "
                   f"rsm={params.get('rsm')}  l2={params.get('l2_leaf_reg')}"
                   f"{'  ★ BEST' if is_best else ''}")

    print(f"  Best AUROC: {best_score:.4f}")
    print(f"  Best params: {best_params}")
    return dict(best_params)
