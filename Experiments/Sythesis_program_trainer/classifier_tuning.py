"""
classifier_tuning.py

Randomized hyperparameter search for XGBoost and CatBoost classifiers.
"""

import argparse
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

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


def _tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    args: argparse.Namespace,
) -> Dict:
    """Random search over XGBoost hyperparameters, return best config."""

    param_space = {
        "n_estimators": [100, 300, 500, 1000],
        "max_depth": [3, 4, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 3.0, 5.0, 10.0],
    }

    rng = np.random.RandomState(args.seed)
    n_trials = args.tune_iter
    objective = "multi:softprob" if num_classes > 2 else "binary:logistic"
    eval_metric = "mlogloss" if num_classes > 2 else "logloss"

    print(f"\nXGBoost hyperparameter tuning ({n_trials} trials) ...")

    best_f1 = -1.0
    best_params = {}
    results = []

    for trial in range(n_trials):
        config = {k: rng.choice(v) for k, v in param_space.items()}

        xgb_params = dict(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            learning_rate=float(config["learning_rate"]),
            subsample=float(config["subsample"]),
            colsample_bytree=float(config["colsample_bytree"]),
            min_child_weight=int(config["min_child_weight"]),
            gamma=float(config["gamma"]),
            reg_lambda=float(config["reg_lambda"]),
            objective=objective,
            eval_metric=eval_metric,
            use_label_encoder=False,
            random_state=args.seed,
            n_jobs=-1,
            early_stopping_rounds=args.xgb_early_stopping,
        )
        if num_classes > 2:
            xgb_params["num_class"] = num_classes

        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_preds = clf.predict(X_val)
        trial_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        trial_acc = balanced_accuracy_score(y_val, val_preds)

        results.append({**config, "f1": trial_f1, "balanced_acc": trial_acc})
        is_best = trial_f1 > best_f1
        if is_best:
            best_f1 = trial_f1
            best_params = dict(config)

        print(f"  Trial {trial+1:3d}/{n_trials}  depth={config['max_depth']}  "
              f"lr={config['learning_rate']:.3f}  subsample={config['subsample']:.1f}  "
              f"colsample={config['colsample_bytree']:.1f}  mcw={config['min_child_weight']}  "
              f"gamma={config['gamma']:.1f}  lambda={config['reg_lambda']:.1f}  "
              f"F1={trial_f1:.4f}  Acc={trial_acc:.4f}{'  ★ BEST' if is_best else ''}")

    print(f"\n  Best trial F1: {best_f1:.4f}")
    print(f"  Best params: {best_params}")

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print(f"\n  Top 5 configs:")
    print(results_df.head().to_string(index=False))

    return {
        "n_estimators": int(best_params["n_estimators"]),
        "max_depth": int(best_params["max_depth"]),
        "learning_rate": float(best_params["learning_rate"]),
        "subsample": float(best_params["subsample"]),
        "colsample_bytree": float(best_params["colsample_bytree"]),
    }


def _tune_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    args: argparse.Namespace,
) -> Dict:
    """Random search over CatBoost hyperparameters, return best config."""
    param_space = {
        "iterations": [100],
        "depth": [3, 6, 9],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1.0, 5.0, 10.0, 20.0],
        "auto_class_weights": ["Balanced", "SqrtBalanced"],
        # "random_strength": [0.5, 1.0, 2.0],
        # "bagging_temperature": [0.0, 0.5, 1.0, 2.0],
    }

    rng = np.random.RandomState(args.seed)
    n_trials = args.tune_iter
    loss_fn = "MultiClass" if num_classes > 2 else "Logloss"
    eval_metric = "TotalF1:average=Weighted" if num_classes > 2 else "F1"

    print(f"\nCatBoost hyperparameter tuning ({n_trials} trials) ...")

    best_f1 = -1.0
    best_params = {}
    results = []

    for trial in range(n_trials):
        config = {k: rng.choice(v) for k, v in param_space.items()}
        auto_cw = None if config["auto_class_weights"] == "None" else config["auto_class_weights"]

        cb_params = dict(
            iterations=int(config["iterations"]),
            depth=int(config["depth"]),
            learning_rate=float(config["learning_rate"]),
            l2_leaf_reg=float(config["l2_leaf_reg"]),
            auto_class_weights=auto_cw,
            loss_function=loss_fn,
            eval_metric=eval_metric,
            random_seed=args.seed,
            verbose=0,
            early_stopping_rounds=args.cb_early_stopping,
        )

        clf = CatBoostClassifier(**cb_params)
        clf.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
        )

        val_preds = clf.predict(X_val).astype(int).ravel()
        trial_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        trial_acc = balanced_accuracy_score(y_val, val_preds)

        results.append({**config, "f1": trial_f1, "balanced_acc": trial_acc})
        is_best = trial_f1 > best_f1
        if is_best:
            best_f1 = trial_f1
            best_params = dict(config)

        print(f"  Trial {trial+1:3d}/{n_trials}  depth={config['depth']}  "
              f"lr={config['learning_rate']:.3f}  l2={config['l2_leaf_reg']:.1f}  "
              f"cw={config['auto_class_weights']:<12s}  "
              f"F1={trial_f1:.4f}  Acc={trial_acc:.4f}{'  ★ BEST' if is_best else ''}")

    print(f"\n  Best trial F1: {best_f1:.4f}")
    print(f"  Best params: {best_params}")

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print(f"\n  Top 5 configs:")
    print(results_df.head().to_string(index=False))

    return {
        "iterations": int(best_params["iterations"]),
        "depth": int(best_params["depth"]),
        "learning_rate": float(best_params["learning_rate"]),
        "l2_leaf_reg": float(best_params["l2_leaf_reg"]),
        "auto_class_weights": str(best_params["auto_class_weights"]),
    }
