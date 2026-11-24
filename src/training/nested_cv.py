"""Nested cross-validation for Phase 3.

Design
------
* Outer loop: 5-fold GroupKFold on states (primary) or shuffled KFold (reference).
* Inner loop: 4-fold of the same kind, used only by GridSearchCV.
* A state can never appear in both train and test of an outer fold. The inner
  split is built from the outer training rows only, so held-out states are never
  visible during tuning either.
* Every pipeline step (median imputation, scaling) is fitted inside the training
  portion, never on the full dataset.

The module returns out-of-fold predictions so that a single set of predictions
per district backs the pooled metrics, the error analysis and the plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold

from src.evaluation.metrics import METRIC_NAMES, pooled_metrics, summarise_folds
from src.models.registry import ModelSpec

OUTER_SPLITS = 5
INNER_SPLITS = 4


def build_outer_folds(
    groups: np.ndarray, scheme: str, n_splits: int = OUTER_SPLITS, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups)
    if scheme == "grouped":
        splitter = GroupKFold(n_splits=n_splits)
        return [
            (tr, te)
            for tr, te in splitter.split(np.zeros(len(groups)), groups=groups)
        ]
    if scheme == "random":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in splitter.split(np.arange(len(groups)))]
    raise ValueError(f"unknown scheme: {scheme!r}")


def _inner_splitter(scheme: str, n_splits: int, seed: int = 0):
    if scheme == "grouped":
        return GroupKFold(n_splits=n_splits)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def run_model(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    spec: ModelSpec,
    outer_folds: list[tuple[np.ndarray, np.ndarray]],
    scheme: str,
    inner_splits: int = INNER_SPLITS,
    n_jobs: int = 1,
) -> tuple[np.ndarray, list[dict], dict]:
    """Fit one model across the outer folds; return OOF predictions and records."""
    if X.isna().any().any():
        raise AssertionError(f"{spec.name}: input matrix contains missing values")

    oof = np.full(len(y), np.nan)
    fold_rows: list[dict] = []

    for fold_id, (train_idx, test_idx) in enumerate(outer_folds, start=1):
        X_train, y_train = X.iloc[train_idx], y[train_idx]

        if spec.param_grid:
            search = GridSearchCV(
                spec.estimator,
                spec.param_grid,
                cv=_inner_splitter(scheme, inner_splits),
                scoring="neg_mean_absolute_error",
                refit=True,
                n_jobs=n_jobs,
            )
            search.fit(
                X_train,
                y_train,
                groups=groups[train_idx] if scheme == "grouped" else None,
            )
            estimator = search.best_estimator_
            best_params = dict(search.best_params_)
            n_candidates = len(search.cv_results_["params"])
            best_inner_mae = float(-search.best_score_)
        else:
            spec.estimator.fit(X_train, y_train)
            estimator = spec.estimator
            best_params = {}
            n_candidates = 1
            best_inner_mae = float("nan")

        predictions = estimator.predict(X.iloc[test_idx])
        if not np.all(np.isfinite(predictions)):
            raise AssertionError(f"{spec.name}: non-finite predictions in fold {fold_id}")
        oof[test_idx] = predictions

        record = {
            "fold": fold_id,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_states_train": int(len(np.unique(groups[train_idx]))),
            "n_states_test": int(len(np.unique(groups[test_idx]))),
            "n_candidates": int(n_candidates),
            "best_inner_mae": best_inner_mae,
            "best_params": best_params,
        }
        from src.evaluation.metrics import regression_metrics

        record.update(regression_metrics(y[test_idx], predictions))
        fold_rows.append(record)

    if np.isnan(oof).any():
        raise AssertionError(f"{spec.name}: some districts have no out-of-fold prediction")
    return oof, fold_rows, pooled_metrics(y, oof)


def run_configuration(
    label: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    keys: pd.Series,
    models: dict[str, ModelSpec],
    scheme: str,
    seed: int = 42,
    n_jobs: int = 1,
    log=print,
) -> tuple[list[dict], list[dict], list[pd.DataFrame], dict]:
    """Run every model on one configuration and one CV scheme."""
    outer_folds = build_outer_folds(groups, scheme, OUTER_SPLITS, seed)
    summary_rows: list[dict] = []
    fold_rows: list[dict] = []
    oof_frames: list[pd.DataFrame] = []
    best_params: dict = {}

    for name, spec in models.items():
        oof, folds, pooled = run_model(
            X, y, groups, spec, outer_folds, scheme, INNER_SPLITS, n_jobs
        )
        summary = {
            "configuration": label,
            "scheme": scheme,
            "model": name,
            "n_districts": int(len(y)),
            "n_features": int(X.shape[1]),
            "n_outer_folds": len(outer_folds),
            "n_candidates": int(spec.n_configs),
            "complexity_rank": spec.complexity_rank,
            **summarise_folds(folds),
            **pooled,
        }
        summary_rows.append(summary)

        for record in folds:
            record.update({"configuration": label, "scheme": scheme, "model": name})
        fold_rows.extend(folds)

        oof_frames.append(
            pd.DataFrame(
                {
                    "configuration": label,
                    "scheme": scheme,
                    "model": name,
                    "key": keys.to_numpy(),
                    "actual": y,
                    "predicted": oof,
                }
            )
        )
        best_params[name] = {
            "per_fold": [r["best_params"] for r in folds],
            "n_configs": spec.n_configs,
            "param_grid": spec.param_grid,
        }
        log(
            f"  [{label}/{scheme}] {name:<24} MAE {summary['mae_mean']:.4f} "
            f"± {summary['mae_sd'] if summary['mae_sd'] == summary['mae_sd'] else float('nan'):.4f}  "
            f"R2 {summary['r2_mean']:.4f}  rho {summary['spearman_mean']:.4f}"
        )

    return summary_rows, fold_rows, oof_frames, best_params
