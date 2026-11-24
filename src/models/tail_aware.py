"""Tail-aware experiment arms for the authorised Phase 4 investigation.

Implements exactly the five registered arms in
``reports/proposals/EXTREME_DISTRICT_REMEDY_PROPOSAL.md`` and nothing else:

===== ==========================================================================
E0    reference: the ratified Phase 3 Ridge OOF predictions, reused and never refitted
E1    weighted Ridge, threshold q75 of training targets, lambda = 2
E2    weighted Ridge, threshold q75, lambda = 4
E3    weighted Ridge, threshold q90, lambda = 4
E4a   linear quantile regression, tau = 0.75
E4b   linear quantile regression, tau = 0.90
===== ==========================================================================

Isolation contract
------------------
Weights are computed *inside* whichever fit consumes them:

* for an inner split, the threshold is the ``quantile`` of that split's
  **inner-training** targets, and the weights come from those targets only;
* for the outer refit, the threshold and weights are **recomputed from the
  complete outer training set** after the hyperparameter is fixed;
* outer test targets never enter a threshold, a weight, a preprocessing
  statistic, a hyperparameter choice or a model fit.

The inner search is implemented by hand rather than with ``GridSearchCV``:
scikit-learn accepts one weight vector per search, which cannot express
"recompute the threshold for every inner training split". Doing it explicitly
also makes the isolation testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
INNER_SPLITS = 4


@dataclass(frozen=True)
class TailArm:
    """One registered experiment arm."""

    name: str
    kind: str  # "weighted_ridge" | "quantile"
    alpha_grid: tuple[float, ...]
    quantile: float | None = None  # weighting threshold quantile
    lam: float | None = None  # weight boost for districts at/above the threshold
    tau: float | None = None  # conditional quantile being estimated
    description: str = ""
    extra: dict = field(default_factory=dict, compare=False)

    @property
    def is_weighted(self) -> bool:
        return self.kind == "weighted_ridge"

    def as_dict(self) -> dict:
        return {
            "arm": self.name,
            "kind": self.kind,
            "alpha_grid": list(self.alpha_grid),
            "quantile": self.quantile,
            "lambda": self.lam,
            "tau": self.tau,
            "description": self.description,
        }


def build_arms() -> dict[str, TailArm]:
    """The registered grid. No arm or hyperparameter may be added at runtime."""
    return {
        "E1": TailArm(
            name="E1",
            kind="weighted_ridge",
            alpha_grid=(30.0, 100.0, 300.0),
            quantile=0.75,
            lam=2.0,
            description="Mild weighting: top training quartile receives 3x weight (normalised mean 1).",
        ),
        "E2": TailArm(
            name="E2",
            kind="weighted_ridge",
            alpha_grid=(30.0, 100.0, 300.0),
            quantile=0.75,
            lam=4.0,
            description="Strong weighting: top training quartile receives 5x weight (normalised mean 1).",
        ),
        "E3": TailArm(
            name="E3",
            kind="weighted_ridge",
            alpha_grid=(30.0, 100.0, 300.0),
            quantile=0.90,
            lam=4.0,
            description="Threshold weighting: top training decile receives 5x weight (normalised mean 1).",
        ),
        "E4a": TailArm(
            name="E4a",
            kind="quantile",
            alpha_grid=(0.001, 0.01, 0.1),
            tau=0.75,
            description="Linear quantile regression at tau = 0.75 (conditional quantile, not a mean).",
        ),
        "E4b": TailArm(
            name="E4b",
            kind="quantile",
            alpha_grid=(0.001, 0.01, 0.1),
            tau=0.90,
            description="Linear quantile regression at tau = 0.90 (conditional quantile, not a mean).",
        ),
    }


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #


def weighting_threshold(y_train, quantile: float) -> float:
    """Quantile of the *training* targets supplied to this call."""
    values = np.asarray(y_train, dtype=float)
    if values.size == 0:
        raise ValueError("cannot compute a weighting threshold from an empty target vector")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1), got {quantile!r}")
    return float(np.quantile(values, quantile))


def sample_weights(y_train, quantile: float, lam: float) -> tuple[np.ndarray, float]:
    """Weights from the supplied training targets only, normalised to mean 1.

    Returns ``(weights, threshold)``. The caller decides which target vector to
    pass: an inner-training fold or the complete outer-training fold.
    """
    values = np.asarray(y_train, dtype=float)
    threshold = weighting_threshold(values, quantile)
    weights = 1.0 + float(lam) * (values >= threshold).astype(float)
    weights = weights / weights.mean()
    return weights, threshold


def weighted_row_count(y_train, threshold: float) -> int:
    return int(np.sum(np.asarray(y_train, dtype=float) >= threshold))


# --------------------------------------------------------------------------- #
# estimators and losses
# --------------------------------------------------------------------------- #


def build_estimator(arm: TailArm, alpha: float) -> Pipeline:
    """Imputation and scaling are fitted inside the pipeline, per fit."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if arm.kind == "weighted_ridge":
        steps.append(("model", Ridge(alpha=float(alpha))))
    elif arm.kind == "quantile":
        steps.append(
            (
                "model",
                QuantileRegressor(
                    quantile=float(arm.tau), alpha=float(alpha), solver="highs"
                ),
            )
        )
    else:
        raise ValueError(f"unknown arm kind: {arm.kind!r}")
    return Pipeline(steps)


def fit_estimator(estimator: Pipeline, X, y, weights: np.ndarray | None = None) -> Pipeline:
    if weights is None:
        estimator.fit(X, y)
    else:
        estimator.fit(X, y, model__sample_weight=np.asarray(weights, dtype=float))
    return estimator


def pinball_loss(y_true, y_pred, tau: float) -> float:
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(tau * residual, (tau - 1.0) * residual)))


def selection_score(arm: TailArm, y_true, y_pred) -> float:
    """The registered inner-selection criterion for this arm."""
    if arm.kind == "quantile":
        return pinball_loss(y_true, y_pred, float(arm.tau))
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


# --------------------------------------------------------------------------- #
# inner search
# --------------------------------------------------------------------------- #


def inner_folds(groups_train, n_splits: int = INNER_SPLITS):
    """GroupKFold over the outer-training states only."""
    groups_train = np.asarray(groups_train)
    features = np.zeros((len(groups_train), 1))
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(features, groups=groups_train))


def inner_search(
    X_train,
    y_train,
    groups_train,
    arm: TailArm,
    fold_id: int,
    n_splits: int = INNER_SPLITS,
    diagnostics: list | None = None,
) -> tuple[float, list[dict]]:
    """Choose alpha on inner folds, recomputing weights per inner training split.

    Returns ``(best_alpha, table)``. Every candidate fit is recorded in
    ``diagnostics`` when a list is supplied.
    """
    splits = inner_folds(groups_train, n_splits)
    table: list[dict] = []

    for alpha in arm.alpha_grid:
        scores: list[float] = []
        for inner_id, (inner_train, inner_valid) in enumerate(splits, start=1):
            X_fit, y_fit = X_train.iloc[inner_train], np.asarray(y_train)[inner_train]
            if arm.is_weighted:
                # threshold and weights come from THIS inner-training split only
                weights, threshold = sample_weights(y_fit, arm.quantile, arm.lam)
                n_weighted = weighted_row_count(y_fit, threshold)
                weight_mean = float(weights.mean())
            else:
                weights, threshold, n_weighted, weight_mean = None, float("nan"), 0, float("nan")

            estimator = fit_estimator(build_estimator(arm, alpha), X_fit, y_fit, weights)
            predictions = estimator.predict(X_train.iloc[inner_valid])
            if not np.all(np.isfinite(predictions)):
                raise AssertionError(f"{arm.name}: non-finite inner prediction (fold {fold_id})")
            score = selection_score(arm, np.asarray(y_train)[inner_valid], predictions)
            scores.append(score)

            if diagnostics is not None:
                diagnostics.append(
                    {
                        "arm": arm.name,
                        "fold": fold_id,
                        "stage": "inner",
                        "inner_split": inner_id,
                        "alpha": float(alpha),
                        "threshold": threshold,
                        "n_train": int(len(inner_train)),
                        "n_valid": int(len(inner_valid)),
                        "n_weighted": n_weighted,
                        "weight_mean": weight_mean,
                        "inner_score": score,
                        "selected": False,
                    }
                )

        table.append(
            {
                "alpha": float(alpha),
                "mean_score": float(np.mean(scores)),
                "sd_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else float("nan"),
            }
        )

    best = min(table, key=lambda row: row["mean_score"])
    return best["alpha"], table


# --------------------------------------------------------------------------- #
# one outer fold
# --------------------------------------------------------------------------- #


def fit_fold(
    X,
    y,
    groups,
    train_idx,
    test_idx,
    arm: TailArm,
    fold_id: int,
    n_splits: int = INNER_SPLITS,
    diagnostics: list | None = None,
    return_estimator: bool = False,
):
    """Tune inside the training rows, then refit once and predict the test rows.

    ``return_estimator=True`` additionally returns the fitted outer pipeline, used
    by the isolation tests to compare coefficients rather than predictions.
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    X_train, y_train = X.iloc[train_idx], y[train_idx]

    best_alpha, table = inner_search(
        X_train, y_train, groups[train_idx], arm, fold_id, n_splits, diagnostics
    )

    weights, threshold, n_weighted, weight_mean = None, float("nan"), 0, float("nan")
    if arm.is_weighted:
        # recomputed on the COMPLETE outer training set, after alpha is fixed
        weights, threshold = sample_weights(y_train, arm.quantile, arm.lam)
        n_weighted = weighted_row_count(y_train, threshold)
        weight_mean = float(weights.mean())

    estimator = fit_estimator(build_estimator(arm, best_alpha), X_train, y_train, weights)
    predictions = estimator.predict(X.iloc[test_idx])
    if not np.all(np.isfinite(predictions)):
        raise AssertionError(f"{arm.name}: non-finite prediction in fold {fold_id}")

    if diagnostics is not None:
        diagnostics.append(
            {
                "arm": arm.name,
                "fold": fold_id,
                "stage": "outer_refit",
                "inner_split": 0,
                "alpha": float(best_alpha),
                "threshold": threshold,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(test_idx)),
                "n_weighted": n_weighted,
                "weight_mean": weight_mean,
                "inner_score": float("nan"),
                "selected": True,
            }
        )

    record = {
        "arm": arm.name,
        "fold": fold_id,
        "best_alpha": float(best_alpha),
        "threshold": threshold,
        "n_weighted": n_weighted,
        "weight_mean": weight_mean,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_states_train": int(len(np.unique(groups[train_idx]))),
        "n_states_test": int(len(np.unique(groups[test_idx]))),
        "inner_table": table,
    }
    if return_estimator:
        return predictions, record, estimator
    return predictions, record


def run_arm(
    arm: TailArm,
    X,
    y,
    groups,
    folds,
    n_splits: int = INNER_SPLITS,
    diagnostics: list | None = None,
    require_full_coverage: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """Run one arm across the frozen outer folds. Returns OOF predictions and records."""
    y = np.asarray(y, dtype=float)
    oof = np.full(len(y), np.nan)
    records: list[dict] = []

    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        predictions, record = fit_fold(
            X, y, groups, train_idx, test_idx, arm, fold_id, n_splits, diagnostics
        )
        oof[test_idx] = predictions
        records.append(record)

    if require_full_coverage and not np.all(np.isfinite(oof)):
        raise AssertionError(f"{arm.name}: some districts have no out-of-fold prediction")
    return oof, records
