"""Baseline models and metrics for Phase 2.

Three baselines, all fitted strictly inside each training fold:

* ``national_mean`` — training-fold mean.
* ``state_mean``    — training-fold mean per state. Under grouped CV a held-out
  state is unseen, so it falls back to the training-fold global mean. That is
  reported explicitly: it is why the state-mean baseline is weak under grouped CV.
* ``knn``           — k-nearest districts on six headline features, scaled with
  statistics from the training fold only.

No advanced model is trained in this phase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

METRIC_NAMES = ("mae", "rmse", "r2", "spearman")
STATE_FALLBACK_PREFIX = "state_mean"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    result = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": np.nan,
    }
    if np.std(y_pred) > 0 and np.std(y_true) > 0:
        result["spearman"] = float(
            pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
        )
    return result


def predict_national_mean(train_idx, test_idx, y, state, X) -> np.ndarray:
    return np.full(len(test_idx), float(np.mean(y[train_idx])))


def predict_state_mean(train_idx, test_idx, y, state, X) -> np.ndarray:
    sums: dict[str, list[float]] = {}
    for group, value in zip(state[train_idx], y[train_idx]):
        sums.setdefault(str(group), []).append(float(value))
    means = {group: float(np.mean(values)) for group, values in sums.items()}
    fallback = float(np.mean(y[train_idx]))
    return np.array([means.get(str(group), fallback) for group in state[test_idx]])


def predict_knn(train_idx, test_idx, y, state, X) -> np.ndarray:
    """k-NN on the supplied headline features; scaling statistics come from train only."""
    scaler = StandardScaler().fit(X.iloc[train_idx])
    model = KNeighborsRegressor(n_neighbors=15, weights="distance")
    model.fit(scaler.transform(X.iloc[train_idx]), y[train_idx])
    return model.predict(scaler.transform(X.iloc[test_idx]))


# ``knn_headline`` and ``knn_all_features`` share one implementation; they differ
# only in which feature frame is passed, so the A/B comparison tests the added
# features rather than a different algorithm.
BASELINES = {
    "national_mean": predict_national_mean,
    "state_mean": predict_state_mean,
    "knn_headline": predict_knn,
    "knn_all_features": predict_knn,
}


def run_baselines(
    label: str,
    y: np.ndarray,
    state: np.ndarray,
    X: pd.DataFrame,
    folds_by_scheme: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    n_districts: int,
    knn_X: pd.DataFrame | None = None,
) -> list[dict]:
    """Evaluate every baseline on every fold scheme; return long-format rows.

    ``knn_X`` restricts the k-NN baseline to a small headline feature set; the
    mean baselines ignore features entirely.
    """
    knn_X = X if knn_X is None else knn_X
    inputs = {
        "national_mean": X,
        "state_mean": X,
        "knn_headline": knn_X,
        "knn_all_features": X,
    }
    rows: list[dict] = []
    for scheme, folds in folds_by_scheme.items():
        for model_name, predictor in BASELINES.items():
            per_fold: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
            for train_idx, test_idx in folds:
                preds = predictor(train_idx, test_idx, y, state, inputs[model_name])
                scores = regression_metrics(y[test_idx], preds)
                for metric in METRIC_NAMES:
                    per_fold[metric].append(scores[metric])
            for metric, values in per_fold.items():
                array = np.array(values, dtype=float)
                defined = array[np.isfinite(array)]
                rows.append(
                    {
                        "configuration": label,
                        "n_districts": n_districts,
                        "scheme": scheme,
                        "model": model_name,
                        "metric": metric,
                        "mean": round(float(defined.mean()), 4) if len(defined) else None,
                        "sd": round(float(defined.std(ddof=1)), 4) if len(defined) > 1 else None,
                        "n_folds": len(values),
                        "n_folds_defined": int(len(defined)),
                    }
                )
    return rows


def summarise(rows: list[dict]) -> pd.DataFrame:
    """Wide table for the report: configuration x scheme x model."""
    frame = pd.DataFrame(rows)
    wide = frame.pivot_table(
        index=["configuration", "n_districts", "scheme", "model"],
        columns="metric",
        values="mean",
    ).reset_index()
    order = ["configuration", "n_districts", "scheme", "model", "mae", "rmse", "r2", "spearman"]
    return wide[[c for c in order if c in wide.columns]].sort_values(
        ["configuration", "scheme", "model"]
    )
