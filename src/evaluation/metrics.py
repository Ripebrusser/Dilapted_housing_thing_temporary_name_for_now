"""Regression metrics and aggregation for Phase 3.

``regression_metrics`` is imported from the Phase 2 baseline module rather than
reimplemented, so both phases score predictions identically.

Conventions:
* MAE and RMSE are in percentage points (the target's unit).
* ``spearman`` is ``NaN`` when predictions are constant (rank correlation is then
  undefined); it is never silently replaced by zero.
* Negative R² is retained and reported as-is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.baselines import regression_metrics  # re-export, single implementation

METRIC_NAMES = ("mae", "rmse", "r2", "spearman")
HIGHER_IS_BETTER = {"mae": False, "rmse": False, "r2": True, "spearman": True}

__all__ = ["regression_metrics", "METRIC_NAMES", "summarise_folds", "pooled_metrics", "metric_table"]


def summarise_folds(fold_rows: list[dict], metrics: tuple[str, ...] = METRIC_NAMES) -> dict:
    """Mean and sample SD across outer folds, ignoring undefined values per metric."""
    summary: dict[str, float] = {}
    for metric in metrics:
        values = np.array([row.get(metric, np.nan) for row in fold_rows], dtype=float)
        defined = values[np.isfinite(values)]
        summary[f"{metric}_mean"] = float(defined.mean()) if len(defined) else np.nan
        summary[f"{metric}_sd"] = (
            float(defined.std(ddof=1)) if len(defined) > 1 else np.nan
        )
        summary[f"{metric}_n_defined"] = int(len(defined))
    return summary


def pooled_metrics(y_true: np.ndarray, oof: np.ndarray) -> dict:
    """Metrics over all out-of-fold predictions pooled (one prediction per district)."""
    return {f"pooled_{k}": v for k, v in regression_metrics(y_true, oof).items()}


def metric_table(rows: list[dict], sort_by: str = "mae_mean") -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if sort_by in frame.columns:
        frame = frame.sort_values(sort_by, ascending=HIGHER_IS_BETTER.get(sort_by.replace("_mean", ""), False))
    return frame.reset_index(drop=True)
