"""Ridge coefficient stability across outer folds (Configuration B).

Decision 3 of the Phase 3 ratification requires coefficient stability to be
investigated before any strong statement about an individual predictor is made.
This script refits Ridge inside each outer fold with that fold's chosen alpha,
reads the coefficients on the standardised scale, and reports sign consistency,
magnitude spread and rank stability.

Read-only with respect to Phase 1/2 data and the completed sweep results.

    py src/evaluation/coefficient_stability.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.registry import build_models  # noqa: E402
from src.training.nested_cv import build_outer_folds  # noqa: E402
from src.training.run_nested_cv import build_configurations  # noqa: E402

METRICS = ROOT / "evaluation" / "metrics"
CONFIG_ID = "b628"
LABEL = "B_augmented_628"


def main() -> int:
    config = build_configurations([CONFIG_ID])[CONFIG_ID]
    payload = json.loads(
        (METRICS / "phase3_best_parameters.json").read_text(encoding="utf-8")
    )
    per_fold = payload["grouped_best_parameters"]["best_params"][LABEL]["ridge"]["per_fold"]

    outer = build_outer_folds(config["groups"], "grouped")
    columns: dict[str, np.ndarray] = {}
    alphas: list[float] = []

    for index, (train_idx, _) in enumerate(outer):
        estimator = clone(build_models()["ridge"].estimator)
        params = per_fold[index] or {}
        if params:
            estimator.set_params(**params)
        estimator.fit(config["X"].iloc[train_idx], config["y"][train_idx])
        columns[f"fold_{index + 1}"] = estimator.named_steps["model"].coef_.copy()
        alphas.append(float(estimator.named_steps["model"].alpha))

    matrix = pd.DataFrame(columns, index=config["X"].columns)
    positive = (matrix > 0).sum(axis=1)
    table = pd.DataFrame(
        {
            "configuration": LABEL,
            "model": "ridge",
            "feature": matrix.index,
            "coef_mean": matrix.mean(axis=1).to_numpy(),
            "coef_sd": matrix.std(axis=1, ddof=1).to_numpy(),
            "coef_min": matrix.min(axis=1).to_numpy(),
            "coef_max": matrix.max(axis=1).to_numpy(),
            "folds_positive": positive.to_numpy(),
            "sign_consistent": positive.isin([0, len(matrix.columns)]).to_numpy(),
            "rank_sd": matrix.rank(ascending=False).std(axis=1, ddof=1).to_numpy(),
        }
    )
    table["abs_coef_mean"] = table["coef_mean"].abs()
    table["sd_over_mean"] = (
        table["coef_sd"] / table["abs_coef_mean"].replace(0, np.nan)
    )
    table = table.sort_values("abs_coef_mean", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))

    out = METRICS / "phase3_coefficient_stability.csv"
    table.to_csv(out, index=False)

    print(f"Ridge on {LABEL}: {len(table)} predictors, alphas per fold = {alphas}")
    print(f"sign-consistent across all 5 folds: {int(table['sign_consistent'].sum())} "
          f"of {len(table)}")
    print(f"median sd/|mean| across folds: {table['sd_over_mean'].median():.2f}")
    print()
    print("Top 20 by |mean standardised coefficient|")
    print(
        table.head(20)[
            ["rank", "feature", "coef_mean", "coef_sd", "coef_min", "coef_max",
             "folds_positive", "sign_consistent", "rank_sd"]
        ].round(3).to_string(index=False)
    )
    print()
    print("Largest sign flips (features whose coefficient changes sign)")
    flipped = table[~table["sign_consistent"]].reindex(
        table[~table["sign_consistent"]]["abs_coef_mean"]
        .sort_values(ascending=False).index
    )
    print(
        flipped.head(10)[
            ["feature", "coef_mean", "coef_sd", "coef_min", "coef_max", "folds_positive"]
        ].round(3).to_string(index=False)
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
