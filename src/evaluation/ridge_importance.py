"""Permutation importance for the ratified primary model (Ridge, Configuration B).

The Phase 3 sweep computed permutation importance for the selected benchmark
(SVR-RBF) and for extra trees, so a reader looking for the *primary* model's
importance found nothing (audit defect D-21). This script fills that gap without
touching any frozen artefact: it writes one new CSV.

Protocol — identical to the existing Phase 3 importance run
-----------------------------------------------------------
* the five frozen grouped outer folds, no reshuffling;
* each fold's Ridge (alpha = 100, the ratified value) is fitted on that fold's
  training rows only, with the validated preprocessing pipeline;
* importance is measured on that fold's **held-out** rows by shuffling one
  feature at a time and recomputing mean absolute error;
* 20 shuffles per feature per fold, seed 42;
* a feature counts as stable when its importance is positive in at least four
  of the five folds.

Output: ``evaluation/metrics/phase3_ridge_permutation_importance.csv``

Usage: py -3.14 src/evaluation/ridge_importance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.nested_cv import build_outer_folds  # noqa: E402
from src.training.run_nested_cv import build_configurations  # noqa: E402

METRICS = ROOT / "evaluation" / "metrics"
OUTPUT = METRICS / "phase3_ridge_permutation_importance.csv"
ALPHA = 100.0
N_REPEATS = 20
SEED = 42
CONFIGURATION = "B_augmented_628"


def pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=ALPHA)),
        ]
    )


def main() -> int:
    config = build_configurations(["b628"])["b628"]
    X, y = config["X"], np.asarray(config["y"], dtype=float)
    folds = build_outer_folds(config["groups"], "grouped")

    if X.shape != (628, 77):
        raise SystemExit(f"STOP: expected 628x77, got {X.shape}")

    per_fold = []
    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        model = pipeline().fit(X.iloc[train_idx], y[train_idx])
        result = permutation_importance(
            model,
            X.iloc[test_idx],
            y[test_idx],
            scoring="neg_mean_absolute_error",
            n_repeats=N_REPEATS,
            random_state=SEED,
        )
        # positive importance = shuffling the feature increased MAE
        per_fold.append(pd.Series(result.importances_mean, index=X.columns, name=fold_id))

    table = pd.concat(per_fold, axis=1)
    summary = pd.DataFrame(
        {
            "configuration": CONFIGURATION,
            "model": "ridge",
            "feature": table.index,
            "importance_mean": table.mean(axis=1).to_numpy(),
            "importance_sd": table.std(axis=1, ddof=1).to_numpy(),
            "n_folds_positive": (table > 0).sum(axis=1).to_numpy(),
            "rank_mean": table.rank(ascending=False).mean(axis=1).to_numpy(),
            "rank_sd": table.rank(ascending=False).std(axis=1, ddof=1).to_numpy(),
        }
    ).sort_values("importance_mean", ascending=False)
    summary.insert(2, "rank", np.arange(1, len(summary) + 1))
    summary = summary[
        [
            "configuration", "model", "rank", "feature", "importance_mean",
            "importance_sd", "n_folds_positive", "rank_mean", "rank_sd",
        ]
    ]

    if not np.all(np.isfinite(summary["importance_mean"])):
        raise SystemExit("STOP: non-finite importance value")
    if len(summary) != 77 or summary["feature"].duplicated().any():
        raise SystemExit("STOP: expected 77 unique features")

    summary.to_csv(OUTPUT, index=False)

    # the trace of the score must be non-positive by construction
    print(f"wrote {OUTPUT.relative_to(ROOT)}  ({len(summary)} features, {N_REPEATS} shuffles x 5 folds)")
    print(f"stable (positive in >=4/5 folds): {int((summary['n_folds_positive'] >= 4).sum())} of 77")
    print()
    print(summary.head(10)[["rank", "feature", "importance_mean", "n_folds_positive"]]
          .round(4).to_string(index=False))
    print()
    top_coef = pd.read_csv(METRICS / "phase3_coefficient_stability.csv")
    top_coef = top_coef[(top_coef["configuration"] == CONFIGURATION) & (top_coef["model"] == "ridge")]
    top_coef = set(top_coef.nlargest(10, "abs_coef_mean")["feature"])
    top_perm = set(summary.head(10)["feature"])
    print(f"overlap with the 10 largest |Ridge coefficient|: {len(top_coef & top_perm)}/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
