"""Model comparison, ranking and selection for Phase 3.

Selection is deliberately not "highest R²": models are ranked by grouped
nested-CV MAE (lower is better), with RMSE as the tie-break. Complexity and
stability are reported alongside so the choice can be justified in prose rather
than asserted by the metric alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["configuration", "model"]

COMPARISON_COLUMNS = [
    "configuration",
    "model",
    "n_districts",
    "n_features",
    "mae_mean",
    "mae_sd",
    "rmse_mean",
    "rmse_sd",
    "r2_mean",
    "r2_sd",
    "spearman_mean",
    "spearman_sd",
    "pooled_mae",
    "pooled_rmse",
    "pooled_r2",
    "pooled_spearman",
    "n_candidates",
    "complexity_rank",
]


def load_phase2_reference(metrics_dir) -> dict[str, float]:
    """Grouped-CV MAE of the Phase 2 k-NN all-feature baseline, per configuration."""
    wide = pd.read_csv(metrics_dir / "phase2_baselines_wide.csv")
    grouped = wide[
        (wide["scheme"] == "grouped_by_state") & (wide["model"] == "knn_all_features")
    ]
    return {
        row["configuration"]: float(row["mae"])
        for _, row in grouped.iterrows()
    }


def build_comparison(
    grouped: pd.DataFrame,
    random_summary: pd.DataFrame | None,
    phase2_reference: dict[str, float],
) -> pd.DataFrame:
    """Wide comparison table: grouped primary metrics + random reference + flags."""
    comparison = grouped[COMPARISON_COLUMNS].copy()

    if random_summary is not None and len(random_summary):
        reference = random_summary[
            KEY + ["mae_mean", "rmse_mean", "r2_mean", "spearman_mean", "pooled_mae", "pooled_r2"]
        ].rename(
            columns={
                "mae_mean": "random_mae_mean",
                "rmse_mean": "random_rmse_mean",
                "r2_mean": "random_r2_mean",
                "spearman_mean": "random_spearman_mean",
                "pooled_mae": "random_pooled_mae",
                "pooled_r2": "random_pooled_r2",
            }
        )
        comparison = comparison.merge(reference, on=KEY, how="left")

    dummy = (
        comparison[comparison["model"] == "dummy_mean"]
        .set_index("configuration")["mae_mean"]
    )
    comparison["dummy_mae"] = comparison["configuration"].map(dummy)
    comparison["mae_improvement_vs_dummy_pct"] = (
        100.0 * (comparison["dummy_mae"] - comparison["mae_mean"]) / comparison["dummy_mae"]
    )

    comparison["phase2_knn_mae"] = comparison["configuration"].map(phase2_reference)
    comparison["mae_improvement_vs_phase2_knn_pct"] = (
        100.0
        * (comparison["phase2_knn_mae"] - comparison["mae_mean"])
        / comparison["phase2_knn_mae"]
    )
    comparison["beats_phase2_knn"] = comparison["mae_mean"] < comparison["phase2_knn_mae"]

    comparison["rank"] = (
        comparison.groupby("configuration")["mae_mean"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    selected = (
        comparison.sort_values(["configuration", "mae_mean", "rmse_mean"])
        .groupby("configuration")
        .head(1)
        .index
    )
    comparison["selected"] = comparison.index.isin(selected)
    return comparison.sort_values(["configuration", "rank"]).reset_index(drop=True)


def ab_paired_comparison(
    folds: pd.DataFrame,
    a_label: str = "A_restricted_628",
    b_label: str = "B_augmented_628",
) -> pd.DataFrame:
    """Paired fold-by-fold MAE difference B - A on the shared 628 districts.

    Negative delta means Configuration B had lower fold MAE. With five folds this
    is descriptive, not a significance test; the win count is reported so the
    reader can see whether the gain is consistent or driven by one fold.
    """
    a = (
        folds[folds["configuration"] == a_label]
        .set_index(["model", "fold"])["mae"]
        .rename("a_mae")
    )
    b = (
        folds[folds["configuration"] == b_label]
        .set_index(["model", "fold"])["mae"]
        .rename("b_mae")
    )
    paired = pd.concat([a, b], axis=1).dropna().reset_index()
    if paired.empty:
        return pd.DataFrame(
            columns=[
                "model", "a_mae", "b_mae", "delta_mean", "delta_sd",
                "b_better_folds", "n_folds", "delta_pct",
            ]
        )

    paired["delta"] = paired["b_mae"] - paired["a_mae"]
    summary = (
        paired.groupby("model")
        .agg(
            a_mae=("a_mae", "mean"),
            b_mae=("b_mae", "mean"),
            delta_mean=("delta", "mean"),
            delta_sd=("delta", "std"),
            b_better_folds=("delta", lambda values: int((values < 0).sum())),
            n_folds=("delta", "size"),
        )
        .reset_index()
    )
    summary["delta_pct"] = 100.0 * summary["delta_mean"] / summary["a_mae"]
    return summary.sort_values("delta_mean").reset_index(drop=True)


def fold_stability(grouped: pd.DataFrame) -> pd.DataFrame:
    """Coefficient of variation of fold MAE: how much performance swings by fold."""
    frame = grouped[["configuration", "model", "mae_mean", "mae_sd"]].copy()
    frame["mae_cv"] = frame["mae_sd"] / frame["mae_mean"]
    return frame.sort_values(["configuration", "mae_cv"]).reset_index(drop=True)


def selection_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    chosen = comparison[comparison["selected"]].copy()
    chosen["beats_dummy"] = chosen["mae_mean"] < chosen["dummy_mae"]
    return chosen[
        [
            "configuration",
            "model",
            "n_districts",
            "n_features",
            "mae_mean",
            "mae_sd",
            "rmse_mean",
            "r2_mean",
            "spearman_mean",
            "random_mae_mean",
            "beats_phase2_knn",
            "beats_dummy",
            "complexity_rank",
        ]
    ].reset_index(drop=True)
