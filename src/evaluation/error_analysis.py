"""Residual, error and permutation-importance analysis for Phase 3.

Every function here consumes **out-of-fold** predictions only. Training-set
predictions never enter these tables or plots.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from src.training.nested_cv import OUTER_SPLITS, build_outer_folds  # noqa: E402

CONFIG_IDS = {
    "A_primary_640": "a640",
    "A_restricted_628": "a628",
    "B_augmented_628": "b628",
}


def load_districts(processed_dir) -> pd.DataFrame:
    frame = pd.read_csv(processed_dir / "district_analysis.csv")
    return frame[["key", "state", "district", "target_deprivation_rate_pct"]].rename(
        columns={"target_deprivation_rate_pct": "target"}
    )


def residual_table(
    oof: pd.DataFrame, configuration: str, model: str, districts: pd.DataFrame
) -> pd.DataFrame:
    """Per-district residuals. ``residual = actual - predicted`` (> 0 = underprediction)."""
    predictions = oof[
        (oof["configuration"] == configuration) & (oof["model"] == model)
    ][["key", "actual", "predicted"]]
    frame = predictions.merge(districts, on="key", how="left", validate="one_to_one")
    if frame["state"].isna().any():
        raise AssertionError(f"{configuration}/{model}: districts failed to join")
    frame["residual"] = frame["actual"] - frame["predicted"]
    frame["abs_error"] = frame["residual"].abs()
    frame["configuration"] = configuration
    frame["model"] = model
    return frame


def error_summary(frame: pd.DataFrame) -> dict:
    return {
        "n": int(len(frame)),
        "mae": float(frame["abs_error"].mean()),
        "rmse": float(np.sqrt((frame["residual"] ** 2).mean())),
        "bias": float(frame["residual"].mean()),
        "n_overpredicted": int((frame["residual"] < 0).sum()),
        "n_underpredicted": int((frame["residual"] > 0).sum()),
        "p95_abs_error": float(frame["abs_error"].quantile(0.95)),
        "max_abs_error": float(frame["abs_error"].max()),
    }


def errors_by_state(frame: pd.DataFrame, min_districts: int = 1) -> pd.DataFrame:
    grouped = (
        frame.groupby("state")
        .agg(
            n=("residual", "size"),
            mae=("abs_error", "mean"),
            bias=("residual", "mean"),
            max_abs_error=("abs_error", "max"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["n"] >= min_districts]
    return grouped.sort_values("mae", ascending=False).reset_index(drop=True)


def errors_by_target_band(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    banded = frame.copy()
    banded["band"] = pd.qcut(banded["actual"], q=bins, duplicates="drop")
    grouped = (
        banded.groupby("band", observed=True)
        .agg(
            n=("residual", "size"),
            mean_actual=("actual", "mean"),
            mae=("abs_error", "mean"),
            bias=("residual", "mean"),
        )
        .reset_index()
    )
    grouped["band"] = grouped["band"].astype(str)
    return grouped


def extreme_target_summary(frame: pd.DataFrame, quantile: float = 0.95) -> pd.DataFrame:
    cut = frame["actual"].quantile(quantile)
    extreme = frame[frame["actual"] >= cut]
    rest = frame[frame["actual"] < cut]
    rows = []
    for label, group in [
        (f"extreme (top {int((1 - quantile) * 100)}%)", extreme),
        ("remainder", rest),
    ]:
        rows.append(
            {
                "group": label,
                "n": int(len(group)),
                "mean_actual": float(group["actual"].mean()),
                "mae": float(group["abs_error"].mean()),
                "bias": float(group["residual"].mean()),
                "share_underpredicted": float((group["residual"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def largest_errors(frame: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    ordered = frame.sort_values("abs_error", ascending=False).head(n)
    return ordered[
        ["key", "state", "district", "actual", "predicted", "residual", "abs_error"]
    ].reset_index(drop=True)


def permutation_importance_table(
    config_id: str,
    model: str,
    specs: dict,
    grouped_params: dict,
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance refitted per outer fold on held-out data.

    The estimator is rebuilt from the fold's own best hyperparameters and fitted
    on that fold's training rows; importance is measured on the held-out rows.
    """
    from src.training.run_nested_cv import build_configurations

    built = build_configurations([config_id])[config_id]
    X, y, groups = built["X"], built["y"], built["groups"]
    folds = build_outer_folds(groups, "grouped", OUTER_SPLITS, seed)
    per_fold = grouped_params["best_params"][built["label"]][model]["per_fold"]

    columns: dict[str, np.ndarray] = {}
    for index, (train_idx, test_idx) in enumerate(folds):
        estimator = clone(specs[model].estimator)
        if per_fold[index]:
            estimator.set_params(**per_fold[index])
        estimator.fit(X.iloc[train_idx], y[train_idx])
        result = permutation_importance(
            estimator,
            X.iloc[test_idx],
            y[test_idx],
            n_repeats=n_repeats,
            random_state=seed,
            scoring="neg_mean_absolute_error",
        )
        columns[f"fold_{index + 1}"] = result.importances_mean

    frame = pd.DataFrame(columns, index=X.columns)
    ranks = frame.rank(ascending=False, axis=0)
    out = pd.DataFrame(
        {
            "feature": frame.index,
            "importance_mean": frame.mean(axis=1).to_numpy(),
            "importance_sd": frame.std(axis=1, ddof=1).to_numpy(),
            "n_folds_positive": (frame > 0).sum(axis=1).to_numpy(),
            "rank_mean": ranks.mean(axis=1).to_numpy(),
            "rank_sd": ranks.std(axis=1, ddof=1).to_numpy(),
        }
    )
    out = out.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.insert(0, "model", model)
    out.insert(0, "configuration", built["label"])
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_predicted_vs_actual(frame: pd.DataFrame, path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(frame["actual"], frame["predicted"], s=16, alpha=0.6, edgecolor="none")
    low = float(min(frame["actual"].min(), frame["predicted"].min()))
    high = float(max(frame["actual"].max(), frame["predicted"].max()))
    ax.plot([low, high], [low, high], color="black", lw=1, ls="--")
    ax.set_xlabel("Actual dilapidated share (%)")
    ax.set_ylabel("Out-of-fold prediction (%)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residual_distribution(frame: pd.DataFrame, path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(frame["residual"], bins=40, color="#4c72b0", edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(frame["residual"].mean(), color="crimson", lw=1, ls="--",
               label=f"mean {frame['residual'].mean():.2f}")
    ax.set_xlabel("Residual (actual - predicted, pp)")
    ax.set_ylabel("Districts")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals_vs_predicted(frame: pd.DataFrame, path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.scatter(frame["predicted"], frame["residual"], s=16, alpha=0.6, edgecolor="none")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Out-of-fold prediction (%)")
    ax.set_ylabel("Residual (pp)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mae_by_fold(folds: pd.DataFrame, configuration: str, models: list[str], path) -> None:
    subset = folds[(folds["configuration"] == configuration) & (folds["model"].isin(models))]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for model, group in subset.groupby("model"):
        group = group.sort_values("fold")
        ax.plot(group["fold"], group["mae"], marker="o", label=model)
    ax.set_xticks(sorted(subset["fold"].unique()))
    ax.set_xlabel("Outer fold (held-out states)")
    ax.set_ylabel("Fold MAE (pp)")
    ax.set_title(configuration, fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(comparison: pd.DataFrame, path) -> None:
    configurations = list(comparison["configuration"].unique())
    fig, axes = plt.subplots(1, len(configurations), figsize=(4.6 * len(configurations), 5.4), sharex=False)
    if len(configurations) == 1:
        axes = [axes]
    for ax, configuration in zip(axes, configurations):
        subset = comparison[comparison["configuration"] == configuration].sort_values(
            "mae_mean", ascending=False
        )
        colors = ["#c44e52" if flag else "#4c72b0" for flag in subset["selected"]]
        ax.barh(subset["model"], subset["mae_mean"], xerr=subset["mae_sd"], color=colors)
        ax.set_title(configuration, fontsize=9)
        ax.set_xlabel("Grouped nested-CV MAE (pp)")
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ab_comparison(ab: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ordered = ab.sort_values("delta_mean")
    ax.barh(ordered["model"], ordered["delta_mean"], color="#55a868")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Paired fold MAE change, B - A (pp; negative = B better)")
    ax.set_title("Configuration B vs A on the same 628 districts", fontsize=10)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(importance: pd.DataFrame, path, top_n: int = 15) -> None:
    subset = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(subset) + 1.4))
    ax.barh(subset["feature"], subset["importance_mean"], xerr=subset["importance_sd"], color="#4c72b0")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Permutation importance (MAE increase when shuffled, pp)")
    ax.set_title(f"{importance['configuration'].iloc[0]} / {importance['model'].iloc[0]}", fontsize=10)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
