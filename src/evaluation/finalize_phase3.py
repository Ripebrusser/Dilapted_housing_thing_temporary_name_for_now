"""Assemble Phase 3 deliverables from the nested-CV artefacts.

Reads the ``_phase3_*`` intermediates written by ``run_nested_cv.py`` and writes
the deliverable filenames, the comparison table, the error analysis and the plots.

    py src/evaluation/finalize_phase3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation import error_analysis as ea  # noqa: E402
from src.evaluation.compare import (  # noqa: E402
    ab_paired_comparison,
    build_comparison,
    fold_stability,
    load_phase2_reference,
    selection_summary,
)
from src.models.registry import build_models  # noqa: E402

METRICS = ROOT / "evaluation" / "metrics"
PREDICTIONS = ROOT / "evaluation" / "predictions"
PLOTS = ROOT / "evaluation" / "plots"
ERRORS = ROOT / "evaluation" / "error_analysis"
PROCESSED = ROOT / "datasets" / "processed"


def main() -> int:
    PLOTS.mkdir(parents=True, exist_ok=True)
    ERRORS.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)

    grouped = pd.read_csv(METRICS / "_phase3_grouped_summary.csv")
    random_summary = pd.read_csv(METRICS / "_phase3_random_summary.csv")
    grouped_folds = pd.read_csv(METRICS / "_phase3_grouped_folds.csv")
    random_folds = pd.read_csv(METRICS / "_phase3_random_folds.csv")
    grouped_oof = pd.read_csv(PREDICTIONS / "_phase3_grouped_oof.csv")
    random_oof = pd.read_csv(PREDICTIONS / "_phase3_random_oof.csv")
    grouped_params = json.loads(
        (METRICS / "_phase3_grouped_params.json").read_text(encoding="utf-8")
    )
    random_params = json.loads(
        (METRICS / "_phase3_random_params.json").read_text(encoding="utf-8")
    )

    # --- required result tables -------------------------------------------
    grouped.to_csv(METRICS / "phase3_nested_cv_results.csv", index=False)
    random_summary.to_csv(METRICS / "phase3_random_cv_results.csv", index=False)
    pd.concat([grouped_folds, random_folds], ignore_index=True).to_csv(
        METRICS / "phase3_fold_results.csv", index=False
    )
    pd.concat([grouped_oof, random_oof], ignore_index=True).to_csv(
        PREDICTIONS / "phase3_oof_predictions.csv", index=False
    )

    comparison = build_comparison(
        grouped, random_summary, load_phase2_reference(METRICS)
    )
    comparison.to_csv(METRICS / "phase3_model_comparison.csv", index=False)
    fold_stability(grouped).to_csv(METRICS / "phase3_fold_stability.csv", index=False)

    ab = ab_paired_comparison(grouped_folds)
    ab.to_csv(METRICS / "phase3_ab_comparison.csv", index=False)

    selected = selection_summary(comparison)
    selected.to_csv(METRICS / "phase3_selected_models.csv", index=False)

    best_payload = {
        "selection_rule": (
            "minimum grouped nested-CV mean MAE per configuration; "
            "RMSE used only as a tie-break"
        ),
        "selected": {
            row["configuration"]: {
                "model": row["model"],
                "grouped_mae_mean": row["mae_mean"],
                "grouped_mae_sd": row["mae_sd"],
                "grouped_rmse_mean": row["rmse_mean"],
                "grouped_r2_mean": row["r2_mean"],
                "random_mae_mean": row.get("random_mae_mean"),
            }
            for _, row in selected.iterrows()
        },
        "grouped_best_parameters": grouped_params,
        "random_best_parameters": random_params,
    }
    (METRICS / "phase3_best_parameters.json").write_text(
        json.dumps(best_payload, indent=2, default=str), encoding="utf-8"
    )

    # --- error analysis on out-of-fold predictions -------------------------
    districts = ea.load_districts(PROCESSED)
    specs = build_models()
    residuals = []
    summaries = []
    states = []
    bands = []
    extremes = []
    largest = []
    importances = []

    for _, row in selected.iterrows():
        configuration, model = row["configuration"], row["model"]
        frame = ea.residual_table(grouped_oof, configuration, model, districts)
        residuals.append(frame)
        summary = ea.error_summary(frame)
        summary.update({"configuration": configuration, "model": model})
        summaries.append(summary)
        by_state = ea.errors_by_state(frame)
        by_state["configuration"] = configuration
        by_state["model"] = model
        states.append(by_state)
        by_band = ea.errors_by_target_band(frame)
        by_band["configuration"] = configuration
        by_band["model"] = model
        bands.append(by_band)
        extreme = ea.extreme_target_summary(frame)
        extreme["configuration"] = configuration
        extreme["model"] = model
        extremes.append(extreme)
        top = ea.largest_errors(frame, n=20)
        top["configuration"] = configuration
        top["model"] = model
        largest.append(top)

        importance = ea.permutation_importance_table(
            ea.CONFIG_IDS[configuration], model, specs, grouped_params
        )
        importances.append(importance)

        suffix = ea.CONFIG_IDS[configuration]
        ea.plot_predicted_vs_actual(
            frame, PLOTS / f"phase3_predicted_vs_actual_{suffix}.png",
            f"{configuration} / {model}",
        )
        ea.plot_residual_distribution(
            frame, PLOTS / f"phase3_residual_distribution_{suffix}.png",
            f"Residuals: {configuration} / {model}",
        )
        ea.plot_residuals_vs_predicted(
            frame, PLOTS / f"phase3_residuals_vs_predicted_{suffix}.png",
            f"Residuals vs prediction: {configuration} / {model}",
        )
        top_models = (
            comparison[comparison["configuration"] == configuration]
            .sort_values("rank")["model"].head(4).tolist()
        )
        ea.plot_mae_by_fold(
            grouped_folds, configuration, top_models,
            PLOTS / f"phase3_mae_by_fold_{suffix}.png",
        )
        ea.plot_importance(
            importance, PLOTS / f"phase3_importance_{suffix}.png"
        )

    pd.concat(residuals, ignore_index=True).to_csv(
        ERRORS / "phase3_residuals_oof.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(ERRORS / "phase3_error_summary.csv", index=False)
    pd.concat(states, ignore_index=True).to_csv(
        ERRORS / "phase3_errors_by_state.csv", index=False
    )
    pd.concat(bands, ignore_index=True).to_csv(
        ERRORS / "phase3_errors_by_target_band.csv", index=False
    )
    pd.concat(extremes, ignore_index=True).to_csv(
        ERRORS / "phase3_extreme_target_summary.csv", index=False
    )
    pd.concat(largest, ignore_index=True).to_csv(
        ERRORS / "phase3_largest_errors.csv", index=False
    )
    pd.concat(importances, ignore_index=True).to_csv(
        METRICS / "phase3_permutation_importance.csv", index=False
    )

    ea.plot_model_comparison(comparison, PLOTS / "phase3_model_comparison_grouped.png")
    if not ab.empty:
        ea.plot_ab_comparison(ab, PLOTS / "phase3_ab_comparison.png")

    print("Selection (grouped nested-CV MAE):")
    print(selected.to_string(index=False))
    print("\nA vs B on the same 628 districts (paired fold MAE, B - A):")
    print(ab.to_string(index=False))
    print("\nError summary (out-of-fold):")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nWrote {len(list(PLOTS.glob('*.png')))} plots to {PLOTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
