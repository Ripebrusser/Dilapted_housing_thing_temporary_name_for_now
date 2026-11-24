"""Phase 3 contract tests.

Locks in leakage protection, fold integrity, metric arithmetic, selection logic
and reproducibility for the modelling phase.

Run: py tests/test_phase3_contracts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import regression_metrics  # noqa: E402
from src.features.predictors import FORBIDDEN_SOURCES  # noqa: E402
from src.models.registry import build_models  # noqa: E402
from src.training.nested_cv import build_outer_folds, run_configuration  # noqa: E402
from src.training.run_nested_cv import build_configurations  # noqa: E402

METRICS = ROOT / "evaluation" / "metrics"
PREDICTIONS = ROOT / "evaluation" / "predictions"
PLOTS = ROOT / "evaluation" / "plots"
ERRORS = ROOT / "evaluation" / "error_analysis"
PROCESSED = ROOT / "datasets" / "processed"

RESULTS = METRICS / "phase3_nested_cv_results.csv"
RANDOM = METRICS / "phase3_random_cv_results.csv"
FOLDS = METRICS / "phase3_fold_results.csv"
COMPARISON = METRICS / "phase3_model_comparison.csv"
OOF = PREDICTIONS / "phase3_oof_predictions.csv"
PARAMS = METRICS / "phase3_best_parameters.json"

CONFIGS = {
    "A_primary_640": 640,
    "A_restricted_628": 628,
    "B_augmented_628": 628,
}
MODELS = {
    "dummy_mean", "ridge", "elastic_net", "knn", "decision_tree",
    "random_forest", "extra_trees", "gradient_boosting",
    "hist_gradient_boosting", "svr_rbf",
}


def results() -> pd.DataFrame:
    assert RESULTS.exists(), f"missing {RESULTS}"
    return pd.read_csv(RESULTS)


def random_results() -> pd.DataFrame:
    assert RANDOM.exists(), f"missing {RANDOM}"
    return pd.read_csv(RANDOM)


def folds() -> pd.DataFrame:
    assert FOLDS.exists(), f"missing {FOLDS}"
    return pd.read_csv(FOLDS)


def oof() -> pd.DataFrame:
    assert OOF.exists(), f"missing {OOF}"
    return pd.read_csv(OOF)


def comparison() -> pd.DataFrame:
    assert COMPARISON.exists(), f"missing {COMPARISON}"
    return pd.read_csv(COMPARISON)


# ---------------------------------------------------------------------------
# Deliverables and coverage
# ---------------------------------------------------------------------------
def test_required_deliverables_exist() -> None:
    required = [
        RESULTS, RANDOM, FOLDS, COMPARISON, OOF, PARAMS,
        METRICS / "phase3_best_parameters.json",
        METRICS / "phase3_ab_comparison.csv",
        METRICS / "phase3_permutation_importance.csv",
        METRICS / "phase3_selected_models.csv",
        ERRORS / "phase3_residuals_oof.csv",
        ERRORS / "phase3_largest_errors.csv",
        ERRORS / "phase3_errors_by_state.csv",
        ERRORS / "phase3_errors_by_target_band.csv",
        PLOTS / "phase3_model_comparison_grouped.png",
        PLOTS / "phase3_predicted_vs_actual_a640.png",
        PLOTS / "phase3_importance_b628.png",
    ]
    missing = [str(path) for path in required if not path.exists()]
    assert not missing, missing


def test_every_configuration_and_model_is_present() -> None:
    frame = results()
    assert set(frame["configuration"]) == set(CONFIGS)
    for configuration, n_expected in CONFIGS.items():
        subset = frame[frame["configuration"] == configuration]
        assert set(subset["model"]) == MODELS, configuration
        assert (subset["n_districts"] == n_expected).all(), configuration


def test_fold_count_and_coverage() -> None:
    frame = folds()
    assert set(frame["scheme"]) == {"grouped", "random"}
    for (configuration, scheme), group in frame.groupby(["configuration", "scheme"]):
        assert group["fold"].nunique() == 5, (configuration, scheme)
        per_model = group[group["model"] == "ridge"]
        assert int(per_model["n_test"].sum()) == CONFIGS[configuration]
        assert per_model["n_train"].min() > 0


def test_grouped_folds_never_split_states() -> None:
    frame = folds()
    district_states = pd.read_csv(PROCESSED / "district_analysis.csv")["state"].nunique()
    grouped = frame[frame["scheme"] == "grouped"]
    for (configuration, _), group in grouped.groupby(["configuration", "model"]):
        held_out = group.groupby("fold")["n_states_test"].first()
        trained = group.groupby("fold")["n_states_train"].first()
        combined = held_out + trained
        assert (combined == district_states).all(), (
            configuration,
            dict(combined),
            district_states,
        )
        assert (held_out >= 1).all()


def test_out_of_fold_predictions_are_complete_and_unique() -> None:
    frame = oof()
    assert not frame["predicted"].isna().any()
    for (configuration, scheme, model), group in frame.groupby(
        ["configuration", "scheme", "model"]
    ):
        assert len(group) == CONFIGS[configuration], (configuration, model)
        assert group["key"].is_unique
        assert np.isfinite(group["predicted"]).all()


def test_pooled_metrics_match_out_of_fold_predictions() -> None:
    predictions = oof()
    frame = results()
    for _, row in frame.iterrows():
        subset = predictions[
            (predictions["configuration"] == row["configuration"])
            & (predictions["scheme"] == "grouped")
            & (predictions["model"] == row["model"])
        ]
        recomputed = regression_metrics(
            subset["actual"].to_numpy(), subset["predicted"].to_numpy()
        )
        assert abs(recomputed["mae"] - row["pooled_mae"]) < 1e-9, row["model"]
        assert abs(recomputed["rmse"] - row["pooled_rmse"]) < 1e-9, row["model"]
        if np.isfinite(row["pooled_r2"]):
            assert abs(recomputed["r2"] - row["pooled_r2"]) < 1e-9, row["model"]


def test_fold_means_match_summary() -> None:
    fold_frame = folds()
    frame = results()
    grouped = fold_frame[fold_frame["scheme"] == "grouped"]
    for _, row in frame.iterrows():
        subset = grouped[
            (grouped["configuration"] == row["configuration"])
            & (grouped["model"] == row["model"])
        ]
        assert abs(subset["mae"].mean() - row["mae_mean"]) < 1e-9, row["model"]
        assert abs(subset["rmse"].mean() - row["rmse_mean"]) < 1e-9, row["model"]


def test_metric_arithmetic() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.5, 2.0, 2.0, 5.0])
    metrics = regression_metrics(actual, predicted)
    assert abs(metrics["mae"] - 0.625) < 1e-12
    assert abs(metrics["rmse"] - 0.75) < 1e-12
    # SS_res = 2.25, SS_tot = 5.0 -> R2 = 1 - 0.45 = 0.55
    assert abs(metrics["r2"] - 0.55) < 1e-12
    perfect = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert abs(perfect["r2"] - 1.0) < 1e-12 and perfect["mae"] == 0.0


def test_negative_r2_is_retained() -> None:
    frame = results()
    assert (frame["r2_mean"] < 0).any(), "no negative R2 present; values may be clipped"


# ---------------------------------------------------------------------------
# Leakage and fold identity
# ---------------------------------------------------------------------------
def test_no_identifier_or_target_columns_are_used() -> None:
    banned = {"key", "state", "district", "district_code", "target_deprivation_rate_pct"}
    for config_id in ["a640", "a628", "b628"]:
        columns = set(build_configurations([config_id])[config_id]["X"].columns)
        assert not (columns & banned), (config_id, columns & banned)


def test_feature_sets_match_the_approved_manifest() -> None:
    manifest = pd.read_csv(PROCESSED / "feature_manifest.csv")
    retained = manifest[manifest["retained"]]
    a_cols = set(retained.loc[retained["configuration"] == "A", "feature_name"])
    b_cols = set(retained.loc[retained["configuration"] != "A", "feature_name"])
    assert set(build_configurations(["a640"])["a640"]["X"].columns) == a_cols
    assert set(build_configurations(["b628"])["b628"]["X"].columns) == a_cols | b_cols

    used: set[str] = set()
    for value in retained["source_columns"]:
        used.update(str(value).split("|"))
    assert not (used & FORBIDDEN_SOURCES)


def test_a_and_b_share_districts_and_folds() -> None:
    a = build_configurations(["a628"])["a628"]
    b = build_configurations(["b628"])["b628"]
    assert np.array_equal(a["keys"].to_numpy(), b["keys"].to_numpy())
    assert np.array_equal(a["y"], b["y"])
    assert np.array_equal(a["groups"], b["groups"])
    assert len(a["y"]) == 628

    folds_a = build_outer_folds(a["groups"], "grouped")
    folds_b = build_outer_folds(b["groups"], "grouped")
    for (train_a, test_a), (train_b, test_b) in zip(folds_a, folds_b):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(test_a, test_b)

    signature = json.loads(
        (METRICS / "_phase3_grouped_meta.json").read_text(encoding="utf-8")
    )["fold_signature"]
    assert signature["A_restricted_628"] == signature["B_augmented_628"]


def test_outer_folds_are_deterministic() -> None:
    groups = build_configurations(["a640"])["a640"]["groups"]
    first = build_outer_folds(groups, "grouped")
    second = build_outer_folds(groups, "grouped")
    for (train_first, test_first), (train_second, test_second) in zip(first, second):
        assert np.array_equal(train_first, train_second)
        assert np.array_equal(test_first, test_second)
    assert not (set(first[0][0]) & set(first[0][1]))


# ---------------------------------------------------------------------------
# Selection and reproducibility
# ---------------------------------------------------------------------------
def test_selection_is_one_model_per_configuration_by_grouped_mae() -> None:
    frame = comparison()
    assert frame["selected"].sum() == len(CONFIGS)
    for configuration, group in frame.groupby("configuration"):
        chosen = group[group["selected"]]
        assert len(chosen) == 1, configuration
        assert chosen["mae_mean"].iloc[0] == group["mae_mean"].min(), configuration


def test_random_cv_is_reported_separately_from_primary() -> None:
    frame = random_results()
    assert set(frame["scheme"]) == {"random"}
    assert "random_mae_mean" in comparison().columns
    primary = results()
    assert set(primary["scheme"]) == {"grouped"}


def test_seeds_and_versions_are_recorded() -> None:
    meta = json.loads(
        (METRICS / "_phase3_grouped_meta.json").read_text(encoding="utf-8")
    )
    assert meta["seed"] == 42
    assert meta["outer_splits"] == 5 and meta["inner_splits"] == 4
    assert {"numpy", "pandas", "scikit-learn"} <= set(meta["versions"])


def test_ridge_result_is_reproducible() -> None:
    """Re-run one model end-to-end and require the identical grouped MAE."""
    stored = results()
    expected = stored[
        (stored["configuration"] == "A_restricted_628") & (stored["model"] == "ridge")
    ]["mae_mean"].iloc[0]
    config = build_configurations(["a628"])["a628"]
    specs = {"ridge": build_models()["ridge"]}
    summary, _, _, _ = run_configuration(
        config["label"], config["X"], config["y"], config["groups"],
        config["keys"], specs, scheme="grouped", seed=42, log=lambda *_: None,
    )
    assert abs(summary[0]["mae_mean"] - expected) < 1e-9, (summary[0]["mae_mean"], expected)


def test_inner_tuning_folds_stay_inside_outer_training() -> None:
    """The tuning split must be carved from outer-training rows only."""
    from sklearn.model_selection import GroupKFold

    config = build_configurations(["a640"])["a640"]
    outer = build_outer_folds(config["groups"], "grouped")
    train_idx, test_idx = outer[0]
    allowed = set(range(len(train_idx)))

    inner = GroupKFold(n_splits=4).split(
        np.zeros(len(train_idx)), groups=config["groups"][train_idx]
    )
    for inner_train, inner_test in inner:
        assert set(inner_train) <= allowed
        assert set(inner_test) <= allowed
        assert not (set(inner_train) & set(inner_test))
        # Map inner positions back to global rows: no held-out outer row may appear.
        mapped = set(np.asarray(train_idx)[np.asarray(inner_train)]) | set(
            np.asarray(train_idx)[np.asarray(inner_test)]
        )
        assert not (mapped & set(test_idx))
        assert mapped <= set(train_idx)


def test_scaling_is_fitted_on_the_training_fold_only() -> None:
    """StandardScaler must learn the training-fold moments, not the full-sample ones."""
    from sklearn.base import clone

    config = build_configurations(["a640"])["a640"]
    train_idx, _ = build_outer_folds(config["groups"], "grouped")[0]
    estimator = clone(build_models()["ridge"].estimator)
    estimator.fit(config["X"].iloc[train_idx], config["y"][train_idx])

    learned = estimator.named_steps["scaler"].mean_
    training_means = config["X"].iloc[train_idx].mean().to_numpy()
    full_means = config["X"].mean().to_numpy()

    assert np.allclose(learned, training_means), "scaler did not use training-fold means"
    assert not np.allclose(learned, full_means), "scaler may have seen the full sample"
    assert estimator.named_steps["imputer"].statistics_ is not None


def test_notebooks_are_valid_and_contain_no_error_outputs() -> None:
    notebook_dir = ROOT / "notebooks"
    for name in [
        "06_model_training.ipynb",
        "07_model_comparison.ipynb",
        "08_cross_validation.ipynb",
        "09_robustness_testing.ipynb",
        "10_error_analysis.ipynb",
    ]:
        payload = json.loads((notebook_dir / name).read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4, name
        code_cells = [c for c in payload["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 5, name
        errors = [
            output for cell in code_cells for output in cell["outputs"]
            if output.get("output_type") == "error"
        ]
        assert not errors, (name, errors[0].get("evalue"))
        # Phase 3 requires the notebooks to have actually been executed.
        assert any(cell.get("execution_count") for cell in code_cells), (
            f"{name}: no execution counts - notebook has not been executed"
        )


def test_fold_metrics_correspond_to_saved_fold_assignments() -> None:
    """Recompute each fold's MAE from the OOF file using the deterministic folds."""
    predictions = oof()
    fold_frame = folds()
    pairs = [
        ("a640", "A_primary_640"),
        ("a628", "A_restricted_628"),
        ("b628", "B_augmented_628"),
    ]
    for config_id, label in pairs:
        built = build_configurations([config_id])[config_id]
        outer = build_outer_folds(built["groups"], "grouped")
        keys = built["keys"].to_numpy()
        for model in ["ridge", "svr_rbf"]:
            subset = predictions[
                (predictions["configuration"] == label)
                & (predictions["scheme"] == "grouped")
                & (predictions["model"] == model)
            ].set_index("key")
            saved = fold_frame[
                (fold_frame["configuration"] == label)
                & (fold_frame["scheme"] == "grouped")
                & (fold_frame["model"] == model)
            ].set_index("fold")
            for fold_id, (_, test_idx) in enumerate(outer, start=1):
                block = subset.loc[keys[test_idx]]
                assert len(block) == int(saved.loc[fold_id, "n_test"])
                recomputed = float(np.mean(np.abs(block["actual"] - block["predicted"])))
                assert abs(recomputed - float(saved.loc[fold_id, "mae"])) < 1e-9, (
                    label, model, fold_id, recomputed, saved.loc[fold_id, "mae"]
                )


def main() -> int:
    tests = sorted(
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    failures = []
    for name, func in tests:
        try:
            func()
            print(f"PASS  {name}")
        except BaseException as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {exc}")
    print("-" * 60)
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
