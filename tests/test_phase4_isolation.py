"""Phase 4 isolation and boundary tests.

These are prerequisites: no experimental result may be reported until every
check here passes. They verify that

* the registered arms and their grids are exactly the five in the proposal;
* sample weights are defined as registered and normalised to mean 1;
* weights reach the estimator as the row-aligned vector they should be;
* an inner split's validation targets cannot reach that split's weights;
* outer test targets cannot reach any fold decision (alpha, threshold, weights,
  coefficients);
* the outer folds reproduce the frozen Phase 3 Ridge predictions;
* the frozen evaluation thresholds and per-fold extreme counts are unchanged.

Run: py tests/test_phase4_isolation.py
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import regression_metrics  # noqa: E402
from src.models.tail_aware import (  # noqa: E402
    build_arms,
    build_estimator,
    fit_estimator,
    fit_fold,
    inner_folds,
    inner_search,
    sample_weights,
    weighted_row_count,
)
from src.training.nested_cv import build_outer_folds  # noqa: E402
from src.training.run_nested_cv import build_configurations  # noqa: E402

PREDICTIONS = ROOT / "evaluation" / "predictions"
PHASE3_OOF = PREDICTIONS / "phase3_oof_predictions.csv"

P90 = 9.200
P95 = 11.200
FOLD_P90 = {1: 11, 2: 20, 3: 21, 4: 10, 5: 2}
FOLD_P95 = {1: 4, 2: 12, 3: 14, 4: 5, 5: 0}


@lru_cache(maxsize=1)
def data():
    config = build_configurations(["b628"])["b628"]
    folds = build_outer_folds(config["groups"], "grouped")
    return config, folds


@lru_cache(maxsize=1)
def phase3_reference() -> pd.Series:
    frame = pd.read_csv(PHASE3_OOF)
    frame = frame[
        (frame["configuration"] == "B_augmented_628")
        & (frame["scheme"] == "grouped")
        & (frame["model"] == "ridge")
    ]
    assert len(frame) == 628, f"expected 628 reference rows, found {len(frame)}"
    return frame.set_index("key")["predicted"]


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #


def test_registered_arms_are_exactly_the_five_proposed():
    arms = build_arms()
    assert set(arms) == {"E1", "E2", "E3", "E4a", "E4b"}, sorted(arms)
    weighted = {name: (arm.quantile, arm.lam) for name, arm in arms.items() if arm.is_weighted}
    assert weighted == {"E1": (0.75, 2.0), "E2": (0.75, 4.0), "E3": (0.90, 4.0)}, weighted
    for name in ("E1", "E2", "E3"):
        assert arms[name].alpha_grid == (30.0, 100.0, 300.0), arms[name].alpha_grid
    assert arms["E4a"].tau == 0.75 and arms["E4b"].tau == 0.90
    for name in ("E4a", "E4b"):
        assert arms[name].alpha_grid == (0.001, 0.01, 0.1), arms[name].alpha_grid


def test_weight_values_match_registration():
    """E1 -> 0.667/2.0, E2 -> 0.5/2.5, E3 -> 0.714/3.57 on a quartile-declining vector."""
    y = np.arange(1.0, 101.0)  # q75 = 75.75, q90 = 90.9
    arms = build_arms()
    for name, expected in (("E1", (2 / 3, 2.0)), ("E2", (0.5, 2.5)), ("E3", (10 / 14, 50 / 14))):
        arm = arms[name]
        weights, threshold = sample_weights(y, arm.quantile, arm.lam)
        low, high = weights[y < threshold][0], weights[y >= threshold][0]
        assert np.isclose(weights.mean(), 1.0), (name, weights.mean())
        assert np.isclose(low, expected[0]) and np.isclose(high, expected[1]), (name, low, high)
        assert weighted_row_count(y, threshold) == int(np.sum(y >= threshold))


def test_weights_are_normalised_and_bounded():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 20.0])
    for name in ("E1", "E2", "E3"):
        arm = build_arms()[name]
        weights, threshold = sample_weights(y, arm.quantile, arm.lam)
        assert np.isclose(weights.mean(), 1.0)
        assert weights.min() > 0
        assert np.all(np.isfinite(weights))
        # every district at/above the threshold outweighs every district below it
        below = weights[y < threshold]
        above = weights[y >= threshold]
        assert len(below) and len(above)
        assert above.min() > below.max(), (name, above.min(), below.max())


# --------------------------------------------------------------------------- #
# routing and isolation
# --------------------------------------------------------------------------- #


def test_weights_reach_the_estimator_as_a_row_aligned_vector():
    """Weighted pipeline fit must equal a direct weighted Ridge on the same design."""
    config, _ = data()
    X, y = config["X"], config["y"]
    train_idx = np.arange(0, 400)
    X_train, y_train = X.iloc[train_idx], y[train_idx]

    arm = build_arms()["E2"]
    weights, _ = sample_weights(y_train, arm.quantile, arm.lam)
    pipeline = fit_estimator(build_estimator(arm, 100.0), X_train, y_train, weights)

    design = pipeline[:-1].transform(X_train)
    direct = Ridge(alpha=100.0).fit(design, y_train, sample_weight=weights)

    assert np.allclose(pipeline[-1].coef_, direct.coef_, atol=1e-10, rtol=0)
    assert np.isclose(pipeline[-1].intercept_, direct.intercept_, atol=1e-10)


def test_inner_validation_targets_cannot_reach_that_split_weights():
    """Corrupt an inner split's validation targets; its own weights must not move."""
    config, folds = data()
    X, y, groups = config["X"], config["y"], config["groups"]
    train_idx, _ = folds[0]
    X_train, y_train = X.iloc[train_idx], np.asarray(y)[train_idx]
    groups_train = groups[train_idx]

    splits = inner_folds(groups_train)
    arm = build_arms()["E2"]

    for split_id, (_, valid_idx) in enumerate(splits, start=1):
        diagnostics_a: list[dict] = []
        inner_search(X_train, y_train, groups_train, arm, 1, diagnostics=diagnostics_a)

        corrupted = y_train.copy()
        corrupted[valid_idx] = 999.0  # arbitrary values in the validation rows
        diagnostics_b: list[dict] = []
        inner_search(X_train, corrupted, groups_train, arm, 1, diagnostics=diagnostics_b)

        rows_a = [r for r in diagnostics_a if r["inner_split"] == split_id]
        rows_b = [r for r in diagnostics_b if r["inner_split"] == split_id]
        assert rows_a and len(rows_a) == len(rows_b) == len(arm.alpha_grid)

        for row_a, row_b in zip(rows_a, rows_b, strict=True):
            assert np.isclose(row_a["threshold"], row_b["threshold"]), (
                f"split {split_id}: threshold changed when validation targets changed")
            assert row_a["n_weighted"] == row_b["n_weighted"]
            assert np.isclose(row_a["weight_mean"], row_b["weight_mean"])


def test_inner_search_is_deterministic():
    config, folds = data()
    X, y, groups = config["X"], config["y"], config["groups"]
    train_idx, _ = folds[1]
    X_train, y_train = X.iloc[train_idx], np.asarray(y)[train_idx]

    for name in ("E1", "E4a"):
        arm = build_arms()[name]
        alpha_a, table_a = inner_search(X_train, y_train, groups[train_idx], arm, 2)
        alpha_b, table_b = inner_search(X_train, y_train, groups[train_idx], arm, 2)
        assert alpha_a == alpha_b, (name, alpha_a, alpha_b)
        assert [row["mean_score"] for row in table_a] == [row["mean_score"] for row in table_b]


def test_outer_test_targets_cannot_reach_any_fold_decision():
    """Overwrite the outer test targets: alpha, weights, coefficients must be identical."""
    config, folds = data()
    X, y, groups = config["X"], config["y"], config["groups"]
    train_idx, test_idx = folds[0]

    for name in ("E2", "E4b"):
        arm = build_arms()[name]
        predictions_a, record_a, estimator_a = fit_fold(
            X, y, groups, train_idx, test_idx, arm, 1, return_estimator=True
        )

        corrupted = np.asarray(y, dtype=float).copy()
        corrupted[test_idx] = 999.0
        predictions_b, record_b, estimator_b = fit_fold(
            X, corrupted, groups, train_idx, test_idx, arm, 1, return_estimator=True
        )

        assert record_a["best_alpha"] == record_b["best_alpha"], (
            f"{name}: alpha depended on test targets")
        assert np.isclose(record_a["threshold"], record_b["threshold"], equal_nan=True)
        assert record_a["n_weighted"] == record_b["n_weighted"]
        assert np.isclose(record_a["weight_mean"], record_b["weight_mean"], equal_nan=True)
        assert np.allclose(estimator_a[-1].coef_, estimator_b[-1].coef_, atol=0, rtol=0), (
            f"{name}: coefficients changed when test targets changed")
        # the model is unchanged, so its predictions on the uncorrupted rows are too
        assert np.allclose(
            estimator_a.predict(X.iloc[test_idx]), estimator_b.predict(X.iloc[test_idx]),
            atol=0, rtol=0,
        )
        assert np.allclose(predictions_a, predictions_b, atol=0, rtol=0)


# --------------------------------------------------------------------------- #
# fold identity and frozen definitions
# --------------------------------------------------------------------------- #


def test_outer_folds_reproduce_the_frozen_phase3_ridge():
    """Uniform Ridge(alpha=100) on these folds must match the Phase 3 predictions."""
    config, folds = data()
    X, y = config["X"], config["y"]
    keys = config["keys"].to_numpy()
    reference = phase3_reference()

    arm = build_arms()["E1"]
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in folds:
        estimator = fit_estimator(build_estimator(arm, 100.0), X.iloc[train_idx], y[train_idx])
        oof[test_idx] = estimator.predict(X.iloc[test_idx])

    expected = reference.loc[keys].to_numpy()
    difference = np.abs(oof - expected)
    assert difference.max() < 1e-8, f"max |difference| from Phase 3 Ridge: {difference.max():.3e}"
    # and the metric arithmetic is unchanged
    observed = regression_metrics(y, oof)
    frozen = regression_metrics(y, expected)
    assert abs(observed["mae"] - frozen["mae"]) < 1e-9


def test_frozen_evaluation_definitions_are_unchanged():
    config, folds = data()
    y = np.asarray(config["y"], dtype=float)
    assert len(y) == 628
    assert np.isclose(np.quantile(y, 0.90), P90, atol=1e-9), np.quantile(y, 0.90)
    assert np.isclose(np.quantile(y, 0.95), P95, atol=1e-9), np.quantile(y, 0.95)

    counts_p90, counts_p95 = {}, {}
    for fold_id, (_, test_idx) in enumerate(folds, start=1):
        counts_p90[fold_id] = int((y[test_idx] >= P90).sum())
        counts_p95[fold_id] = int((y[test_idx] >= P95).sum())
    assert counts_p90 == FOLD_P90, counts_p90
    assert counts_p95 == FOLD_P95, counts_p95
    assert int((y >= P90).sum()) == 64
    assert int((y >= P95).sum()) == 35


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as error:
            failures += 1
            print(f"FAIL {test.__name__}: {error}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(error).__name__}: {error}")

    passed = len(tests) - failures
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
