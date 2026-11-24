"""Phase 2 contract tests.

Locks in the approved target definition, the feature construction rules and the
fold construction, so later refactors cannot silently break them.

Run: py tests/test_phase2_contracts.py
Compatible with pytest if installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "data"))

from src.data.loaders import (  # noqa: E402
    N_DISTRICTS,
    N_DISTRICTS_WITH_D5,
    PROCESSED,
    load_d1,
    load_d2_total,
    load_d5_households,
)
from src.features.predictors import FORBIDDEN_SOURCES  # noqa: E402
from src.features.target import TARGET_NAME, build_target  # noqa: E402
from src.training.splits import (  # noqa: E402
    assert_folds_valid,
    grouped_folds,
    random_folds,
)

ANALYSIS = PROCESSED / "district_analysis.csv"
MANIFEST = PROCESSED / "feature_manifest.csv"

EXPECTED_A_FEATURES = 43
EXPECTED_B_FEATURES = 34


def analysis() -> pd.DataFrame:
    assert ANALYSIS.exists(), f"missing {ANALYSIS}"
    return pd.read_csv(ANALYSIS)


def manifest() -> pd.DataFrame:
    assert MANIFEST.exists(), f"missing {MANIFEST}"
    return pd.read_csv(MANIFEST)


def retained(config_a: bool) -> list[str]:
    frame = manifest()
    frame = frame[frame["retained"]]
    is_a = frame["configuration"] == "A"
    return frame.loc[is_a if config_a else ~is_a, "feature_name"].tolist()


# ---------------------------------------------------------------------------
# Table shape and target
# ---------------------------------------------------------------------------
def test_table_shape_and_uniqueness() -> None:
    frame = analysis()
    assert len(frame) == N_DISTRICTS
    assert frame["key"].is_unique
    assert frame["district"].notna().all()


def test_districts_match_d1_exactly() -> None:
    assert set(analysis()["key"]) == set(load_d1()["key"])


def test_target_is_reproduced_from_d2() -> None:
    expected = build_target(load_d2_total()).set_index("key")[TARGET_NAME]
    actual = analysis().set_index("key")[TARGET_NAME]
    pd.testing.assert_series_equal(
        actual.sort_index(), expected.sort_index(), check_names=False, rtol=1e-9
    )


def test_target_is_valid_and_not_degenerate() -> None:
    target = analysis()[TARGET_NAME]
    assert target.notna().all()
    assert target.min() >= 0 and target.max() <= 100
    assert target.nunique() > 100, "target collapsed to very few values"
    assert target.std() > 0


def test_no_denied_columns_in_analysis_table() -> None:
    columns = set(analysis().columns)
    forbidden_columns = {
        "Households",
        "Rural_Households",
        "Urban_Households",
        "Condition_of_occupied_census_houses_Dilapidated_Households",
        "Total Number of Dilapidated",
        "Contition_T_Dilapidated",
    }
    assert not (columns & forbidden_columns), columns & forbidden_columns


def test_manifest_never_uses_target_components() -> None:
    used: set[str] = set()
    for value in manifest().loc[manifest()["retained"], "source_columns"]:
        used.update(str(value).split("|"))
    assert not (used & FORBIDDEN_SOURCES)


# ---------------------------------------------------------------------------
# Feature manifest integrity
# ---------------------------------------------------------------------------
def test_every_predictor_is_documented_exactly_once() -> None:
    frame = analysis()
    identifiers = {"key", "state", "district", "district_code", TARGET_NAME, "has_d5_households"}
    predictors = set(frame.columns) - identifiers
    frame_manifest = manifest()
    documented = set(frame_manifest.loc[frame_manifest["retained"], "feature_name"])
    assert predictors == documented, {
        "undocumented": sorted(predictors - documented),
        "documented_but_absent": sorted(documented - predictors),
    }
    assert not frame_manifest["feature_name"].duplicated().any()


def test_every_feature_records_its_calculation() -> None:
    frame = manifest()
    for column in ["formula", "denominator", "denominator_source", "source_dataset", "universe"]:
        assert frame[column].notna().all(), column
        assert (frame[column].astype(str).str.strip() != "").all(), column


def test_feature_counts_match_configuration_contract() -> None:
    assert len(retained(config_a=True)) == EXPECTED_A_FEATURES
    assert len(retained(config_a=False)) == EXPECTED_B_FEATURES


def test_all_features_clear_on_leakage_screen() -> None:
    frame = manifest()
    assert set(frame["leakage_status"].unique()) == {"clear"}


# ---------------------------------------------------------------------------
# Missingness and coverage
# ---------------------------------------------------------------------------
def test_config_a_has_no_missing_values() -> None:
    frame = analysis()
    missing = frame[retained(config_a=True)].isna().sum()
    assert int(missing.sum()) == 0, missing[missing > 0].to_dict()


def test_config_b_missing_only_for_unresolvable_districts() -> None:
    frame = analysis()
    assert int(frame["has_d5_households"].sum()) == N_DISTRICTS_WITH_D5
    absent = ~frame["has_d5_households"].astype(bool)
    for column in retained(config_a=False):
        assert int(frame[column].isna().sum()) == 12, column
        pd.testing.assert_series_equal(
            frame[column].isna(), absent, check_names=False,
        )


def test_no_zero_imputation_of_denominators() -> None:
    """A missing household denominator must stay missing, never 0."""
    frame = analysis()
    absent = ~frame["has_d5_households"].astype(bool)
    for column in retained(config_a=False):
        assert frame.loc[absent, column].isna().all(), column
        assert (frame.loc[absent, column] == 0).sum() == 0, column


# ---------------------------------------------------------------------------
# Value sanity
# ---------------------------------------------------------------------------
def test_config_a_features_are_shares() -> None:
    frame = analysis()
    for column in retained(config_a=True):
        values = frame[column]
        assert values.min() >= -1e-9, f"{column} min {values.min()}"
        assert values.max() <= 1.0 + 1e-9, f"{column} max {values.max()}"


def test_config_b_rates_exceed_one_only_slightly() -> None:
    """D1 asset counts and D5 household counts are different universes per district."""
    frame = analysis()
    overshoot = {}
    for column in retained(config_a=False):
        maximum = float(frame[column].max())
        if maximum > 1.0:
            overshoot[column] = round(maximum, 3)
    assert all(v <= 1.10 for v in overshoot.values()), overshoot


def test_no_single_feature_explains_the_target() -> None:
    frame = analysis()
    target = frame[TARGET_NAME]
    correlations = {}
    for column in retained(config_a=True) + retained(config_a=False):
        pair = pd.concat([frame[column], target], axis=1).dropna()
        correlations[column] = abs(float(pair.iloc[:, 0].corr(pair.iloc[:, 1])))
    worst = max(correlations, key=correlations.get)
    assert correlations[worst] < 0.90, (worst, correlations[worst])


# ---------------------------------------------------------------------------
# Cross-validation splits
# ---------------------------------------------------------------------------
def test_grouped_folds_hold_out_whole_states() -> None:
    frame = analysis()
    folds = grouped_folds(frame["state"].to_numpy())
    assert len(folds) == 5
    assert_folds_valid(folds, len(frame), frame["state"].to_numpy())
    for _, test_idx in folds:
        assert len(np.unique(frame["state"].to_numpy()[test_idx])) >= 1


def test_random_folds_cover_every_row_once() -> None:
    frame = analysis()
    repeats = random_folds(len(frame), repeats=2)
    for folds in repeats.values():
        assert_folds_valid(folds, len(frame))


def test_a_and_b_use_identical_folds() -> None:
    """The comparison is only valid if both configurations see the same splits."""
    frame = analysis()
    subset = frame[frame["has_d5_households"].astype(bool)]
    folds_a = grouped_folds(subset["state"].to_numpy())
    folds_b = grouped_folds(subset["state"].to_numpy())
    for (train_a, test_a), (train_b, test_b) in zip(folds_a, folds_b):
        assert np.array_equal(train_a, train_b) and np.array_equal(test_a, test_b)


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
