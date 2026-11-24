"""Phase 6 tests: persistence, inference contract, predictions, ranking, bands.

These are the tests the audit asked for (defect register D-14/§L): the project
had no persisted model, no inference path and no district output, so there was
nothing to test. They check behaviour that would actually break a deployment:

* the saved pipeline reloads and returns identical predictions;
* the scaler in the saved pipeline is the one fitted on the 628 training rows
  (an inference-time refit would be leakage);
* the input contract rejects missing, non-numeric, NaN, infinite and
  unexpected columns, and accepts a permuted column order;
* the 628-district prediction file, the ranking and the relative bands are
  internally consistent and reproducible from the saved model;
* out-of-range fitted values are flagged rather than clipped.

Run: py -3.14 tests/test_phase6_inference.py
(after `py -3.14 src/prediction/build_predictions.py`)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prediction.build_predictions import (  # noqa: E402
    BAND_LABELS,
    N_BANDS,
    PREDICTION_FILE,
    RANKING_FILE,
    RETROSPECTIVE_FILE,
)
from src.prediction.final_model import (  # noqa: E402
    MANIFEST_PATH,
    METADATA_PATH,
    MODEL_PATH,
    approved_training_data,
    build_pipeline,
    load_artifacts,
)
from src.prediction.inference import (  # noqa: E402
    PREDICTION_COLUMN,
    SchemaError,
    load_scorer,
    predict_districts,
)

TOL = 1e-9


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #


def test_persisted_artifacts_exist_and_metadata_is_complete():
    for path in (MODEL_PATH, METADATA_PATH, MANIFEST_PATH):
        assert path.exists(), f"missing artefact: {path}"
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["model_id"] == "ridge_b_alpha100_v1"
    assert metadata["alpha"] == 100.0
    assert metadata["n_training_districts"] == 628
    assert metadata["n_features"] == 77
    assert len(metadata["features"]) == 77
    assert len(set(metadata["features"])) == 77, "duplicate feature name in the manifest"
    for key in (
        "feature_order_sha256", "feature_matrix_sha256", "training_target_sha256",
        "raw_source_sha256", "dependency_versions", "trained_at_utc",
        "target_definition", "persistence_warning", "validation_reference",
    ):
        assert key in metadata, key
    assert "dilapidated" in metadata["target_definition"]["construction"]
    assert "NOT a measure of individual homelessness" in metadata["target_definition"]["scope"]
    assert metadata["predictions_are_not_validation"] is True
    assert "pickle" in metadata["persistence_warning"]
    assert all(value != "MISSING" for value in metadata["raw_source_sha256"].values())
    manifest = pd.read_csv(MANIFEST_PATH)
    assert manifest["feature"].tolist() == metadata["features"]
    assert manifest["position"].tolist() == list(range(1, 78))


def test_reloaded_model_reproduces_predictions_exactly():
    X, _, _, _ = approved_training_data()
    first, _ = load_artifacts()
    second, _ = load_artifacts()
    a = first.predict(X)
    b = second.predict(X)
    assert np.max(np.abs(a - b)) <= TOL
    # a fresh fit of the same recipe must agree with the persisted one
    fresh = build_pipeline().fit(X, np.asarray(approved_training_data()[1]))
    assert np.max(np.abs(a - fresh.predict(X))) <= 1e-8


def test_saved_scaler_is_the_training_scaler():
    scorer = load_scorer()
    scaler = scorer["pipeline"].named_steps["scaler"]
    assert int(scaler.n_samples_seen_) == 628, scaler.n_samples_seen_
    assert scaler.mean_.shape == (77,)
    X, _, _, _ = approved_training_data()
    assert np.allclose(scaler.mean_, X.to_numpy(dtype=float).mean(axis=0), atol=1e-10)


def test_ridge_alpha_is_the_ratified_value():
    scorer = load_scorer()
    assert scorer["pipeline"].named_steps["model"].alpha == 100.0
    assert scorer["metadata"]["alpha"] == 100.0


# --------------------------------------------------------------------------- #
# inference contract
# --------------------------------------------------------------------------- #


def _frame():
    X, _, keys, groups = approved_training_data()
    frame = X.copy()
    frame.insert(0, "key", keys)
    frame.insert(1, "state", groups)
    return frame


def test_column_order_does_not_change_predictions():
    frame = _frame()
    scorer = load_scorer()
    baseline = predict_districts(frame, scorer=scorer)[PREDICTION_COLUMN].to_numpy()
    shuffled = frame[list(reversed(frame.columns))]
    again = predict_districts(shuffled, scorer=scorer)[PREDICTION_COLUMN].to_numpy()
    assert np.max(np.abs(baseline - again)) <= TOL, "feature order must come from the manifest"


def test_missing_feature_is_rejected():
    frame = _frame().drop(columns=["literacy_rate"])
    try:
        predict_districts(frame)
    except SchemaError as error:
        assert "literacy_rate" in str(error)
        return
    raise AssertionError("a missing feature was accepted")


def test_missing_value_is_rejected():
    frame = _frame()
    frame.loc[frame.index[3], "male_share"] = np.nan
    try:
        predict_districts(frame)
    except SchemaError as error:
        assert "male_share" in str(error)
        return
    raise AssertionError("a NaN feature was accepted")


def test_infinite_value_is_rejected():
    frame = _frame()
    frame.loc[frame.index[7], "male_share"] = np.inf
    try:
        predict_districts(frame)
    except SchemaError:
        return
    raise AssertionError("an infinite feature was accepted")


def test_non_numeric_feature_is_rejected():
    frame = _frame()
    frame["male_share"] = frame["male_share"].astype(str)
    try:
        predict_districts(frame)
    except SchemaError as error:
        assert "non-numeric" in str(error)
        return
    raise AssertionError("a text feature was scored")


def test_unexpected_column_is_rejected_unless_explicitly_allowed():
    frame = _frame()
    frame["secret_leak"] = 1.0
    try:
        predict_districts(frame)
    except SchemaError as error:
        assert "secret_leak" in str(error)
        return
    raise AssertionError("an unexpected column was silently ignored")


def test_extra_columns_can_be_tolerated_and_are_reported():
    frame = _frame()
    frame["geometry_note"] = "rural"
    result = predict_districts(frame, allow_extra_columns=True)
    assert (result["ignored_columns"] == "geometry_note").all()
    assert len(result) == 628


def test_identifiers_are_preserved_and_input_is_not_modified():
    frame = _frame()
    before = frame.copy(deep=True)
    result = predict_districts(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert result["key"].tolist() == frame["key"].tolist()
    assert result["state"].tolist() == frame["state"].tolist()
    assert result[PREDICTION_COLUMN].notna().all()


def test_inference_is_deterministic():
    frame = _frame()
    first = predict_districts(frame)[PREDICTION_COLUMN].to_numpy()
    second = predict_districts(frame)[PREDICTION_COLUMN].to_numpy()
    assert np.array_equal(first, second)


# --------------------------------------------------------------------------- #
# deliverables
# --------------------------------------------------------------------------- #


def test_prediction_file_covers_all_628_districts():
    table = pd.read_csv(PREDICTION_FILE)
    assert len(table) == 628
    assert table["district_key"].is_unique
    assert table["district_key"].str.contains(r"\|\|").all()
    assert table[PREDICTION_COLUMN].notna().all()
    assert np.all(np.isfinite(table[PREDICTION_COLUMN].to_numpy(dtype=float)))
    assert set(table["model_id"]) == {"ridge_b_alpha100_v1"}
    assert table["prediction_type"].str.contains("refit_in_sample").all()
    assert (table["n_features"] == 77).all()
    assert table["state_name"].notna().all()
    assert table["district_name"].notna().all()
    assert "observed" not in " ".join(table.columns).lower(), "actual values must stay out of this file"


def test_predictions_match_the_saved_model():
    table = pd.read_csv(PREDICTION_FILE)
    X, _, keys, _ = approved_training_data()
    scorer = load_scorer()
    expected = scorer["pipeline"].predict(X)
    by_key = dict(zip(keys, expected, strict=True))
    published = table.set_index("district_key")[PREDICTION_COLUMN]
    assert len(published) == 628
    for key, value in published.items():
        assert abs(value - by_key[key]) <= TOL, (key, value, by_key[key])


def test_retrospective_file_separates_observed_values():
    retrospective = pd.read_csv(RETROSPECTIVE_FILE)
    assert len(retrospective) == 628
    assert "observed_rate_pct_2011" in retrospective.columns
    assert retrospective["use"].str.contains("RETROSPECTIVE ONLY").all()
    predictions = pd.read_csv(PREDICTION_FILE)
    assert "observed_rate_pct_2011" not in predictions.columns
    residual = retrospective["residual_observed_minus_predicted"]
    assert np.allclose(
        residual.to_numpy(),
        retrospective["observed_rate_pct_2011"].to_numpy() - retrospective[PREDICTION_COLUMN].to_numpy(),
        atol=1e-9,
    )


def test_out_of_range_values_are_flagged_not_clipped():
    table = pd.read_csv(PREDICTION_FILE)
    values = table[PREDICTION_COLUMN].to_numpy(dtype=float)
    outside = (values < 0) | (values > 100)
    flagged = table["physical_range_flag"].to_numpy() != "ok"
    assert np.array_equal(outside, flagged), "every out-of-range value must be flagged"
    # the flag exists precisely because such values occur; no floor or ceiling was applied
    assert np.all(values > -100) and np.all(values < 100)


def test_ranking_is_sorted_unique_and_complete():
    ranking = pd.read_csv(RANKING_FILE)
    predictions = pd.read_csv(PREDICTION_FILE)
    assert len(ranking) == 628
    assert sorted(ranking["rank"].tolist()) == list(range(1, 629)), "ranks must be 1..628, unique"
    assert set(ranking["district_key"]) == set(predictions["district_key"])
    values = ranking[PREDICTION_COLUMN].to_numpy(dtype=float)
    assert np.all(np.diff(values) <= 1e-12), "ranking must be non-increasing in predicted prevalence"
    assert (np.diff(values) < -1e-9).sum() > 500, "ranking appears not to be sorted at all"
    # tie policy: equal predictions ordered by district key ascending
    for _, group in ranking.groupby(PREDICTION_COLUMN):
        if len(group) > 1:
            assert group["district_key"].tolist() == sorted(group["district_key"].tolist())
    assert ranking["rank_scope"].str.contains("628").all()


def test_relative_bands_are_quintiles_of_the_prediction_only():
    ranking = pd.read_csv(RANKING_FILE)
    assert ranking["relative_band"].nunique() == N_BANDS
    counts = ranking["relative_band"].value_counts()
    assert counts.max() - counts.min() <= 1, counts.to_dict()
    # band membership is exactly the quintile of the predicted value (rank 1 is
    # the HIGHEST prediction, so the band must come from an ascending-value rank)
    positions = ranking[PREDICTION_COLUMN].rank(method="first", ascending=True)
    expected = pd.qcut(positions, N_BANDS, labels=BAND_LABELS).astype(str)
    assert (ranking["relative_band"].to_numpy() == expected.to_numpy()).all()
    # and the bands are monotone in predicted prevalence
    order = {label: index for index, label in enumerate(BAND_LABELS)}
    stats = ranking.groupby("relative_band")[PREDICTION_COLUMN].agg(["min", "max"])
    stats["order"] = [order[label] for label in stats.index]
    stats = stats.sort_values("order")
    assert np.all(np.diff(stats["min"].to_numpy()) > 0)
    # every band's highest member is below the next band's lowest member
    assert np.all(stats["max"].to_numpy()[:-1] <= stats["min"].to_numpy()[1:] + 1e-12)
    assert ranking["band_basis"].str.contains("not a validated risk class").all()


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
