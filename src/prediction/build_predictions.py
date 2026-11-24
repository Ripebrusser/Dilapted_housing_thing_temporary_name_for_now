"""Phase 6 deliverables: district predictions, ranking and relative bands.

Produces three files from the persisted Ridge B model:

``evaluation/predictions/final_predictions_ridge_b_628.csv``
    one row per eligible district with its predicted prevalence.
``evaluation/predictions/district_ranking_ridge_b_628.csv``
    the same districts ranked by predicted prevalence, highest first.
``evaluation/predictions/final_predictions_retrospective_628.csv``
    predictions beside the observed 2011 target, for descriptive comparison
    only. Clearly separated because mixing them invites the reader to treat
    in-sample fit as validation.

Ranking policy
--------------
Descending predicted prevalence; ties broken by district key ascending so the
order is deterministic; ranks are 1..628 with no duplicate values. The rank is
a position in *this* 628-district list, not a national percentile.

Band policy
-----------
``relative_band`` splits the predicted values into five equal-sized groups by
rank (B1 lowest .. B5 highest predicted). These are **relative ranking bands**,
not validated risk classes: no external threshold for "severe" district
prevalence exists in the sources used here, so inventing absolute cut-points
would be unfounded. Bands are computed from predictions only — never from the
observed target.

Usage: py -3.14 src/prediction/build_predictions.py [--refit]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.prediction.final_model import (  # noqa: E402
    METADATA_PATH,
    MODEL_PATH,
    approved_training_data,
    fit_final_model,
    load_artifacts,
    save_artifacts,
)
from src.prediction.inference import PREDICTION_COLUMN, predict_districts  # noqa: E402

PREDICTIONS = ROOT / "evaluation" / "predictions"
PREDICTION_FILE = PREDICTIONS / "final_predictions_ridge_b_628.csv"
RANKING_FILE = PREDICTIONS / "district_ranking_ridge_b_628.csv"
RETROSPECTIVE_FILE = PREDICTIONS / "final_predictions_retrospective_628.csv"
CROSSWALK = ROOT / "datasets" / "metadata" / "district_crosswalk.csv"

BAND_LABELS = ["B1 (lowest predicted)", "B2", "B3", "B4", "B5 (highest predicted)"]
N_BANDS = 5
N_DISTRICTS = 628


def district_frame() -> pd.DataFrame:
    """Approved features plus identifiers and the observed target kept aside."""
    X, y, keys, groups = approved_training_data()
    frame = X.copy()
    frame.insert(0, "key", keys)
    frame.insert(1, "state", groups)

    names = {}
    actual = {}
    if CROSSWALK.exists():
        crosswalk = pd.read_csv(CROSSWALK)
        names = dict(zip(crosswalk["key"], crosswalk["district_name"], strict=False))
        actual = dict(zip(crosswalk["key"], crosswalk["target_deprivation_rate"], strict=False))
    frame.insert(2, "district", [names.get(key, str(key).split("||", 1)[-1]) for key in keys])
    frame.attrs["observed"] = [actual.get(key, np.nan) for key in keys]
    return frame


def main(argv: list[str]) -> int:
    refit = "--refit" in argv
    if refit or not MODEL_PATH.exists():
        pipeline, metadata = fit_final_model()
        save_artifacts(pipeline, metadata)
        print("fitted and saved a fresh model")
    pipeline, metadata = load_artifacts()

    frame = district_frame()
    observed = np.asarray(frame.attrs["observed"], dtype=float)
    features_only = frame.drop(columns=["key", "state", "district"])
    scored = predict_districts(features_only, scorer={"pipeline": pipeline, "metadata": metadata,
                                                      "features": metadata["features"]})
    scored.insert(0, "key", frame["key"].to_numpy())
    scored.insert(1, "district", frame["district"].to_numpy())
    scored.insert(2, "state", frame["state"].to_numpy())
    scored = scored.drop(columns=["units"])

    if len(scored) != N_DISTRICTS:
        raise SystemExit(f"STOP: expected {N_DISTRICTS} predictions, got {len(scored)}")
    if scored["key"].duplicated().any():
        raise SystemExit("STOP: duplicate district keys in the prediction output")
    if not np.all(np.isfinite(scored[PREDICTION_COLUMN].to_numpy())):
        raise SystemExit("STOP: non-finite prediction")

    predictions = scored.rename(columns={"key": "district_key", "state": "state_name", "district": "district_name"})
    predictions["configuration"] = metadata["configuration"]
    predictions["n_features"] = metadata["n_features"]
    predictions["model_version"] = metadata["model_version"]
    predictions["coverage_note"] = "628 of 640 districts; 12 have no resolvable household denominator"
    # Ridge is an unconstrained linear model, so a fitted value can sit just
    # outside the physically admissible [0, 100] range. Those values are kept
    # exactly as computed and flagged: clipping them would hide a real property
    # of the ratified model, and a floor at 0 cannot change the ranking anyway.
    values = predictions[PREDICTION_COLUMN]
    flag = np.where(values < 0, "below 0 (linear extrapolation)",
                    np.where(values > 100, "above 100 (linear extrapolation)", "ok"))
    predictions["physical_range_flag"] = flag

    predictions = predictions[[
        "district_key", "district_name", "state_name", PREDICTION_COLUMN, "physical_range_flag",
        "model_id", "prediction_type", "configuration", "n_features", "model_version", "coverage_note",
    ]]
    predictions.to_csv(PREDICTION_FILE, index=False)
    n_flagged = int((predictions["physical_range_flag"] != "ok").sum())

    ranking = predictions[["district_key", "district_name", "state_name", PREDICTION_COLUMN,
                           "physical_range_flag"]].copy()
    ranking = ranking.sort_values(
        [PREDICTION_COLUMN, "district_key"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    rank_by_position = ranking[PREDICTION_COLUMN].rank(method="first")
    ranking["relative_band"] = pd.qcut(rank_by_position, N_BANDS, labels=BAND_LABELS).astype(str)
    ranking["rank_scope"] = "position among the 628 eligible districts, highest predicted first"
    ranking["band_basis"] = "relative predicted quintile (equal-sized by rank), not a validated risk class"
    ranking.to_csv(RANKING_FILE, index=False)

    retrospective = predictions[["district_key", "district_name", "state_name", PREDICTION_COLUMN,
                                 "physical_range_flag"]].copy()
    retrospective["observed_rate_pct_2011"] = observed
    retrospective["residual_observed_minus_predicted"] = observed - retrospective[PREDICTION_COLUMN]
    retrospective["use"] = "RETROSPECTIVE ONLY - in-sample fitted values, not a validation result"
    retrospective.to_csv(RETROSPECTIVE_FILE, index=False)

    bands = ranking["relative_band"].value_counts().sort_index()
    print(f"predictions  {PREDICTION_FILE.relative_to(ROOT)}  ({len(predictions)} districts)")
    print(f"ranking      {RANKING_FILE.relative_to(ROOT)}  ({ranking['rank'].nunique()} unique ranks)")
    print(f"retrospective{RETROSPECTIVE_FILE.relative_to(ROOT)}  (validation NOT performed here)")
    print()
    print(f"predicted prevalence: {predictions[PREDICTION_COLUMN].min():.3f} to "
          f"{predictions[PREDICTION_COLUMN].max():.3f} pp, mean "
          f"{predictions[PREDICTION_COLUMN].mean():.3f}")
    print(f"outside the physical [0, 100] range: {n_flagged} of {len(predictions)} districts "
          "(reported as computed, not clipped)")
    print("bands:", ", ".join(f"{label}={count}" for label, count in bands.items()))
    print()
    print("top 8 by predicted prevalence:")
    print(ranking.head(8).to_string(index=False))
    print()
    print("bottom 3:")
    print(ranking.tail(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
